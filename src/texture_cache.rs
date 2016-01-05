use app_units::Au;
use device::{MAX_TEXTURE_SIZE, TextureId, TextureFilter};
use euclid::{Point2D, Rect, Size2D};
use fnv::FnvHasher;
use freelist::{FreeList, FreeListItem, FreeListItemId};
use internal_types::{TextureUpdate, TextureUpdateOp, TextureUpdateDetails};
use internal_types::{RasterItem, RenderTargetMode, TextureImage, TextureUpdateList};
use internal_types::{BasicRotationAngle, RectUv};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::collections::hash_state::DefaultState;
use std::mem;
use std::slice::Iter;
use tessellator::BorderCornerTessellation;
use util;
use webrender_traits::{ImageFormat};

/// The number of bytes we're allowed to use for a texture.
const MAX_BYTES_PER_TEXTURE: u32 = 64 * 1024 * 1024;

/// The number of RGBA pixels we're allowed to use for a texture.
const MAX_RGBA_PIXELS_PER_TEXTURE: u32 = MAX_BYTES_PER_TEXTURE / 4;

/// The square root of the number of RGBA pixels we're allowed to use for a texture, rounded down.
/// to the next power of two.
const SQRT_MAX_RGBA_PIXELS_PER_TEXTURE: u32 = 4096;

/// The minimum number of pixels on each side that we require for rects to be classified as
/// "medium" within the free list.
const MINIMUM_MEDIUM_RECT_SIZE: u32 = 16;

/// The minimum number of pixels on each side that we require for rects to be classified as
/// "large" within the free list.
const MINIMUM_LARGE_RECT_SIZE: u32 = 32;

pub type TextureCacheItemId = FreeListItemId;

#[derive(Copy, Clone, Debug)]
pub enum BorderType {
    NoBorder,
    SinglePixel,
}

#[inline]
fn copy_pixels(src: &[u8],
               target: &mut Vec<u8>,
               x: u32,
               y: u32,
               count: u32,
               width: u32,
               bpp: u32) {
    let pixel_index = (y * width + x) * bpp;
    for byte in src.iter().skip(pixel_index as usize).take((count * bpp) as usize) {
        target.push(*byte);
    }
}

/// A texture allocator using the guillotine algorithm with the rectangle merge improvement. See
/// sections 2.2 and 2.2.5 in "A Thousand Ways to Pack the Bin - A Practical Approach to Two-
/// Dimensional Rectangle Bin Packing":
///
///    http://clb.demon.fi/files/RectangleBinPack.pdf
///
/// This approach was chosen because of its simplicity, good performance, and easy support for
/// dynamic texture deallocation.
pub struct TexturePage {
    texture_id: TextureId,
    texture_size: u32,
    free_list: FreeRectList,
    allocations: u32,
    dirty: bool,
}

impl TexturePage {
    pub fn new(texture_id: TextureId, texture_size: u32) -> TexturePage {
        let mut page = TexturePage {
            texture_id: texture_id,
            texture_size: texture_size,
            free_list: FreeRectList::new(),
            allocations: 0,
            dirty: false,
        };
        page.clear();
        page
    }

    fn find_index_of_best_rect_in_bin(&self, bin: FreeListBin, requested_dimensions: &Size2D<u32>)
                                      -> Option<FreeListIndex> {
        let mut smallest_index_and_area = None;
        for (candidate_index, candidate_rect) in self.free_list.iter(bin).enumerate() {
            if !requested_dimensions.fits_inside(&candidate_rect.size) {
                continue
            }

            let candidate_area = candidate_rect.size.width * candidate_rect.size.height;
            smallest_index_and_area = Some((candidate_index, candidate_area));
            break
        }

        smallest_index_and_area.map(|(index, _)| FreeListIndex(bin, index))
    }

    fn find_index_of_best_rect(&self, requested_dimensions: &Size2D<u32>)
                               -> Option<FreeListIndex> {
        match FreeListBin::for_size(requested_dimensions) {
            FreeListBin::Large => {
                self.find_index_of_best_rect_in_bin(FreeListBin::Large, requested_dimensions)
            }
            FreeListBin::Medium => {
                match self.find_index_of_best_rect_in_bin(FreeListBin::Medium,
                                                          requested_dimensions) {
                    Some(index) => Some(index),
                    None => {
                        self.find_index_of_best_rect_in_bin(FreeListBin::Large,
                                                            requested_dimensions)
                    }
                }
            }
            FreeListBin::Small => {
                match self.find_index_of_best_rect_in_bin(FreeListBin::Small,
                                                          requested_dimensions) {
                    Some(index) => Some(index),
                    None => {
                        match self.find_index_of_best_rect_in_bin(FreeListBin::Medium,
                                                                  requested_dimensions) {
                            Some(index) => Some(index),
                            None => {
                                self.find_index_of_best_rect_in_bin(FreeListBin::Large,
                                                                    requested_dimensions)
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn allocate(&mut self,
                    requested_dimensions: &Size2D<u32>,
                    filter: TextureFilter) -> Option<Point2D<u32>> {
        // TODO(gw): For now, anything that requests nearest filtering
        //           just fails to allocate in a texture page, and gets a standalone
        //           texture. This isn't ideal, as it causes lots of batch breaks,
        //           but is probably rare enough that it can be fixed up later (it's also
        //           fairly trivial to implement, just tedious).
        match filter {
            TextureFilter::Nearest => {
                return None;
            }
            TextureFilter::Linear => {}
        }

        // First, try to find a suitable rect in the free list. We choose the smallest such rect
        // in terms of area (Best-Area-Fit, BAF).
        let mut index = self.find_index_of_best_rect(requested_dimensions);

        // If one couldn't be found and we're dirty, coalesce rects and try again.
        if index.is_none() && self.dirty {
            self.coalesce();
            index = self.find_index_of_best_rect(requested_dimensions)
        }

        // If a rect still can't be found, fail.
        let index = match index {
            None => return None,
            Some(index) => index,
        };

        // Remove the rect from the free list and decide how to guillotine it. We choose the split
        // that results in the single largest area (Min Area Split Rule, MINAS).
        let chosen_rect = self.free_list.remove(index);
        let candidate_free_rect_to_right =
            Rect::new(Point2D::new(chosen_rect.origin.x + requested_dimensions.width,
                                   chosen_rect.origin.y),
                      Size2D::new(chosen_rect.size.width - requested_dimensions.width,
                                  requested_dimensions.height));
        let candidate_free_rect_to_bottom =
            Rect::new(Point2D::new(chosen_rect.origin.x,
                                   chosen_rect.origin.y + requested_dimensions.height),
                      Size2D::new(requested_dimensions.width,
                                  chosen_rect.size.height - requested_dimensions.height));
        let candidate_free_rect_to_right_area = candidate_free_rect_to_right.size.width *
            candidate_free_rect_to_right.size.height;
        let candidate_free_rect_to_bottom_area = candidate_free_rect_to_bottom.size.width *
            candidate_free_rect_to_bottom.size.height;

        // Guillotine the rectangle.
        let new_free_rect_to_right;
        let new_free_rect_to_bottom;
        if candidate_free_rect_to_right_area > candidate_free_rect_to_bottom_area {
            new_free_rect_to_right = Rect::new(candidate_free_rect_to_right.origin,
                                               Size2D::new(candidate_free_rect_to_right.size.width,
                                                           chosen_rect.size.height));
            new_free_rect_to_bottom = candidate_free_rect_to_bottom
        } else {
            new_free_rect_to_right = candidate_free_rect_to_right;
            new_free_rect_to_bottom =
                Rect::new(candidate_free_rect_to_bottom.origin,
                          Size2D::new(chosen_rect.size.width,
                                      candidate_free_rect_to_bottom.size.height))
        }

        // Add the guillotined rects back to the free list. If any changes were made, we're now
        // dirty since coalescing might be able to defragment.
        if !util::rect_is_empty(&new_free_rect_to_right) {
            self.free_list.push(&new_free_rect_to_right);
            self.dirty = true
        }
        if !util::rect_is_empty(&new_free_rect_to_bottom) {
            self.free_list.push(&new_free_rect_to_bottom);
            self.dirty = true
        }

        // Bump the allocation counter.
        self.allocations += 1;

        // Return the result.
        Some(chosen_rect.origin)
    }

    #[inline(never)]
    fn coalesce(&mut self) {
        // Iterate to a fixed point.
        let mut free_list = mem::replace(&mut self.free_list, FreeRectList::new()).into_vec();
        let mut changed = false;

        // Combine rects that have the same width and are adjacent.
        let mut new_free_list = Vec::new();
        free_list.sort_by(|a, b| {
            match a.size.width.cmp(&b.size.width) {
                Ordering::Equal => a.origin.x.cmp(&b.origin.x),
                ordering => ordering,
            }
        });
        for work_index in 0..free_list.len() {
            if free_list[work_index].size.width == 0 {
                continue
            }
            for candidate_index in (work_index + 1)..free_list.len() {
                if free_list[work_index].size.width != free_list[candidate_index].size.width ||
                        free_list[work_index].origin.x != free_list[candidate_index].origin.x {
                    break
                }
                if free_list[work_index].origin.y == free_list[candidate_index].max_y() ||
                        free_list[work_index].max_y() == free_list[candidate_index].origin.y {
                    changed = true;
                    free_list[work_index] =
                        free_list[work_index].union(&free_list[candidate_index]);
                    free_list[candidate_index].size.width = 0
                }
                new_free_list.push(free_list[work_index])
            }
            new_free_list.push(free_list[work_index])
        }
        free_list = new_free_list;

        // Combine rects that have the same height and are adjacent.
        let mut new_free_list = Vec::new();
        free_list.sort_by(|a, b| {
            match a.size.height.cmp(&b.size.height) {
                Ordering::Equal => a.origin.y.cmp(&b.origin.y),
                ordering => ordering,
            }
        });
        for work_index in 0..free_list.len() {
            if free_list[work_index].size.height == 0 {
                continue
            }
            for candidate_index in (work_index + 1)..free_list.len() {
                if free_list[work_index].size.height !=
                        free_list[candidate_index].size.height ||
                        free_list[work_index].origin.y != free_list[candidate_index].origin.y {
                    break
                }
                if free_list[work_index].origin.x == free_list[candidate_index].max_x() ||
                        free_list[work_index].max_x() == free_list[candidate_index].origin.x {
                    changed = true;
                    free_list[work_index] =
                        free_list[work_index].union(&free_list[candidate_index]);
                    free_list[candidate_index].size.height = 0
                }
            }
            new_free_list.push(free_list[work_index])
        }
        free_list = new_free_list;

        self.free_list = FreeRectList::from_slice(&free_list[..]);
        self.dirty = changed
    }

    fn clear(&mut self) {
        self.free_list = FreeRectList::new();
        self.free_list.push(&Rect::new(Point2D::new(0, 0),
                                       Size2D::new(self.texture_size, self.texture_size)));
        self.allocations = 0;
        self.dirty = false;
    }

/*
    fn free(&mut self, rect: &Rect<u32>) {
        debug_assert!(self.allocations > 0);
        self.allocations -= 1;
        if self.allocations == 0 {
            self.clear();
            return
        }

        self.free_list.push(rect);
        self.dirty = true
    }*/
}

// TODO(gw): This is used to store data specific to glyphs.
//           Perhaps find a better place to store it.
#[derive(Debug, Clone)]
pub struct TextureCacheItemUserData {
    pub x0: i32,
    pub y0: i32,
}

/// A binning free list. Binning is important to avoid sifting through lots of small strips when
/// allocating many texture items.
struct FreeRectList {
    small: Vec<Rect<u32>>,
    medium: Vec<Rect<u32>>,
    large: Vec<Rect<u32>>,
}

impl FreeRectList {
    fn new() -> FreeRectList {
        FreeRectList {
            small: vec![],
            medium: vec![],
            large: vec![],
        }
    }

    fn from_slice(vector: &[Rect<u32>]) -> FreeRectList {
        let mut free_list = FreeRectList::new();
        for rect in vector {
            free_list.push(rect)
        }
        free_list
    }

    fn push(&mut self, rect: &Rect<u32>) {
        match FreeListBin::for_size(&rect.size) {
            FreeListBin::Small => self.small.push(*rect),
            FreeListBin::Medium => self.medium.push(*rect),
            FreeListBin::Large => self.large.push(*rect),
        }
    }

    fn remove(&mut self, index: FreeListIndex) -> Rect<u32> {
        match index.0 {
            FreeListBin::Small => self.small.swap_remove(index.1),
            FreeListBin::Medium => self.medium.swap_remove(index.1),
            FreeListBin::Large => self.large.swap_remove(index.1),
        }
    }

    fn iter(&self, bin: FreeListBin) -> Iter<Rect<u32>> {
        match bin {
            FreeListBin::Small => self.small.iter(),
            FreeListBin::Medium => self.medium.iter(),
            FreeListBin::Large => self.large.iter(),
        }
    }

    fn into_vec(mut self) -> Vec<Rect<u32>> {
        self.small.extend(self.medium.drain(..));
        self.small.extend(self.large.drain(..));
        self.small
    }
}

#[derive(Debug, Clone, Copy)]
struct FreeListIndex(FreeListBin, usize);

#[derive(Debug, Clone, Copy, PartialEq)]
enum FreeListBin {
    Small,
    Medium,
    Large,
}

impl FreeListBin {
    pub fn for_size(size: &Size2D<u32>) -> FreeListBin {
        if size.width >= MINIMUM_LARGE_RECT_SIZE && size.height >= MINIMUM_LARGE_RECT_SIZE {
            FreeListBin::Large
        } else if size.width >= MINIMUM_MEDIUM_RECT_SIZE &&
                size.height >= MINIMUM_MEDIUM_RECT_SIZE {
            FreeListBin::Medium
        } else {
            FreeListBin::Small
        }
    }
}

#[derive(Debug, Clone)]
pub struct TextureCacheItem {
    // Identifies the texture and array slice
    pub texture_id: TextureId,

    // Custom data - unused by texture cache itself
    pub user_data: TextureCacheItemUserData,

    // The texture coordinates for this item
    pub uv_rect: RectUv,

    // The size of the entire texture (not just the allocated rectangle)
    pub texture_size: Size2D<u32>,

    // The size of the actual allocated rectangle,
    // and the requested size. The allocated size
    // is the same as the requested in most cases,
    // unless the item has a border added for
    // bilinear filtering / texture bleeding purposes.
    pub allocated_rect: Rect<u32>,
    pub requested_rect: Rect<u32>,
}

// Structure squat the width/height fields to maintain the free list information :)
impl FreeListItem for TextureCacheItem {
    fn next_free_id(&self) -> Option<FreeListItemId> {
        if self.requested_rect.size.width == 0 {
            debug_assert!(self.requested_rect.size.height == 0);
            None
        } else {
            debug_assert!(self.requested_rect.size.width == 1);
            Some(FreeListItemId::new(self.requested_rect.size.height))
        }
    }

    fn set_next_free_id(&mut self, id: Option<FreeListItemId>) {
        match id {
            Some(id) => {
                self.requested_rect.size.width = 1;
                self.requested_rect.size.height = id.value();
            }
            None => {
                self.requested_rect.size.width = 0;
                self.requested_rect.size.height = 0;
            }
        }
    }
}

impl TextureCacheItem {
    fn new(texture_id: TextureId,
           user_x0: i32, user_y0: i32,
           allocated_rect: Rect<u32>,
           requested_rect: Rect<u32>,
           texture_size: &Size2D<u32>,
           uv_rect: RectUv)
           -> TextureCacheItem {
        TextureCacheItem {
            texture_id: texture_id,
            texture_size: *texture_size,
            uv_rect: uv_rect,
            user_data: TextureCacheItemUserData {
                x0: user_x0,
                y0: user_y0,
            },
            allocated_rect: allocated_rect,
            requested_rect: requested_rect,
        }
    }

    fn to_image(&self) -> TextureImage {
        let texture_size = texture_size() as f32;
        TextureImage {
            texture_id: self.texture_id,
            texel_uv: Rect::new(Point2D::new(self.uv_rect.top_left.x,
                                             self.uv_rect.top_left.y),
                                Size2D::new(self.uv_rect.bottom_right.x - self.uv_rect.top_left.x,
                                            self.uv_rect.bottom_right.y - self.uv_rect.top_left.y)),
            pixel_uv: Point2D::new((self.uv_rect.top_left.x * texture_size) as u32,
                                   (self.uv_rect.top_left.y * texture_size) as u32),
        }
    }
}

struct TextureCacheArena {
    pages_a8: Vec<TexturePage>,
    pages_rgb8: Vec<TexturePage>,
    pages_rgba8: Vec<TexturePage>,
    alternate_pages_a8: Vec<TexturePage>,
    alternate_pages_rgba8: Vec<TexturePage>,
    //render_target_pages: Vec<TexturePage>,
}

impl TextureCacheArena {
    fn new() -> TextureCacheArena {
        TextureCacheArena {
            pages_a8: Vec::new(),
            pages_rgb8: Vec::new(),
            pages_rgba8: Vec::new(),
            alternate_pages_a8: Vec::new(),
            alternate_pages_rgba8: Vec::new(),
            //render_target_pages: Vec::new(),
        }
    }
}

pub struct TextureCache {
    free_texture_ids: Vec<TextureId>,
    free_texture_levels: HashMap<ImageFormat, Vec<FreeTextureLevel>, DefaultState<FnvHasher>>,
    alternate_free_texture_levels: HashMap<ImageFormat,
                                           Vec<FreeTextureLevel>,
                                           DefaultState<FnvHasher>>,
    //render_target_free_texture_levels: HashMap<ImageFormat,
    //                                           Vec<FreeTextureLevel>,
    //                                           DefaultState<FnvHasher>>,
    items: FreeList<TextureCacheItem>,
    arena: TextureCacheArena,
    pending_updates: TextureUpdateList,
}

#[derive(PartialEq, Eq, Debug)]
pub enum AllocationKind {
    TexturePage,
    Standalone,
}

#[derive(Debug)]
pub struct AllocationResult {
    kind: AllocationKind,
    item: TextureCacheItem,
}

impl TextureCache {
    pub fn new(free_texture_ids: Vec<TextureId>) -> TextureCache {
        TextureCache {
            free_texture_ids: free_texture_ids,
            free_texture_levels: HashMap::with_hash_state(Default::default()),
            alternate_free_texture_levels: HashMap::with_hash_state(Default::default()),
            //render_target_free_texture_levels: HashMap::with_hash_state(Default::default()),
            items: FreeList::new(),
            pending_updates: TextureUpdateList::new(),
            arena: TextureCacheArena::new(),
        }
    }

    pub fn pending_updates(&mut self) -> TextureUpdateList {
        mem::replace(&mut self.pending_updates, TextureUpdateList::new())
    }

    // TODO(gw): This API is a bit ugly (having to allocate an ID and
    //           then use it). But it has to be that way for now due to
    //           how the raster_jobs code works.
    pub fn new_item_id(&mut self) -> TextureCacheItemId {
        let new_item = TextureCacheItem {
            uv_rect: RectUv {
                top_left: Point2D::zero(),
                top_right: Point2D::zero(),
                bottom_left: Point2D::zero(),
                bottom_right: Point2D::zero(),
            },
            user_data: TextureCacheItemUserData {
                x0: 0,
                y0: 0,
            },
            allocated_rect: Rect::zero(),
            requested_rect: Rect::zero(),
            texture_size: Size2D::zero(),
            texture_id: TextureId::invalid(),
        };
        self.items.insert(new_item)
    }

    /*
    pub fn free(&mut self,
                _texture_id: TextureId,
                _uv: &Rect<u32>,
                _kind: TextureCacheItemKind) {
        panic!("can't free texture items yet!");
    }
    */

    pub fn allocate_render_target(&mut self,
                              width: u32,
                              height: u32,
                              format: ImageFormat) -> TextureId {
        let texture_id = self.free_texture_ids.pop().expect("TODO: Handle running out of texture IDs!");
        let op = TextureUpdateOp::Create(width,
                                         height,
                                         format,
                                         TextureFilter::Linear,
                                         RenderTargetMode::RenderTarget,
                                         None);
        let update_op = TextureUpdate {
            id: texture_id,
            op: op,
        };
        self.pending_updates.push(update_op);
        texture_id
    }

    pub fn free_render_target(&mut self, texture_id: TextureId) {
        let op = TextureUpdateOp::Update(0,
                                         0,
                                         0,
                                         0,
                                         TextureUpdateDetails::Blit(Vec::new()));
        let update_op = TextureUpdate {
            id: texture_id,
            op: op,
        };
        self.pending_updates.push(update_op);
        self.free_texture_ids.push(texture_id);
    }

    pub fn allocate(&mut self,
                    image_id: TextureCacheItemId,
                    user_x0: i32,
                    user_y0: i32,
                    requested_width: u32,
                    requested_height: u32,
                    format: ImageFormat,
                    kind: TextureCacheItemKind,
                    border_type: BorderType,
                    filter: TextureFilter)
                    -> AllocationResult {
        let (page_list, mode) = match (format, kind) {
            (ImageFormat::A8, TextureCacheItemKind::Standard) => {
                (&mut self.arena.pages_a8, RenderTargetMode::RenderTarget)
            }
            (ImageFormat::A8, TextureCacheItemKind::Alternate) => {
                (&mut self.arena.alternate_pages_a8, RenderTargetMode::RenderTarget)
            }
            (ImageFormat::RGBA8, TextureCacheItemKind::Standard) => {
                (&mut self.arena.pages_rgba8, RenderTargetMode::RenderTarget)
            }
            (ImageFormat::RGBA8, TextureCacheItemKind::Alternate) => {
                (&mut self.arena.alternate_pages_rgba8, RenderTargetMode::RenderTarget)
            }
            //(ImageFormat::RGBA8, TextureCacheItemKind::RenderTarget) => {
            //    (&mut self.arena.render_target_pages, RenderTargetMode::RenderTarget)
            //}
            (ImageFormat::RGB8, TextureCacheItemKind::Standard) => {
                (&mut self.arena.pages_rgb8, RenderTargetMode::None)
            }
            (ImageFormat::Invalid, TextureCacheItemKind::Standard) |
            (_, TextureCacheItemKind::Alternate) => unreachable!(),
        };

        let border_size = match border_type {
            BorderType::NoBorder => 0,
            BorderType::SinglePixel => 1,
        };
        let requested_size = Size2D::new(requested_width, requested_height);
        let allocation_size = Size2D::new(requested_width + border_size * 2,
                                          requested_height + border_size * 2);
        let location = page_list.last_mut().and_then(|last_page| {
            last_page.allocate(&allocation_size, filter)
        });
        let location = match location {
            Some(location) => location,
            None => {
                // We need a new page.
                let texture_size = texture_size();
                let texture_id = {
                    let free_texture_levels_entry = match kind {
                        TextureCacheItemKind::Standard => self.free_texture_levels.entry(format),
                        TextureCacheItemKind::Alternate => {
                            self.alternate_free_texture_levels.entry(format)
                        }
                        //TextureCacheItemKind::RenderTarget => {
                        //    self.render_target_free_texture_levels.entry(format)
                        //}
                    };
                    let mut free_texture_levels = match free_texture_levels_entry {
                        Entry::Vacant(entry) => entry.insert(Vec::new()),
                        Entry::Occupied(entry) => entry.into_mut(),
                    };
                    if free_texture_levels.is_empty() {
                        create_new_texture_page(&mut self.pending_updates,
                                                &mut self.free_texture_ids,
                                                &mut free_texture_levels,
                                                texture_size,
                                                format,
                                                mode);
                    }
                    let free_texture_level = free_texture_levels.pop().unwrap();
                    free_texture_level.texture_id
                };

                let page = TexturePage::new(texture_id, texture_size);
                page_list.push(page);

                match page_list.last_mut().unwrap().allocate(&allocation_size, filter) {
                    Some(location) => location,
                    None => {
                        // Fall back to standalone texture allocation.
                        let texture_id = self.free_texture_ids
                                             .pop()
                                             .expect("TODO: Handle running out of texture ids!");
                        let cache_item = TextureCacheItem::new(
                            texture_id,
                            user_x0, user_y0,
                            Rect::new(Point2D::zero(), requested_size),
                            Rect::new(Point2D::zero(), requested_size),
                            &requested_size,
                            RectUv {
                                top_left: Point2D::new(0.0, 0.0),
                                top_right: Point2D::new(1.0, 0.0),
                                bottom_left: Point2D::new(0.0, 1.0),
                                bottom_right: Point2D::new(1.0, 1.0),
                            });
                        *self.items.get_mut(image_id) = cache_item;

                        return AllocationResult {
                            item: self.items.get(image_id).clone(),
                            kind: AllocationKind::Standalone,
                        }
                    }
                }
            }
        };

        let page = page_list.last_mut().unwrap();

        let allocated_rect = Rect::new(location, allocation_size);
        let requested_rect = Rect::new(Point2D::new(location.x + border_size,
                                                    location.y + border_size),
                                       requested_size);

        let u0 = requested_rect.origin.x as f32 / page.texture_size as f32;
        let v0 = requested_rect.origin.y as f32 / page.texture_size as f32;
        let u1 = u0 + requested_size.width as f32 / page.texture_size as f32;
        let v1 = v0 + requested_size.height as f32 / page.texture_size as f32;
        let cache_item = TextureCacheItem::new(page.texture_id,
                                               user_x0, user_y0,
                                               allocated_rect,
                                               requested_rect,
                                               &Size2D::new(page.texture_size,
                                                            page.texture_size),
                                               RectUv {
                                                    top_left: Point2D::new(u0, v0),
                                                    top_right: Point2D::new(u1, v0),
                                                    bottom_left: Point2D::new(u0, v1),
                                                    bottom_right: Point2D::new(u1, v1)
                                               });
        *self.items.get_mut(image_id) = cache_item;

        AllocationResult {
            item: self.items.get(image_id).clone(),
            kind: AllocationKind::TexturePage,
        }
    }

    pub fn insert_raster_op(&mut self,
                            image_id: TextureCacheItemId,
                            item: &RasterItem,
                            device_pixel_ratio: f32) {
        let update_op = match item {
            &RasterItem::BorderRadius(ref op) => {
                let rect = Rect::new(Point2D::new(0.0, 0.0),
                                     Size2D::new(op.outer_radius_x.to_f32_px(),
                                                 op.outer_radius_y.to_f32_px()));
                let tessellated_rect = rect.tessellate_border_corner(
                    &Size2D::new(op.outer_radius_x.to_f32_px(), op.outer_radius_y.to_f32_px()),
                    &Size2D::new(op.inner_radius_x.to_f32_px(), op.inner_radius_y.to_f32_px()),
                    BasicRotationAngle::Upright,
                    op.index);
                let width = tessellated_rect.size.width.round() as u32;
                let height = tessellated_rect.size.height.round() as u32;

                let allocation = self.allocate(image_id,
                                               0,
                                               0,
                                               width,
                                               height,
                                               op.image_format,
                                               TextureCacheItemKind::Standard,
                                               BorderType::NoBorder,
                                               TextureFilter::Linear);

                assert!(allocation.kind == AllocationKind::TexturePage);        // TODO: Handle large border radii not fitting in texture cache page

                TextureUpdate {
                    id: allocation.item.texture_id,
                    op: TextureUpdateOp::Update(allocation.item.requested_rect.origin.x,
                                                allocation.item.requested_rect.origin.y,
                                                width,
                                                height,
                                                TextureUpdateDetails::BorderRadius(
                                                    op.outer_radius_x,
                                                    op.outer_radius_y,
                                                    op.inner_radius_x,
                                                    op.inner_radius_y,
                                                    op.index,
                                                    op.inverted)),
                }
            }
            &RasterItem::BoxShadow(ref op) => {
                let device_raster_size =
                    Size2D::new((op.raster_size.0.to_nearest_px() as f32 *
                                 device_pixel_ratio) as u32,
                                (op.raster_size.1.to_nearest_px() as f32 *
                                 device_pixel_ratio) as u32);
                let allocation = self.allocate(image_id,
                                               0,
                                               0,
                                               device_raster_size.width,
                                               device_raster_size.height,
                                               ImageFormat::RGBA8,
                                               TextureCacheItemKind::Standard,
                                               BorderType::NoBorder,
                                               TextureFilter::Linear);

                // TODO(pcwalton): Handle large box shadows not fitting in texture cache page.
                assert!(allocation.kind == AllocationKind::TexturePage);

                TextureUpdate {
                    id: allocation.item.texture_id,
                    op: TextureUpdateOp::Update(
                        allocation.item.requested_rect.origin.x,
                        allocation.item.requested_rect.origin.y,
                        device_raster_size.width,
                        device_raster_size.height,
                        TextureUpdateDetails::BoxShadow(
                            op.blur_radius,
                            op.border_radius,
                            Rect::new(Point2D::new(op.box_rect_origin.0.to_f32_px(),
                                                   op.box_rect_origin.1.to_f32_px()),
                                      Size2D::new(op.box_rect_size.0.to_f32_px(),
                                                  op.box_rect_size.1.to_f32_px())),
                            Point2D::new(op.raster_origin.0.to_f32_px(),
                                         op.raster_origin.1.to_f32_px()),
                            op.inverted)),
                }
            }
        };

        self.pending_updates.push(update_op);
    }

    pub fn add_raw_update(&mut self, id: TextureId, size: Size2D<i32>) {
        self.pending_updates.push(TextureUpdate {
            id: id,
            op: TextureUpdateOp::Update(
                0, 0,
                size.width as u32, size.height as u32,
                TextureUpdateDetails::Raw),
        })
    }

    pub fn update(&mut self,
                  image_id: TextureCacheItemId,
                  width: u32,
                  height: u32,
                  _format: ImageFormat,
                  bytes: Vec<u8>) {
        let existing_item = self.items.get(image_id);

        // TODO(gw): Handle updates to size/format!
        debug_assert!(existing_item.requested_rect.size.width == width);
        debug_assert!(existing_item.requested_rect.size.height == height);

        let op = TextureUpdateOp::Update(existing_item.requested_rect.origin.x,
                                         existing_item.requested_rect.origin.y,
                                         width,
                                         height,
                                         TextureUpdateDetails::Blit(bytes));

        let update_op = TextureUpdate {
            id: existing_item.texture_id,
            op: op,
        };

        self.pending_updates.push(update_op);
    }

    pub fn insert(&mut self,
                  image_id: TextureCacheItemId,
                  x0: i32,
                  y0: i32,
                  width: u32,
                  height: u32,
                  format: ImageFormat,
                  filter: TextureFilter,
                  insert_op: TextureInsertOp,
                  border_type: BorderType) {

        let result = self.allocate(image_id,
                                   x0,
                                   y0,
                                   width,
                                   height,
                                   format,
                                   TextureCacheItemKind::Standard,
                                   border_type,
                                   filter);

        let op = match (result.kind, insert_op) {
            (AllocationKind::TexturePage, TextureInsertOp::Blit(bytes)) => {

                match border_type {
                    BorderType::NoBorder => {}
                    BorderType::SinglePixel => {
                        let bpp = match format {
                            ImageFormat::A8 => 1,
                            ImageFormat::RGB8 => 3,
                            ImageFormat::RGBA8 => 4,
                            ImageFormat::Invalid => unreachable!(),
                        };

                        let mut top_row_bytes = Vec::new();
                        let mut bottom_row_bytes = Vec::new();
                        let mut left_column_bytes = Vec::new();
                        let mut right_column_bytes = Vec::new();

                        copy_pixels(&bytes, &mut top_row_bytes, 0, 0, 1, width, bpp);
                        copy_pixels(&bytes, &mut top_row_bytes, 0, 0, width, width, bpp);
                        copy_pixels(&bytes, &mut top_row_bytes, width-1, 0, 1, width, bpp);

                        copy_pixels(&bytes, &mut bottom_row_bytes, 0, height-1, 1, width, bpp);
                        copy_pixels(&bytes, &mut bottom_row_bytes, 0, height-1, width, width, bpp);
                        copy_pixels(&bytes, &mut bottom_row_bytes, width-1, height-1, 1, width, bpp);

                        for y in 0..height {
                            copy_pixels(&bytes, &mut left_column_bytes, 0, y, 1, width, bpp);
                            copy_pixels(&bytes, &mut right_column_bytes, width-1, y, 1, width, bpp);
                        }

                        let border_update_op_top = TextureUpdate {
                            id: result.item.texture_id,
                            op: TextureUpdateOp::Update(result.item.allocated_rect.origin.x,
                                                        result.item.allocated_rect.origin.y,
                                                        result.item.allocated_rect.size.width,
                                                        1,
                                                        TextureUpdateDetails::Blit(top_row_bytes))
                        };

                        let border_update_op_bottom = TextureUpdate {
                            id: result.item.texture_id,
                            op: TextureUpdateOp::Update(
                                result.item.allocated_rect.origin.x,
                                result.item.allocated_rect.origin.y +
                                    result.item.requested_rect.size.height + 1,
                                result.item.allocated_rect.size.width,
                                1,
                                TextureUpdateDetails::Blit(bottom_row_bytes))
                        };

                        let border_update_op_left = TextureUpdate {
                            id: result.item.texture_id,
                            op: TextureUpdateOp::Update(
                                result.item.allocated_rect.origin.x,
                                result.item.requested_rect.origin.y,
                                1,
                                result.item.requested_rect.size.height,
                                TextureUpdateDetails::Blit(left_column_bytes))
                        };

                        let border_update_op_right = TextureUpdate {
                            id: result.item.texture_id,
                            op: TextureUpdateOp::Update(result.item.allocated_rect.origin.x + result.item.requested_rect.size.width + 1,
                                                        result.item.requested_rect.origin.y,
                                                        1,
                                                        result.item.requested_rect.size.height,
                                                        TextureUpdateDetails::Blit(right_column_bytes))
                        };

                        self.pending_updates.push(border_update_op_top);
                        self.pending_updates.push(border_update_op_bottom);
                        self.pending_updates.push(border_update_op_left);
                        self.pending_updates.push(border_update_op_right);
                    }
                }

                TextureUpdateOp::Update(result.item.requested_rect.origin.x,
                                        result.item.requested_rect.origin.y,
                                        width,
                                        height,
                                        TextureUpdateDetails::Blit(bytes))
            }
            (AllocationKind::TexturePage,
             TextureInsertOp::Blur(bytes, glyph_size, blur_radius)) => {
                let unblurred_glyph_image_id = self.new_item_id();
                let horizontal_blur_image_id = self.new_item_id();
                // TODO(pcwalton): Destroy these!
                self.allocate(unblurred_glyph_image_id,
                              0, 0,
                              glyph_size.width, glyph_size.height,
                              ImageFormat::RGBA8,
                              TextureCacheItemKind::Standard,
                              BorderType::NoBorder,
                              TextureFilter::Linear);
                self.allocate(horizontal_blur_image_id,
                              0, 0,
                              width, height,
                              ImageFormat::RGBA8,
                              TextureCacheItemKind::Alternate,
                              BorderType::NoBorder,
                              TextureFilter::Linear);
                let unblurred_glyph_item = self.get(unblurred_glyph_image_id);
                let horizontal_blur_item = self.get(horizontal_blur_image_id);
                TextureUpdateOp::Update(
                    result.item.requested_rect.origin.x,
                    result.item.requested_rect.origin.y,
                    width,
                    height,
                    TextureUpdateDetails::Blur(bytes,
                                               glyph_size,
                                               blur_radius,
                                               unblurred_glyph_item.to_image(),
                                               horizontal_blur_item.to_image()))
            }
            (AllocationKind::Standalone, TextureInsertOp::Blit(bytes)) => {
                TextureUpdateOp::Create(width,
                                        height,
                                        format,
                                        filter,
                                        RenderTargetMode::None,
                                        Some(bytes))
            }
            (AllocationKind::Standalone, TextureInsertOp::Blur(_, _, _)) => {
                println!("ERROR: Can't blur with a standalone texture yet!");
                return
            }
        };

        let update_op = TextureUpdate {
            id: result.item.texture_id,
            op: op,
        };

        self.pending_updates.push(update_op);
    }

    pub fn get(&self, id: TextureCacheItemId) -> &TextureCacheItem {
        self.items.get(id)
    }

}

fn texture_create_op(texture_size: u32, format: ImageFormat, mode: RenderTargetMode)
                     -> TextureUpdateOp {
    TextureUpdateOp::Create(texture_size, texture_size, format, TextureFilter::Linear, mode, None)
}

pub enum TextureInsertOp {
    Blit(Vec<u8>),
    Blur(Vec<u8>, Size2D<u32>, Au),
}

trait FitsInside {
    fn fits_inside(&self, other: &Self) -> bool;
}

impl FitsInside for Size2D<u32> {
    fn fits_inside(&self, other: &Size2D<u32>) -> bool {
        self.width <= other.width && self.height <= other.height
    }
}

/// FIXME(pcwalton): Would probably be more efficient as a bit vector.
#[derive(Clone, Copy)]
pub struct FreeTextureLevel {
    texture_id: TextureId,
}

fn create_new_texture_page(pending_updates: &mut TextureUpdateList,
                           free_texture_ids: &mut Vec<TextureId>,
                           free_texture_levels: &mut Vec<FreeTextureLevel>,
                           texture_size: u32,
                           format: ImageFormat,
                           mode: RenderTargetMode) {
    let texture_id = free_texture_ids.pop().expect("TODO: Handle running out of texture IDs!");
    let update_op = TextureUpdate {
        id: texture_id,
        op: texture_create_op(texture_size, format, mode),
    };
    pending_updates.push(update_op);

    free_texture_levels.push(FreeTextureLevel {
        texture_id: texture_id,
    })
}

/// Returns the number of pixels on a side we're allowed to use for our texture atlases.
fn texture_size() -> u32 {
    let max_hardware_texture_size = *MAX_TEXTURE_SIZE as u32;
    if max_hardware_texture_size * max_hardware_texture_size > MAX_RGBA_PIXELS_PER_TEXTURE {
        SQRT_MAX_RGBA_PIXELS_PER_TEXTURE
    } else {
        max_hardware_texture_size
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TextureCacheItemKind {
    Standard,
    Alternate,
    //RenderTarget,
}

