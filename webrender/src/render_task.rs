/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{DeviceIntPoint, DeviceIntRect, DeviceIntSize, ImageDescriptor, ImageFormat};
use api::PremultipliedColorF;
use box_shadow::BoxShadowCacheKey;
use clip::ClipWorkItem;
use clip_scroll_tree::CoordinateSystemId;
use device::TextureFilter;
use glyph_rasterizer::GpuGlyphCacheKey;
use gpu_cache::GpuCache;
use gpu_types::PictureType;
use internal_types::{FastHashMap, SavedTargetIndex, SourceTexture};
use pathfinder_partitioner::mesh_library::MeshLibrary;
use picture::ContentOrigin;
use prim_store::{PrimitiveIndex, ImageCacheKey};
#[cfg(feature = "debugger")]
use print_tree::{PrintTreePrinter};
use resource_cache::CacheItem;
use std::{cmp, ops, usize, f32, i32};
use texture_cache::{TextureCache, TextureCacheHandle};
use tiling::{RenderPass, RenderTargetIndex};
use tiling::{RenderTargetKind};

const FLOATS_PER_RENDER_TASK_INFO: usize = 12;
pub const MAX_BLUR_STD_DEVIATION: f32 = 4.0;
pub const MIN_DOWNSCALING_RT_SIZE: i32 = 128;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct RenderTaskId(pub u32); // TODO(gw): Make private when using GPU cache!

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct RenderTaskAddress(pub u32);

#[derive(Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct RenderTaskTree {
    pub tasks: Vec<RenderTask>,
    pub task_data: Vec<RenderTaskData>,
    next_saved: SavedTargetIndex,
}

impl RenderTaskTree {
    pub fn new() -> Self {
        RenderTaskTree {
            tasks: Vec::new(),
            task_data: Vec::new(),
            next_saved: SavedTargetIndex(0),
        }
    }

    pub fn add(&mut self, task: RenderTask) -> RenderTaskId {
        let id = RenderTaskId(self.tasks.len() as u32);
        self.tasks.push(task);
        id
    }

    pub fn max_depth(&self, id: RenderTaskId, depth: usize, max_depth: &mut usize) {
        let depth = depth + 1;
        *max_depth = cmp::max(*max_depth, depth);
        let task = &self.tasks[id.0 as usize];
        for child in &task.children {
            self.max_depth(*child, depth, max_depth);
        }
    }

    pub fn assign_to_passes(
        &self,
        id: RenderTaskId,
        pass_index: usize,
        passes: &mut [RenderPass],
    ) {
        let task = &self.tasks[id.0 as usize];

        for child in &task.children {
            self.assign_to_passes(*child, pass_index - 1, passes);
        }

        // Sanity check - can be relaxed if needed
        match task.location {
            RenderTaskLocation::Fixed(..) => {
                debug_assert!(pass_index == passes.len() - 1);
            }
            RenderTaskLocation::Dynamic(..) |
            RenderTaskLocation::TextureCache(..) => {
                debug_assert!(pass_index < passes.len() - 1);
            }
        }

        // If this task can be shared between multiple
        // passes, render it in the first pass so that
        // it is available to all subsequent passes.
        let pass_index = if task.is_shared() {
            debug_assert!(task.children.is_empty());
            0
        } else {
            pass_index
        };

        let pass = &mut passes[pass_index];
        pass.add_render_task(id, task.get_dynamic_size(), task.target_kind());
    }

    pub fn get_task_address(&self, id: RenderTaskId) -> RenderTaskAddress {
        RenderTaskAddress(id.0)
    }

    pub fn build(&mut self) {
        for task in &self.tasks {
            self.task_data.push(task.write_task_data());
        }
    }

    pub fn save_target(&mut self) -> SavedTargetIndex {
        let id = self.next_saved;
        self.next_saved.0 += 1;
        id
    }
}

impl ops::Index<RenderTaskId> for RenderTaskTree {
    type Output = RenderTask;
    fn index(&self, id: RenderTaskId) -> &RenderTask {
        &self.tasks[id.0 as usize]
    }
}

impl ops::IndexMut<RenderTaskId> for RenderTaskTree {
    fn index_mut(&mut self, id: RenderTaskId) -> &mut RenderTask {
        &mut self.tasks[id.0 as usize]
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum RenderTaskLocation {
    Fixed(DeviceIntRect),
    Dynamic(Option<(DeviceIntPoint, RenderTargetIndex)>, DeviceIntSize),
    TextureCache(SourceTexture, i32, DeviceIntRect),
}

#[derive(Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct CacheMaskTask {
    actual_rect: DeviceIntRect,
    pub clips: Vec<ClipWorkItem>,
    pub coordinate_system_id: CoordinateSystemId,
}

#[derive(Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct PictureTask {
    pub prim_index: PrimitiveIndex,
    pub target_kind: RenderTargetKind,
    pub content_origin: ContentOrigin,
    pub color: PremultipliedColorF,
    pub pic_type: PictureType,
}

#[derive(Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct BlurTask {
    pub blur_std_deviation: f32,
    pub target_kind: RenderTargetKind,
    pub color: PremultipliedColorF,
    pub scale_factor: f32,
}

impl BlurTask {
    #[cfg(feature = "debugger")]
    fn print_with<T: PrintTreePrinter>(&self, pt: &mut T) {
        pt.add_item(format!("std deviation: {}", self.blur_std_deviation));
        pt.add_item(format!("target: {:?}", self.target_kind));
        pt.add_item(format!("scale: {}", self.scale_factor));
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct GlyphTask {
    pub mesh_library: MeshLibrary,
}

// Where the source data for a blit task can be found.
#[derive(Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum BlitSource {
    Image {
        key: ImageCacheKey,
    },
    RenderTask {
        task_id: RenderTaskId,
    },
}

#[derive(Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct BlitTask {
    pub source: BlitSource,
}

#[derive(Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct RenderTaskData {
    pub data: [f32; FLOATS_PER_RENDER_TASK_INFO],
}

#[derive(Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum RenderTaskKind {
    Picture(PictureTask),
    CacheMask(CacheMaskTask),
    VerticalBlur(BlurTask),
    HorizontalBlur(BlurTask),
    Glyph(GlyphTask),
    Readback(DeviceIntRect),
    Scaling(RenderTargetKind),
    Blit(BlitTask),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum ClearMode {
    // Applicable to color and alpha targets.
    Zero,
    One,

    // Applicable to color targets only.
    Transparent,
}

#[derive(Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct RenderTask {
    pub location: RenderTaskLocation,
    pub children: Vec<RenderTaskId>,
    pub kind: RenderTaskKind,
    pub clear_mode: ClearMode,
    pub saved_index: Option<SavedTargetIndex>,
}

impl RenderTask {
    pub fn new_picture(
        location: RenderTaskLocation,
        prim_index: PrimitiveIndex,
        target_kind: RenderTargetKind,
        content_origin: ContentOrigin,
        color: PremultipliedColorF,
        clear_mode: ClearMode,
        children: Vec<RenderTaskId>,
        pic_type: PictureType,
    ) -> Self {
        RenderTask {
            children,
            location,
            kind: RenderTaskKind::Picture(PictureTask {
                prim_index,
                target_kind,
                content_origin,
                color,
                pic_type,
            }),
            clear_mode,
            saved_index: None,
        }
    }

    pub fn new_readback(screen_rect: DeviceIntRect) -> Self {
        RenderTask {
            children: Vec::new(),
            location: RenderTaskLocation::Dynamic(None, screen_rect.size),
            kind: RenderTaskKind::Readback(screen_rect),
            clear_mode: ClearMode::Transparent,
            saved_index: None,
        }
    }

    pub fn new_blit(
        size: DeviceIntSize,
        source: BlitSource,
    ) -> Self {
        let mut children = Vec::new();

        // If this blit uses a render task as a source,
        // ensure it's added as a child task. This will
        // ensure it gets allocated in the correct pass
        // and made available as an input when this task
        // executes.
        if let BlitSource::RenderTask { task_id } = source {
            children.push(task_id);
        }

        RenderTask {
            children,
            location: RenderTaskLocation::Dynamic(None, size),
            kind: RenderTaskKind::Blit(BlitTask {
                source,
            }),
            clear_mode: ClearMode::Transparent,
            saved_index: None,
        }
    }

    pub fn new_mask(
        outer_rect: DeviceIntRect,
        clips: Vec<ClipWorkItem>,
        prim_coordinate_system_id: CoordinateSystemId,
    ) -> Self {
        RenderTask {
            children: Vec::new(),
            location: RenderTaskLocation::Dynamic(None, outer_rect.size),
            kind: RenderTaskKind::CacheMask(CacheMaskTask {
                actual_rect: outer_rect,
                clips,
                coordinate_system_id: prim_coordinate_system_id,
            }),
            clear_mode: ClearMode::One,
            saved_index: None,
        }
    }

    // Construct a render task to apply a blur to a primitive.
    // The render task chain that is constructed looks like:
    //
    //    PrimitiveCacheTask: Draw the primitives.
    //           ^
    //           |
    //    DownscalingTask(s): Each downscaling task reduces the size of render target to
    //           ^            half. Also reduce the std deviation to half until the std
    //           |            deviation less than 4.0.
    //           |
    //           |
    //    VerticalBlurTask: Apply the separable vertical blur to the primitive.
    //           ^
    //           |
    //    HorizontalBlurTask: Apply the separable horizontal blur to the vertical blur.
    //           |
    //           +---- This is stored as the input task to the primitive shader.
    //
    pub fn new_blur(
        blur_std_deviation: f32,
        src_task_id: RenderTaskId,
        render_tasks: &mut RenderTaskTree,
        target_kind: RenderTargetKind,
        clear_mode: ClearMode,
        color: PremultipliedColorF,
    ) -> (Self, f32) {
        // Adjust large std deviation value.
        let mut adjusted_blur_std_deviation = blur_std_deviation;
        let blur_target_size = render_tasks[src_task_id].get_dynamic_size();
        let mut adjusted_blur_target_size = blur_target_size;
        let mut downscaling_src_task_id = src_task_id;
        let mut scale_factor = 1.0;
        while adjusted_blur_std_deviation > MAX_BLUR_STD_DEVIATION {
            if adjusted_blur_target_size.width < MIN_DOWNSCALING_RT_SIZE ||
               adjusted_blur_target_size.height < MIN_DOWNSCALING_RT_SIZE {
                break;
            }
            adjusted_blur_std_deviation *= 0.5;
            scale_factor *= 2.0;
            adjusted_blur_target_size = (blur_target_size.to_f32() / scale_factor).to_i32();
            let downscaling_task = RenderTask::new_scaling(
                target_kind,
                downscaling_src_task_id,
                adjusted_blur_target_size,
            );
            downscaling_src_task_id = render_tasks.add(downscaling_task);
        }
        scale_factor = blur_target_size.width as f32 / adjusted_blur_target_size.width as f32;

        let blur_task_v = RenderTask {
            children: vec![downscaling_src_task_id],
            location: RenderTaskLocation::Dynamic(None, adjusted_blur_target_size),
            kind: RenderTaskKind::VerticalBlur(BlurTask {
                blur_std_deviation: adjusted_blur_std_deviation,
                target_kind,
                color,
                scale_factor,
            }),
            clear_mode,
            saved_index: None,
        };

        let blur_task_v_id = render_tasks.add(blur_task_v);

        let blur_task_h = RenderTask {
            children: vec![blur_task_v_id],
            location: RenderTaskLocation::Dynamic(None, adjusted_blur_target_size),
            kind: RenderTaskKind::HorizontalBlur(BlurTask {
                blur_std_deviation: adjusted_blur_std_deviation,
                target_kind,
                color,
                scale_factor,
            }),
            clear_mode,
            saved_index: None,
        };

        (blur_task_h, scale_factor)
    }

    pub fn new_scaling(
        target_kind: RenderTargetKind,
        src_task_id: RenderTaskId,
        target_size: DeviceIntSize,
    ) -> Self {
        RenderTask {
            children: vec![src_task_id],
            location: RenderTaskLocation::Dynamic(None, target_size),
            kind: RenderTaskKind::Scaling(target_kind),
            clear_mode: match target_kind {
                RenderTargetKind::Color => ClearMode::Transparent,
                RenderTargetKind::Alpha => ClearMode::One,
            },
            saved_index: None,
        }
    }

    pub fn new_glyph(location: RenderTaskLocation,
                     mesh_library: MeshLibrary)
                     -> Self {
        RenderTask {
            children: vec![],
            location: location,
            kind: RenderTaskKind::Glyph(GlyphTask {
                mesh_library: mesh_library,
            }),
            clear_mode: ClearMode::Transparent,
            saved_index: None,
        }
    }

    // Write (up to) 8 floats of data specific to the type
    // of render task that is provided to the GPU shaders
    // via a vertex texture.
    pub fn write_task_data(&self) -> RenderTaskData {
        // NOTE: The ordering and layout of these structures are
        //       required to match both the GPU structures declared
        //       in prim_shared.glsl, and also the uses in submit_batch()
        //       in renderer.rs.
        // TODO(gw): Maybe there's a way to make this stuff a bit
        //           more type-safe. Although, it will always need
        //           to be kept in sync with the GLSL code anyway.

        let (data1, data2) = match self.kind {
            RenderTaskKind::Picture(ref task) => {
                (
                    // Note: has to match `PICTURE_TYPE_*` in shaders
                    // TODO(gw): Instead of using the sign of the picture
                    //           type here, we should consider encoding it
                    //           as a set of flags that get casted here
                    //           and in the shader. This is a bit tidier
                    //           and allows for future expansion of flags.
                    match task.content_origin {
                        ContentOrigin::Local(point) => [
                            point.x, point.y, task.pic_type as u32 as f32,
                        ],
                        ContentOrigin::Screen(point) => [
                            point.x as f32, point.y as f32, -(task.pic_type as u32 as f32),
                        ],
                    },
                    task.color.to_array()
                )
            }
            RenderTaskKind::CacheMask(ref task) => {
                (
                    [
                        task.actual_rect.origin.x as f32,
                        task.actual_rect.origin.y as f32,
                        0.0,
                    ],
                    [0.0; 4],
                )
            }
            RenderTaskKind::VerticalBlur(ref task) |
            RenderTaskKind::HorizontalBlur(ref task) => {
                (
                    [
                        task.blur_std_deviation,
                        task.scale_factor,
                        0.0,
                    ],
                    task.color.to_array()
                )
            }
            RenderTaskKind::Glyph(ref task) => {
                (
                    [
                        1.0,
                        0.0,
                        0.0,
                    ],
                    [0.0; 4],
                )
            }
            RenderTaskKind::Readback(..) |
            RenderTaskKind::Scaling(..) |
            RenderTaskKind::Blit(..) => {
                (
                    [0.0; 3],
                    [0.0; 4],
                )
            }
        };

        let (target_rect, target_index) = self.get_target_rect();

        RenderTaskData {
            data: [
                target_rect.origin.x as f32,
                target_rect.origin.y as f32,
                target_rect.size.width as f32,
                target_rect.size.height as f32,
                target_index.0 as f32,
                data1[0],
                data1[1],
                data1[2],
                data2[0],
                data2[1],
                data2[2],
                data2[3],
            ]
        }
    }

    pub fn get_dynamic_size(&self) -> DeviceIntSize {
        match self.location {
            RenderTaskLocation::Fixed(..) => DeviceIntSize::zero(),
            RenderTaskLocation::Dynamic(_, size) => size,
            RenderTaskLocation::TextureCache(_, _, rect) => rect.size,
        }
    }

    pub fn get_target_rect(&self) -> (DeviceIntRect, RenderTargetIndex) {
        match self.location {
            RenderTaskLocation::Fixed(rect) => {
                (rect, RenderTargetIndex(0))
            }
            // Previously, we only added render tasks after the entire
            // primitive chain was determined visible. This meant that
            // we could assert any render task in the list was also
            // allocated (assigned to passes). Now, we add render
            // tasks earlier, and the picture they belong to may be
            // culled out later, so we can't assert that the task
            // has been allocated.
            // Render tasks that are created but not assigned to
            // passes consume a row in the render task texture, but
            // don't allocate any space in render targets nor
            // draw any pixels.
            // TODO(gw): Consider some kind of tag or other method
            //           to mark a task as unused explicitly. This
            //           would allow us to restore this debug check.
            RenderTaskLocation::Dynamic(Some((origin, target_index)), size) => {
                (DeviceIntRect::new(origin, size), target_index)
            }
            RenderTaskLocation::Dynamic(None, _) => {
                (DeviceIntRect::zero(), RenderTargetIndex(0))
            }
            RenderTaskLocation::TextureCache(_, layer, rect) => {
                (rect, RenderTargetIndex(layer as usize))
            }
        }
    }

    pub fn target_kind(&self) -> RenderTargetKind {
        match self.kind {
            RenderTaskKind::Readback(..) => RenderTargetKind::Color,

            RenderTaskKind::CacheMask(..) => {
                RenderTargetKind::Alpha
            }

            RenderTaskKind::VerticalBlur(ref task_info) |
            RenderTaskKind::HorizontalBlur(ref task_info) => {
                task_info.target_kind
            }

            RenderTaskKind::Glyph(..) => {
                // FIXME(pcwalton): Support color.
                RenderTargetKind::Alpha
            }

            RenderTaskKind::Scaling(target_kind) => {
                target_kind
            }

            RenderTaskKind::Picture(ref task_info) => {
                task_info.target_kind
            }

            RenderTaskKind::Blit(..) => {
                RenderTargetKind::Color
            }
        }
    }

    // Check if this task wants to be made available as an input
    // to all passes (except the first) in the render task tree.
    // To qualify for this, the task needs to have no children / dependencies.
    // Currently, this is only supported for A8 targets, but it can be
    // trivially extended to also support RGBA8 targets in the future
    // if we decide that is useful.
    pub fn is_shared(&self) -> bool {
        match self.kind {
            RenderTaskKind::Picture(..) |
            RenderTaskKind::VerticalBlur(..) |
            RenderTaskKind::Readback(..) |
            RenderTaskKind::HorizontalBlur(..) |
            RenderTaskKind::Scaling(..) |
            RenderTaskKind::Blit(..) |
            RenderTaskKind::Glyph(..) => false,
            RenderTaskKind::CacheMask(..) => true,
        }
    }

    #[cfg(feature = "debugger")]
    pub fn print_with<T: PrintTreePrinter>(&self, pt: &mut T, tree: &RenderTaskTree) -> bool {
        match self.kind {
            RenderTaskKind::Picture(ref task) => {
                pt.new_level(format!("Picture of {:?}", task.prim_index));
                pt.add_item(format!("kind: {:?}", task.target_kind));
            }
            RenderTaskKind::CacheMask(ref task) => {
                pt.new_level(format!("CacheMask with {} clips", task.clips.len()));
                pt.add_item(format!("rect: {:?}", task.actual_rect));
            }
            RenderTaskKind::VerticalBlur(ref task) => {
                pt.new_level("VerticalBlur".to_owned());
                task.print_with(pt);
            }
            RenderTaskKind::HorizontalBlur(ref task) => {
                pt.new_level("HorizontalBlur".to_owned());
                task.print_with(pt);
            }
            RenderTaskKind::Readback(ref rect) => {
                pt.new_level("Readback".to_owned());
                pt.add_item(format!("rect: {:?}", rect));
            }
            RenderTaskKind::Scaling(ref kind) => {
                pt.new_level("Scaling".to_owned());
                pt.add_item(format!("kind: {:?}", kind));
            }
            RenderTaskKind::Blit(ref task) => {
                pt.new_level("Blit".to_owned());
                pt.add_item(format!("source: {:?}", task.source));
            }
        }

        pt.add_item(format!("clear to: {:?}", self.clear_mode));

        for &child_id in &self.children {
            if tree[child_id].print_with(pt, tree) {
                pt.add_item(format!("self: {:?}", child_id))
            }
        }

        pt.end_level();
        true
    }

    /// Mark this render task for keeping the results alive up until the end of the frame.
    pub fn mark_for_saving(&mut self) {
        match self.location {
            RenderTaskLocation::Fixed(..) |
            RenderTaskLocation::Dynamic(..) => {
                self.saved_index = Some(SavedTargetIndex::PENDING);
            }
            RenderTaskLocation::TextureCache(..) => {
                panic!("Unable to mark a permanently cached task for saving!");
            }
        }
    }
}

#[derive(Debug, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum RenderTaskCacheKeyKind {
    BoxShadow(BoxShadowCacheKey),
    Image(ImageCacheKey),
    Glyph(GpuGlyphCacheKey),
}

#[derive(Debug, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct RenderTaskCacheKey {
    pub size: DeviceIntSize,
    pub kind: RenderTaskCacheKeyKind,
}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
struct RenderTaskCacheEntry {
    handle: TextureCacheHandle,
}

// A cache of render tasks that are stored in the texture
// cache for usage across frames.
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct RenderTaskCache {
    entries: FastHashMap<RenderTaskCacheKey, RenderTaskCacheEntry>,
}

impl RenderTaskCache {
    pub fn new() -> Self {
        RenderTaskCache {
            entries: FastHashMap::default(),
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn begin_frame(
        &mut self,
        texture_cache: &mut TextureCache,
    ) {
        // Drop any items from the cache that have been
        // evicted from the texture cache.
        //
        // This isn't actually necessary for the texture
        // cache to be able to evict old render tasks.
        // It will evict render tasks as required, since
        // the access time in the texture cache entry will
        // be stale if this task hasn't been requested
        // for a while.
        //
        // Nonetheless, we should remove stale entries
        // from here so that this hash map doesn't
        // grow indefinitely!
        self.entries.retain(|_, value| {
            texture_cache.is_allocated(&value.handle)
        });
    }

    pub fn request_render_task<F>(
        &mut self,
        key: RenderTaskCacheKey,
        texture_cache: &mut TextureCache,
        gpu_cache: &mut GpuCache,
        render_tasks: &mut RenderTaskTree,
        mut f: F,
    ) -> TextureCacheHandle where F: FnMut(&mut RenderTaskTree) -> (RenderTaskId, [f32; 3], bool) {
        // Get the texture cache handle for this cache key,
        // or create one.
        let cache_entry = self.entries
                              .entry(key)
                              .or_insert(RenderTaskCacheEntry {
                                  handle: TextureCacheHandle::new(),
                              });

        // Check if this texture cache handle is valie.
        if texture_cache.request(&mut cache_entry.handle, gpu_cache) {
            // Invoke user closure to get render task chain
            // to draw this into the texture cache.
            let (render_task_id, user_data, is_opaque) = f(render_tasks);
            let render_task = &mut render_tasks[render_task_id];

            // Select the right texture page to allocate from.
            let image_format = match render_task.target_kind() {
                RenderTargetKind::Color => ImageFormat::BGRA8,
                RenderTargetKind::Alpha => ImageFormat::R8,
            };

            // Find out what size to alloc in the texture cache.
            let size = match render_task.location {
                RenderTaskLocation::Fixed(..) |
                RenderTaskLocation::TextureCache(..) => {
                    panic!("BUG: dynamic task was expected");
                }
                RenderTaskLocation::Dynamic(_, size) => size,
            };

            // TODO(gw): Support color tasks in the texture cache,
            //           and perhaps consider if we can determine
            //           if some tasks are opaque as an optimization.
            let descriptor = ImageDescriptor::new(
                size.width as u32,
                size.height as u32,
                image_format,
                is_opaque,
            );

            // Allocate space in the texture cache, but don't supply
            // and CPU-side data to be uploaded.
            texture_cache.update(
                &mut cache_entry.handle,
                descriptor,
                TextureFilter::Linear,
                None,
                user_data,
                None,
                gpu_cache,
            );

            // Get the allocation details in the texture cache, and store
            // this in the render task. The renderer will draw this
            // task into the appropriate layer and rect of the texture
            // cache on this frame.
            let (texture_id, texture_layer, uv_rect) =
                texture_cache.get_cache_location(&cache_entry.handle);

            render_task.location = RenderTaskLocation::TextureCache(
                texture_id,
                texture_layer,
                uv_rect.to_i32()
            );
        }

        // Finally, return the texture cache handle that we know
        // is now up to date.
        cache_entry.handle.clone()
    }
}
