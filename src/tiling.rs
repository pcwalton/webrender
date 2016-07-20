/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use app_units::{Au};
use batch_builder::{BorderSideHelpers, BoxShadowMetrics};
use device::{TextureId, TextureFilter};
use euclid::{Point2D, Rect, Matrix4D, Size2D, Point4D};
use fnv::FnvHasher;
use frame::FrameId;
use internal_types::{AxisDirection, Glyph, GlyphKey, DevicePixel};
use layer::Layer;
use renderer::{BLUR_INFLATION_FACTOR};
use resource_cache::ResourceCache;
use resource_list::ResourceList;
use std::cmp;
use std::collections::{HashMap};
use std::f32;
use std::mem;
use std::hash::{BuildHasherDefault};
use texture_cache::TexturePage;
use util::{self, rect_from_points, rect_from_points_f, MatrixHelpers, subtract_rect};
use webrender_traits::{ColorF, FontKey, ImageKey, ImageRendering, ComplexClipRegion};
use webrender_traits::{BorderDisplayItem, BorderStyle, ItemRange, AuxiliaryLists, BorderRadius};
use webrender_traits::{BoxShadowClipMode, PipelineId, ScrollLayerId};

const PRIMITIVES_PER_BUFFER: u8 = 16;

struct RenderTargetContext<'a> {
    layers: &'a Vec<StackingContext>,
    resource_cache: &'a ResourceCache,
    device_pixel_ratio: f32,
    frame_id: FrameId,
    clips: &'a Vec<Clip>,
    alpha_batch_max_tiles: usize,
    alpha_batch_max_layers: usize,
}

pub struct AlphaBatchRenderTask {
    pub batches: Vec<PrimitiveBatch>,
    pub layer_ubo: Vec<PackedLayer>,
    pub tile_ubo: Vec<PackedTile>,
    screen_tile_layers: Vec<ScreenTileLayer>,
    layer_to_ubo_map: Vec<Option<LayerUboIndex>>,
    tile_to_ubo_map: Vec<Option<u32>>,
}

#[derive(Clone, Copy, Debug)]
pub struct LayerUboIndex(u32);

impl AlphaBatchRenderTask {
    fn new(ctx: &RenderTargetContext) -> AlphaBatchRenderTask {
        let mut layer_to_ubo_map = Vec::new();
        for _ in 0..ctx.layers.len() {
            layer_to_ubo_map.push(None);
        }

        AlphaBatchRenderTask {
            batches: Vec::new(),
            layer_ubo: Vec::new(),
            tile_ubo: Vec::new(),
            screen_tile_layers: Vec::new(),
            layer_to_ubo_map: layer_to_ubo_map,
            tile_to_ubo_map: vec![],
        }
    }

    fn add_screen_tile_layer(&mut self,
                             target_rect: Rect<DevicePixel>,
                             screen_tile_layer: ScreenTileLayer,
                             ctx: &RenderTargetContext) -> Option<ScreenTileLayer> {
        if self.tile_ubo.len() == ctx.alpha_batch_max_tiles {
            return Some(screen_tile_layer);
        }

        let global_tile_index = screen_tile_layer.global_tile_index;
        while self.tile_to_ubo_map.len() < global_tile_index + 1 {
            self.tile_to_ubo_map.push(None)
        }
        debug_assert!(self.tile_to_ubo_map[global_tile_index].is_none());
        self.tile_to_ubo_map[global_tile_index] = Some(self.tile_ubo.len() as u32);
        self.tile_ubo.push(PackedTile {
            target_rect: target_rect,
            actual_rect: screen_tile_layer.actual_rect,
        });

        let StackingContextIndex(si) = screen_tile_layer.layer_index;
        if self.layer_to_ubo_map[si].is_none() {
            if self.layer_ubo.len() == ctx.alpha_batch_max_layers {
                return Some(screen_tile_layer);
            }

            let index = LayerUboIndex(self.layer_ubo.len() as u32);
            let sc = &ctx.layers[si];
            self.layer_ubo.push(PackedLayer {
                padding: [0, 0],
                transform: sc.transform,
                inv_transform: sc.transform.invert(),
                screen_vertices: sc.xf_rect.as_ref().unwrap().vertices,
                blend_info: [sc.opacity, 0.0],
            });
            self.layer_to_ubo_map[si] = Some(index);
        }

        self.screen_tile_layers.push(screen_tile_layer);
        None
    }

    fn build(&mut self,
             ctx: &RenderTargetContext,
             frame_primitives: &mut FramePackedPrimList,
             screen_tiles: &ScreenTileMap) {
        debug_assert!(self.layer_ubo.len() <= ctx.alpha_batch_max_layers);
        debug_assert!(self.tile_ubo.len() <= ctx.alpha_batch_max_tiles);

        // Build batches
        let mut batch = None;

        // Pull next primitive
        for screen_tile_layer in &mut self.screen_tile_layers {
            let tile_location = screen_tile_layer.tile_location();
            if screen_tile_layer.first_prim_sample_index ==
                    screen_tile_layer.last_prim_sample_index {
                continue
            }

            let StackingContextIndex(si) = screen_tile_layer.layer_index;
            let layer = &ctx.layers[si];

            let global_tile_index = screen_tile_layer.global_tile_index;
            let tile_index = self.tile_to_ubo_map[global_tile_index]
                                 .expect("Tile not in tile-to-UBO map?!");

            let first_prim_sample_index = screen_tile_layer.first_prim_sample_index;
            let last_prim_sample_index = screen_tile_layer.last_prim_sample_index;
            let packed_primitive_indices = screen_tiles.primitive_indices(&tile_location,
                                                                          last_prim_sample_index,
                                                                          first_prim_sample_index);
            // TODO(pcwalton): Fix this. We need to iterate over the range and the primitive
            // indices simultaneously. Put this logic in the iterator, probably.
            for packed_primitive_index in packed_primitive_indices {
                // FIXME(pcwalton): Don't clone?
                let packed_primitive =
                    frame_primitives.clone_packed_primitive(packed_primitive_index);
                if !packed_primitive.intersects(&screen_tile_layer.actual_rect,
                                                &layer.transform,
                                                ctx.device_pixel_ratio) {
                    continue
                }
                // FIXME(pcwalton): Use the real transformed rect kind!
                let transform_kind = TransformedRectKind::AxisAligned;
                loop {
                    let new_batch_needed = batch.is_none();
                    if new_batch_needed {
                        batch = Some(PrimitiveBatch::new(&packed_primitive,
                                                         transform_kind,
                                                         frame_primitives.color_texture_id))
                    }
                    let successfully_added = {
                        let batch = batch.as_mut().unwrap();
                        batch.class() == packed_primitive_index.class() &&
                            batch.push_item_if_possible(
                                packed_primitive_index.class_specific_index(),
                                frame_primitives,
                                &self.layer_to_ubo_map,
                                transform_kind,
                                screen_tile_layer.layer_index,
                                tile_index)
                    };
                    debug_assert!(!new_batch_needed || successfully_added);
                    if successfully_added {
                        break
                    }

                    self.batches.push(mem::replace(&mut batch, None).expect("No batch present?!"))
                }
            }
        }

        if let Some(batch) = batch {
            self.batches.push(batch)
        }
    }
}

pub struct RenderTarget {
    pub is_framebuffer: bool,
    page_allocator: TexturePage,
    tasks: Vec<RenderTask>,

    pub alpha_batch_tasks: Vec<AlphaBatchRenderTask>,
    pub composite_batches: HashMap<CompositeBatchKey,
                                   Vec<CompositeTile>,
                                   BuildHasherDefault<FnvHasher>>,
}

impl RenderTarget {
    fn new(is_framebuffer: bool) -> RenderTarget {
        RenderTarget {
            is_framebuffer: is_framebuffer,
            page_allocator: TexturePage::new(TextureId(0), RENDERABLE_CACHE_SIZE.0 as u32),
            tasks: Vec::new(),

            alpha_batch_tasks: Vec::new(),
            composite_batches: HashMap::with_hasher(Default::default()),
        }
    }

    fn add_render_task(&mut self, task: RenderTask) {
        self.tasks.push(task);
    }

    fn build(&mut self,
             ctx: &RenderTargetContext,
             frame_primitives: &mut FramePackedPrimList,
             screen_tiles: &ScreenTileMap) {
        // Step through each task, adding to batches as appropriate.
        let mut alpha_batch_tasks = Vec::new();
        let mut current_alpha_batch_task = AlphaBatchRenderTask::new(ctx);

        for task in self.tasks.drain(..) {
            let target_rect = task.get_target_rect();

            match task.kind {
                RenderTaskKind::AlphaBatch(screen_tile_layer) => {
                    if let Some(screen_tile_layer) =
                        current_alpha_batch_task.add_screen_tile_layer(target_rect,
                                                                       screen_tile_layer,
                                                                       ctx) {
                        let old_task = mem::replace(&mut current_alpha_batch_task,
                                                    AlphaBatchRenderTask::new(ctx));
                        alpha_batch_tasks.push(old_task);

                        let result =
                            current_alpha_batch_task.add_screen_tile_layer(target_rect,
                                                                           screen_tile_layer,
                                                                           ctx);
                        debug_assert!(result.is_none());
                    }
                }
                RenderTaskKind::Composite(info) => {
                    let mut composite_tile = CompositeTile::new(&target_rect);
                    debug_assert!(info.layer_indices.len() == task.child_locations.len());
                    for (i, (layer_index, location)) in info.layer_indices
                                                            .iter()
                                                            .zip(task.child_locations.iter())
                                                            .enumerate() {
                        let opacity = layer_index.map_or(1.0, |layer_index| {
                            let StackingContextIndex(si) = layer_index;
                            ctx.layers[si].opacity
                        });
                        composite_tile.src_rects[i] = *location;
                        composite_tile.blend_info[i] = opacity;
                    }
                    let shader = CompositeShader::from_cover(info.layer_indices.len());
                    let key = CompositeBatchKey::new(shader);
                    let batch = self.composite_batches.entry(key).or_insert_with(|| {
                        Vec::new()
                    });
                    batch.push(composite_tile);
                }
            }
        }

        if !current_alpha_batch_task.screen_tile_layers.is_empty() {
            alpha_batch_tasks.push(current_alpha_batch_task);
        }
        for task in &mut alpha_batch_tasks {
            task.build(ctx, frame_primitives, screen_tiles);
        }
        self.alpha_batch_tasks = alpha_batch_tasks;
    }
}

pub struct RenderPhase {
    pub targets: Vec<RenderTarget>,
}

impl RenderPhase {
    fn new(max_target_count: usize) -> RenderPhase {
        let mut targets = Vec::with_capacity(max_target_count);
        for index in 0..max_target_count {
            targets.push(RenderTarget::new(index == max_target_count-1));
        }

        RenderPhase {
            targets: targets,
        }
    }

    fn add_compiled_screen_tile(&mut self,
                                mut tile: CompiledScreenTile) -> Option<CompiledScreenTile> {
        debug_assert!(tile.required_target_count <= self.targets.len());

        let ok = tile.main_render_task.alloc_if_required(self.targets.len() - 1,
                                                         &mut self.targets);

        if ok {
            tile.main_render_task.assign_to_targets(self.targets.len() - 1,
                                                    &mut self.targets);
            None
        } else {
            Some(tile)
        }
    }

    fn build(&mut self,
             ctx: &RenderTargetContext,
             frame_primitives: &mut FramePackedPrimList,
             screen_tiles: &ScreenTileMap) {
        for target in &mut self.targets {
            target.build(ctx, frame_primitives, screen_tiles);
        }
    }
}

#[derive(Debug)]
enum RenderTaskLocation {
    Fixed(Rect<DevicePixel>),
    Dynamic(Option<Point2D<DevicePixel>>, Size2D<DevicePixel>),
}

#[derive(Debug)]
enum RenderTaskKind {
    AlphaBatch(ScreenTileLayer),
    Composite(CompositeTileInfo),
}

#[derive(Debug)]
struct RenderTask {
    location: RenderTaskLocation,
    children: Vec<RenderTask>,
    child_locations: Vec<Rect<DevicePixel>>,
    kind: RenderTaskKind,
}

impl RenderTask {
    fn from_layer(layer: ScreenTileLayer, location: RenderTaskLocation) -> RenderTask {
        RenderTask {
            children: Vec::new(),
            child_locations: Vec::new(),
            location: location,
            kind: RenderTaskKind::AlphaBatch(layer),
        }
    }

    fn composite(layers: Vec<RenderTask>,
                 location: RenderTaskLocation,
                 layer_indices: Vec<Option<StackingContextIndex>>) -> RenderTask {
        RenderTask {
            children: layers,
            child_locations: Vec::new(),
            location: location,
            kind: RenderTaskKind::Composite(CompositeTileInfo {
                layer_indices: layer_indices,
            }),
        }
    }

    fn get_target_rect(&self) -> Rect<DevicePixel> {
        match self.location {
            RenderTaskLocation::Fixed(rect) => rect,
            RenderTaskLocation::Dynamic(origin, size) => {
                Rect::new(origin.expect("Should have been allocated by now!"),
                          size)
            }
        }
    }

    fn assign_to_targets(mut self,
                         target_index: usize,
                         targets: &mut Vec<RenderTarget>) {
        for child in self.children.drain(..) {
            self.child_locations.push(child.get_target_rect());
            child.assign_to_targets(target_index - 1, targets);
        }

        // Sanity check - can be relaxed if needed
        match self.location {
            RenderTaskLocation::Fixed(..) => debug_assert!(target_index == targets.len() - 1),
            RenderTaskLocation::Dynamic(..) => debug_assert!(target_index < targets.len() - 1),
        }

        let target = &mut targets[target_index];
        target.add_render_task(self);
    }

    fn alloc_if_required(&mut self,
                         target_index: usize,
                         targets: &mut Vec<RenderTarget>) -> bool {
        match self.location {
            RenderTaskLocation::Fixed(..) => {}
            RenderTaskLocation::Dynamic(ref mut origin, ref size) => {
                let target = &mut targets[target_index];

                let alloc_size = Size2D::new(size.width.0 as u32,
                                             size.height.0 as u32);

                let alloc_origin = target.page_allocator
                                         .allocate(&alloc_size, TextureFilter::Linear);

                match alloc_origin {
                    Some(alloc_origin) => {
                        *origin = Some(Point2D::new(DevicePixel(alloc_origin.x as i32),
                                                    DevicePixel(alloc_origin.y as i32)));
                    }
                    None => {
                        return false;
                    }
                }
            }
        }

        for child in &mut self.children {
            if !child.alloc_if_required(target_index - 1,
                                        targets) {
                return false;
            }
        }

        true
    }

    fn max_depth(&self,
                 depth: usize,
                 max_depth: &mut usize) {
        let depth = depth + 1;
        *max_depth = cmp::max(*max_depth, depth);
        for child in &self.children {
            child.max_depth(depth, max_depth);
        }
    }
}

pub const SCREEN_TILE_SIZE: usize = 64;
pub const RENDERABLE_CACHE_SIZE: DevicePixel = DevicePixel(2048);
pub const MAX_LAYERS_PER_PASS: usize = 8;

#[allow(non_camel_case_types)]
#[derive(Debug, Eq, PartialEq, Copy, Clone, Hash)]
pub enum CompositeShader {
    Prim1,
    Prim2,
    Prim3,
    Prim4,
    Prim5,
    Prim6,
    Prim7,
    Prim8,
}

impl CompositeShader {
    fn from_cover(size: usize) -> CompositeShader {
        match size {
            1 => CompositeShader::Prim1,
            2 => CompositeShader::Prim2,
            3 => CompositeShader::Prim3,
            4 => CompositeShader::Prim4,
            5 => CompositeShader::Prim5,
            6 => CompositeShader::Prim6,
            7 => CompositeShader::Prim7,
            8 => CompositeShader::Prim8,
            _ => panic!("todo - other shader?"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DebugRect {
    pub label: String,
    pub color: ColorF,
    pub rect: Rect<DevicePixel>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TransformedRectKind {
    AxisAligned,
    Complex,
}

#[derive(Debug, Clone)]
struct TransformedRect {
    local_rect: Rect<f32>,
    bounding_rect: Rect<DevicePixel>,
    vertices: [Point4D<f32>; 4],
    kind: TransformedRectKind,
}

impl TransformedRect {
    fn new(rect: &Rect<f32>,
           transform: &Matrix4D<f32>,
           device_pixel_ratio: f32) -> TransformedRect {

        let kind = if transform.can_losslessly_transform_and_perspective_project_a_2d_rect() {
            TransformedRectKind::AxisAligned
        } else {
            TransformedRectKind::Complex
        };

        match kind {
            TransformedRectKind::AxisAligned => {
                let v0 = transform.transform_point(&rect.origin);
                let v1 = transform.transform_point(&rect.top_right());
                let v2 = transform.transform_point(&rect.bottom_left());
                let v3 = transform.transform_point(&rect.bottom_right());

                let screen_min_dp = Point2D::new(DevicePixel((v0.x * device_pixel_ratio).floor() as i32),
                                                 DevicePixel((v0.y * device_pixel_ratio).floor() as i32));
                let screen_max_dp = Point2D::new(DevicePixel((v3.x * device_pixel_ratio).ceil() as i32),
                                                 DevicePixel((v3.y * device_pixel_ratio).ceil() as i32));

                let screen_rect_dp = Rect::new(screen_min_dp, Size2D::new(screen_max_dp.x - screen_min_dp.x,
                                                                          screen_max_dp.y - screen_min_dp.y));

                TransformedRect {
                    local_rect: *rect,
                    vertices: [
                        Point4D::new(v0.x, v0.y, 0.0, 1.0),
                        Point4D::new(v1.x, v1.y, 0.0, 1.0),
                        Point4D::new(v2.x, v2.y, 0.0, 1.0),
                        Point4D::new(v3.x, v3.y, 0.0, 1.0),
                    ],
                    bounding_rect: screen_rect_dp,
                    kind: kind,
                }
            }
            TransformedRectKind::Complex => {
                let vertices = [
                    transform.transform_point4d(&Point4D::new(rect.origin.x,
                                                              rect.origin.y,
                                                              0.0,
                                                              1.0)),
                    transform.transform_point4d(&Point4D::new(rect.bottom_left().x,
                                                              rect.bottom_left().y,
                                                              0.0,
                                                              1.0)),
                    transform.transform_point4d(&Point4D::new(rect.bottom_right().x,
                                                              rect.bottom_right().y,
                                                              0.0,
                                                              1.0)),
                    transform.transform_point4d(&Point4D::new(rect.top_right().x,
                                                              rect.top_right().y,
                                                              0.0,
                                                              1.0)),
                ];

                let mut screen_min: Point2D<f32> = Point2D::new( 10000000.0,  10000000.0);
                let mut screen_max: Point2D<f32> = Point2D::new(-10000000.0, -10000000.0);

                for vertex in &vertices {
                    let inv_w = 1.0 / vertex.w;
                    let vx = vertex.x * inv_w;
                    let vy = vertex.y * inv_w;
                    screen_min.x = screen_min.x.min(vx);
                    screen_min.y = screen_min.y.min(vy);
                    screen_max.x = screen_max.x.max(vx);
                    screen_max.y = screen_max.y.max(vy);
                }

                let screen_min_dp = Point2D::new(DevicePixel((screen_min.x * device_pixel_ratio).floor() as i32),
                                                 DevicePixel((screen_min.y * device_pixel_ratio).floor() as i32));
                let screen_max_dp = Point2D::new(DevicePixel((screen_max.x * device_pixel_ratio).ceil() as i32),
                                                 DevicePixel((screen_max.y * device_pixel_ratio).ceil() as i32));

                let screen_rect_dp = Rect::new(screen_min_dp, Size2D::new(screen_max_dp.x - screen_min_dp.x,
                                                                          screen_max_dp.y - screen_min_dp.y));

                TransformedRect {
                    local_rect: *rect,
                    vertices: vertices,
                    bounding_rect: screen_rect_dp,
                    kind: kind,
                }
            }
        }
    }
}

#[derive(Debug)]
struct RectanglePrimitive {
    color: ColorF,
}

#[derive(Debug)]
struct TextPrimitive {
    color: ColorF,
    font_key: FontKey,
    size: Au,
    blur_radius: Au,
    glyph_range: ItemRange,
}

#[derive(Debug)]
struct BoxShadowPrimitive {
    src_rect: Rect<f32>,
    bs_rect: Rect<f32>,
    color: ColorF,
    blur_radius: f32,
    spread_radius: f32,
    border_radius: f32,
    clip_mode: BoxShadowClipMode,
    metrics: BoxShadowMetrics,
}

#[derive(Debug)]
struct BorderPrimitive {
    tl_outer: Point2D<f32>,
    tl_inner: Point2D<f32>,
    tr_outer: Point2D<f32>,
    tr_inner: Point2D<f32>,
    bl_outer: Point2D<f32>,
    bl_inner: Point2D<f32>,
    br_outer: Point2D<f32>,
    br_inner: Point2D<f32>,
    left_width: f32,
    top_width: f32,
    right_width: f32,
    bottom_width: f32,
    radius: BorderRadius,
    left_color: ColorF,
    top_color: ColorF,
    right_color: ColorF,
    bottom_color: ColorF,
}

#[derive(Debug)]
struct ImagePrimitive {
    image_key: ImageKey,
    image_rendering: ImageRendering,
}

#[derive(Debug)]
struct GradientPrimitive {
    stops_range: ItemRange,
    dir: AxisDirection,
}

#[derive(Debug)]
enum PrimitiveDetails {
    Rectangle(RectanglePrimitive),
    Text(TextPrimitive),
    Image(ImagePrimitive),
    Border(BorderPrimitive),
    Gradient(GradientPrimitive),
    BoxShadow(BoxShadowPrimitive),
}

#[derive(Clone, Debug)]
enum PackedPrimitive {
    Rectangle(PackedRectanglePrimitive),
    RectangleClip(PackedRectanglePrimitiveClip),
    Glyph(PackedGlyphPrimitive),
    Image(PackedImagePrimitive),
    Border(PackedBorderPrimitive),
    BoxShadow(PackedBoxShadowPrimitive),
    Gradient(PackedGradientPrimitive),
}

impl PackedPrimitive {
    fn intersects(&self,
                  query_rect: &Rect<DevicePixel>,
                  transform: &Matrix4D<f32>,
                  device_pixel_ratio: f32)
                  -> bool {
        let local_rect = match *self {
            PackedPrimitive::Rectangle(ref rectangle) => rectangle.local_rect,
            PackedPrimitive::RectangleClip(ref rectangle_clip) => rectangle_clip.local_rect,
            PackedPrimitive::Border(ref border) => border.local_rect,
            PackedPrimitive::BoxShadow(ref box_shadow) => box_shadow.local_rect,
            PackedPrimitive::Glyph(ref glyph) => glyph.local_rect,
            PackedPrimitive::Image(ref image) => image.local_rect,
            PackedPrimitive::Gradient(ref gradient) => gradient.local_rect,
        };
        transform.transform_rect(&local_rect).intersects(&Rect::new(
                Point2D::new((query_rect.origin.x.0 as f32) / device_pixel_ratio,
                             (query_rect.origin.y.0 as f32) / device_pixel_ratio),
                Size2D::new((query_rect.size.width.0 as f32) / device_pixel_ratio,
                            (query_rect.size.height.0 as f32) / device_pixel_ratio)))
    }
}

#[derive(Copy, Clone, Debug)]
struct PackedPrimitiveRange {
    start: ClassSpecificPackedPrimitiveIndex,
    end: ClassSpecificPackedPrimitiveIndex,
    class: PrimitiveClass,
}

impl PackedPrimitiveRange {
    fn new(start: PackedPrimitiveIndex, end: PackedPrimitiveIndex) -> PackedPrimitiveRange {
        debug_assert!(start.class() == end.class());
        PackedPrimitiveRange {
            start: start.class_specific_index(),
            end: end.class_specific_index(),
            class: start.class(),
        }
    }

    fn empty() -> PackedPrimitiveRange {
        PackedPrimitiveRange {
            start: ClassSpecificPackedPrimitiveIndex::zero(),
            end: ClassSpecificPackedPrimitiveIndex::zero(),
            class: PrimitiveClass::Rectangle,
        }
    }

    fn end(&self) -> PackedPrimitiveIndex {
        PackedPrimitiveIndex::new(self.class, self.end)
    }
}

#[derive(Copy, Clone, Debug)]
struct LayerPackedPrimitiveRangeStartOffsets {
    rectangles: usize,
    rectangles_clip: usize,
    borders: usize,
    box_shadows: usize,
    text: usize,
    images: usize,
    gradients: usize,
}

struct FramePackedDataForPrimitive<PrimitiveType> {
    data: Vec<PrimitiveType>,
}

impl<PrimitiveType> FramePackedDataForPrimitive<PrimitiveType> {
    fn new() -> FramePackedDataForPrimitive<PrimitiveType> {
        FramePackedDataForPrimitive {
            data: vec![],
        }
    }
}

struct FramePackedPrimList {
    color_texture_id: TextureId,
    rectangles: FramePackedDataForPrimitive<PackedRectanglePrimitive>,
    rectangles_clip: FramePackedDataForPrimitive<PackedRectanglePrimitiveClip>,
    glyphs: FramePackedDataForPrimitive<PackedGlyphPrimitive>,
    images: FramePackedDataForPrimitive<PackedImagePrimitive>,
    borders: FramePackedDataForPrimitive<PackedBorderPrimitive>,
    box_shadows: FramePackedDataForPrimitive<PackedBoxShadowPrimitive>,
    gradients: FramePackedDataForPrimitive<PackedGradientPrimitive>,
}

impl FramePackedPrimList {
    fn new() -> FramePackedPrimList {
        FramePackedPrimList {
            color_texture_id: TextureId(0),
            rectangles: FramePackedDataForPrimitive::new(),
            rectangles_clip: FramePackedDataForPrimitive::new(),
            glyphs: FramePackedDataForPrimitive::new(),
            images: FramePackedDataForPrimitive::new(),
            borders: FramePackedDataForPrimitive::new(),
            box_shadows: FramePackedDataForPrimitive::new(),
            gradients: FramePackedDataForPrimitive::new(),
        }
    }

    // FIXME(pcwalton): Don't clone?
    fn clone_packed_primitive(&self, packed_primitive_index: PackedPrimitiveIndex)
                              -> PackedPrimitive {
        let index = packed_primitive_index.class_specific_index().0 as usize;
        match packed_primitive_index.class() {
            PrimitiveClass::Rectangle => {
                PackedPrimitive::Rectangle(self.rectangles.data[index].clone())
            }
            PrimitiveClass::RectangleClip => {
                PackedPrimitive::RectangleClip(self.rectangles_clip.data[index].clone())
            }
            PrimitiveClass::Border => PackedPrimitive::Border(self.borders.data[index].clone()),
            PrimitiveClass::BoxShadow => {
                PackedPrimitive::BoxShadow(self.box_shadows.data[index].clone())
            }
            PrimitiveClass::Text => PackedPrimitive::Glyph(self.glyphs.data[index].clone()),
            PrimitiveClass::Image => PackedPrimitive::Image(self.images.data[index].clone()),
            PrimitiveClass::Gradient => {
                PackedPrimitive::Gradient(self.gradients.data[index].clone())
            }
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct PrimitiveIndex(usize);

#[derive(Debug)]
struct Primitive {
    rect: Rect<f32>,
    clip_index: Option<ClipIndex>,
    xf_rect: Option<TransformedRect>,
    details: PrimitiveDetails,
}

impl Primitive {
    #[inline(never)]
    fn pack(&self,
            batch: &mut FramePackedPrimList,
            auxiliary_lists: &AuxiliaryLists,
            _transform_kind: TransformedRectKind,
            ctx: &RenderTargetContext)
            -> PackedPrimitiveRange {
        /*if transform_kind != batch.transform_kind {
            return false;
        }*/

        let packed_primitive_range;
        match self.details {
            PrimitiveDetails::Rectangle(ref details) => {
                match self.clip_index {
                    Some(clip_index) => {
                        let start_index = PackedPrimitiveIndex::new(
                            PrimitiveClass::RectangleClip,
                            ClassSpecificPackedPrimitiveIndex(batch.rectangles_clip.data.len() as
                                                              u16));

                        let ClipIndex(clip_index) = clip_index;
                        batch.rectangles_clip.data.push(PackedRectanglePrimitiveClip {
                            common: PackedPrimitiveInfo {
                                padding: 0,
                                part: PrimitivePart::Invalid,
                                local_clip_rect: self.rect,
                            },
                            local_rect: self.rect,
                            color: details.color,
                            clip: ctx.clips[clip_index].clone(),
                        });

                        let end_index = PackedPrimitiveIndex::new(
                            PrimitiveClass::RectangleClip,
                            ClassSpecificPackedPrimitiveIndex(batch.rectangles_clip.data.len() as
                                                              u16));
                        packed_primitive_range = PackedPrimitiveRange::new(start_index, end_index);
                    }
                    None => {
                        let start_index = PackedPrimitiveIndex::new(
                            PrimitiveClass::Rectangle,
                            ClassSpecificPackedPrimitiveIndex(batch.rectangles.data.len() as u16));

                        batch.rectangles.data.push(PackedRectanglePrimitive {
                            common: PackedPrimitiveInfo {
                                padding: 0,
                                part: PrimitivePart::Invalid,
                                local_clip_rect: self.rect,
                            },
                            local_rect: self.rect,
                            color: details.color,
                        });

                        let end_index = PackedPrimitiveIndex::new(
                            PrimitiveClass::Rectangle,
                            ClassSpecificPackedPrimitiveIndex(batch.rectangles.data.len() as
                                                              u16));
                        packed_primitive_range = PackedPrimitiveRange::new(start_index, end_index);
                    }
                }
            }
            PrimitiveDetails::Image(ref details) => {
                let start_index = PackedPrimitiveIndex::new(
                    PrimitiveClass::Image,
                    ClassSpecificPackedPrimitiveIndex(batch.images.data.len() as u16));

                let image_info = ctx.resource_cache.get_image(details.image_key,
                                                              details.image_rendering,
                                                              ctx.frame_id);
                let uv_rect = image_info.uv_rect();

                // TODO(gw): Need a general solution to handle multiple texture pages per tile in
                // WR2!
                assert!(batch.color_texture_id == TextureId(0) ||
                        batch.color_texture_id == image_info.texture_id);
                batch.color_texture_id = image_info.texture_id;

                batch.images.data.push(PackedImagePrimitive {
                    common: PackedPrimitiveInfo {
                        padding: 0,
                        part: PrimitivePart::Invalid,
                        local_clip_rect: self.rect,
                    },
                    local_rect: self.rect,
                    st0: uv_rect.top_left,
                    st1: uv_rect.bottom_right,
                    stretch_size: self.rect.size,
                });

                let end_index = PackedPrimitiveIndex::new(
                    PrimitiveClass::Image,
                    ClassSpecificPackedPrimitiveIndex(batch.images.data.len() as u16));
                packed_primitive_range = PackedPrimitiveRange::new(start_index, end_index);
            }
            PrimitiveDetails::Border(ref details) => {
                let start_index = PackedPrimitiveIndex::new(
                    PrimitiveClass::Border,
                    ClassSpecificPackedPrimitiveIndex(batch.borders.data.len() as u16));

                let (left_width, right_width) = (details.left_width, details.right_width);
                let inner_radius = BorderRadius {
                    top_left: Size2D::new(details.radius.top_left.width - left_width,
                                          details.radius.top_left.width - left_width),
                    top_right: Size2D::new(details.radius.top_right.width - right_width,
                                           details.radius.top_right.width - right_width),
                    bottom_left: Size2D::new(details.radius.bottom_left.width - left_width,
                                             details.radius.bottom_left.width - left_width),
                    bottom_right: Size2D::new(details.radius.bottom_right.width - right_width,
                                              details.radius.bottom_right.width - right_width),
                };

                let _clip = Clip::from_border_radius(&self.rect,
                                                     &details.radius,
                                                     &inner_radius);

                batch.borders.data.push(PackedBorderPrimitive {
                    common: PackedPrimitiveInfo {
                        padding: 0,
                        part: PrimitivePart::TopLeft,
                        local_clip_rect: self.rect,
                    },
                    local_rect: rect_from_points_f(details.tl_outer.x,
                                                   details.tl_outer.y,
                                                   details.tl_inner.x,
                                                   details.tl_inner.y),
                    color0: details.top_color,
                    color1: details.left_color,
                    outer_radius_x: details.radius.top_left.width,
                    outer_radius_y: details.radius.top_left.height,
                    inner_radius_x: inner_radius.top_left.width,
                    inner_radius_y: inner_radius.top_left.height,
                });

                batch.borders.data.push(PackedBorderPrimitive {
                    common: PackedPrimitiveInfo {
                        padding: 0,
                        part: PrimitivePart::TopRight,
                        local_clip_rect: self.rect,
                    },
                    local_rect: rect_from_points_f(details.tr_inner.x,
                                                   details.tr_outer.y,
                                                   details.tr_outer.x,
                                                   details.tr_inner.y),
                    color0: details.right_color,
                    color1: details.top_color,
                    outer_radius_x: details.radius.top_right.width,
                    outer_radius_y: details.radius.top_right.height,
                    inner_radius_x: inner_radius.top_right.width,
                    inner_radius_y: inner_radius.top_right.height,
                });

                batch.borders.data.push(PackedBorderPrimitive {
                    common: PackedPrimitiveInfo {
                        padding: 0,
                        part: PrimitivePart::BottomLeft,
                        local_clip_rect: self.rect,
                    },
                    local_rect: rect_from_points_f(details.bl_outer.x,
                                                   details.bl_inner.y,
                                                   details.bl_inner.x,
                                                   details.bl_outer.y),
                    color0: details.left_color,
                    color1: details.bottom_color,
                    outer_radius_x: details.radius.bottom_left.width,
                    outer_radius_y: details.radius.bottom_left.height,
                    inner_radius_x: inner_radius.bottom_left.width,
                    inner_radius_y: inner_radius.bottom_left.height,
                });

                batch.borders.data.push(PackedBorderPrimitive {
                    common: PackedPrimitiveInfo {
                        padding: 0,
                        part: PrimitivePart::BottomRight,
                        local_clip_rect: self.rect,
                    },
                    local_rect: rect_from_points_f(details.br_inner.x,
                                                   details.br_inner.y,
                                                   details.br_outer.x,
                                                   details.br_outer.y),
                    color0: details.right_color,
                    color1: details.bottom_color,
                    outer_radius_x: details.radius.bottom_right.width,
                    outer_radius_y: details.radius.bottom_right.height,
                    inner_radius_x: inner_radius.bottom_right.width,
                    inner_radius_y: inner_radius.bottom_right.height,
                });

                batch.borders.data.push(PackedBorderPrimitive {
                    common: PackedPrimitiveInfo {
                        padding: 0,
                        part: PrimitivePart::Left,
                        local_clip_rect: self.rect,
                    },
                    local_rect: rect_from_points_f(details.tl_outer.x,
                                                   details.tl_inner.y,
                                                   details.tl_outer.x + details.left_width,
                                                   details.bl_inner.y),
                    color0: details.left_color,
                    color1: details.left_color,
                    outer_radius_x: 0.0,
                    outer_radius_y: 0.0,
                    inner_radius_x: 0.0,
                    inner_radius_y: 0.0,
                });

                batch.borders.data.push(PackedBorderPrimitive {
                    common: PackedPrimitiveInfo {
                        padding: 0,
                        part: PrimitivePart::Right,
                        local_clip_rect: self.rect,
                    },
                    local_rect: rect_from_points_f(details.tr_outer.x - details.right_width,
                                                   details.tr_inner.y,
                                                   details.br_outer.x,
                                                   details.br_inner.y),
                    color0: details.right_color,
                    color1: details.right_color,
                    outer_radius_x: 0.0,
                    outer_radius_y: 0.0,
                    inner_radius_x: 0.0,
                    inner_radius_y: 0.0,
                });

                batch.borders.data.push(PackedBorderPrimitive {
                    common: PackedPrimitiveInfo {
                        padding: 0,
                        part: PrimitivePart::Top,
                        local_clip_rect: self.rect,
                    },
                    local_rect: rect_from_points_f(details.tl_inner.x,
                                                   details.tl_outer.y,
                                                   details.tr_inner.x,
                                                   details.tr_outer.y + details.top_width),
                    color0: details.top_color,
                    color1: details.top_color,
                    outer_radius_x: 0.0,
                    outer_radius_y: 0.0,
                    inner_radius_x: 0.0,
                    inner_radius_y: 0.0,
                });

                batch.borders.data.push(PackedBorderPrimitive {
                    common: PackedPrimitiveInfo {
                        padding: 0,
                        part: PrimitivePart::Bottom,
                        local_clip_rect: self.rect,
                    },
                    local_rect: rect_from_points_f(details.bl_inner.x,
                                                   details.bl_outer.y - details.bottom_width,
                                                   details.br_inner.x,
                                                   details.br_outer.y),
                    color0: details.bottom_color,
                    color1: details.bottom_color,
                    outer_radius_x: 0.0,
                    outer_radius_y: 0.0,
                    inner_radius_x: 0.0,
                    inner_radius_y: 0.0,
                });

                let end_index = PackedPrimitiveIndex::new(
                    PrimitiveClass::Border,
                    ClassSpecificPackedPrimitiveIndex(batch.borders.data.len() as u16));
                packed_primitive_range = PackedPrimitiveRange::new(start_index, end_index);
            }
            PrimitiveDetails::Gradient(ref details) => {
                let start_index = PackedPrimitiveIndex::new(
                    PrimitiveClass::Gradient,
                    ClassSpecificPackedPrimitiveIndex(batch.gradients.data.len() as u16));

                let stops = auxiliary_lists.gradient_stops(&details.stops_range);
                for i in 0..(stops.len() - 1) {
                    let (prev_stop, next_stop) = (&stops[i], &stops[i + 1]);
                    let piece_origin;
                    let piece_size;
                    match details.dir {
                        AxisDirection::Horizontal => {
                            let prev_x = util::lerp(self.rect.origin.x, self.rect.max_x(), prev_stop.offset);
                            let next_x = util::lerp(self.rect.origin.x, self.rect.max_x(), next_stop.offset);
                            piece_origin = Point2D::new(prev_x, self.rect.origin.y);
                            piece_size = Size2D::new(next_x - prev_x, self.rect.size.height);
                        }
                        AxisDirection::Vertical => {
                            let prev_y = util::lerp(self.rect.origin.y, self.rect.max_y(), prev_stop.offset);
                            let next_y = util::lerp(self.rect.origin.y, self.rect.max_y(), next_stop.offset);
                            piece_origin = Point2D::new(self.rect.origin.x, prev_y);
                            piece_size = Size2D::new(self.rect.size.width, next_y - prev_y);
                        }
                    }

                    let piece_rect = Rect::new(piece_origin, piece_size);

                    batch.gradients.data.push(PackedGradientPrimitive {
                        common: PackedPrimitiveInfo {
                            padding: 0,
                            part: PrimitivePart::Bottom,
                            local_clip_rect: self.rect,
                        },
                        local_rect: piece_rect,
                        color0: prev_stop.color,
                        color1: next_stop.color,
                        padding: [0, 0, 0],
                        dir: details.dir,
                    });
                }

                let end_index = PackedPrimitiveIndex::new(
                    PrimitiveClass::Gradient,
                    ClassSpecificPackedPrimitiveIndex(batch.gradients.data.len() as u16));
                packed_primitive_range = PackedPrimitiveRange::new(start_index, end_index);
            }
            PrimitiveDetails::BoxShadow(ref details) => {
                let start_index = PackedPrimitiveIndex::new(
                    PrimitiveClass::BoxShadow,
                    ClassSpecificPackedPrimitiveIndex(batch.box_shadows.data.len() as u16));

                let mut rects = Vec::new();
                subtract_rect(&self.rect, &details.src_rect, &mut rects);

                for rect in rects {
                    batch.box_shadows.data.push(PackedBoxShadowPrimitive {
                        common: PackedPrimitiveInfo {
                            padding: 0,
                            part: PrimitivePart::Invalid,
                            local_clip_rect: self.rect,
                        },
                        local_rect: rect,
                        color: details.color,

                        border_radii: Point2D::new(details.border_radius, details.border_radius),
                        blur_radius: details.blur_radius,
                        inverted: 0.0,
                        bs_rect: details.bs_rect,
                        src_rect: details.src_rect,
                    });
                }

                let end_index = PackedPrimitiveIndex::new(
                    PrimitiveClass::BoxShadow,
                    ClassSpecificPackedPrimitiveIndex(batch.box_shadows.data.len() as u16));
                packed_primitive_range = PackedPrimitiveRange::new(start_index, end_index);
            }
            PrimitiveDetails::Text(ref details) => {
                let start_index = PackedPrimitiveIndex::new(
                    PrimitiveClass::Text,
                    ClassSpecificPackedPrimitiveIndex(batch.glyphs.data.len() as u16));

                let src_glyphs = auxiliary_lists.glyph_instances(&details.glyph_range);
                let mut glyph_key = GlyphKey::new(details.font_key,
                                                  details.size,
                                                  details.blur_radius,
                                                  src_glyphs[0].index);
                let blur_offset = details.blur_radius.to_f32_px() * (BLUR_INFLATION_FACTOR as f32) / 2.0;

                for glyph in src_glyphs {
                    glyph_key.index = glyph.index;
                    let image_info = ctx.resource_cache.get_glyph(&glyph_key, ctx.frame_id);
                    if let Some(image_info) = image_info {
                        // TODO(gw): Need a general solution to handle multiple texture pages per
                        // tile in WR2!
                        assert!(batch.color_texture_id == TextureId(0) ||
                                batch.color_texture_id == image_info.texture_id);
                        batch.color_texture_id = image_info.texture_id;

                        let x = glyph.x + image_info.user_data.x0 as f32 / ctx.device_pixel_ratio -
                            blur_offset;
                        let y = glyph.y - image_info.user_data.y0 as f32 / ctx.device_pixel_ratio -
                            blur_offset;

                        let width = image_info.requested_rect.size.width as f32 /
                            ctx.device_pixel_ratio;
                        let height = image_info.requested_rect.size.height as f32 /
                            ctx.device_pixel_ratio;

                        let uv_rect = image_info.uv_rect();

                        batch.glyphs.data.push(PackedGlyphPrimitive {
                            common: PackedPrimitiveInfo {
                                padding: 0,
                                part: PrimitivePart::Invalid,
                                local_clip_rect: self.rect,
                            },
                            local_rect: Rect::new(Point2D::new(x, y),
                                                  Size2D::new(width, height)),
                            color: details.color,
                            st0: uv_rect.top_left,
                            st1: uv_rect.bottom_right,
                        });
                    }
                }

                let end_index = PackedPrimitiveIndex::new(
                    PrimitiveClass::Text,
                    ClassSpecificPackedPrimitiveIndex(batch.glyphs.data.len() as u16));
                packed_primitive_range = PackedPrimitiveRange::new(start_index, end_index);
            }
        }
        packed_primitive_range
    }
}

#[repr(u32)]
#[derive(Debug, Copy, Clone)]
enum PrimitivePart {
    Invalid = 0,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    Top,
    Left,
    Bottom,
    Right,
}

#[derive(Debug, Clone)]
pub struct CompositeTile {
    pub screen_rect: Rect<DevicePixel>,
    pub src_rects: [Rect<DevicePixel>; MAX_LAYERS_PER_PASS],
    pub blend_info: [f32; MAX_LAYERS_PER_PASS],
}

impl CompositeTile {
    fn new(rect: &Rect<DevicePixel>) -> CompositeTile {
        CompositeTile {
            screen_rect: *rect,
            src_rects: unsafe { mem::uninitialized() },
            blend_info: [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ],
        }
    }
}

#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub struct CompositeBatchKey {
    pub shader: CompositeShader,
    //pub samplers: [TextureId; MAX_PRIMS_PER_COMPOSITE],
}

impl CompositeBatchKey {
    fn new(shader: CompositeShader,
           //samplers: [TextureId; MAX_PRIMS_PER_COMPOSITE]
           ) -> CompositeBatchKey {
        CompositeBatchKey {
            shader: shader,
            //samplers: samplers,
        }
    }
}

#[derive(Debug)]
pub struct PackedTile {
    actual_rect: Rect<DevicePixel>,
    target_rect: Rect<DevicePixel>,
}

#[derive(Debug)]
pub struct PackedLayer {
    transform: Matrix4D<f32>,
    inv_transform: Matrix4D<f32>,
    screen_vertices: [Point4D<f32>; 4],
    blend_info: [f32; 2],
    padding: [u32; 2],
}

#[derive(Debug, Clone)]
pub struct PackedPrimitiveInfo {
    part: PrimitivePart,
    padding: u32,
    local_clip_rect: Rect<f32>,
}

#[derive(Debug, Clone)]
pub struct PackedRectanglePrimitiveClip {
    common: PackedPrimitiveInfo,
    local_rect: Rect<f32>,
    color: ColorF,
    clip: Clip,
}

#[derive(Debug, Clone)]
pub struct PackedRectanglePrimitive {
    pub common: PackedPrimitiveInfo,
    pub local_rect: Rect<f32>,
    color: ColorF,
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct TilePackedPrimitive<P> {
    pub layer_index: LayerUboIndex,
    pub tile_index: u32,
    pub primitive: P,
}

impl<P> TilePackedPrimitive<P> {
    fn new(layer_index: LayerUboIndex, tile_index: u32, primitive: P) -> TilePackedPrimitive<P> {
        TilePackedPrimitive {
            layer_index: layer_index,
            tile_index: tile_index,
            primitive: primitive,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PackedGlyphPrimitive {
    pub common: PackedPrimitiveInfo,
    pub local_rect: Rect<f32>,
    color: ColorF,
    st0: Point2D<f32>,
    st1: Point2D<f32>,
}

#[derive(Debug, Clone)]
pub struct PackedImagePrimitive {
    common: PackedPrimitiveInfo,
    local_rect: Rect<f32>,
    st0: Point2D<f32>,
    st1: Point2D<f32>,
    stretch_size: Size2D<f32>,
}

#[derive(Debug, Clone)]
pub struct PackedGradientPrimitive {
    common: PackedPrimitiveInfo,
    local_rect: Rect<f32>,
    color0: ColorF,
    color1: ColorF,
    dir: AxisDirection,
    padding: [u32; 3],
}

#[derive(Debug, Clone)]
pub struct PackedBorderPrimitive {
    common: PackedPrimitiveInfo,
    local_rect: Rect<f32>,
    color0: ColorF,
    color1: ColorF,
    outer_radius_x: f32,
    outer_radius_y: f32,
    inner_radius_x: f32,
    inner_radius_y: f32,
}

#[derive(Debug, Clone)]
pub struct PackedBoxShadowPrimitive {
    common: PackedPrimitiveInfo,
    local_rect: Rect<f32>,
    color: ColorF,
    border_radii: Point2D<f32>,
    blur_radius: f32,
    inverted: f32,
    bs_rect: Rect<f32>,
    src_rect: Rect<f32>,
}

/// Top 3 bits: primitive batch type; bottom 13 bits: class-specific index.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PackedPrimitiveIndex(u16);

impl PackedPrimitiveIndex {
    fn new(class: PrimitiveClass, index: ClassSpecificPackedPrimitiveIndex)
           -> PackedPrimitiveIndex {
        debug_assert!(index.0 < (1 << 13));
        PackedPrimitiveIndex(((class as u16) << 13) | (index.0 as u16))
    }

    fn class(self) -> PrimitiveClass {
        PrimitiveClass::from((self.0 >> 13) as u8)
    }

    fn class_specific_index(self) -> ClassSpecificPackedPrimitiveIndex {
        ClassSpecificPackedPrimitiveIndex(self.0 & !(7 << 13))
    }
}

#[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
pub struct ClassSpecificPackedPrimitiveIndex(u16);

impl ClassSpecificPackedPrimitiveIndex {
    fn zero() -> ClassSpecificPackedPrimitiveIndex {
        ClassSpecificPackedPrimitiveIndex(0)
    }

    fn dec(&mut self) {
        *self = ClassSpecificPackedPrimitiveIndex(self.0 - 1)
    }
}

#[derive(Debug)]
pub enum PrimitiveBatchData {
    Rectangles(Vec<TilePackedPrimitive<PackedRectanglePrimitive>>),
    RectanglesClip(Vec<TilePackedPrimitive<PackedRectanglePrimitiveClip>>),
    Borders(Vec<TilePackedPrimitive<PackedBorderPrimitive>>),
    BoxShadows(Vec<TilePackedPrimitive<PackedBoxShadowPrimitive>>),
    Text(Vec<TilePackedPrimitive<PackedGlyphPrimitive>>),
    Image(Vec<TilePackedPrimitive<PackedImagePrimitive>>),
    Gradient(Vec<TilePackedPrimitive<PackedGradientPrimitive>>),
}

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum PrimitiveClass {
    Rectangle = 0,
    RectangleClip = 1,
    Border = 2,
    BoxShadow = 3,
    Text = 4,
    Image = 5,
    Gradient = 6,
}

impl PrimitiveClass {
    fn from(n: u8) -> PrimitiveClass {
        match n {
            0 => PrimitiveClass::Rectangle,
            1 => PrimitiveClass::RectangleClip,
            2 => PrimitiveClass::Border,
            3 => PrimitiveClass::BoxShadow,
            4 => PrimitiveClass::Text,
            5 => PrimitiveClass::Image,
            6 => PrimitiveClass::Gradient,
            _ => panic!("Not a valid primitive class number!"),
        }
    }
}

#[derive(Debug)]
pub struct PrimitiveBatch {
    pub transform_kind: TransformedRectKind,
    pub color_texture_id: TextureId,        // TODO(gw): Expand to sampler array to handle all glyphs!
    pub data: PrimitiveBatchData,
}

impl PrimitiveBatch {
    fn new(prim: &PackedPrimitive,
           transform_kind: TransformedRectKind,
           color_texture_id: TextureId)
           -> PrimitiveBatch {
        let data = match *prim {
            PackedPrimitive::Rectangle(..) => PrimitiveBatchData::Rectangles(vec![]),
            PackedPrimitive::RectangleClip(..) => PrimitiveBatchData::RectanglesClip(vec![]),
            PackedPrimitive::Border(..) => PrimitiveBatchData::Borders(vec![]),
            PackedPrimitive::BoxShadow(..) => PrimitiveBatchData::BoxShadows(vec![]),
            PackedPrimitive::Glyph(..) => PrimitiveBatchData::Text(vec![]),
            PackedPrimitive::Image(..) => PrimitiveBatchData::Image(vec![]),
            PackedPrimitive::Gradient(..) => PrimitiveBatchData::Gradient(vec![]),
        };

        PrimitiveBatch {
            color_texture_id: color_texture_id,
            transform_kind: transform_kind,
            data: data,
        }
    }

    fn push_item_if_possible(&mut self,
                             packed_primitive_index: ClassSpecificPackedPrimitiveIndex,
                             frame_primitives: &FramePackedPrimList,
                             layer_to_ubo_map: &[Option<LayerUboIndex>],
                             _transform_kind: TransformedRectKind,
                             layer_index: StackingContextIndex,
                             tile_index: u32)
                             -> bool {
        let layer_ubo_index =
            layer_to_ubo_map[layer_index.0].expect("No entry for layer in layer-to-UBO map?!");

        // FIXME(pcwalton): Stop cloning?
        match self.data {
            PrimitiveBatchData::Rectangles(ref mut primitives) => {
                let rectangle = frame_primitives.rectangles
                                                .data[packed_primitive_index.0 as usize]
                                                .clone();
                primitives.push(TilePackedPrimitive::new(layer_ubo_index, tile_index, rectangle));
            }
            PrimitiveBatchData::RectanglesClip(ref mut primitives) => {
                let rectangle = frame_primitives.rectangles_clip
                                                .data[packed_primitive_index.0 as usize]
                                                .clone();
                primitives.push(TilePackedPrimitive::new(layer_ubo_index, tile_index, rectangle))
            }
            PrimitiveBatchData::Borders(ref mut primitives) => {
                let border = frame_primitives.borders.data[packed_primitive_index.0 as usize]
                                                     .clone();
                primitives.push(TilePackedPrimitive::new(layer_ubo_index, tile_index, border))
            }
            PrimitiveBatchData::BoxShadows(ref mut primitives) => {
                let box_shadow = frame_primitives.box_shadows
                                                 .data[packed_primitive_index.0 as usize]
                                                 .clone();
                primitives.push(TilePackedPrimitive::new(layer_ubo_index, tile_index, box_shadow))
            }
            PrimitiveBatchData::Text(ref mut primitives) => {
                let glyph = frame_primitives.glyphs.data[packed_primitive_index.0 as usize]
                                                   .clone();
                primitives.push(TilePackedPrimitive::new(layer_ubo_index, tile_index, glyph))
            }
            PrimitiveBatchData::Image(ref mut primitives) => {
                let image = frame_primitives.images.data[packed_primitive_index.0 as usize]
                                                   .clone();
                primitives.push(TilePackedPrimitive::new(layer_ubo_index, tile_index, image))
            }
            PrimitiveBatchData::Gradient(ref mut primitives) => {
                let gradient = frame_primitives.gradients.data[packed_primitive_index.0 as usize]
                                                         .clone();
                primitives.push(TilePackedPrimitive::new(layer_ubo_index, tile_index, gradient))
            }
        }
        true
    }

    fn class(&self) -> PrimitiveClass {
        match self.data {
            PrimitiveBatchData::Rectangles(_) |
            PrimitiveBatchData::RectanglesClip(_) => PrimitiveClass::Rectangle,
            PrimitiveBatchData::Borders(_) => PrimitiveClass::Border,
            PrimitiveBatchData::BoxShadows(_) => PrimitiveClass::BoxShadow,
            PrimitiveBatchData::Text(_) => PrimitiveClass::Text,
            PrimitiveBatchData::Image(_) => PrimitiveClass::Image,
            PrimitiveBatchData::Gradient(_) => PrimitiveClass::Gradient,
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct StackingContextChunkIndex(usize);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct StackingContextIndex(usize);

struct StackingContext {
    pipeline_id: PipelineId,
    local_transform: Matrix4D<f32>,
    local_rect: Rect<f32>,
    local_offset: Point2D<f32>,
    primitives: Vec<Primitive>,
    scroll_layer_id: ScrollLayerId,
    opacity: f32,
    transform: Matrix4D<f32>,
    xf_rect: Option<TransformedRect>,
}

impl StackingContext {
    fn build_resource_list(&self,
                           resource_list: &mut ResourceList,
                           //index_buffer: &Vec<PrimitiveIndex>,
                           auxiliary_lists: &AuxiliaryLists) {
        for prim in &self.primitives {
            match prim.details {
                PrimitiveDetails::Rectangle(..) => {}
                PrimitiveDetails::Gradient(..) => {}
                PrimitiveDetails::Border(..) => {}
                PrimitiveDetails::BoxShadow(..) => {}
                PrimitiveDetails::Image(ref details) => {
                   resource_list.add_image(details.image_key,
                                            details.image_rendering);
                }
                PrimitiveDetails::Text(ref details) => {
                    let glyphs = auxiliary_lists.glyph_instances(&details.glyph_range);
                    for glyph in glyphs {
                        let glyph = Glyph::new(details.size, details.blur_radius, glyph.index);
                        resource_list.add_glyph(details.font_key, glyph);
                    }
                }
            }
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct ClipIndex(usize);

#[derive(Debug, Clone)]
pub struct ClipCorner {
    rect: Rect<f32>,
    outer_radius_x: f32,
    outer_radius_y: f32,
    inner_radius_x: f32,
    inner_radius_y: f32,
}

#[derive(Debug, Clone)]
pub struct Clip {
    rect: Rect<f32>,
    top_left: ClipCorner,
    top_right: ClipCorner,
    bottom_left: ClipCorner,
    bottom_right: ClipCorner,
}

#[derive(Debug, Clone)]
pub struct ClearTile {
    pub rect: Rect<DevicePixel>,
}

#[derive(Clone, Copy)]
pub struct FrameBuilderConfig {
    max_prim_layers: usize,
    max_prim_tiles: usize,
}

impl FrameBuilderConfig {
    pub fn new(max_prim_layers: usize,
               max_prim_tiles: usize) -> FrameBuilderConfig {
        FrameBuilderConfig {
            max_prim_layers: max_prim_layers,
            max_prim_tiles: max_prim_tiles,
        }
    }
}

pub struct FrameBuilder {
    screen_rect: Rect<i32>,
    layers: Vec<StackingContext>,
    layer_stack: Vec<StackingContextIndex>,
    clips: Vec<Clip>,
    clip_stack: Vec<ClipIndex>,
    device_pixel_ratio: f32,
    debug: bool,
    config: FrameBuilderConfig,
}

pub struct Frame {
    pub debug_rects: Vec<DebugRect>,
    pub cache_size: Size2D<f32>,
    pub phases: Vec<RenderPhase>,
    pub clear_tiles: Vec<ClearTile>,
}

impl Clip {
    pub fn from_clip_region(clip: &ComplexClipRegion) -> Clip {
        Clip {
            rect: clip.rect,
            top_left: ClipCorner {
                rect: Rect::new(Point2D::new(clip.rect.origin.x, clip.rect.origin.y),
                                Size2D::new(clip.radii.top_left.width, clip.radii.top_left.height)),
                outer_radius_x: clip.radii.top_left.width,
                outer_radius_y: clip.radii.top_left.height,
                inner_radius_x: 0.0,
                inner_radius_y: 0.0,
            },
            top_right: ClipCorner {
                rect: Rect::new(Point2D::new(clip.rect.origin.x + clip.rect.size.width - clip.radii.top_right.width,
                                             clip.rect.origin.y),
                                Size2D::new(clip.radii.top_right.width, clip.radii.top_right.height)),
                outer_radius_x: clip.radii.top_right.width,
                outer_radius_y: clip.radii.top_right.height,
                inner_radius_x: 0.0,
                inner_radius_y: 0.0,
            },
            bottom_left: ClipCorner {
                rect: Rect::new(Point2D::new(clip.rect.origin.x,
                                             clip.rect.origin.y + clip.rect.size.height - clip.radii.bottom_left.height),
                                Size2D::new(clip.radii.bottom_left.width, clip.radii.bottom_left.height)),
                outer_radius_x: clip.radii.bottom_left.width,
                outer_radius_y: clip.radii.bottom_left.height,
                inner_radius_x: 0.0,
                inner_radius_y: 0.0,
            },
            bottom_right: ClipCorner {
                rect: Rect::new(Point2D::new(clip.rect.origin.x + clip.rect.size.width - clip.radii.bottom_right.width,
                                             clip.rect.origin.y + clip.rect.size.height - clip.radii.bottom_right.height),
                                Size2D::new(clip.radii.bottom_right.width, clip.radii.bottom_right.height)),
                outer_radius_x: clip.radii.bottom_right.width,
                outer_radius_y: clip.radii.bottom_right.height,
                inner_radius_x: 0.0,
                inner_radius_y: 0.0,
            },
        }
    }

    pub fn from_border_radius(rect: &Rect<f32>,
                              outer_radius: &BorderRadius,
                              inner_radius: &BorderRadius) -> Clip {
        Clip {
            rect: *rect,
            top_left: ClipCorner {
                rect: Rect::new(Point2D::new(rect.origin.x, rect.origin.y),
                                Size2D::new(outer_radius.top_left.width, outer_radius.top_left.height)),
                outer_radius_x: outer_radius.top_left.width,
                outer_radius_y: outer_radius.top_left.height,
                inner_radius_x: inner_radius.top_left.width,
                inner_radius_y: inner_radius.top_left.height,
            },
            top_right: ClipCorner {
                rect: Rect::new(Point2D::new(rect.origin.x + rect.size.width - outer_radius.top_right.width,
                                             rect.origin.y),
                                Size2D::new(outer_radius.top_right.width, outer_radius.top_right.height)),
                outer_radius_x: outer_radius.top_right.width,
                outer_radius_y: outer_radius.top_right.height,
                inner_radius_x: inner_radius.top_right.width,
                inner_radius_y: inner_radius.top_right.height,
            },
            bottom_left: ClipCorner {
                rect: Rect::new(Point2D::new(rect.origin.x,
                                             rect.origin.y + rect.size.height - outer_radius.bottom_left.height),
                                Size2D::new(outer_radius.bottom_left.width, outer_radius.bottom_left.height)),
                outer_radius_x: outer_radius.bottom_left.width,
                outer_radius_y: outer_radius.bottom_left.height,
                inner_radius_x: inner_radius.bottom_left.width,
                inner_radius_y: inner_radius.bottom_left.height,
            },
            bottom_right: ClipCorner {
                rect: Rect::new(Point2D::new(rect.origin.x + rect.size.width - outer_radius.bottom_right.width,
                                             rect.origin.y + rect.size.height - outer_radius.bottom_right.height),
                                Size2D::new(outer_radius.bottom_right.width, outer_radius.bottom_right.height)),
                outer_radius_x: outer_radius.bottom_right.width,
                outer_radius_y: outer_radius.bottom_right.height,
                inner_radius_x: inner_radius.bottom_right.width,
                inner_radius_y: inner_radius.bottom_right.height,
            },
        }
    }
}

#[derive(Debug)]
struct CompositeTileInfo {
    layer_indices: Vec<Option<StackingContextIndex>>,
}

#[derive(Debug)]
struct ScreenTileLayer {
    actual_rect: Rect<DevicePixel>,
    layer_index: StackingContextIndex,
    /// First primitive index in the screen tile map for this tile.
    first_prim_sample_index: u8,
    /// Last primitive index in the screen tile map for this tile.
    last_prim_sample_index: u8,
    global_tile_index: usize,
    layer_opacity: f32,
    is_opaque: bool,
}

impl ScreenTileLayer {
    fn tile_location(&self) -> Point2D<u32> {
        Point2D {
            x: ((self.actual_rect.origin.x.0 as usize) / SCREEN_TILE_SIZE) as u32,
            y: ((self.actual_rect.origin.y.0 as usize) / SCREEN_TILE_SIZE) as u32,
        }
    }

    fn prim_count(&self) -> u8 {
        self.last_prim_sample_index - self.first_prim_sample_index
    }

    fn compile(&mut self,
               _layer: &StackingContext,
               _tile_prim_indices: PrimitiveIndices,
               _screen_rect: &Rect<DevicePixel>) {
        // FIXME(pcwalton)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ScreenTileIndex(usize);

#[derive(Debug)]
struct CompiledScreenTile {
    main_render_task: RenderTask,
    required_target_count: usize,
}

impl CompiledScreenTile {
    fn new(main_render_task: RenderTask) -> CompiledScreenTile {
        let mut required_target_count = 0;
        main_render_task.max_depth(0, &mut required_target_count);

        CompiledScreenTile {
            main_render_task: main_render_task,
            required_target_count: required_target_count,
        }
    }
}

#[derive(Debug)]
struct ScreenTile {
    rect: Rect<DevicePixel>,
    layers: Vec<ScreenTileLayer>,
}

impl ScreenTile {
    fn new(rect: Rect<DevicePixel>) -> ScreenTile {
        ScreenTile {
            rect: rect,
            layers: Vec::new(),
        }
    }

    fn layer_count(&self) -> usize {
        self.layers.len()
    }

    fn compile(&mut self) -> Option<CompiledScreenTile> {
        if self.layers.len() == 0 {
            return None
        }

        // TODO(gw): If a single had blending, fall through to the
        //           compositing case below. Investigate if it's
        //           worth having a special path for this?
        if self.layers.len() == 1 && self.layers[0].layer_opacity == 1.0 {
            let task = RenderTask::from_layer(self.layers.pop().unwrap(),
                                              RenderTaskLocation::Fixed(self.rect));
            return Some(CompiledScreenTile::new(task))
        }

        let mut layer_indices_in_current_layer = Vec::new();
        let mut tasks_in_current_layer = Vec::new();

        for layer in self.layers.drain(..) {
            if tasks_in_current_layer.len() == MAX_LAYERS_PER_PASS {
                let composite_location = RenderTaskLocation::Dynamic(None, self.rect.size);
                let tasks_to_composite = mem::replace(&mut tasks_in_current_layer, Vec::new());
                let layers_to_composite = mem::replace(&mut layer_indices_in_current_layer,
                                                       Vec::new());
                let composite_task = RenderTask::composite(tasks_to_composite,
                                                           composite_location,
                                                           layers_to_composite);
                debug_assert!(tasks_in_current_layer.is_empty());
                tasks_in_current_layer.push(composite_task);
                layer_indices_in_current_layer.push(None);
            }

            layer_indices_in_current_layer.push(Some(layer.layer_index));
            let layer_task =
                RenderTask::from_layer(layer, RenderTaskLocation::Dynamic(None, self.rect.size));
            tasks_in_current_layer.push(layer_task);
        }

        debug_assert!(!tasks_in_current_layer.is_empty());
        let main_task = RenderTask::composite(tasks_in_current_layer,
                                              RenderTaskLocation::Fixed(self.rect),
                                              layer_indices_in_current_layer);

        Some(CompiledScreenTile::new(main_task))
    }
}

impl FrameBuilder {
    pub fn new(viewport_size: Size2D<f32>,
               device_pixel_ratio: f32,
               debug: bool,
               config: FrameBuilderConfig) -> FrameBuilder {
        let viewport_size = Size2D::new(viewport_size.width as i32, viewport_size.height as i32);
        FrameBuilder {
            screen_rect: Rect::new(Point2D::zero(), viewport_size),
            layers: Vec::new(),
            layer_stack: Vec::new(),
            device_pixel_ratio: device_pixel_ratio,
            debug: debug,
            clips: Vec::new(),
            clip_stack: Vec::new(),
            config: config,
        }
    }

    pub fn set_clip(&mut self, clip: Clip) {
        let clip_index = ClipIndex(self.clips.len());
        // TODO(gw): Handle intersecting clips!
        self.clips.push(clip);
        self.clip_stack.push(clip_index);
    }

    pub fn clear_clip(&mut self) {
        self.clip_stack.pop().unwrap();
    }

    fn add_primitive(&mut self,
                     rect: &Rect<f32>,
                     details: PrimitiveDetails) {
        let current_layer = *self.layer_stack.last().unwrap();
        let StackingContextIndex(layer_index) = current_layer;
        let layer = &mut self.layers[layer_index as usize];

        let prim = Primitive {
            rect: *rect,
            xf_rect: None,
            clip_index: self.clip_stack.last().map(|i| *i),
            details: details,
        };
        layer.primitives.push(prim);
    }

    pub fn push_layer(&mut self,
                      rect: Rect<f32>,
                      _clip_rect: Rect<f32>,
                      transform: Matrix4D<f32>,
                      opacity: f32,
                      pipeline_id: PipelineId,
                      scroll_layer_id: ScrollLayerId,
                      offset: Point2D<f32>) {
        let sc = StackingContext {
            primitives: Vec::new(),
            local_rect: rect,
            local_transform: transform,
            local_offset: offset,
            scroll_layer_id: scroll_layer_id,
            opacity: opacity,
            pipeline_id: pipeline_id,
            xf_rect: None,
            transform: Matrix4D::identity(),
        };

        self.layer_stack.push(StackingContextIndex(self.layers.len()));
        self.layers.push(sc);
    }

    pub fn pop_layer(&mut self) {
        self.layer_stack.pop();
    }

    pub fn add_solid_rectangle(&mut self,
                               rect: &Rect<f32>,
                               color: &ColorF) {
        if color.a == 0.0 {
            return;
        }

        let prim = RectanglePrimitive {
            color: *color,
        };

        self.add_primitive(rect, PrimitiveDetails::Rectangle(prim));
    }

    pub fn add_border(&mut self,
                      rect: Rect<f32>,
                      border: &BorderDisplayItem) {
        let radius = &border.radius;
        let left = &border.left;
        let right = &border.right;
        let top = &border.top;
        let bottom = &border.bottom;

        if (left.style != BorderStyle::Solid && left.style != BorderStyle::None) ||
           (top.style != BorderStyle::Solid && top.style != BorderStyle::None) ||
           (bottom.style != BorderStyle::Solid && bottom.style != BorderStyle::None) ||
           (right.style != BorderStyle::Solid && right.style != BorderStyle::None) {
            println!("TODO: Other border styles {:?} {:?} {:?} {:?}", left.style, top.style, bottom.style, right.style);
            return;
        }

        let tl_outer = Point2D::new(rect.origin.x, rect.origin.y);
        let tl_inner = tl_outer + Point2D::new(radius.top_left.width.max(left.width),
                                               radius.top_left.height.max(top.width));

        let tr_outer = Point2D::new(rect.origin.x + rect.size.width, rect.origin.y);
        let tr_inner = tr_outer + Point2D::new(-radius.top_right.width.max(right.width),
                                               radius.top_right.height.max(top.width));

        let bl_outer = Point2D::new(rect.origin.x, rect.origin.y + rect.size.height);
        let bl_inner = bl_outer + Point2D::new(radius.bottom_left.width.max(left.width),
                                               -radius.bottom_left.height.max(bottom.width));

        let br_outer = Point2D::new(rect.origin.x + rect.size.width,
                                    rect.origin.y + rect.size.height);
        let br_inner = br_outer - Point2D::new(radius.bottom_right.width.max(right.width),
                                               radius.bottom_right.height.max(bottom.width));

        let left_color = left.border_color(1.0, 2.0/3.0, 0.3, 0.7);
        let top_color = top.border_color(1.0, 2.0/3.0, 0.3, 0.7);
        let right_color = right.border_color(2.0/3.0, 1.0, 0.7, 0.3);
        let bottom_color = bottom.border_color(2.0/3.0, 1.0, 0.7, 0.3);

        let prim = BorderPrimitive {
            tl_outer: tl_outer,
            tl_inner: tl_inner,
            tr_outer: tr_outer,
            tr_inner: tr_inner,
            bl_outer: bl_outer,
            bl_inner: bl_inner,
            br_outer: br_outer,
            br_inner: br_inner,
            radius: radius.clone(),
            left_width: left.width,
            top_width: top.width,
            bottom_width: bottom.width,
            right_width: right.width,
            left_color: left_color,
            top_color: top_color,
            bottom_color: bottom_color,
            right_color: right_color,
        };

        self.add_primitive(&rect, PrimitiveDetails::Border(prim));
    }

    pub fn add_gradient(&mut self,
                        rect: Rect<f32>,
                        start_point: Point2D<f32>,
                        end_point: Point2D<f32>,
                        stops: ItemRange) {
        // Fast paths for axis-aligned gradients:
        if start_point.x == end_point.x {
            let rect = Rect::new(Point2D::new(rect.origin.x, start_point.y),
                                 Size2D::new(rect.size.width, end_point.y - start_point.y));
            let prim = GradientPrimitive {
                stops_range: stops,
                dir: AxisDirection::Vertical,
            };
            self.add_primitive(&rect, PrimitiveDetails::Gradient(prim));
        } else if start_point.y == end_point.y {
            let rect = Rect::new(Point2D::new(start_point.x, rect.origin.y),
                                 Size2D::new(end_point.x - start_point.x, rect.size.height));
            let prim = GradientPrimitive {
                stops_range: stops,
                dir: AxisDirection::Horizontal,
            };
            self.add_primitive(&rect, PrimitiveDetails::Gradient(prim));
        } else {
            println!("TODO: Angle gradients! {:?} {:?} {:?}", start_point, end_point, stops);
        }
    }

    pub fn add_text(&mut self,
                    rect: Rect<f32>,
                    font_key: FontKey,
                    size: Au,
                    blur_radius: Au,
                    color: &ColorF,
                    glyph_range: ItemRange) {
        if color.a == 0.0 {
            return
        }

        let prim = TextPrimitive {
            color: *color,
            font_key: font_key,
            size: size,
            blur_radius: blur_radius,
            glyph_range: glyph_range,
        };

        self.add_primitive(&rect, PrimitiveDetails::Text(prim));
    }

    pub fn add_box_shadow(&mut self,
                          box_bounds: &Rect<f32>,
                          box_offset: &Point2D<f32>,
                          color: &ColorF,
                          blur_radius: f32,
                          spread_radius: f32,
                          border_radius: f32,
                          clip_mode: BoxShadowClipMode) {
        if color.a == 0.0 {
            return
        }

        let bs_rect = compute_box_shadow_rect(box_bounds,
                                              box_offset,
                                              spread_radius);

        let metrics = BoxShadowMetrics::new(&bs_rect,
                                            border_radius,
                                            blur_radius);

        let prim_rect = Rect::new(metrics.tl_outer,
                                  Size2D::new(metrics.br_outer.x - metrics.tl_outer.x,
                                              metrics.br_outer.y - metrics.tl_outer.y));

        let prim = BoxShadowPrimitive {
            metrics: metrics,
            src_rect: *box_bounds,
            bs_rect: bs_rect,
            color: *color,
            blur_radius: blur_radius,
            spread_radius: spread_radius,
            border_radius: border_radius,
            clip_mode: clip_mode,
        };

        self.add_primitive(&prim_rect, PrimitiveDetails::BoxShadow(prim));
    }

    pub fn add_image(&mut self,
                     rect: Rect<f32>,
                     _stretch_size: &Size2D<f32>,
                     image_key: ImageKey,
                     image_rendering: ImageRendering) {
        let prim = ImagePrimitive {
            image_key: image_key,
            image_rendering: image_rendering,
        };

        self.add_primitive(&rect, PrimitiveDetails::Image(prim));
    }

    fn cull_layers(&mut self,
                   screen_rect: &Rect<DevicePixel>,
                   layer_map: &HashMap<ScrollLayerId, Layer, BuildHasherDefault<FnvHasher>>) {
        // Remove layers that are transparent.
        self.layers.retain(|layer| {
            layer.opacity > 0.0
        });

        // Build layer screen rects.
        // TODO(gw): This can be done earlier once update_layer_transforms() is fixed.
        for layer in &mut self.layers {
            let scroll_layer = &layer_map[&layer.scroll_layer_id];
            let offset_transform = Matrix4D::identity().translate(layer.local_offset.x,
                                                                  layer.local_offset.y,
                                                                  0.0);
            let transform = scroll_layer.world_transform
                                        .as_ref()
                                        .unwrap()
                                        .mul(&layer.local_transform)
                                        .mul(&offset_transform);
            layer.transform = transform;
            layer.xf_rect = Some(TransformedRect::new(&layer.local_rect,
                                                      &transform,
                                                      self.device_pixel_ratio));
        }

        self.layers.retain(|layer| {
            layer.xf_rect
                 .as_ref()
                 .unwrap()
                 .bounding_rect
                 .intersects(&screen_rect)
        });

        for layer in &mut self.layers {
            for prim in &mut layer.primitives {
                prim.xf_rect = Some(TransformedRect::new(&prim.rect,
                                                         &layer.transform,
                                                         self.device_pixel_ratio));
            }

            layer.primitives.retain(|prim| {
                prim.xf_rect
                    .as_ref()
                    .unwrap()
                    .bounding_rect
                    .intersects(&screen_rect)
            });
        }
    }

    fn pack_primitives(&self,
                       pipeline_auxiliary_lists: &HashMap<PipelineId,
                                                          AuxiliaryLists,
                                                          BuildHasherDefault<FnvHasher>>,
                       screen_tiles: &mut ScreenTileMap,
                       ctx: &RenderTargetContext)
                       -> FramePackedPrimList {
        let mut packed_prim_list = FramePackedPrimList::new();
        for (layer_index, layer) in self.layers.iter().enumerate() {
            for (primitive_index_in_layer, prim) in layer.primitives.iter().enumerate() {
                let packed_primitive_range =
                    prim.pack(&mut packed_prim_list,
                              &pipeline_auxiliary_lists[&layer.pipeline_id],
                              layer.xf_rect.as_ref().unwrap().kind,
                              ctx);
                let primitive_range_index =
                    PackedPrimitiveRangeIndex(screen_tiles.primitive_range_index_map.len() as u32);
                screen_tiles.primitive_range_index_map.push(packed_primitive_range);
                screen_tiles.add_packed_primitive_range(layer_index as u32,
                                                        primitive_index_in_layer as u32,
                                                        primitive_range_index)
            }
        }
        packed_prim_list
    }

    fn assign_prims_to_screen_tiles(&self,
                                    screen_tiles: &mut ScreenTileMap,
                                    debug_rects: &mut Vec<DebugRect>) { //-> usize {
        // TODO(gw): This can be made much faster - calculate tile indices and
        //           assign in a loop.
        let mut next_global_tile_index = 0;
        for y in 0..screen_tiles.tile_size.height {
            for x in 0..screen_tiles.tile_size.width {
                let tile_location = Point2D::new(x, y);
                let tile_rect = screen_tiles.tile_metadata(&tile_location).rect;
                let mut prim_count = 0;
                for (layer_index, layer) in self.layers.iter().enumerate() {
                    let layer_index = StackingContextIndex(layer_index);
                    let layer_rect = layer.xf_rect.as_ref().unwrap().bounding_rect;
                    if !layer_rect.intersects(&tile_rect) {
                        continue
                    }

                    let mut tile_layer = ScreenTileLayer {
                        actual_rect: tile_rect,
                        layer_index: layer_index,
                        first_prim_sample_index: screen_tiles.primitive_count(&tile_location),
                        last_prim_sample_index: screen_tiles.primitive_count(&tile_location),
                        layer_opacity: layer.opacity,
                        global_tile_index: next_global_tile_index,
                        is_opaque: false,
                    };

                    next_global_tile_index += 1;

                    for (prim_index_in_layer, prim) in layer.primitives.iter().enumerate().rev() {
                        let prim_rect = &prim.xf_rect.as_ref().unwrap().bounding_rect;
                        if !prim_rect.intersects(&tile_rect) {
                            continue
                        }

                        let prim_range_index =
                            screen_tiles.layer_packed_primitive_range_index_map[layer_index.0 as
                                                                                usize]
                                        .packed_primitive_range_index_map[prim_index_in_layer as
                                                                          usize];
                        screen_tiles.add_prim_index(&tile_location, prim_range_index);
                        tile_layer.last_prim_sample_index += 1;
                        prim_count += 1;
                    }

                    if tile_layer.prim_count() == 0 {
                        continue
                    }

                    let last_prim_sample_index = tile_layer.last_prim_sample_index;
                    let first_prim_sample_index = tile_layer.first_prim_sample_index;
                    tile_layer.compile(layer,
                                       screen_tiles.primitive_indices(&tile_location,
                                                                      last_prim_sample_index,
                                                                      first_prim_sample_index),
                                       &screen_tiles.tile_metadata(&tile_location).rect);

                    let tile_metadata = screen_tiles.tile_metadata_mut(&tile_location);
                    if tile_layer.is_opaque {
                        tile_metadata.layers.clear()
                    }
                    tile_metadata.layers.push(tile_layer)
                }

                if self.debug {
                    let tile_metadata = screen_tiles.tile_metadata(&tile_location);
                    debug_rects.push(DebugRect {
                        label: format!("{}|{}", tile_metadata.layer_count(), prim_count),
                        color: ColorF::new(1.0, 0.0, 0.0, 1.0),
                        rect: tile_metadata.rect,
                    })
                }
            }
        }

        //pass_count
    }

    fn build_resource_list(&mut self,
                           resource_cache: &mut ResourceCache,
                           frame_id: FrameId,
                           pipeline_auxiliary_lists: &HashMap<PipelineId, AuxiliaryLists, BuildHasherDefault<FnvHasher>>) {
        let mut resource_list = ResourceList::new(self.device_pixel_ratio);

        // Non-visible layers have been removed by now
        for layer in &self.layers {
            let auxiliary_lists = pipeline_auxiliary_lists.get(&layer.pipeline_id)
                                                          .expect("No auxiliary lists?!");

            // Non-visible chunks have also been removed by now
            layer.build_resource_list(&mut resource_list,
                                      //&layer.primitives,
                                      auxiliary_lists);
        }

        resource_cache.add_resource_list(&resource_list,
                                         frame_id);
        resource_cache.raster_pending_glyphs(frame_id);
    }

    fn create_context<'a>(&'a self, resource_cache: &'a ResourceCache, frame_id: FrameId)
                          -> RenderTargetContext<'a> {
        RenderTargetContext {
            layers: &self.layers,
            resource_cache: resource_cache,
            device_pixel_ratio: self.device_pixel_ratio,
            frame_id: frame_id,
            clips: &self.clips,
            alpha_batch_max_layers: self.config.max_prim_layers,
            alpha_batch_max_tiles: self.config.max_prim_tiles,
        }
    }

    pub fn build(&mut self,
                 resource_cache: &mut ResourceCache,
                 frame_id: FrameId,
                 pipeline_auxiliary_lists: &HashMap<PipelineId,
                                                    AuxiliaryLists,
                                                    BuildHasherDefault<FnvHasher>>,
                 layer_map: &HashMap<ScrollLayerId, Layer, BuildHasherDefault<FnvHasher>>)
                 -> Frame {
        let screen_rect = Rect::new(Point2D::zero(),
                                    Size2D::new(DevicePixel::new(self.screen_rect.size.width as f32, self.device_pixel_ratio),
                                                DevicePixel::new(self.screen_rect.size.height as f32, self.device_pixel_ratio)));

        self.cull_layers(&screen_rect, layer_map);

        let mut debug_rects = Vec::new();

        self.build_resource_list(resource_cache, frame_id, pipeline_auxiliary_lists);

        let mut screen_tiles = ScreenTileMap::new(&self.screen_rect, self.device_pixel_ratio);
        let mut frame_packed_prim_list = {
            let ctx = self.create_context(resource_cache, frame_id);
            self.pack_primitives(pipeline_auxiliary_lists, &mut screen_tiles, &ctx)
        };

        self.assign_prims_to_screen_tiles(&mut screen_tiles, &mut debug_rects);

        let mut clear_tiles = Vec::new();

        // Build list of passes, target allocs that each tile needs.
        let mut compiled_screen_tiles = Vec::new();
        for screen_tile in &mut screen_tiles.tile_metadata {
            let rect = screen_tile.rect;
            match screen_tile.compile() {
                Some(compiled_screen_tile) => {
                    compiled_screen_tiles.push(compiled_screen_tile);
                }
                None => {
                    clear_tiles.push(ClearTile {
                        rect: rect,
                    });
                }
            }
        }

        let mut phases = Vec::new();

        let ctx = self.create_context(resource_cache, frame_id);
        if !compiled_screen_tiles.is_empty() {
            // Sort by pass count to minimize render target switches.
            compiled_screen_tiles.sort_by(|a, b| {
                let a_passes = a.required_target_count;
                let b_passes = b.required_target_count;
                b_passes.cmp(&a_passes)
            });

            // Do the allocations now, assigning each tile to a render
            // phase as required.

            let mut current_phase = RenderPhase::new(compiled_screen_tiles[0].required_target_count);

            for compiled_screen_tile in compiled_screen_tiles {
                if let Some(failed_tile) = current_phase.add_compiled_screen_tile(compiled_screen_tile) {
                    let full_phase = mem::replace(&mut current_phase,
                                                  RenderPhase::new(failed_tile.required_target_count));
                    phases.push(full_phase);

                    let result = current_phase.add_compiled_screen_tile(failed_tile);
                    assert!(result.is_none(), "TODO: Handle single tile not fitting in render phase.");
                }
            }

            phases.push(current_phase);

            for phase in &mut phases {
                phase.build(&ctx, &mut frame_packed_prim_list, &screen_tiles);
            }
        }

        Frame {
            debug_rects: debug_rects,
            phases: phases,
            clear_tiles: clear_tiles,
            cache_size: Size2D::new(RENDERABLE_CACHE_SIZE.0 as f32,
                                    RENDERABLE_CACHE_SIZE.0 as f32),
        }
    }
}

fn compute_box_shadow_rect(box_bounds: &Rect<f32>,
                           box_offset: &Point2D<f32>,
                           spread_radius: f32)
                           -> Rect<f32> {
    let mut rect = (*box_bounds).clone();
    rect.origin.x += box_offset.x;
    rect.origin.y += box_offset.y;
    rect.inflate(spread_radius, spread_radius)
}

#[derive(Clone, Copy, Debug)]
pub struct PackedPrimitiveRangeIndex(u32);

pub struct ScreenPrimitiveIndexBuffer {
    index_buffer: Vec<PackedPrimitiveRangeIndex>,
}

impl ScreenPrimitiveIndexBuffer {
    fn new(tile_size: &Size2D<u32>) -> ScreenPrimitiveIndexBuffer {
        let length = tile_size.height * tile_size.width * (PRIMITIVES_PER_BUFFER as u32);
        ScreenPrimitiveIndexBuffer {
            index_buffer: vec![PackedPrimitiveRangeIndex(0); length as usize],
        }
    }
}

struct LayerPackedPrimitiveRangeIndexMap {
    packed_primitive_range_index_map: Vec<PackedPrimitiveRangeIndex>,
}

impl LayerPackedPrimitiveRangeIndexMap {
    fn new() -> LayerPackedPrimitiveRangeIndexMap {
        LayerPackedPrimitiveRangeIndexMap {
            packed_primitive_range_index_map: vec![],
        }
    }
}

pub struct ScreenTileMap {
    index_buffers: Vec<ScreenPrimitiveIndexBuffer>,
    primitive_count_buffer: Vec<u8>,
    tile_metadata: Vec<ScreenTile>,
    tile_size: Size2D<u32>,
    layer_packed_primitive_range_index_map: Vec<LayerPackedPrimitiveRangeIndexMap>,
    /// Map from `PackedPrimitiveRangeIndex` to `PackedPrimitiveRange`.
    primitive_range_index_map: Vec<PackedPrimitiveRange>,
}

impl ScreenTileMap {
    pub fn new(screen_rect: &Rect<i32>, device_pixel_ratio: f32) -> ScreenTileMap {
        let dp_size = Size2D::new(DevicePixel::new(screen_rect.size.width as f32,
                                                   device_pixel_ratio),
                                  DevicePixel::new(screen_rect.size.height as f32,
                                                   device_pixel_ratio));

        let x_tile_size = DevicePixel(SCREEN_TILE_SIZE as i32);
        let y_tile_size = DevicePixel(SCREEN_TILE_SIZE as i32);
        let x_tile_count = (dp_size.width + x_tile_size - DevicePixel(1)).0 / x_tile_size.0;
        let y_tile_count = (dp_size.height + y_tile_size - DevicePixel(1)).0 / y_tile_size.0;

        // Build screen space tiles.
        let mut screen_tiles = Vec::new();
        for y in 0..y_tile_count {
            let y0 = DevicePixel(y * y_tile_size.0);
            let y1 = y0 + y_tile_size;

            for x in 0..x_tile_count {
                let x0 = DevicePixel(x * x_tile_size.0);
                let x1 = x0 + x_tile_size;

                let tile_rect = rect_from_points(x0, y0, x1, y1);

                screen_tiles.push(ScreenTile::new(tile_rect));
            }
        }

        ScreenTileMap {
            index_buffers: vec![],
            primitive_count_buffer: vec![0; (y_tile_count as usize) * (x_tile_count as usize)],
            tile_metadata: screen_tiles,
            tile_size: Size2D::new(x_tile_count as u32, y_tile_count as u32),
            layer_packed_primitive_range_index_map: vec![],
            primitive_range_index_map: vec![],
        }
    }

    /// Returns an iterator over primitive indices, in *front to back* (i.e. reverse) order.
    fn primitive_indices<'a,'b>(&'a self,
                                tile_location: &'b Point2D<u32>,
                                starting_depth: u8,
                                ending_depth: u8)
                                -> PrimitiveIndices<'a> {
        let primitive_count = self.primitive_count(tile_location);
        debug_assert!(starting_depth <= primitive_count);
        PrimitiveIndices {
            primitive_range: PackedPrimitiveRange::empty(),
            primitive_depth: starting_depth,
            primitive_depth_limit: ending_depth,
            tile_index: self.tile_index(tile_location),
            tile_map: self,
        }
    }

    fn primitive_count(&self, tile_location: &Point2D<u32>) -> u8 {
        self.primitive_count_buffer[self.tile_index(tile_location) as usize]
    }

    fn tile_index(&self, tile_location: &Point2D<u32>) -> u32 {
        tile_location.y * self.tile_size.width + tile_location.x
    }

    fn tile_metadata(&self, tile_location: &Point2D<u32>) -> &ScreenTile {
        &self.tile_metadata[self.tile_index(tile_location) as usize]
    }

    fn tile_metadata_mut<'a,'b>(&'a mut self, tile_location: &'b Point2D<u32>)
                                -> &'a mut ScreenTile {
        let tile_index = self.tile_index(tile_location);
        &mut self.tile_metadata[tile_index as usize]
    }

    fn add_prim_index(&mut self,
                      tile_location: &Point2D<u32>,
                      primitive_range_index: PackedPrimitiveRangeIndex) {
        let tile_index = self.tile_index(tile_location);
        let primitive_count = self.primitive_count(tile_location);
        let (buffer_index, index_in_buffer) =
            self.buffer_index_and_index_in_buffer(tile_index, primitive_count);
        while self.index_buffers.len() <= buffer_index {
            self.index_buffers.push(ScreenPrimitiveIndexBuffer::new(&self.tile_size))
        }
        self.index_buffers[buffer_index].index_buffer[index_in_buffer] = primitive_range_index;
        self.primitive_count_buffer[tile_index as usize] += 1;
    }

    fn add_packed_primitive_range(&mut self,
                                  layer_index: u32,
                                  primitive_index_in_layer: u32,
                                  primitive_range_index: PackedPrimitiveRangeIndex) {
        while (layer_index as usize) >= self.layer_packed_primitive_range_index_map.len() {
            self.layer_packed_primitive_range_index_map
                .push(LayerPackedPrimitiveRangeIndexMap::new())
        }
        let mut layer_packed_primitive_range_index_map =
            &mut self.layer_packed_primitive_range_index_map[layer_index as usize];
        while (primitive_index_in_layer as usize) >=
                layer_packed_primitive_range_index_map.packed_primitive_range_index_map.len() {
            layer_packed_primitive_range_index_map.packed_primitive_range_index_map
                                                  .push(PackedPrimitiveRangeIndex(0))
        }
        layer_packed_primitive_range_index_map.packed_primitive_range_index_map[
            primitive_index_in_layer as usize] = primitive_range_index
    }

    fn buffer_index_and_index_in_buffer(&self, tile_index: u32, primitive_depth: u8)
                                        -> (usize, usize) {
        let primitives_per_buffer = PRIMITIVES_PER_BUFFER as u32;
        let primitive_depth = primitive_depth as u32;
        let buffer_index = primitive_depth / primitives_per_buffer;
        let index_in_buffer = tile_index * primitives_per_buffer + primitive_depth %
            primitives_per_buffer;
        (buffer_index as usize, index_in_buffer as usize)
    }
}

pub struct PrimitiveIndices<'a> {
    primitive_range: PackedPrimitiveRange,
    primitive_depth: u8,
    primitive_depth_limit: u8,
    tile_index: u32,
    tile_map: &'a ScreenTileMap,
}

impl<'a> Iterator for PrimitiveIndices<'a> {
    type Item = PackedPrimitiveIndex;

    fn next(&mut self) -> Option<PackedPrimitiveIndex> {
        while self.primitive_range.start == self.primitive_range.end {
            if self.primitive_depth == self.primitive_depth_limit {
                return None
            }
            self.primitive_depth -= 1;

            let (buffer_index, index_in_buffer) =
                self.tile_map.buffer_index_and_index_in_buffer(self.tile_index,
                                                               self.primitive_depth);
            let primitive_range_index = self.tile_map
                                            .index_buffers[buffer_index]
                                            .index_buffer[index_in_buffer];
            self.primitive_range =
                self.tile_map.primitive_range_index_map[primitive_range_index.0 as usize];
        }

        self.primitive_range.end.dec();
        Some(self.primitive_range.end())
    }
}

