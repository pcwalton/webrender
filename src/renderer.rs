/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use batch::RasterBatch;
use debug_render::DebugRenderer;
use device::{Device, FileWatcherHandler, GpuProfile, OffscreenDevice, ProgramId, ProgramLoader};
use device::{TextureFilter, TextureId, UniformLocation, VAOId, VertexFormat, VertexUsageHint};
use euclid::{Matrix4D, Point2D, Rect, Size2D};
use gleam::gl;
use internal_types::{RendererFrame, ResultMsg, TextureUpdateOp};
use internal_types::{TextureUpdateDetails, TextureUpdateList, PackedVertex, RenderTargetMode};
use internal_types::{ORTHO_NEAR_PLANE, ORTHO_FAR_PLANE, DevicePixel};
use internal_types::{AxisDirection, PackedVertexForTextureCacheUpdate, CompositionOp};
use ipc_channel::ipc;
use profiler::{Profiler, BackendProfileCounters};
use profiler::{RendererProfileTimers, RendererProfileCounters};
use render_backend::RenderBackend;
use std::f32;
use std::mem;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use texture_cache::{BorderType, TextureCache, TextureInsertOp};
use tiling::{Frame, FrameBuilderConfig, Pass};
use time::precise_time_ns;
use webrender_traits::{ColorF, Epoch, PipelineId, RenderNotifier};
use webrender_traits::{ImageFormat, MixBlendMode, RenderApiSender};
use offscreen_gl_context::{NativeGLContext, NativeGLContextMethods};

pub const BLUR_INFLATION_FACTOR: u32 = 3;
pub const MAX_RASTER_OP_SIZE: u32 = 2048;

// TODO(gw): HACK! Need to support lighten/darken mix-blend-mode properly on android...
const UBO_BIND_PRIMITIVES: u32 = 1;
const UBO_BIND_LAYERS: u32 = 2;
const UBO_BIND_TILES: u32 = 4;

const UBO_BIND_CACHE_ITEMS: u32 = 5;
const UBO_BIND_COMPOSITE_TILES: u32 = 6;

#[derive(Clone, Copy)]
struct VertexBuffer {
    vao_id: VAOId,
}

pub trait CompositionOpHelpers {
    fn needs_framebuffer(&self) -> bool;
}

impl CompositionOpHelpers for CompositionOp {
    fn needs_framebuffer(&self) -> bool {
        match *self {
            CompositionOp::MixBlend(MixBlendMode::Normal) => unreachable!(),

            CompositionOp::MixBlend(MixBlendMode::Screen) |
            CompositionOp::MixBlend(MixBlendMode::Overlay) |
            CompositionOp::MixBlend(MixBlendMode::ColorDodge) |
            CompositionOp::MixBlend(MixBlendMode::ColorBurn) |
            CompositionOp::MixBlend(MixBlendMode::HardLight) |
            CompositionOp::MixBlend(MixBlendMode::SoftLight) |
            CompositionOp::MixBlend(MixBlendMode::Difference) |
            CompositionOp::MixBlend(MixBlendMode::Exclusion) |
            CompositionOp::MixBlend(MixBlendMode::Hue) |
            CompositionOp::MixBlend(MixBlendMode::Saturation) |
            CompositionOp::MixBlend(MixBlendMode::Color) |
            CompositionOp::MixBlend(MixBlendMode::Luminosity) => true,
            CompositionOp::Filter(_) |
            CompositionOp::MixBlend(MixBlendMode::Multiply) |
            CompositionOp::MixBlend(MixBlendMode::Darken) |
            CompositionOp::MixBlend(MixBlendMode::Lighten) => false,
        }
    }
}

struct FileWatcher {
    notifier: Arc<Mutex<Option<Box<RenderNotifier>>>>,
    result_tx: Sender<ResultMsg>,
}

impl FileWatcherHandler for FileWatcher {
    fn file_changed(&self, path: PathBuf) {
        self.result_tx.send(ResultMsg::RefreshShader(path)).ok();
        let mut notifier = self.notifier.lock();
        notifier.as_mut().unwrap().as_mut().unwrap().new_frame_ready();
    }
}

fn create_composite_shader(name: &'static str, device: &mut Device) -> ProgramId {
    let program_id = device.create_program(name, "composite_shared");

    let tiles_index = gl::get_uniform_block_index(program_id.0, "Tiles");
    gl::uniform_block_binding(program_id.0, tiles_index, UBO_BIND_COMPOSITE_TILES);

    let layer_index = gl::get_uniform_block_index(program_id.0, "Layers");
    gl::uniform_block_binding(program_id.0, layer_index, UBO_BIND_LAYERS);

    let prim_index = gl::get_uniform_block_index(program_id.0, "Primitives");
    gl::uniform_block_binding(program_id.0, prim_index, UBO_BIND_PRIMITIVES);

    println!("CompositeShader {}: tiles={} layers={} prims={}", name, tiles_index, layer_index, prim_index);

    program_id
}

fn create_cache_shader(name: &'static str, device: &mut Device) -> ProgramId {
    let program_id = device.create_program(name, "cache_shared");

    let layer_index = gl::get_uniform_block_index(program_id.0, "Layers");
    gl::uniform_block_binding(program_id.0, layer_index, UBO_BIND_LAYERS);

    let item_index = gl::get_uniform_block_index(program_id.0, "Items");
    gl::uniform_block_binding(program_id.0, item_index, UBO_BIND_CACHE_ITEMS);

    println!("CacheShader {}: items={} layers={}", name, item_index, layer_index);

    program_id
}

fn create_special_shader(name: &'static str, device: &mut Device) -> ProgramId {
    let program_id = device.create_program(name, "special_shared");

    let tile_index = gl::get_uniform_block_index(program_id.0, "Tiles");
    gl::uniform_block_binding(program_id.0, tile_index, UBO_BIND_TILES);

    println!("SpecialShader {}: tiles={}", name, tile_index);

    program_id
}

pub struct Renderer {
    result_rx: Receiver<ResultMsg>,
    device: Device,
    pending_texture_updates: Vec<TextureUpdateList>,
    pending_shader_updates: Vec<PathBuf>,
    current_frame: Option<RendererFrame>,
    device_pixel_ratio: f32,
    raster_batches: Vec<RasterBatch>,
    raster_op_vao: Option<VAOId>,

    box_shadow_program_id: ProgramId,

    blur_program_id: ProgramId,
    u_direction: UniformLocation,

    cs_border: ProgramId,
    cs_rectangle: ProgramId,
    cs_box_shadow: ProgramId,
    cs_complex: ProgramId,
    cs_text: ProgramId,
    cs_gradient: ProgramId,

    composite_shaders: [ProgramId; 8],
    tile_error_shader: ProgramId,
    tile_clear_shader: ProgramId,

    notifier: Arc<Mutex<Option<Box<RenderNotifier>>>>,

    enable_profiler: bool,
    enable_msaa: bool,
    debug: DebugRenderer,
    backend_profile_counters: BackendProfileCounters,
    profile_counters: RendererProfileCounters,
    profiler: Profiler,
    last_time: u64,

    max_raster_op_size: u32,
    raster_op_target_a8: TextureId,
    raster_op_target_rgba8: TextureId,
    text_composite_target: TextureId,

    gpu_profile_paint: GpuProfile,
    gpu_profile_composite: GpuProfile,
    quad_vao_id: VAOId,
}

impl Renderer {
    pub fn new(options: RendererOptions) -> (Renderer, RenderApiSender) {
        let (api_tx, api_rx) = ipc::channel().unwrap();
        let (payload_tx, payload_rx) = ipc::bytes_channel().unwrap();
        let (result_tx, result_rx) = channel();

        let notifier = Arc::new(Mutex::new(None));

        let file_watch_handler = FileWatcher {
            result_tx: result_tx.clone(),
            notifier: notifier.clone(),
        };

        let mut device = Device::new(options.resource_path.clone(),
                                     options.device_pixel_ratio,
                                     Box::new(file_watch_handler));
        device.begin_frame();

        let box_shadow_program_id = device.create_program("box_shadow", "shared_other");
        let blur_program_id = device.create_program("blur", "shared_other");
        let max_raster_op_size = MAX_RASTER_OP_SIZE * options.device_pixel_ratio as u32;

        let cs_text = create_cache_shader("cs_text", &mut device);
        let cs_rectangle = create_cache_shader("cs_rectangle", &mut device);
        let cs_box_shadow = create_cache_shader("cs_box_shadow", &mut device);
        let cs_complex = create_cache_shader("cs_complex", &mut device);
        let cs_border = create_cache_shader("cs_border", &mut device);
        let cs_gradient = create_cache_shader("cs_gradient", &mut device);

        let tile_error_shader = create_special_shader("ps_error", &mut device);
        let tile_clear_shader = create_special_shader("ps_clear", &mut device);

        let composite_shaders: [ProgramId; 8] = [
            create_composite_shader("cs_p1", &mut device),
            create_composite_shader("cs_p2", &mut device),
            create_composite_shader("cs_p3", &mut device),
            create_composite_shader("cs_p4", &mut device),
            create_composite_shader("cs_p5", &mut device),
            create_composite_shader("cs_p6", &mut device),
            create_composite_shader("cs_p7", &mut device),
            create_composite_shader("cs_p8", &mut device),
        ];

        let texture_ids = device.create_texture_ids(1024);
        let mut texture_cache = TextureCache::new(texture_ids);
        let white_pixels: Vec<u8> = vec![
            0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff,
        ];
        let mask_pixels: Vec<u8> = vec![
            0xff, 0xff,
            0xff, 0xff,
        ];
        // TODO: Ensure that the white texture can never get evicted when the cache supports LRU eviction!
        let white_image_id = texture_cache.new_item_id();
        texture_cache.insert(white_image_id,
                             0,
                             0,
                             2,
                             2,
                             ImageFormat::RGBA8,
                             TextureFilter::Linear,
                             TextureInsertOp::Blit(white_pixels),
                             BorderType::SinglePixel);

        let dummy_mask_image_id = texture_cache.new_item_id();
        texture_cache.insert(dummy_mask_image_id,
                             0,
                             0,
                             2,
                             2,
                             ImageFormat::A8,
                             TextureFilter::Linear,
                             TextureInsertOp::Blit(mask_pixels),
                             BorderType::SinglePixel);

        let debug_renderer = DebugRenderer::new(&mut device);

        let raster_op_target_a8 = device.create_texture_ids(1)[0];
        device.init_texture(raster_op_target_a8,
                            max_raster_op_size,
                            max_raster_op_size,
                            ImageFormat::A8,
                            TextureFilter::Nearest,
                            RenderTargetMode::RenderTarget,
                            None);

        let raster_op_target_rgba8 = device.create_texture_ids(1)[0];
        device.init_texture(raster_op_target_rgba8,
                            max_raster_op_size,
                            max_raster_op_size,
                            ImageFormat::RGBA8,
                            TextureFilter::Nearest,
                            RenderTargetMode::RenderTarget,
                            None);

        let x0 = 0.0;
        let y0 = 0.0;
        let x1 = 1.0;
        let y1 = 1.0;

        // TODO(gw): Consider separate VBO for quads vs border corners if VS ever shows up in profile!
        let quad_indices: [u16; 6] = [ 0, 1, 2, 2, 1, 3 ];
        let quad_vertices = [
            PackedVertex {
                pos: [x0, y0],
            },
            PackedVertex {
                pos: [x1, y0],
            },
            PackedVertex {
                pos: [x0, y1],
            },
            PackedVertex {
                pos: [x1, y1],
            },
        ];

        let quad_vao_id = device.create_vao(VertexFormat::Triangles, None);
        device.bind_vao(quad_vao_id);
        device.update_vao_indices(quad_vao_id, &quad_indices, VertexUsageHint::Static);
        device.update_vao_main_vertices(quad_vao_id, &quad_vertices, VertexUsageHint::Static);

        device.end_frame();

        let backend_notifier = notifier.clone();

        // We need a reference to the webrender context from the render backend in order to share
        // texture ids
        let context_handle = NativeGLContext::current_handle();

        let config = FrameBuilderConfig::new();

        let debug = options.debug;
        let (device_pixel_ratio, enable_aa) = (options.device_pixel_ratio, options.enable_aa);
        let payload_tx_for_backend = payload_tx.clone();
        let program_loader = ProgramLoader::from_device(&device);
        let stencil_routing = options.stencil_routing;
        thread::spawn(move || {
            let offscreen_device = OffscreenDevice::new(program_loader).expect("Couldn't create \
                                                                               an offscreen \
                                                                               device for the \
                                                                               render backend!");
            let mut backend = RenderBackend::new(api_rx,
                                                 payload_rx,
                                                 payload_tx_for_backend,
                                                 result_tx,
                                                 device_pixel_ratio,
                                                 white_image_id,
                                                 texture_cache,
                                                 enable_aa,
                                                 backend_notifier,
                                                 context_handle,
                                                 config,
                                                 offscreen_device,
                                                 debug,
                                                 stencil_routing);
            backend.run();
        });

        let mut renderer = Renderer {
            result_rx: result_rx,
            device: device,
            current_frame: None,
            raster_batches: Vec::new(),
            raster_op_vao: None,
            pending_texture_updates: Vec::new(),
            pending_shader_updates: Vec::new(),
            device_pixel_ratio: options.device_pixel_ratio,
            box_shadow_program_id: box_shadow_program_id,
            blur_program_id: blur_program_id,
            tile_error_shader: tile_error_shader,
            tile_clear_shader: tile_clear_shader,
            cs_rectangle: cs_rectangle,
            cs_box_shadow: cs_box_shadow,
            cs_complex: cs_complex,
            cs_border: cs_border,
            cs_gradient: cs_gradient,
            cs_text: cs_text,
            composite_shaders: composite_shaders,
            u_direction: UniformLocation::invalid(),
            notifier: notifier,
            debug: debug_renderer,
            backend_profile_counters: BackendProfileCounters::new(),
            profile_counters: RendererProfileCounters::new(),
            profiler: Profiler::new(),
            enable_profiler: options.enable_profiler,
            enable_msaa: options.enable_msaa,
            last_time: 0,
            raster_op_target_a8: raster_op_target_a8,
            raster_op_target_rgba8: raster_op_target_rgba8,
            text_composite_target: TextureId(0),
            max_raster_op_size: max_raster_op_size,
            gpu_profile_paint: GpuProfile::new(),
            gpu_profile_composite: GpuProfile::new(),
            quad_vao_id: quad_vao_id,
        };

        renderer.update_uniform_locations();

        let sender = RenderApiSender::new(api_tx, payload_tx);
        (renderer, sender)
    }

    #[cfg(any(target_os = "android", target_os = "gonk"))]
    fn enable_msaa(&self, _: bool) {
    }

    #[cfg(not(any(target_os = "android", target_os = "gonk")))]
    fn enable_msaa(&self, enable_msaa: bool) {
        if self.enable_msaa {
            if enable_msaa {
                gl::enable(gl::MULTISAMPLE);
            } else {
                gl::disable(gl::MULTISAMPLE);
            }
        }
    }

    fn update_uniform_locations(&mut self) {
        self.u_direction = self.device.get_uniform_location(self.blur_program_id, "uDirection");
    }

    pub fn set_render_notifier(&self, notifier: Box<RenderNotifier>) {
        let mut notifier_arc = self.notifier.lock().unwrap();
        *notifier_arc = Some(notifier);
    }

    pub fn current_epoch(&self, pipeline_id: PipelineId) -> Option<Epoch> {
        self.current_frame.as_ref().and_then(|frame| {
            frame.pipeline_epoch_map.get(&pipeline_id).map(|epoch| *epoch)
        })
    }

    pub fn update(&mut self) {
        // Pull any pending results and return the most recent.
        while let Ok(msg) = self.result_rx.try_recv() {
            match msg {
                ResultMsg::UpdateTextureCache(update_list) => {
                    self.pending_texture_updates.push(update_list);
                }
                ResultMsg::NewFrame(frame, profile_counters) => {
                    self.backend_profile_counters = profile_counters;
                    self.current_frame = Some(frame);
                }
                ResultMsg::RefreshShader(path) => {
                    self.pending_shader_updates.push(path);
                }
            }
        }
    }

    pub fn render(&mut self, framebuffer_size: Size2D<u32>) {
        let mut profile_timers = RendererProfileTimers::new();

        // Block CPU waiting for last frame's GPU profiles to arrive.
        // In general this shouldn't block unless heavily GPU limited.
        let paint_ns = self.gpu_profile_paint.get();
        let composite_ns = self.gpu_profile_composite.get();

        profile_timers.cpu_time.profile(|| {
            self.device.begin_frame();

            gl::disable(gl::SCISSOR_TEST);
            //gl::clear_color(1.0, 1.0, 1.0, 0.0);
            //gl::clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT | gl::STENCIL_BUFFER_BIT);

            //self.update_shaders();
            self.update_texture_cache();
            self.draw_frame(framebuffer_size);
        });

        let current_time = precise_time_ns();
        let ns = current_time - self.last_time;
        self.profile_counters.frame_time.set(ns);

        profile_timers.gpu_time_paint.set(paint_ns);
        profile_timers.gpu_time_composite.set(composite_ns);

        let gpu_ns = paint_ns + composite_ns;
        profile_timers.gpu_time_total.set(gpu_ns);

        if self.enable_profiler {
            self.profiler.draw_profile(&self.backend_profile_counters,
                                       &self.profile_counters,
                                       &profile_timers,
                                       &mut self.debug);
        }

        self.profile_counters.reset();
        self.profile_counters.frame_counter.inc();

        let debug_size = Size2D::new(framebuffer_size.width as u32,
                                     framebuffer_size.height as u32);
        self.debug.render(&mut self.device, &debug_size);
        self.device.end_frame();
        self.last_time = current_time;
    }

    pub fn layers_are_bouncing_back(&self) -> bool {
        match self.current_frame {
            None => false,
            Some(ref current_frame) => !current_frame.layers_bouncing_back.is_empty(),
        }
    }

/*
    fn update_shaders(&mut self) {
        let update_uniforms = !self.pending_shader_updates.is_empty();

        for path in self.pending_shader_updates.drain(..) {
            panic!("todo");
            //self.device.refresh_shader(path);
        }

        if update_uniforms {
            self.update_uniform_locations();
        }
    }
*/

    fn update_texture_cache(&mut self) {
        let mut pending_texture_updates = mem::replace(&mut self.pending_texture_updates, vec![]);
        for update_list in pending_texture_updates.drain(..) {
            for update in update_list.updates {
                match update.op {
                    TextureUpdateOp::Create(width, height, format, filter, mode, maybe_bytes) => {
                        // TODO: clean up match
                        match maybe_bytes {
                            Some(bytes) => {
                                self.device.init_texture(update.id,
                                                         width,
                                                         height,
                                                         format,
                                                         filter,
                                                         mode,
                                                         Some(bytes.as_slice()));
                            }
                            None => {
                                self.device.init_texture(update.id,
                                                         width,
                                                         height,
                                                         format,
                                                         filter,
                                                         mode,
                                                         None);
                            }
                        }
                    }
                    TextureUpdateOp::Grow(new_width,
                                          new_height,
                                          format,
                                          filter,
                                          mode) => {
                        self.device.resize_texture(update.id,
                                                   new_width,
                                                   new_height,
                                                   format,
                                                   filter,
                                                   mode);
                    }
                    TextureUpdateOp::Update(x, y, width, height, details) => {
                        match details {
                            TextureUpdateDetails::Raw => {
                                self.device.update_raw_texture(update.id, x, y, width, height);
                            }
                            TextureUpdateDetails::Blit(bytes) => {
                                self.device.update_texture(
                                    update.id,
                                    x,
                                    y,
                                    width, height,
                                    bytes.as_slice());
                            }
                            TextureUpdateDetails::Blur(bytes,
                                                       glyph_size,
                                                       radius,
                                                       unblurred_glyph_texture_image,
                                                       horizontal_blur_texture_image,
                                                       border_type) => {
                                let radius =
                                    f32::ceil(radius.to_f32_px() * self.device_pixel_ratio) as u32;
                                self.device.update_texture(
                                    unblurred_glyph_texture_image.texture_id,
                                    unblurred_glyph_texture_image.pixel_uv.x,
                                    unblurred_glyph_texture_image.pixel_uv.y,
                                    glyph_size.width,
                                    glyph_size.height,
                                    bytes.as_slice());

                                let blur_program_id = self.blur_program_id;

                                let white = ColorF::new(1.0, 1.0, 1.0, 1.0);
                                let (width, height) = (width as f32, height as f32);

                                let zero_point = Point2D::new(0.0, 0.0);
                                let dest_texture_size = Size2D::new(width as f32, height as f32);
                                let source_texture_size = Size2D::new(glyph_size.width as f32,
                                                                      glyph_size.height as f32);
                                let blur_radius = radius as f32;

                                self.add_rect_to_raster_batch(horizontal_blur_texture_image.texture_id,
                                                              unblurred_glyph_texture_image.texture_id,
                                                              blur_program_id,
                                                              Some(AxisDirection::Horizontal),
                                                              &Rect::new(horizontal_blur_texture_image.pixel_uv,
                                                                         Size2D::new(width as u32, height as u32)),
                                                              border_type,
                                                              |texture_rect| {
                                    [
                                        PackedVertexForTextureCacheUpdate::new(
                                            &texture_rect.origin,
                                            &white,
                                            &Point2D::new(0.0, 0.0),
                                            &zero_point,
                                            &zero_point,
                                            &unblurred_glyph_texture_image.texel_uv.origin,
                                            &unblurred_glyph_texture_image.texel_uv.bottom_right(),
                                            &dest_texture_size,
                                            &source_texture_size,
                                            blur_radius),
                                        PackedVertexForTextureCacheUpdate::new(
                                            &texture_rect.top_right(),
                                            &white,
                                            &Point2D::new(1.0, 0.0),
                                            &zero_point,
                                            &zero_point,
                                            &unblurred_glyph_texture_image.texel_uv.origin,
                                            &unblurred_glyph_texture_image.texel_uv.bottom_right(),
                                            &dest_texture_size,
                                            &source_texture_size,
                                            blur_radius),
                                        PackedVertexForTextureCacheUpdate::new(
                                            &texture_rect.bottom_left(),
                                            &white,
                                            &Point2D::new(0.0, 1.0),
                                            &zero_point,
                                            &zero_point,
                                            &unblurred_glyph_texture_image.texel_uv.origin,
                                            &unblurred_glyph_texture_image.texel_uv.bottom_right(),
                                            &dest_texture_size,
                                            &source_texture_size,
                                            blur_radius),
                                        PackedVertexForTextureCacheUpdate::new(
                                            &texture_rect.bottom_right(),
                                            &white,
                                            &Point2D::new(1.0, 1.0),
                                            &zero_point,
                                            &zero_point,
                                            &unblurred_glyph_texture_image.texel_uv.origin,
                                            &unblurred_glyph_texture_image.texel_uv.bottom_right(),
                                            &dest_texture_size,
                                            &source_texture_size,
                                            blur_radius),
                                    ]
                                });

                                let source_texture_size = Size2D::new(width as f32, height as f32);

                                self.add_rect_to_raster_batch(update.id,
                                                              horizontal_blur_texture_image.texture_id,
                                                              blur_program_id,
                                                              Some(AxisDirection::Vertical),
                                                              &Rect::new(Point2D::new(x as u32, y as u32),
                                                                         Size2D::new(width as u32, height as u32)),
                                                              border_type,
                                                              |texture_rect| {
                                    [
                                        PackedVertexForTextureCacheUpdate::new(
                                            &texture_rect.origin,
                                            &white,
                                            &Point2D::new(0.0, 0.0),
                                            &zero_point,
                                            &zero_point,
                                            &horizontal_blur_texture_image.texel_uv.origin,
                                            &horizontal_blur_texture_image.texel_uv.bottom_right(),
                                            &dest_texture_size,
                                            &source_texture_size,
                                            blur_radius),
                                        PackedVertexForTextureCacheUpdate::new(
                                            &texture_rect.top_right(),
                                            &white,
                                            &Point2D::new(1.0, 0.0),
                                            &zero_point,
                                            &zero_point,
                                            &horizontal_blur_texture_image.texel_uv.origin,
                                            &horizontal_blur_texture_image.texel_uv.bottom_right(),
                                            &dest_texture_size,
                                            &source_texture_size,
                                            blur_radius),
                                        PackedVertexForTextureCacheUpdate::new(
                                            &texture_rect.bottom_left(),
                                            &white,
                                            &Point2D::new(0.0, 1.0),
                                            &zero_point,
                                            &zero_point,
                                            &horizontal_blur_texture_image.texel_uv.origin,
                                            &horizontal_blur_texture_image.texel_uv.bottom_right(),
                                            &dest_texture_size,
                                            &source_texture_size,
                                            blur_radius),
                                        PackedVertexForTextureCacheUpdate::new(
                                            &texture_rect.bottom_right(),
                                            &white,
                                            &Point2D::new(1.0, 1.0),
                                            &zero_point,
                                            &zero_point,
                                            &horizontal_blur_texture_image.texel_uv.origin,
                                            &horizontal_blur_texture_image.texel_uv.bottom_right(),
                                            &dest_texture_size,
                                            &source_texture_size,
                                            blur_radius),
                                    ]
                                });
                            }
                            TextureUpdateDetails::BoxShadow(blur_radius,
                                                            border_radius,
                                                            box_rect_size,
                                                            raster_origin,
                                                            inverted,
                                                            border_type) => {
                                self.update_texture_cache_for_box_shadow(
                                    update.id,
                                    &Rect::new(Point2D::new(x, y),
                                               Size2D::new(width, height)),
                                    &Rect::new(
                                        Point2D::new(raster_origin.x, raster_origin.y),
                                        Size2D::new(box_rect_size.width, box_rect_size.height)),
                                    blur_radius,
                                    border_radius,
                                    inverted,
                                    border_type)
                            }
                        }
                    }
                }
            }
        }

        self.flush_raster_batches();
    }

    fn update_texture_cache_for_box_shadow(&mut self,
                                           update_id: TextureId,
                                           texture_rect: &Rect<u32>,
                                           box_rect: &Rect<DevicePixel>,
                                           blur_radius: DevicePixel,
                                           border_radius: DevicePixel,
                                           inverted: bool,
                                           border_type: BorderType) {
        debug_assert!(border_type == BorderType::SinglePixel);
        let box_shadow_program_id = self.box_shadow_program_id;

        let blur_radius = blur_radius.as_f32();

        let color = if inverted {
            ColorF::new(1.0, 1.0, 1.0, 0.0)
        } else {
            ColorF::new(1.0, 1.0, 1.0, 1.0)
        };

        let zero_point = Point2D::new(0.0, 0.0);
        let zero_size = Size2D::new(0.0, 0.0);

        self.add_rect_to_raster_batch(update_id,
                                      TextureId(0),
                                      box_shadow_program_id,
                                      None,
                                      &texture_rect,
                                      border_type,
                                      |texture_rect| {
            let box_rect_top_left = Point2D::new(box_rect.origin.x.as_f32() + texture_rect.origin.x,
                                                 box_rect.origin.y.as_f32() + texture_rect.origin.y);
            let box_rect_bottom_right = Point2D::new(box_rect_top_left.x + box_rect.size.width.as_f32(),
                                                     box_rect_top_left.y + box_rect.size.height.as_f32());
            let border_radii = Point2D::new(border_radius.as_f32(),
                                            border_radius.as_f32());

            [
                PackedVertexForTextureCacheUpdate::new(&texture_rect.origin,
                                                       &color,
                                                       &zero_point,
                                                       &border_radii,
                                                       &zero_point,
                                                       &box_rect_top_left,
                                                       &box_rect_bottom_right,
                                                       &zero_size,
                                                       &zero_size,
                                                       blur_radius),
                PackedVertexForTextureCacheUpdate::new(&texture_rect.top_right(),
                                                       &color,
                                                       &zero_point,
                                                       &border_radii,
                                                       &zero_point,
                                                       &box_rect_top_left,
                                                       &box_rect_bottom_right,
                                                       &zero_size,
                                                       &zero_size,
                                                       blur_radius),
                PackedVertexForTextureCacheUpdate::new(&texture_rect.bottom_left(),
                                                       &color,
                                                       &zero_point,
                                                       &border_radii,
                                                       &zero_point,
                                                       &box_rect_top_left,
                                                       &box_rect_bottom_right,
                                                       &zero_size,
                                                       &zero_size,
                                                       blur_radius),
                PackedVertexForTextureCacheUpdate::new(&texture_rect.bottom_right(),
                                                       &color,
                                                       &zero_point,
                                                       &border_radii,
                                                       &zero_point,
                                                       &box_rect_top_left,
                                                       &box_rect_bottom_right,
                                                       &zero_size,
                                                       &zero_size,
                                                       blur_radius),
            ]
        });
    }

    fn add_rect_to_raster_batch<F>(&mut self,
                                   dest_texture_id: TextureId,
                                   color_texture_id: TextureId,
                                   program_id: ProgramId,
                                   blur_direction: Option<AxisDirection>,
                                   dest_rect: &Rect<u32>,
                                   border_type: BorderType,
                                   f: F)
                                   where F: Fn(&Rect<f32>) -> [PackedVertexForTextureCacheUpdate; 4] {
        // FIXME(pcwalton): Use a hash table if this linear search shows up in the profile.
        for batch in &mut self.raster_batches {
            if batch.add_rect_if_possible(dest_texture_id,
                                          color_texture_id,
                                          program_id,
                                          blur_direction,
                                          dest_rect,
                                          border_type,
                                          &f) {
                return;
            }
        }

        let raster_op_target = if self.device.texture_has_alpha(dest_texture_id) {
            self.raster_op_target_rgba8
        } else {
            self.raster_op_target_a8
        };

        let mut raster_batch = RasterBatch::new(raster_op_target,
                                                self.max_raster_op_size,
                                                program_id,
                                                blur_direction,
                                                color_texture_id,
                                                dest_texture_id);

        let added = raster_batch.add_rect_if_possible(dest_texture_id,
                                                      color_texture_id,
                                                      program_id,
                                                      blur_direction,
                                                      dest_rect,
                                                      border_type,
                                                      &f);
        debug_assert!(added);
        self.raster_batches.push(raster_batch);
    }

    fn flush_raster_batches(&mut self) {
        let batches = mem::replace(&mut self.raster_batches, vec![]);
        if !batches.is_empty() {
            //println!("flushing {:?} raster batches", batches.len());

            gl::disable(gl::DEPTH_TEST);
            gl::disable(gl::SCISSOR_TEST);

            // Disable MSAA here for raster ops
            self.enable_msaa(false);

            let projection = Matrix4D::ortho(0.0,
                                             self.max_raster_op_size as f32,
                                             0.0,
                                             self.max_raster_op_size as f32,
                                             ORTHO_NEAR_PLANE,
                                             ORTHO_FAR_PLANE);

            // All horizontal blurs must complete before anything else.
            let mut remaining_batches = vec![];
            for batch in batches.into_iter() {
                if batch.blur_direction != Some(AxisDirection::Horizontal) {
                    remaining_batches.push(batch);
                    continue
                }

                self.set_up_gl_state_for_texture_cache_update(batch.page_allocator.texture_id(),
                                                              batch.color_texture_id,
                                                              batch.program_id,
                                                              batch.blur_direction,
                                                              &projection);
                self.perform_gl_texture_cache_update(batch);
            }

            // Flush the remaining batches.
            for batch in remaining_batches.into_iter() {
                self.set_up_gl_state_for_texture_cache_update(batch.page_allocator.texture_id(),
                                                              batch.color_texture_id,
                                                              batch.program_id,
                                                              batch.blur_direction,
                                                              &projection);
                self.perform_gl_texture_cache_update(batch);
            }
        }
    }

    fn set_up_gl_state_for_texture_cache_update(&mut self,
                                                target_texture_id: TextureId,
                                                color_texture_id: TextureId,
                                                program_id: ProgramId,
                                                blur_direction: Option<AxisDirection>,
                                                projection: &Matrix4D<f32>) {
        if !self.device.texture_has_alpha(target_texture_id) {
            gl::enable(gl::BLEND);
            gl::blend_func(gl::SRC_ALPHA, gl::ZERO);
        } else {
            gl::disable(gl::BLEND);
        }

        self.device.bind_render_target(Some(target_texture_id));
        gl::viewport(0, 0, self.max_raster_op_size as gl::GLint, self.max_raster_op_size as gl::GLint);

        self.device.bind_program(program_id, &projection);

        self.device.bind_color_texture(color_texture_id);
        self.device.bind_mask_texture(TextureId(0));

        match blur_direction {
            Some(AxisDirection::Horizontal) => {
                self.device.set_uniform_2f(self.u_direction, 1.0, 0.0)
            }
            Some(AxisDirection::Vertical) => {
                self.device.set_uniform_2f(self.u_direction, 0.0, 1.0)
            }
            None => {}
        }
    }

    fn perform_gl_texture_cache_update(&mut self, batch: RasterBatch) {
        let vao_id = match self.raster_op_vao {
            Some(ref mut vao_id) => *vao_id,
            None => {
                let vao_id = self.device.create_vao(VertexFormat::RasterOp, None);
                self.raster_op_vao = Some(vao_id);
                vao_id
            }
        };
        self.device.bind_vao(vao_id);

        self.device.update_vao_indices(vao_id, &batch.indices[..], VertexUsageHint::Dynamic);
        self.device.update_vao_main_vertices(vao_id,
                                             &batch.vertices[..],
                                             VertexUsageHint::Dynamic);

        self.profile_counters.vertices.add(batch.indices.len());
        self.profile_counters.draw_calls.inc();

        //println!("drawing triangles due to GL texture cache update");
        self.device.draw_triangles_u16(0, batch.indices.len() as gl::GLint);

        for blit_job in batch.blit_jobs {
            self.device.read_framebuffer_rect(blit_job.dest_texture_id,
                                              blit_job.dest_origin.x as i32,
                                              blit_job.dest_origin.y as i32,
                                              blit_job.src_origin.x as i32,
                                              blit_job.src_origin.y as i32,
                                              blit_job.size.width as i32,
                                              blit_job.size.height as i32);

            match blit_job.border_type {
                BorderType::SinglePixel => {
                    // Single pixel corners
                    self.device.read_framebuffer_rect(blit_job.dest_texture_id,
                                                      blit_job.dest_origin.x as i32 - 1,
                                                      blit_job.dest_origin.y as i32 - 1,
                                                      blit_job.src_origin.x as i32,
                                                      blit_job.src_origin.y as i32,
                                                      1,
                                                      1);

                    self.device.read_framebuffer_rect(blit_job.dest_texture_id,
                                                      (blit_job.dest_origin.x + blit_job.size.width) as i32,
                                                      blit_job.dest_origin.y as i32 - 1,
                                                      (blit_job.src_origin.x + blit_job.size.width) as i32 - 1,
                                                      blit_job.src_origin.y as i32,
                                                      1,
                                                      1);

                    self.device.read_framebuffer_rect(blit_job.dest_texture_id,
                                                      blit_job.dest_origin.x as i32 - 1,
                                                      (blit_job.dest_origin.y + blit_job.size.height) as i32,
                                                      blit_job.src_origin.x as i32,
                                                      (blit_job.src_origin.y + blit_job.size.height) as i32 - 1,
                                                      1,
                                                      1);

                    self.device.read_framebuffer_rect(blit_job.dest_texture_id,
                                                      (blit_job.dest_origin.x + blit_job.size.width) as i32,
                                                      (blit_job.dest_origin.y + blit_job.size.height) as i32,
                                                      (blit_job.src_origin.x + blit_job.size.width) as i32 - 1,
                                                      (blit_job.src_origin.y + blit_job.size.height) as i32 - 1,
                                                      1,
                                                      1);

                    // Horizontal edges
                    self.device.read_framebuffer_rect(blit_job.dest_texture_id,
                                                      blit_job.dest_origin.x as i32,
                                                      blit_job.dest_origin.y as i32 - 1,
                                                      blit_job.src_origin.x as i32,
                                                      blit_job.src_origin.y as i32,
                                                      blit_job.size.width as i32,
                                                      1);

                    self.device.read_framebuffer_rect(blit_job.dest_texture_id,
                                                      blit_job.dest_origin.x as i32,
                                                      (blit_job.dest_origin.y + blit_job.size.height) as i32,
                                                      blit_job.src_origin.x as i32,
                                                      (blit_job.src_origin.y + blit_job.size.height) as i32 - 1,
                                                      blit_job.size.width as i32,
                                                      1);

                    // Vertical edges
                    self.device.read_framebuffer_rect(blit_job.dest_texture_id,
                                                      blit_job.dest_origin.x as i32 - 1,
                                                      blit_job.dest_origin.y as i32,
                                                      blit_job.src_origin.x as i32,
                                                      blit_job.src_origin.y as i32,
                                                      1,
                                                      blit_job.size.height as i32);

                    self.device.read_framebuffer_rect(blit_job.dest_texture_id,
                                                      (blit_job.dest_origin.x + blit_job.size.width) as i32,
                                                      blit_job.dest_origin.y as i32,
                                                      (blit_job.src_origin.x + blit_job.size.width) as i32 - 1,
                                                      blit_job.src_origin.y as i32,
                                                      1,
                                                      blit_job.size.height as i32);

                }
                BorderType::_NoBorder => {}
            }
        }
    }

/*
    fn compute_layer_viewport_in_render_target_space(&self, layer: &DrawLayer) -> Rect<f32> {
        let layer_origin = Point2D::new(layer.origin.x * self.device_pixel_ratio,
                                        layer.origin.y * self.device_pixel_ratio);
        let layer_size = Size2D::new(layer.size.width * self.device_pixel_ratio,
                                     layer.size.height * self.device_pixel_ratio);
        Rect::new(layer_origin, layer_size)
    }

    fn calculate_layer_viewports(&self,
                                 layer: &DrawLayer,
                                 render_context: &RenderContext,
                                 viewport: &Rect<f32>,
                                 layer_viewports: &mut HashMap<RenderTargetId, Rect<f32>>) {
        layer_viewports.insert(layer.id, *viewport);

        let mut indices_of_composited_children = BitSet::new();
        for command in &layer.commands {
            match *command {
                DrawCommand::CompositeBatch(ref info) => {
                    for job in &info.jobs {
                        indices_of_composited_children.insert(job.child_layer_index.0 as usize);
                        let child_layer = &layer.child_layers[job.child_layer_index.0 as usize];
                        if let Some(ref new_viewport) = job.world_transform
                                                           .transform_rect(&job.rect)
                                                           .intersection(viewport) {
                            self.calculate_layer_viewports(child_layer,
                                                           render_context,
                                                           new_viewport,
                                                           layer_viewports)
                        }
                    }
                }
                DrawCommand::Batch(_) | DrawCommand::Clear(_) => {}
            }
        }

        for (index, child_layer) in layer.child_layers.iter().enumerate() {
            if !indices_of_composited_children.contains(index) {
                // FIXME(pcwalton): This is probably wrong for nested composites.
                let layer_viewport =
                    self.compute_layer_viewport_in_render_target_space(child_layer);
                println!("calculate_layer_viewports: child layer_viewport={:?}", layer_viewport);
                if let Some(ref new_viewport) = layer_viewport.intersection(viewport) {
                    self.calculate_layer_viewports(child_layer,
                                                   render_context,
                                                   new_viewport,
                                                   layer_viewports)
                }
            }
        }
    }*/

    fn draw_prim_cache_pass(&mut self,
                            render_pass: &Pass,
                            target_size: &Size2D<f32>) {
        self.device.bind_render_target(Some(self.text_composite_target));
        gl::viewport(0,
                     0,
                     target_size.width as i32,
                     target_size.height as i32);

        gl::clear_color(1.0, 1.0, 1.0, 0.0);
        gl::clear(gl::COLOR_BUFFER_BIT);

        let projection = Matrix4D::ortho(0.0,
                                         target_size.width,
                                         0.0,
                                         target_size.height,
                                         ORTHO_NEAR_PLANE,
                                         ORTHO_FAR_PLANE);

        for pass in &render_pass.prim_cache_passes {
            if !pass.gradients.is_empty() {
                self.device.bind_program(self.cs_gradient, &projection);
                self.device.bind_vao(self.quad_vao_id);

                // TODO(gw): Select chunk size based on max ubo size queried from device!
                for chunk in pass.gradients.chunks(512) {
                    let ubos = gl::gen_buffers(1);
                    let ubo = ubos[0];

                    gl::bind_buffer(gl::UNIFORM_BUFFER, ubo);
                    gl::buffer_data(gl::UNIFORM_BUFFER, &chunk, gl::STATIC_DRAW);
                    gl::bind_buffer_base(gl::UNIFORM_BUFFER, UBO_BIND_CACHE_ITEMS, ubo);

                    self.device.draw_indexed_triangles_instanced_u16(6, chunk.len() as gl::GLint);

                    gl::delete_buffers(&ubos);
                }
            }

            if !pass.borders.is_empty() {
                self.device.bind_program(self.cs_border, &projection);
                self.device.bind_vao(self.quad_vao_id);

                // TODO(gw): Select chunk size based on max ubo size queried from device!
                for chunk in pass.borders.chunks(1024) {
                    let ubos = gl::gen_buffers(1);
                    let ubo = ubos[0];

                    gl::bind_buffer(gl::UNIFORM_BUFFER, ubo);
                    gl::buffer_data(gl::UNIFORM_BUFFER, &chunk, gl::STATIC_DRAW);
                    gl::bind_buffer_base(gl::UNIFORM_BUFFER, UBO_BIND_CACHE_ITEMS, ubo);

                    self.device.draw_indexed_triangles_instanced_u16(6, chunk.len() as gl::GLint);

                    gl::delete_buffers(&ubos);
                }
            }

            if !pass.box_shadows.is_empty() {
                self.device.bind_program(self.cs_box_shadow, &projection);
                self.device.bind_vao(self.quad_vao_id);

                // TODO(gw): Select chunk size based on max ubo size queried from device!
                for chunk in pass.box_shadows.chunks(1024) {
                    let ubos = gl::gen_buffers(1);
                    let ubo = ubos[0];

                    gl::bind_buffer(gl::UNIFORM_BUFFER, ubo);
                    gl::buffer_data(gl::UNIFORM_BUFFER, &chunk, gl::STATIC_DRAW);
                    gl::bind_buffer_base(gl::UNIFORM_BUFFER, UBO_BIND_CACHE_ITEMS, ubo);

                    self.device.draw_indexed_triangles_instanced_u16(6, chunk.len() as gl::GLint);

                    gl::delete_buffers(&ubos);
                }
            }

            if !pass.rectangles.is_empty() {
                self.device.bind_program(self.cs_rectangle, &projection);
                self.device.bind_vao(self.quad_vao_id);

                // TODO(gw): Select chunk size based on max ubo size queried from device!
                for chunk in pass.rectangles.chunks(1024) {
                    let ubos = gl::gen_buffers(1);
                    let ubo = ubos[0];

                    gl::bind_buffer(gl::UNIFORM_BUFFER, ubo);
                    gl::buffer_data(gl::UNIFORM_BUFFER, &chunk, gl::STATIC_DRAW);
                    gl::bind_buffer_base(gl::UNIFORM_BUFFER, UBO_BIND_CACHE_ITEMS, ubo);

                    self.device.draw_indexed_triangles_instanced_u16(6, chunk.len() as gl::GLint);

                    gl::delete_buffers(&ubos);
                }
            }

            if !pass.complex.is_empty() {
                self.device.bind_program(self.cs_complex, &projection);
                self.device.bind_vao(self.quad_vao_id);

                // TODO(gw): Select chunk size based on max ubo size queried from device!
                for chunk in pass.complex.chunks(256) {
                    let ubos = gl::gen_buffers(1);
                    let ubo = ubos[0];

                    gl::bind_buffer(gl::UNIFORM_BUFFER, ubo);
                    gl::buffer_data(gl::UNIFORM_BUFFER, &chunk, gl::STATIC_DRAW);
                    gl::bind_buffer_base(gl::UNIFORM_BUFFER, UBO_BIND_CACHE_ITEMS, ubo);

                    self.device.draw_indexed_triangles_instanced_u16(6, chunk.len() as gl::GLint);

                    gl::delete_buffers(&ubos);
                }
            }

            if !pass.glyphs.is_empty() {
                self.device.bind_program(self.cs_text, &projection);
                self.device.bind_color_texture(render_pass.text_texture_id);
                self.device.bind_vao(self.quad_vao_id);

                // TODO(gw): Select chunk size based on max ubo size queried from device!
                for chunk in pass.glyphs.chunks(1024) {
                    let glyphs_ubos = gl::gen_buffers(1);
                    let glyphs_ubo = glyphs_ubos[0];

                    gl::bind_buffer(gl::UNIFORM_BUFFER, glyphs_ubo);
                    gl::buffer_data(gl::UNIFORM_BUFFER, &chunk, gl::STATIC_DRAW);
                    gl::bind_buffer_base(gl::UNIFORM_BUFFER, UBO_BIND_CACHE_ITEMS, glyphs_ubo);

                    self.device.draw_indexed_triangles_instanced_u16(6, chunk.len() as gl::GLint);

                    gl::delete_buffers(&glyphs_ubos);
                }
            }
        }
    }

    fn add_debug_rect(&mut self,
                      p0: Point2D<DevicePixel>,
                      p1: Point2D<DevicePixel>,
                      label: &str,
                      c: &ColorF) {
        let tile_x0 = p0.x;
        let tile_y0 = p0.y;
        let tile_x1 = p1.x;
        let tile_y1 = p1.y;

        //let c = &ColorF::new(1.0, 0.0, 1.0, 1.0);

        self.debug.add_line(tile_x0,
                            tile_y0,
                            c,//&ColorF::new(0.0, 0.0, 0.0, 1.0),
                            tile_x1,
                            tile_y0,
                            c);//&ColorF::new(0.0, 0.0, 0.0, 1.0));
        self.debug.add_line(tile_x0,
                            tile_y1,
                            c,//&ColorF::new(0.0, 0.0, 0.0, 1.0),
                            tile_x1,
                            tile_y1,
                            c);//&ColorF::new(0.0, 0.0, 0.0, 1.0));
        self.debug.add_line(tile_x0,
                            tile_y0,
                            c,//&ColorF::new(0.0, 0.0, 0.0, 1.0),
                            tile_x0,
                            tile_y1,
                            c);//&ColorF::new(0.0, 0.0, 0.0, 1.0));
        self.debug.add_line(tile_x1,
                            tile_y0,
                            c,//&ColorF::new(0.0, 0.0, 0.0, 1.0),
                            tile_x1,
                            tile_y1,
                            c);//&ColorF::new(0.0, 0.0, 0.0, 1.0));
        if label.len() > 0 {
            self.debug.add_text((tile_x0.0 as f32 + tile_x1.0 as f32) * 0.5,
                                (tile_y0.0 as f32 + tile_y1.0 as f32) * 0.5,
                                label,
                                c);//&ColorF::new(0.0, 0.0, 0.0, 1.0));
        }
    }

    fn draw_tile_frame(&mut self,
                       frame: &Frame,
                       framebuffer_size: &Size2D<u32>) {
        //println!("render {} debug rects", frame.debug_rects.len());

        for debug_rect in frame.debug_rects.iter().rev() {
            self.add_debug_rect(debug_rect.rect.origin,
                                debug_rect.rect.bottom_right(),
                                &debug_rect.label,
                                &debug_rect.color);
        }

        //gl::clear_color(1.0, 1.0, 1.0, 1.0);
        //gl::clear(gl::COLOR_BUFFER_BIT);

        if frame.passes.is_empty() {
            return;
        }

        let projection = Matrix4D::ortho(0.0,
                                         framebuffer_size.width as f32,
                                         framebuffer_size.height as f32,
                                         0.0,
                                         ORTHO_NEAR_PLANE,
                                         ORTHO_FAR_PLANE);

        gl::depth_mask(false);
        gl::disable(gl::STENCIL_TEST);
        gl::disable(gl::BLEND);

        self.gpu_profile_paint.begin();
        self.gpu_profile_paint.end();
        self.gpu_profile_composite.begin();

        let misc_ubos = gl::gen_buffers(1);
        let layer_ubo = misc_ubos[0];

        gl::bind_buffer(gl::UNIFORM_BUFFER, layer_ubo);
        gl::buffer_data(gl::UNIFORM_BUFFER, &frame.layer_ubo.items, gl::STATIC_DRAW);
        gl::bind_buffer_base(gl::UNIFORM_BUFFER, UBO_BIND_LAYERS, layer_ubo);

        if self.text_composite_target == TextureId(0) {
            self.text_composite_target = self.device.create_texture_ids(1)[0];
            self.device.init_texture(self.text_composite_target,
                                     frame.prim_cache_size.width as u32,
                                     frame.prim_cache_size.height as u32,
                                     ImageFormat::RGBA8,
                                     TextureFilter::Linear,
                                     RenderTargetMode::RenderTarget,
                                     None);
        }

        for pass in &frame.passes {
            self.draw_prim_cache_pass(pass,
                                      &frame.prim_cache_size);

            self.device.bind_render_target(None);
            gl::viewport(0, 0, framebuffer_size.width as i32, framebuffer_size.height as i32);
            self.device.bind_vao(self.quad_vao_id);

            for tile_batch in &pass.tile_batches {
                let prim_ubos = gl::gen_buffers(1);
                let prim_ubo = prim_ubos[0];

                gl::bind_buffer(gl::UNIFORM_BUFFER, prim_ubo);
                gl::buffer_data(gl::UNIFORM_BUFFER, &tile_batch.primitives, gl::STATIC_DRAW);
                gl::bind_buffer_base(gl::UNIFORM_BUFFER, UBO_BIND_PRIMITIVES, prim_ubo);

                for (key, tiles) in &tile_batch.batches {
                    let shader = self.composite_shaders[key.shader as usize];
                    self.device.bind_program(shader, &projection);
                    self.device.bind_cache_texture(self.text_composite_target);
                    for (i, texture_id) in key.samplers.iter().enumerate() {
                        let texture_id = match *texture_id {
                            TextureId(0) => self.text_composite_target,
                            other_texture => other_texture,
                        };
                        self.device.bind_layer_texture(i, texture_id);
                    }

                    for batch in tiles.chunks(512) {
                        let ubos = gl::gen_buffers(1);
                        let ubo = ubos[0];

                        gl::bind_buffer(gl::UNIFORM_BUFFER, ubo);
                        gl::buffer_data(gl::UNIFORM_BUFFER, &batch, gl::STATIC_DRAW);
                        gl::bind_buffer_base(gl::UNIFORM_BUFFER, UBO_BIND_COMPOSITE_TILES, ubo);

                        self.device.draw_indexed_triangles_instanced_u16(6, batch.len() as i32);
                        self.profile_counters.vertices.add(6 * batch.len());
                        self.profile_counters.draw_calls.inc();

                        gl::delete_buffers(&ubos);
                    }
                }

                gl::delete_buffers(&prim_ubos);
            }
        }

        gl::delete_buffers(&misc_ubos);

        if !frame.error_tiles.is_empty() {
            for batch in frame.error_tiles.chunks(1024) {
                let ubos = gl::gen_buffers(1);
                let ubo = ubos[0];

                gl::bind_buffer(gl::UNIFORM_BUFFER, ubo);
                gl::buffer_data(gl::UNIFORM_BUFFER, &batch, gl::STATIC_DRAW);
                gl::bind_buffer_base(gl::UNIFORM_BUFFER, UBO_BIND_TILES, ubo);

                self.device.bind_program(self.tile_error_shader, &projection);
                self.device.draw_indexed_triangles_instanced_u16(6, batch.len() as i32);
                self.profile_counters.vertices.add(6 * batch.len());
                self.profile_counters.draw_calls.inc();

                gl::delete_buffers(&ubos);
            }
        }

        if !frame.clear_tiles.is_empty() {
            for batch in frame.clear_tiles.chunks(1024) {
                let ubos = gl::gen_buffers(1);
                let ubo = ubos[0];

                gl::bind_buffer(gl::UNIFORM_BUFFER, ubo);
                gl::buffer_data(gl::UNIFORM_BUFFER, &batch, gl::STATIC_DRAW);
                gl::bind_buffer_base(gl::UNIFORM_BUFFER, UBO_BIND_TILES, ubo);

                self.device.bind_program(self.tile_clear_shader, &projection);
                self.device.draw_indexed_triangles_instanced_u16(6, batch.len() as i32);
                self.profile_counters.vertices.add(6 * batch.len());
                self.profile_counters.draw_calls.inc();

                gl::delete_buffers(&ubos);
            }
        }

        self.gpu_profile_composite.end();
    }

    fn draw_frame(&mut self, framebuffer_size: Size2D<u32>) {
        if let Some(frame) = self.current_frame.take() {
            // TODO: cache render targets!

            // TODO(gw): Doesn't work well with transforms.
            //           Look into this...
            gl::disable(gl::DEPTH_TEST);
            gl::disable(gl::SCISSOR_TEST);
            gl::disable(gl::BLEND);

            if let Some(ref frame) = frame.frame {
                self.draw_tile_frame(frame, &framebuffer_size);
            }

            // Restore frame - avoid borrow checker!
            self.current_frame = Some(frame);
        }
    }
}

#[derive(Clone, Debug)]
pub struct RendererOptions {
    pub device_pixel_ratio: f32,
    pub resource_path: PathBuf,
    pub enable_aa: bool,
    pub enable_msaa: bool,
    pub enable_profiler: bool,
    pub debug: bool,
    pub stencil_routing: bool,
}
