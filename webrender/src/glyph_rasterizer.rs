/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#[cfg(test)]
use api::{IdNamespace, LayoutPoint};
use api::{ColorF, ColorU, DevicePoint, DeviceUintSize};
use api::{FontInstanceFlags, FontInstancePlatformOptions};
use api::{FontKey, FontRenderMode, FontTemplate, FontVariation};
use api::{GlyphDimensions, GlyphKey, SubpixelDirection};
use api::{ImageData, ImageDescriptor, ImageFormat, LayerToWorldTransform};
use app_units::Au;
use device::TextureFilter;
use euclid::{TypedPoint2D, TypedSize2D, TypedVector2D};
use glyph_cache::{CachedGlyphData, CachedGlyphInfo, GlyphCache};
use gpu_cache::GpuCache;
use internal_types::{FastHashSet, ResourceCacheError};
use pathfinder_font_renderer;
use pathfinder_partitioner::mesh_library::MeshLibrary;
use platform::font::FontContext;
use profiler::TextureCacheProfileCounters;
use rayon::ThreadPool;
use rayon::prelude::*;
use render_task::{RenderTask, RenderTaskCache, RenderTaskCacheKey, RenderTaskCacheKeyKind};
use render_task::{RenderTaskLocation, RenderTaskTree};
use std::cmp;
use std::collections::hash_map::Entry;
use std::f32;
use std::hash::{Hash, Hasher};
use std::mem;
use std::sync::{Arc, Mutex, MutexGuard, RwLock};
use std::sync::mpsc::{channel, Receiver, Sender};
use texture_cache::{TextureCache, TextureCacheHandle};
#[cfg(test)]
use thread_profiler::register_thread_with_profiler;
use tiling::{RenderPass, RenderTargetKind};
use webrender_api::{DeviceIntPoint, DeviceIntSize, DevicePixel};

/// Should match macOS 10.13 High Sierra.
/// 
/// We multiply by sqrt(2) to compensate for the fact that dilation amounts are relative to the
/// pixel square on macOS and relative to the vertex normal in Pathfinder.
const STEM_DARKENING_FACTOR_X: f32 = 0.0121 * f32::consts::SQRT_2;
const STEM_DARKENING_FACTOR_Y: f32 = 0.0121 * 1.25 * f32::consts::SQRT_2;

/// Likewise, should match macOS 10.13 High Sierra.
const MAX_STEM_DARKENING_AMOUNT: f32 = 0.3 * f32::consts::SQRT_2;

type PathfinderFontContext = pathfinder_font_renderer::FontContext<FontKey>;

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct FontTransform {
    pub scale_x: f32,
    pub skew_x: f32,
    pub skew_y: f32,
    pub scale_y: f32,
}

// Floats don't impl Hash/Eq/Ord...
impl Eq for FontTransform {}
impl Ord for FontTransform {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.partial_cmp(other).unwrap_or(cmp::Ordering::Equal)
    }
}
impl Hash for FontTransform {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Note: this is inconsistent with the Eq impl for -0.0 (don't care).
        self.scale_x.to_bits().hash(state);
        self.skew_x.to_bits().hash(state);
        self.skew_y.to_bits().hash(state);
        self.scale_y.to_bits().hash(state);
    }
}

impl FontTransform {
    const QUANTIZE_SCALE: f32 = 1024.0;

    pub fn new(scale_x: f32, skew_x: f32, skew_y: f32, scale_y: f32) -> Self {
        FontTransform { scale_x, skew_x, skew_y, scale_y }
    }

    pub fn identity() -> Self {
        FontTransform::new(1.0, 0.0, 0.0, 1.0)
    }

    pub fn is_identity(&self) -> bool {
        *self == FontTransform::identity()
    }

    pub fn quantize(&self) -> Self {
        FontTransform::new(
            (self.scale_x * Self::QUANTIZE_SCALE).round() / Self::QUANTIZE_SCALE,
            (self.skew_x * Self::QUANTIZE_SCALE).round() / Self::QUANTIZE_SCALE,
            (self.skew_y * Self::QUANTIZE_SCALE).round() / Self::QUANTIZE_SCALE,
            (self.scale_y * Self::QUANTIZE_SCALE).round() / Self::QUANTIZE_SCALE,
        )
    }

    pub fn determinant(&self) -> f64 {
        self.scale_x as f64 * self.scale_y as f64 - self.skew_y as f64 * self.skew_x as f64
    }

    pub fn compute_scale(&self) -> Option<(f64, f64)> {
        let det = self.determinant();
        if det != 0.0 {
            let x_scale = (self.scale_x as f64).hypot(self.skew_y as f64);
            let y_scale = det.abs() / x_scale;
            Some((x_scale, y_scale))
        } else {
            None
        }
    }

    pub fn pre_scale(&self, scale_x: f32, scale_y: f32) -> Self {
        FontTransform::new(
            self.scale_x * scale_x,
            self.skew_x * scale_y,
            self.skew_y * scale_x,
            self.scale_y * scale_y,
        )
    }

    pub fn invert_scale(&self, x_scale: f64, y_scale: f64) -> Self {
        self.pre_scale(x_scale.recip() as f32, y_scale.recip() as f32)
    }

    pub fn synthesize_italics(&self, skew_factor: f32) -> Self {
        FontTransform::new(
            self.scale_x,
            self.skew_x - self.scale_x * skew_factor,
            self.skew_y,
            self.scale_y - self.skew_y * skew_factor,
        )
    }

    pub fn swap_xy(&self) -> Self {
        FontTransform::new(self.skew_x, self.scale_x, self.scale_y, self.skew_y)
    }

    pub fn flip_x(&self) -> Self {
        FontTransform::new(-self.scale_x, self.skew_x, -self.skew_y, self.scale_y)
    }

    pub fn flip_y(&self) -> Self {
        FontTransform::new(self.scale_x, -self.skew_y, self.skew_y, -self.scale_y)
    }
}

impl<'a> From<&'a LayerToWorldTransform> for FontTransform {
    fn from(xform: &'a LayerToWorldTransform) -> Self {
        FontTransform::new(xform.m11, xform.m21, xform.m12, xform.m22)
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug, Ord, PartialOrd)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct FontInstance {
    pub font_key: FontKey,
    // The font size is in *device* pixels, not logical pixels.
    // It is stored as an Au since we need sub-pixel sizes, but
    // can't store as a f32 due to use of this type as a hash key.
    // TODO(gw): Perhaps consider having LogicalAu and DeviceAu
    //           or something similar to that.
    pub size: Au,
    pub color: ColorU,
    pub bg_color: ColorU,
    pub render_mode: FontRenderMode,
    pub subpx_dir: SubpixelDirection,
    pub flags: FontInstanceFlags,
    pub platform_options: Option<FontInstancePlatformOptions>,
    pub variations: Vec<FontVariation>,
    pub transform: FontTransform,
}

impl FontInstance {
    pub fn new(
        font_key: FontKey,
        size: Au,
        color: ColorF,
        bg_color: ColorU,
        render_mode: FontRenderMode,
        subpx_dir: SubpixelDirection,
        flags: FontInstanceFlags,
        platform_options: Option<FontInstancePlatformOptions>,
        variations: Vec<FontVariation>,
    ) -> Self {
        FontInstance {
            font_key,
            size,
            color: color.into(),
            bg_color,
            render_mode,
            subpx_dir,
            flags,
            platform_options,
            variations,
            transform: FontTransform::identity(),
        }
    }

    pub fn get_subpx_offset(&self, glyph: &GlyphKey) -> (f64, f64) {
        match self.subpx_dir {
            SubpixelDirection::None => (0.0, 0.0),
            SubpixelDirection::Horizontal => (glyph.subpixel_offset.into(), 0.0),
            SubpixelDirection::Vertical => (0.0, glyph.subpixel_offset.into()),
        }
    }

    pub fn get_alpha_glyph_format(&self) -> GlyphFormat {
        if self.transform.is_identity() { GlyphFormat::Alpha } else { GlyphFormat::TransformedAlpha }
    }

    pub fn get_subpixel_glyph_format(&self) -> GlyphFormat {
        if self.transform.is_identity() { GlyphFormat::Subpixel } else { GlyphFormat::TransformedSubpixel }
    }

    #[allow(dead_code)]
    pub fn get_glyph_format(&self) -> GlyphFormat {
        match self.render_mode {
            FontRenderMode::Mono | FontRenderMode::Alpha => self.get_alpha_glyph_format(),
            FontRenderMode::Subpixel => self.get_subpixel_glyph_format(),
        }
    }

    #[allow(dead_code)]
    pub fn get_extra_strikes(&self, x_scale: f64) -> usize {
        if self.flags.contains(FontInstanceFlags::SYNTHETIC_BOLD) {
            let mut bold_offset = self.size.to_f64_px() / 48.0;
            if bold_offset < 1.0 {
                bold_offset = 0.25 + 0.75 * bold_offset;
            }
            (bold_offset * x_scale).max(1.0).round() as usize
        } else {
            0
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[allow(dead_code)]
pub enum GlyphFormat {
    Alpha,
    TransformedAlpha,
    Subpixel,
    TransformedSubpixel,
    Bitmap,
    ColorBitmap,
}

impl GlyphFormat {
    pub fn ignore_color(self) -> Self {
        match self {
            GlyphFormat::ColorBitmap => GlyphFormat::Bitmap,
            _ => self,
        }
    }
}

pub struct RasterizedGlyph {
    pub top: f32,
    pub left: f32,
    pub width: u32,
    pub height: u32,
    pub scale: f32,
    pub format: GlyphFormat,
    pub bytes: Vec<u8>,
}

pub struct FontContexts {
    // These worker are mostly accessed from their corresponding worker threads.
    // The goal is that there should be no noticeable contention on the muteces.
    worker_contexts: Vec<Mutex<FontContext>>,

    // This worker should be accessed by threads that don't belong to thre thread pool
    // (in theory that's only the render backend thread so no contention expected either).
    shared_context: Mutex<FontContext>,

    pathfinder_context: Box<Mutex<PathfinderFontContext>>,

    // Stored here as a convenience to get the current thread index.
    workers: Arc<ThreadPool>,
}

impl FontContexts {
    /// Get access to the font context associated to the current thread.
    pub fn lock_current_context(&self) -> MutexGuard<FontContext> {
        let id = self.current_worker_id();
        self.lock_context(id)
    }

    /// Get access to any particular font context.
    ///
    /// The id is ```Some(i)``` where i is an index between 0 and num_worker_contexts
    /// for font contexts associated to the thread pool, and None for the shared
    /// global font context for use outside of the thread pool.
    pub fn lock_context(&self, id: Option<usize>) -> MutexGuard<FontContext> {
        match id {
            Some(index) => self.worker_contexts[index].lock().unwrap(),
            None => self.shared_context.lock().unwrap(),
        }
    }

    /// Get access to the font context usable outside of the thread pool.
    pub fn lock_shared_context(&self) -> MutexGuard<FontContext> {
        self.shared_context.lock().unwrap()
    }

    pub fn lock_pathfinder_context(&self) -> MutexGuard<PathfinderFontContext> {
        self.pathfinder_context.lock().unwrap()
    }

    // number of contexts associated to workers
    pub fn num_worker_contexts(&self) -> usize {
        self.worker_contexts.len()
    }

    fn current_worker_id(&self) -> Option<usize> {
        self.workers.current_thread_index()
    }
}

pub struct GlyphRasterizer {
    workers: Arc<ThreadPool>,
    font_contexts: Arc<FontContexts>,

    // Maintain a set of glyphs that have been requested this
    // frame. This ensures the glyph thread won't rasterize
    // the same glyph more than once in a frame. This is required
    // because the glyph cache hash table is not updated
    // until the end of the frame when we wait for glyph requests
    // to be resolved.
    pending_glyphs: FastHashSet<GlyphRequest>,

    // Receives the rendered glyphs.
    glyph_rx: Receiver<Vec<GlyphRasterJob>>,
    glyph_tx: Sender<Vec<GlyphRasterJob>>,

    // We defer removing fonts to the end of the frame so that:
    // - this work is done outside of the critical path,
    // - we don't have to worry about the ordering of events if a font is used on
    //   a frame where it is used (although it seems unlikely).
    fonts_to_remove: Vec<FontKey>,

    next_gpu_glyph_cache_key: GpuGlyphCacheKey,
}

impl GlyphRasterizer {
    pub fn new(workers: Arc<ThreadPool>) -> Result<Self, ResourceCacheError> {
        let (glyph_tx, glyph_rx) = channel();

        let num_workers = workers.current_num_threads();
        let mut contexts = Vec::with_capacity(num_workers);

        let shared_context = FontContext::new()?;

        for _ in 0 .. num_workers {
            contexts.push(Mutex::new(FontContext::new()?));
        }

        let pathfinder_context = PathfinderFontContext::new().map_err(|_| {
            ResourceCacheError::new("Failed to create the Pathfinder font context!".to_owned())
        })?;

        Ok(GlyphRasterizer {
            font_contexts: Arc::new(FontContexts {
                worker_contexts: contexts,
                shared_context: Mutex::new(shared_context),
                pathfinder_context: Box::new(Mutex::new(pathfinder_context)),
                workers: Arc::clone(&workers),
            }),
            pending_glyphs: FastHashSet::default(),
            glyph_rx,
            glyph_tx,
            workers,
            fonts_to_remove: Vec::new(),
            next_gpu_glyph_cache_key: GpuGlyphCacheKey(0),
        })
    }

    pub fn add_font(&mut self, font_key: FontKey, template: FontTemplate) {
        let font_contexts = Arc::clone(&self.font_contexts);
        // It's important to synchronously add the font for the shared context because
        // we use it to check that fonts have been properly added when requesting glyphs.
        font_contexts
            .lock_shared_context()
            .add_font(&font_key, &template);

        // TODO: this locks each font context while adding the font data, probably not a big deal,
        // but if there is contention on this lock we could easily have a queue of per-context
        // operations to add and delete fonts, and have these queues lazily processed by each worker
        // before rendering a glyph.
        // We can also move this into a worker to free up some cycles in the calling (render backend)
        // thread.
        for i in 0 .. font_contexts.num_worker_contexts() {
            font_contexts
                .lock_context(Some(i))
                .add_font(&font_key, &template);
        }

        font_contexts.lock_pathfinder_context().add_font(&font_key, &template);
    }

    pub fn delete_font(&mut self, font_key: FontKey) {
        self.fonts_to_remove.push(font_key);
    }

    pub fn prepare_font(&self, font: &mut FontInstance) {
        FontContext::prepare_font(font);
    }

    pub fn request_glyphs(
        &mut self,
        glyph_cache: &mut GlyphCache,
        font: FontInstance,
        glyph_keys: &[GlyphKey],
        texture_cache: &mut TextureCache,
        gpu_cache: &mut GpuCache,
    ) {
        assert!(
            self.font_contexts
                .lock_shared_context()
                .has_font(&font.font_key)
        );
        let mut glyphs = Vec::new();

        let glyph_key_cache = glyph_cache.get_glyph_key_cache_for_font_mut(font.clone());

        // select glyphs that have not been requested yet.
        for key in glyph_keys {
            match glyph_key_cache.entry(key.clone()) {
                Entry::Occupied(mut entry) => {
                    if let Ok(Some(ref mut glyph_info)) = *entry.get_mut() {
                        if texture_cache.request(&mut glyph_info.texture_cache_handle, gpu_cache) {
                            if let CachedGlyphData::Memory(ref bytes) = glyph_info.glyph_bytes {
                                // This case gets hit when we have already rasterized
                                // the glyph and stored it in CPU memory, the the glyph
                                // has been evicted from the texture cache. In which case
                                // we need to re-upload it to the GPU.
                                texture_cache.update(
                                    &mut glyph_info.texture_cache_handle,
                                    ImageDescriptor {
                                        width: glyph_info.size.width,
                                        height: glyph_info.size.height,
                                        stride: None,
                                        format: ImageFormat::BGRA8,
                                        is_opaque: false,
                                        offset: 0,
                                    },
                                    TextureFilter::Linear,
                                    Some(ImageData::Raw((*bytes).clone())),
                                    [glyph_info.offset.x, glyph_info.offset.y, glyph_info.scale],
                                    None,
                                    gpu_cache,
                                );
                            }
                        }
                    }
                }
                Entry::Vacant(..) => {
                    let request = GlyphRequest::new(&font, key);
                    if self.pending_glyphs.insert(request.clone()) {
                        glyphs.push(request);
                    }
                }
            }
        }

        if glyphs.is_empty() {
            return;
        }

        self.request_glyphs_from_pathfinder(glyphs);
        /*
        let font_contexts = Arc::clone(&self.font_contexts);
        let glyph_tx = self.glyph_tx.clone();

        // spawn an async task to get off of the render backend thread as early as
        // possible and in that task use rayon's fork join dispatch to rasterize the
        // glyphs in the thread pool.
        self.workers.spawn(move || {
            let jobs = glyphs
                .par_iter()
                .map(|request: &GlyphRequest| {
                    profile_scope!("glyph-raster");
                    let mut context = font_contexts.lock_current_context();
                    let job = GlyphRasterJob {
                        request: request.clone(),
                        result: context.rasterize_glyph(&request.font, &request.key),
                    };

                    // Sanity check.
                    if let Some(ref glyph) = job.result {
                        let bpp = 4; // We always render glyphs in 32 bits RGBA format.
                        assert_eq!(
                            glyph.bytes.len(),
                            bpp * (glyph.width * glyph.height) as usize
                        );
                    }

                    job
                })
                .collect();

            glyph_tx.send(jobs).unwrap();
        });*/
    }

    fn request_glyphs_from_pathfinder(&mut self, glyphs: Vec<GlyphRequest>) {
        let mut font_context = self.font_contexts.lock_pathfinder_context();
        let mesh_library = MeshLibrary::new();

        let raster_jobs: Vec<_> = glyphs.into_iter().map(|glyph| {
            let pathfinder_font_instance = pathfinder_font_renderer::FontInstance {
                font_key: glyph.font.font_key.clone(),
                size: glyph.font.size,
            };
            let pathfinder_subpixel_offset =
                pathfinder_font_renderer::SubpixelOffset(glyph.key.subpixel_offset as u8);
            let glyph_subpixel_offset: f64 = glyph.key.subpixel_offset.into();
            eprintln!("pathfinder_subpixel_offset={:?} my subpixel offset={:?}",
                      pathfinder_subpixel_offset,
                      glyph_subpixel_offset);
            let pathfinder_glyph_key =
                pathfinder_font_renderer::GlyphKey::new(glyph.key.index,
                                                        pathfinder_subpixel_offset);
            let raster_result = match (font_context.glyph_outline(&pathfinder_font_instance,
                                                                  &pathfinder_glyph_key),
                                       font_context.glyph_dimensions(&pathfinder_font_instance,
                                                                     &pathfinder_glyph_key,
                                                                     false)) {
                (Ok(outline), Ok(dimensions)) => {
                    // TODO(pcwalton)
                    eprintln!("request_glyphs_from_pathfinder(): Glyph outline {:?}/{:?} fetched \
                               OK: dimensions={:?}",
                            glyph.key.index,
                            glyph.key.subpixel_offset,
                            dimensions);
                    let mut mesh_library = MeshLibrary::new();
                    mesh_library.push_stencil_segments(1, outline.iter());
                    mesh_library.push_stencil_normals(1, outline.iter());
                    let glyph_origin = TypedPoint2D::new(dimensions.origin.x,
                                                         -dimensions.origin.y);
                    let glyph_size = TypedSize2D::from_untyped(&dimensions.size.to_i32());
                    eprintln!("request_glyphs_from_pathfinder(): glyph origin={:?} \
                               glyph size={:?}",
                              glyph_origin,
                              glyph_size);
                    // FIXME(pcwalton): Support vertical subpixel offsets.
                    // FIXME(pcwalton): Embolden amount should be 0 on macOS if "Use LCD font
                    // smoothing" is unchecked in System Preferences.

                    GlyphRasterResult::Mesh(Mesh {
                        mesh_library: mesh_library,
                        origin: glyph_origin,
                        dimensions: glyph_size,
                        subpixel_offset: TypedPoint2D::new(glyph_subpixel_offset as f32, 0.0),
                        render_mode: glyph.font.render_mode,
                        embolden_amount: compute_embolden_amount(glyph.font.size.to_f32_px()),
                    })
                }
                _ => {
                    eprintln!("request_glyphs_from_pathfinder(): Failed to get glyph outline!");
                    // TODO(pcwalton): Fall back to CPU rendering.
                    GlyphRasterResult::LoadFailed
                }
            };

            GlyphRasterJob {
                request: glyph,
                result: raster_result
            }
        }).collect();

        drop(self.glyph_tx.send(raster_jobs));
    }

    pub fn get_glyph_dimensions(
        &mut self,
        font: &FontInstance,
        glyph_key: &GlyphKey,
    ) -> Option<GlyphDimensions> {
        self.font_contexts
            .lock_shared_context()
            .get_glyph_dimensions(font, glyph_key)
    }

    pub fn get_glyph_index(&mut self, font_key: FontKey, ch: char) -> Option<u32> {
        self.font_contexts
            .lock_shared_context()
            .get_glyph_index(font_key, ch)
    }

    pub fn resolve_glyphs(
        &mut self,
        glyph_cache: &mut GlyphCache,
        texture_cache: &mut TextureCache,
        gpu_cache: &mut GpuCache,
        render_task_cache: &mut RenderTaskCache,
        render_tasks: &mut RenderTaskTree,
        alpha_glyph_pass: &mut RenderPass,
        color_glyph_pass: &mut RenderPass,
        _texture_cache_profile: &mut TextureCacheProfileCounters,
    ) {
        let mut rasterized_glyphs = Vec::with_capacity(self.pending_glyphs.len());

        // Pull rasterized glyphs from the queue.

        while !self.pending_glyphs.is_empty() {
            // TODO: rather than blocking until all pending glyphs are available
            // we could try_recv and steal work from the thread pool to take advantage
            // of the fact that this thread is alive and we avoid the added latency
            // of blocking it.
            let raster_jobs = self.glyph_rx
                .recv()
                .expect("BUG: Should be glyphs pending!");
            for job in raster_jobs {
                debug_assert!(self.pending_glyphs.contains(&job.request));
                self.pending_glyphs.remove(&job.request);

                rasterized_glyphs.push(job);
            }
        }

        // Ensure that the glyphs are always processed in the same
        // order for a given text run (since iterating a hash set doesn't
        // guarantee order). This can show up as very small float inaccuacry
        // differences in rasterizers due to the different coordinates
        // that text runs get associated with by the texture cache allocator.
        rasterized_glyphs.sort_by(|a, b| a.request.cmp(&b.request));

        // Update the caches.
        for job in rasterized_glyphs {
            let glyph_info = match job.result {
                GlyphRasterResult::LoadFailed => None,
                GlyphRasterResult::Bitmap(ref glyph) if glyph.width == 0 || glyph.height == 0 => {
                    None
                }
                GlyphRasterResult::Bitmap(glyph) => {
                    assert_eq!((glyph.left.fract(), glyph.top.fract()), (0.0, 0.0));
                    let glyph_bytes = Arc::new(glyph.bytes);
                    let mut texture_cache_handle = TextureCacheHandle::new();
                    texture_cache.request(&mut texture_cache_handle, gpu_cache);
                    texture_cache.update(
                        &mut texture_cache_handle,
                        ImageDescriptor {
                            width: glyph.width,
                            height: glyph.height,
                            stride: None,
                            format: ImageFormat::BGRA8,
                            is_opaque: false,
                            offset: 0,
                        },
                        TextureFilter::Linear,
                        Some(ImageData::Raw(glyph_bytes.clone())),
                        [glyph.left, -glyph.top, glyph.scale],
                        None,
                        gpu_cache,
                    );
                    Some(CachedGlyphInfo {
                        texture_cache_handle,
                        glyph_bytes: CachedGlyphData::Memory(glyph_bytes),
                        size: DeviceUintSize::new(glyph.width, glyph.height),
                        offset: DevicePoint::new(glyph.left, -glyph.top),
                        scale: glyph.scale,
                        format: glyph.format,
                    })
                }
                GlyphRasterResult::Mesh(mesh) => {
                    // TODO(pcwalton)
                    let render_task_cache_key = RenderTaskCacheKey {
                        size: mesh.dimensions,
                        kind: RenderTaskCacheKeyKind::Glyph(self.next_gpu_glyph_cache_key),
                    };
                    self.next_gpu_glyph_cache_key.0 += 1;

                    let mut mesh_library = Some(mesh.mesh_library);
                    let Mesh {
                        dimensions,
                        origin,
                        subpixel_offset,
                        render_mode,
                        embolden_amount,
                        ..
                    } = mesh;

                    let texture_cache_handle =
                        render_task_cache.request_render_task(render_task_cache_key,
                                                              texture_cache,
                                                              gpu_cache,
                                                              render_tasks,
                                                              |render_tasks| {
                            let location = RenderTaskLocation::Dynamic(None, dimensions);
                            let mesh_library = mesh_library.take().unwrap();
                            let glyph_render_task = RenderTask::new_glyph(location,
                                                                          mesh_library,
                                                                          &origin,
                                                                          &subpixel_offset,
                                                                          render_mode,
                                                                          &embolden_amount);
                            let root_task_id = render_tasks.add(glyph_render_task);
                            let target_kind = match render_mode {
                                FontRenderMode::Mono | FontRenderMode::Alpha => {
                                    alpha_glyph_pass.add_render_task(root_task_id,
                                                                     dimensions,
                                                                     RenderTargetKind::Alpha);
                                }
                                FontRenderMode::Subpixel => {
                                    color_glyph_pass.add_render_task(root_task_id,
                                                                     dimensions,
                                                                     RenderTargetKind::Color);
                                }
                            };
                            eprintln!("resolve_glyphs(): added task ID {:?}", root_task_id);
                            (root_task_id, [
                                origin.x as f32,
                                (origin.y - dimensions.height) as f32,
                                1.0
                             ], false)
                        });
                    eprintln!("resolve_glyphs(): requesting render task for mesh: size={:?}",
                              dimensions);
                    let glyph_format = match render_mode {
                        FontRenderMode::Mono | FontRenderMode::Alpha => GlyphFormat::Alpha,
                        FontRenderMode::Subpixel => GlyphFormat::Subpixel,
                    };
                    Some(CachedGlyphInfo {
                        texture_cache_handle: texture_cache_handle,
                        glyph_bytes: CachedGlyphData::Gpu,
                        size: TypedSize2D::new(dimensions.width as u32,
                                               dimensions.height as u32),
                        offset: DevicePoint::new(origin.x as f32, -origin.y as f32),
                        scale: 1.0,
                        format: glyph_format,
                    })
                }
            };

            let glyph_key_cache = glyph_cache.get_glyph_key_cache_for_font_mut(job.request.font);
            glyph_key_cache.insert(job.request.key, Ok(glyph_info));
        }

        // Now that we are done with the critical path (rendering the glyphs),
        // we can schedule removing the fonts if needed.
        if !self.fonts_to_remove.is_empty() {
            let font_contexts = Arc::clone(&self.font_contexts);
            let fonts_to_remove = mem::replace(&mut self.fonts_to_remove, Vec::new());
            self.workers.spawn(move || {
                for font_key in &fonts_to_remove {
                    font_contexts.lock_shared_context().delete_font(font_key);
                }
                for i in 0 .. font_contexts.num_worker_contexts() {
                    let mut context = font_contexts.lock_context(Some(i));
                    for font_key in &fonts_to_remove {
                        context.delete_font(font_key);
                    }
                }
            });
        }
    }

    #[cfg(feature = "replay")]
    pub fn reset(&mut self) {
        //TODO: any signals need to be sent to the workers?
        self.pending_glyphs.clear();
        self.fonts_to_remove.clear();
    }
}

trait AddFont {
    fn add_font(&mut self, font_key: &FontKey, template: &FontTemplate);
}

impl AddFont for FontContext {
    fn add_font(&mut self, font_key: &FontKey, template: &FontTemplate) {
        match template {
            &FontTemplate::Raw(ref bytes, index) => {
                self.add_raw_font(&font_key, bytes.clone(), index);
            }
            &FontTemplate::Native(ref native_font_handle) => {
                self.add_native_font(&font_key, (*native_font_handle).clone());
            }
        }
    }
}

impl AddFont for PathfinderFontContext {
    fn add_font(&mut self, font_key: &FontKey, template: &FontTemplate) {
        match template {
            &FontTemplate::Raw(ref bytes, index) => {
                drop(self.add_font_from_memory(&font_key, bytes.clone(), index));
            }
            &FontTemplate::Native(ref native_font_handle) => {
                drop(self.add_native_font(&font_key, (*native_font_handle).clone().0));
            }
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug, Ord, PartialOrd)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct GlyphRequest {
    pub key: GlyphKey,
    pub font: FontInstance,
}

impl GlyphRequest {
    pub fn new(font: &FontInstance, key: &GlyphKey) -> Self {
        GlyphRequest {
            key: key.clone(),
            font: font.clone(),
        }
    }
}

struct GlyphRasterJob {
    request: GlyphRequest,
    result: GlyphRasterResult,
}

enum GlyphRasterResult {
    LoadFailed,
    Bitmap(RasterizedGlyph),
    Mesh(Mesh),
}

struct Mesh {
    mesh_library: MeshLibrary,
    origin: DeviceIntPoint,
    subpixel_offset: TypedPoint2D<f32, DevicePixel>,
    dimensions: DeviceIntSize,
    render_mode: FontRenderMode,
    embolden_amount: TypedVector2D<f32, DevicePixel>,
}

#[derive(Debug, Copy, Clone, Eq, Hash, PartialEq)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct GpuGlyphCacheKey(pub u32);

#[test]
fn rasterize_200_glyphs() {
    // This test loads a font from disc, the renders 4 requests containing
    // 50 glyphs each, deletes the font and waits for the result.

    use rayon::ThreadPoolBuilder;
    use std::fs::File;
    use std::io::Read;

    let worker = ThreadPoolBuilder::new()
        .thread_name(|idx|{ format!("WRWorker#{}", idx) })
        .start_handler(move |idx| {
            register_thread_with_profiler(format!("WRWorker#{}", idx));
        })
        .build();
    let workers = Arc::new(worker.unwrap());
    let mut glyph_rasterizer = GlyphRasterizer::new(workers).unwrap();
    let mut glyph_cache = GlyphCache::new();
    let mut gpu_cache = GpuCache::new();
    let mut texture_cache = TextureCache::new(2048);

    let mut font_file =
        File::open("../wrench/reftests/text/VeraBd.ttf").expect("Couldn't open font file");
    let mut font_data = vec![];
    font_file
        .read_to_end(&mut font_data)
        .expect("failed to read font file");

    let font_key = FontKey::new(IdNamespace(0), 0);
    glyph_rasterizer.add_font(font_key, FontTemplate::Raw(Arc::new(font_data), 0));

    let font = FontInstance::new(
        font_key,
        Au::from_px(32),
        ColorF::new(0.0, 0.0, 0.0, 1.0),
        ColorU::new(0, 0, 0, 0),
        FontRenderMode::Subpixel,
        SubpixelDirection::Horizontal,
        Default::default(),
        None,
        Vec::new(),
    );

    let mut glyph_keys = Vec::with_capacity(200);
    for i in 0 .. 200 {
        glyph_keys.push(GlyphKey::new(
            i,
            LayoutPoint::zero(),
            font.render_mode,
            font.subpx_dir,
        ));
    }

    for i in 0 .. 4 {
        glyph_rasterizer.request_glyphs(
            &mut glyph_cache,
            font.clone(),
            &glyph_keys[(50 * i) .. (50 * (i + 1))],
            &mut texture_cache,
            &mut gpu_cache,
        );
    }

    glyph_rasterizer.delete_font(font_key);

    glyph_rasterizer.resolve_glyphs(
        &mut glyph_cache,
        &mut TextureCache::new(4096),
        &mut gpu_cache,
        &mut TextureCacheProfileCounters::new(),
    );
}

fn compute_embolden_amount(ppem: f32) -> TypedVector2D<f32, DevicePixel> {
    TypedVector2D::new(f32::min(ppem * STEM_DARKENING_FACTOR_X, MAX_STEM_DARKENING_AMOUNT),
                       f32::min(ppem * STEM_DARKENING_FACTOR_Y, MAX_STEM_DARKENING_AMOUNT))
}
