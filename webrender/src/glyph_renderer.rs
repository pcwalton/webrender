/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! GPU glyph rasterization using Pathfinder.

use api::{DeviceUintSize, ImageFormat, TextureTarget};
use batch::BatchTextures;
use debug_colors;
use device::{Device, Texture, TextureFilter, VAO, VertexAttribute};
use device::{VertexAttributeKind, VertexDescriptor};
use euclid::{Point2D, Transform3D};
use internal_types::{RenderTargetInfo, SourceTexture};
use profiler::GpuProfileTag;
use render_task::RenderTaskTree;
use renderer::{ImageBufferKind, LazilyCompiledShader, Renderer, RendererError};
use renderer::{RendererStats, ShaderKind, VertexArrayKind};
use tiling::GlyphJob;

const GPU_TAG_GLYPH_STENCIL: GpuProfileTag = GpuProfileTag {
    label: "Glyph Stencil",
    color: debug_colors::STEELBLUE,
};
const GPU_TAG_GLYPH_COVER: GpuProfileTag = GpuProfileTag {
    label: "Glyph Cover",
    color: debug_colors::LIGHTSTEELBLUE,
};

pub const DESC_VECTOR_STENCIL: VertexDescriptor = VertexDescriptor {
    vertex_attributes: &[
        VertexAttribute {
            name: "aPosition",
            count: 2,
            kind: VertexAttributeKind::F32,
        },
    ],
    instance_attributes: &[
        VertexAttribute {
            name: "aFromPosition",
            count: 2,
            kind: VertexAttributeKind::F32,
        },
        VertexAttribute {
            name: "aCtrlPosition",
            count: 2,
            kind: VertexAttributeKind::F32,
        },
        VertexAttribute {
            name: "aToPosition",
            count: 2,
            kind: VertexAttributeKind::F32,
        },
        VertexAttribute {
            name: "aPathID",
            count: 1,
            kind: VertexAttributeKind::U16,
        },
        VertexAttribute {
            name: "aPad",
            count: 1,
            kind: VertexAttributeKind::U16,
        },
    ],
};

pub const DESC_VECTOR_COVER: VertexDescriptor = VertexDescriptor {
    vertex_attributes: &[
        VertexAttribute {
            name: "aPosition",
            count: 2,
            kind: VertexAttributeKind::F32,
        },
    ],
    instance_attributes: &[
        VertexAttribute {
            name: "aBounds",
            count: 4,
            kind: VertexAttributeKind::I32,
        },
    ],
};

pub struct GlyphRenderer {
    pub vector_stencil_vao: VAO,
    pub vector_cover_vao: VAO,

    // These are Pathfinder shaders, used for rendering vector graphics.
    vector_stencil: LazilyCompiledShader,
    vector_cover: LazilyCompiledShader,
}

impl GlyphRenderer {
    pub fn new(device: &mut Device, prim_vao: &VAO, precache_shaders: bool)
               -> Result<GlyphRenderer, RendererError> {
        let vector_stencil_vao = device.create_vao_with_new_instances(&DESC_VECTOR_STENCIL,
                                                                      prim_vao);
        let vector_cover_vao = device.create_vao_with_new_instances(&DESC_VECTOR_COVER, prim_vao);

        // Load Pathfinder vector graphics shaders.
        let vector_stencil = try!{
            LazilyCompiledShader::new(ShaderKind::VectorStencil,
                                      "pf_vector_stencil",
                                      &[ImageBufferKind::Texture2D.get_feature_string()],
                                      device,
                                      precache_shaders)
        };
        let vector_cover = try!{
            LazilyCompiledShader::new(ShaderKind::VectorCover,
                                      "pf_vector_cover",
                                      &[ImageBufferKind::Texture2D.get_feature_string()],
                                      device,
                                      precache_shaders)
        };

        Ok(GlyphRenderer {
            vector_stencil_vao,
            vector_cover_vao,
            vector_stencil,
            vector_cover,
        })
    }
}

impl Renderer {
    /// Renders glyphs using the vector graphics shaders (Pathfinder).
    pub fn stencil_glyphs(&mut self,
                          glyphs: &[GlyphJob],
                          projection: &Transform3D<f32>,
                          target_size: &DeviceUintSize,
                          render_tasks: &RenderTaskTree,
                          stats: &mut RendererStats)
                          -> Option<Texture> {
        if glyphs.is_empty() {
            return None
        }

        let _timer = self.gpu_profile.start_timer(GPU_TAG_GLYPH_STENCIL);

        // Initialize path info.
        // TODO(pcwalton): Cache this texture!
        let mut path_info_texture = self.device.create_texture(TextureTarget::Default,
                                                               ImageFormat::RGBAF32);

        let mut path_info_texels = Vec::with_capacity(glyphs.len() * 12);
        for glyph in glyphs {
            let rect = glyph.target_rect.to_f32()
                                        .translate(&-glyph.origin.to_f32().to_vector())
                                        .translate(&glyph.subpixel_offset.to_vector());
            path_info_texels.extend_from_slice(&[
                1.0, 0.0, 0.0, -1.0,
                rect.origin.x, rect.max_y(), 0.0, 0.0,
                rect.size.width, rect.size.height, 0.0, 0.0,
            ]);
        }

        self.device.init_texture(&mut path_info_texture,
                                 3,
                                 glyphs.len() as u32,
                                 TextureFilter::Nearest,
                                 None,
                                 1,
                                 Some(&path_info_texels));

        self.glyph_renderer.vector_stencil.bind(&mut self.device,
                                                projection,
                                                0,
                                                &mut self.renderer_errors);

        let path_info_external_texture = path_info_texture.to_external();
        let batch_textures =
            BatchTextures::color(SourceTexture::Custom(path_info_external_texture));

        // Initialize temporary framebuffer.
        // FIXME(pcwalton): Cache this too!
        // FIXME(pcwalton): Use RF32, not RGBAF32!
        let mut stencil_texture = self.device.create_texture(TextureTarget::Default,
                                                             ImageFormat::RGBAF32);
        self.device.init_texture::<f32>(&mut stencil_texture,
                                        target_size.width,
                                        target_size.height,
                                        TextureFilter::Nearest,
                                        Some(RenderTargetInfo {
                                            has_depth: false,
                                        }),
                                        1,
                                        None);
        self.device.bind_draw_target(Some((&stencil_texture, 0)), Some(*target_size));
        //self.device.clear_target(Some([0.0, 0.0, 0.0, 0.0]), None, None);

        self.device.set_blend(true);
        self.device.set_blend_mode_subpixel_pass1();

        let mut instance_data = vec![];
        for (path_id, glyph) in glyphs.iter().enumerate() {
            instance_data.extend(glyph.mesh_library.stencil_segments.iter().map(|segment| {
                VectorStencilInstanceAttrs {
                    from_position: segment.from,
                    ctrl_position: segment.ctrl,
                    to_position: segment.to,
                    path_id: path_id as u16,
                }
            }));
        }

        self.draw_instanced_batch(&instance_data,
                                  VertexArrayKind::VectorStencil,
                                  &batch_textures,
                                  stats);

        self.device.delete_texture(path_info_texture);

        //self.device.delete_texture(stencil_texture);

        Some(stencil_texture)
    }

    /// Blits glyphs from the stencil texture to the texture cache.
    ///
    /// Deletes the stencil texture at the end.
    /// FIXME(pcwalton): This is bad. Cache it somehow.
    pub fn cover_glyphs(&mut self,
                        stencil_texture: Texture,
                        glyphs: &[GlyphJob],
                        projection: &Transform3D<f32>,
                        render_tasks: &RenderTaskTree,
                        stats: &mut RendererStats) {
        debug_assert!(!glyphs.is_empty());

        let _timer = self.gpu_profile.start_timer(GPU_TAG_GLYPH_COVER);

        self.glyph_renderer.vector_cover.bind(&mut self.device,
                                              projection,
                                              0,
                                              &mut self.renderer_errors);

        let instance_data: Vec<_> = glyphs.iter().map(|glyph| glyph.target_rect).collect();

        let stencil_external_texture = stencil_texture.to_external();
        let batch_textures = BatchTextures::color(SourceTexture::Custom(stencil_external_texture));

        self.draw_instanced_batch(&instance_data,
                                  VertexArrayKind::VectorCover,
                                  &batch_textures,
                                  stats);

        self.device.delete_texture(stencil_texture);
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct VectorStencilInstanceAttrs {
    from_position: Point2D<f32>,
    ctrl_position: Point2D<f32>,
    to_position: Point2D<f32>,
    path_id: u16,
}
