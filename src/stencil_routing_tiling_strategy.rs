/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use device::{OffscreenDevice, ProgramId, ProgramLoader};
use euclid::{Point2D, Point3D, Rect, Size2D};
use gleam::gl::{self, GLfloat, GLint, GLuint};
use internal_types::DevicePixel;
use std::mem;
use std::rc::Rc;
use tiling::{BuiltTile, RenderableId, RenderableInstanceId, RenderableList, TileRange};
use tiling::{TilingStrategy};

const SAMPLES: GLuint = 8;
const TILE_SIZE: GLuint = 16;

static QUAD_VERTICES: [Point2D<f32>; 4] = [
    Point2D { x: -1.0, y: -1.0 },
    Point2D { x:  1.0, y: -1.0 },
    Point2D { x: -1.0, y:  1.0 },
    Point2D { x:  1.0, y:  1.0 },
];

#[allow(dead_code)]
pub struct StencilRoutingTilingStrategySharedState {
    device: OffscreenDevice,
    clear_program: ProgramId,
    clear_position_attribute: GLint,
    clear_vbo: GLuint,
    clear_vao: GLuint,
    transparent_program: ProgramId,
    transparent_position_attribute: GLint,
    transparent_renderable_id_attribute: GLint,
    transparent_framebuffer_size_uniform: GLint,
    transparent_screen_size_uniform: GLint,
    transparent_tile_size_uniform: GLint,
    transparent_vbo: GLuint,
    transparent_vao: GLuint,
    extract_program: ProgramId,
    extract_position_attribute: GLint,
    extract_texture_uniform: GLint,
    extract_samples_uniform: GLint,
    extract_vbo: GLuint,
    extract_vao: GLuint,
    samples_passed_query: GLuint,
}

impl Drop for StencilRoutingTilingStrategySharedState {
    fn drop(&mut self) {
        gl::delete_queries(&[self.samples_passed_query]);
        gl::delete_vertex_arrays(&[self.extract_vao]);
        gl::delete_buffers(&[self.extract_vbo]);
        gl::delete_program(self.extract_program.0);
        gl::delete_vertex_arrays(&[self.transparent_vao]);
        gl::delete_buffers(&[self.transparent_vbo]);
        gl::delete_program(self.transparent_program.0);
        gl::delete_vertex_arrays(&[self.clear_vao]);
        gl::delete_buffers(&[self.clear_vbo]);
        gl::delete_program(self.clear_program.0);
    }
}

impl StencilRoutingTilingStrategySharedState {
    pub fn new(mut device: OffscreenDevice) -> StencilRoutingTilingStrategySharedState {
        device.make_current().expect("Couldn't make the offscreen rendering context current!");

        let clear_program = device.create_program("sr_clear");
        let clear_position_attribute = gl::get_attrib_location(clear_program.0, "aPosition");

        let clear_vao = gl::gen_vertex_arrays(1)[0];
        let clear_vbo = gl::gen_buffers(1)[0];
        gl::bind_vertex_array(clear_vao);
        gl::bind_buffer(gl::ARRAY_BUFFER, clear_vbo);
        gl::buffer_data(gl::ARRAY_BUFFER, &QUAD_VERTICES, gl::STATIC_DRAW);
        gl::use_program(clear_program.0);
        gl::vertex_attrib_pointer(clear_position_attribute as GLuint,
                                  2,
                                  gl::FLOAT,
                                  false,
                                  mem::size_of::<Point2D<f32>>() as GLint,
                                  0);
        gl::enable_vertex_attrib_array(clear_position_attribute as GLuint);

        let transparent_program = device.create_program("sr_transparent");
        let transparent_position_attribute = gl::get_attrib_location(transparent_program.0,
                                                                     "aPosition");
        let transparent_renderable_id_attribute = gl::get_attrib_location(transparent_program.0,
                                                                          "aRenderableId");
        let transparent_framebuffer_size_uniform = gl::get_uniform_location(transparent_program.0,
                                                                            "uFramebufferSize");
        let transparent_screen_size_uniform = gl::get_uniform_location(transparent_program.0,
                                                                       "uScreenSize");
        let transparent_tile_size_uniform = gl::get_uniform_location(transparent_program.0,
                                                                     "uTileSize");

        let transparent_vao = gl::gen_vertex_arrays(1)[0];
        let transparent_vbo = gl::gen_buffers(1)[0];
        gl::bind_vertex_array(transparent_vao);
        gl::bind_buffer(gl::ARRAY_BUFFER, transparent_vbo);
        gl::use_program(transparent_program.0);
        gl::vertex_attrib_pointer(transparent_renderable_id_attribute as GLuint,
                                  4,
                                  gl::UNSIGNED_BYTE,
                                  true,
                                  mem::size_of::<TileVertex>() as GLint,
                                  0);
        gl::vertex_attrib_pointer(transparent_position_attribute as GLuint,
                                  3,
                                  gl::UNSIGNED_SHORT,
                                  false,
                                  mem::size_of::<TileVertex>() as GLint,
                                  4);
        gl::enable_vertex_attrib_array(transparent_renderable_id_attribute as GLuint);
        gl::enable_vertex_attrib_array(transparent_position_attribute as GLuint);

        let extract_program = device.create_program("sr_extract");
        let extract_position_attribute = gl::get_attrib_location(extract_program.0, "aPosition");
        let extract_texture_uniform = gl::get_uniform_location(extract_program.0, "uTexture");
        let extract_samples_uniform = gl::get_uniform_location(extract_program.0, "uSamples");

        let extract_vao = gl::gen_vertex_arrays(1)[0];
        let extract_vbo = gl::gen_buffers(1)[0];
        gl::bind_vertex_array(extract_vao);
        gl::bind_buffer(gl::ARRAY_BUFFER, extract_vbo);
        gl::buffer_data(gl::ARRAY_BUFFER, &QUAD_VERTICES, gl::STATIC_DRAW);
        gl::use_program(extract_program.0);
        gl::vertex_attrib_pointer(extract_position_attribute as GLuint,
                                  2,
                                  gl::FLOAT,
                                  false,
                                  mem::size_of::<Point2D<f32>>() as GLint,
                                  0);
        gl::enable_vertex_attrib_array(extract_position_attribute as GLuint);

        let samples_passed_query = gl::gen_queries(1)[0];

        StencilRoutingTilingStrategySharedState {
            device: device,
            clear_program: clear_program,
            clear_position_attribute: clear_position_attribute,
            clear_vbo: clear_vbo,
            clear_vao: clear_vao,
            transparent_program: transparent_program,
            transparent_position_attribute: transparent_position_attribute,
            transparent_renderable_id_attribute: transparent_renderable_id_attribute,
            transparent_framebuffer_size_uniform: transparent_framebuffer_size_uniform,
            transparent_screen_size_uniform: transparent_screen_size_uniform,
            transparent_tile_size_uniform: transparent_tile_size_uniform,
            transparent_vbo: transparent_vbo,
            transparent_vao: transparent_vao,
            extract_program: extract_program,
            extract_position_attribute: extract_position_attribute,
            extract_texture_uniform: extract_texture_uniform,
            extract_samples_uniform: extract_samples_uniform,
            extract_vbo: extract_vbo,
            extract_vao: extract_vao,
            samples_passed_query: samples_passed_query,
        }
    }
}

struct StencilRoutingFramebuffers {
    multisample_tiles_texture: GLuint,
    multisample_tiles_renderbuffer: GLuint,
    multisample_tiles_framebuffer: GLuint,
    extracted_tiles_texture: GLuint,
    extracted_tiles_framebuffer: GLuint,
}

impl Drop for StencilRoutingFramebuffers {
    fn drop(&mut self) {
        gl::delete_framebuffers(&[self.extracted_tiles_framebuffer]);
        gl::delete_textures(&[self.extracted_tiles_texture]);
        gl::delete_framebuffers(&[self.multisample_tiles_framebuffer]);
        gl::delete_renderbuffers(&[self.multisample_tiles_renderbuffer]);
        gl::delete_textures(&[self.multisample_tiles_texture]);
    }
}

impl StencilRoutingFramebuffers {
    fn new(tiles_framebuffer_size: &Size2D<u32>, extracted_tiles_framebuffer_size: &Size2D<u32>)
           -> StencilRoutingFramebuffers {
        let multisample_tiles_texture = gl::gen_textures(1)[0];
        gl::bind_texture(gl::TEXTURE_2D_MULTISAMPLE, multisample_tiles_texture);
        gl::tex_image_2d_multisample(gl::TEXTURE_2D_MULTISAMPLE,
                                     SAMPLES as GLint,
                                     gl::RGBA,
                                     tiles_framebuffer_size.width as GLint,
                                     tiles_framebuffer_size.height as GLint,
                                     true);

        let multisample_tiles_renderbuffer = gl::gen_renderbuffers(1)[0];
        gl::bind_renderbuffer(gl::RENDERBUFFER, multisample_tiles_renderbuffer);
        gl::renderbuffer_storage_multisample(gl::RENDERBUFFER,
                                             SAMPLES as GLint,
                                             gl::DEPTH24_STENCIL8,
                                             tiles_framebuffer_size.width as GLint,
                                             tiles_framebuffer_size.height as GLint);

        let multisample_tiles_framebuffer = gl::gen_framebuffers(1)[0];
        gl::bind_framebuffer(gl::FRAMEBUFFER, multisample_tiles_framebuffer);
        gl::framebuffer_texture_2d(gl::FRAMEBUFFER,
                                   gl::COLOR_ATTACHMENT0,
                                   gl::TEXTURE_2D_MULTISAMPLE,
                                   multisample_tiles_texture,
                                   0);
        gl::framebuffer_renderbuffer(gl::FRAMEBUFFER,
                                     gl::DEPTH_STENCIL_ATTACHMENT,
                                     gl::RENDERBUFFER,
                                     multisample_tiles_renderbuffer);

        let extracted_tiles_texture = gl::gen_textures(1)[0];
        gl::bind_texture(gl::TEXTURE_2D, extracted_tiles_texture);
        gl::tex_parameter_i(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as GLint);
        gl::tex_parameter_i(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as GLint);
        gl::tex_image_2d(gl::TEXTURE_2D,
                         0,
                         gl::RGBA as GLint,
                         extracted_tiles_framebuffer_size.width as GLint,
                         extracted_tiles_framebuffer_size.height as GLint,
                         0,
                         gl::RGBA,
                         gl::UNSIGNED_BYTE,
                         None);

        let extracted_tiles_framebuffer = gl::gen_framebuffers(1)[0];
        gl::bind_framebuffer(gl::FRAMEBUFFER, extracted_tiles_framebuffer);
        gl::framebuffer_texture_2d(gl::FRAMEBUFFER,
                                   gl::COLOR_ATTACHMENT0,
                                   gl::TEXTURE_2D,
                                   extracted_tiles_texture,
                                   0);

        StencilRoutingFramebuffers {
            multisample_tiles_texture: multisample_tiles_texture,
            multisample_tiles_renderbuffer: multisample_tiles_renderbuffer,
            multisample_tiles_framebuffer: multisample_tiles_framebuffer,
            extracted_tiles_texture: extracted_tiles_texture,
            extracted_tiles_framebuffer: extracted_tiles_framebuffer,
        }
    }
}

pub struct StencilRoutingTilingStrategy {
    shared: Rc<StencilRoutingTilingStrategySharedState>,
    screen_size: Size2D<DevicePixel>,
    renderables: Vec<RenderableId>,
    transparent_vertices: Vec<TileVertex>,
}

impl StencilRoutingTilingStrategy {
    pub fn new(shared_state: Rc<StencilRoutingTilingStrategySharedState>,
               screen_size: &Size2D<DevicePixel>)
               -> StencilRoutingTilingStrategy {
        StencilRoutingTilingStrategy {
            shared: shared_state,
            screen_size: *screen_size,
            renderables: vec![],
            transparent_vertices: vec![],
        }
    }

    fn tiles_framebuffer_size(&self) -> Size2D<u32> {
        Size2D::new((self.screen_size.width.0 as f32 / TILE_SIZE as f32).ceil() as u32,
                    (self.screen_size.height.0 as f32 / TILE_SIZE as f32).ceil() as u32)
    }

    fn extracted_tiles_framebuffer_size(&self) -> Size2D<u32> {
        let tiles_framebuffer_size = self.tiles_framebuffer_size();
        Size2D::new(tiles_framebuffer_size.width * SAMPLES, tiles_framebuffer_size.height)
    }

    fn add_renderable(&mut self, bounding_rect: &Rect<DevicePixel>, id: RenderableId) {
        self.renderables.push(id);

        let bounding_rect = Rect::new(Point2D::new(bounding_rect.origin.x.0 as u16,
                                                   bounding_rect.origin.y.0 as u16),
                                      Size2D::new(bounding_rect.size.width.0 as u16,
                                                  bounding_rect.size.height.0 as u16));
        //println!("bounding rect={:?}", bounding_rect);
        let depth = self.transparent_vertices.len() as u16;
        let top_left = TileVertex {
            renderable_id: id.0,
            position: Point3D::new(bounding_rect.origin.x, bounding_rect.origin.y, depth),
        };
        let top_right = TileVertex {
            renderable_id: id.0,
            position: Point3D::new(bounding_rect.max_x(), bounding_rect.origin.y, depth),
        };
        let bottom_right = TileVertex {
            renderable_id: id.0,
            position: Point3D::new(bounding_rect.max_x(), bounding_rect.max_y(), depth),
        };
        let bottom_left = TileVertex {
            renderable_id: id.0,
            position: Point3D::new(bounding_rect.origin.x, bounding_rect.max_y(), depth),
        };
        //println!("added renderable id: {:?}", id.0);
        self.transparent_vertices
            .extend(&[top_left, top_right, bottom_left, top_right, bottom_left, bottom_right]);
    }
}

impl TilingStrategy for StencilRoutingTilingStrategy {
    fn add_renderables(&mut self, renderable_list: &RenderableList) {
        for (renderable_index, renderable) in renderable_list.renderables.iter().enumerate() {
            let renderable_id = RenderableId(renderable_index as u32);
            self.add_renderable(&renderable.bounding_rect, RenderableId(renderable_id.0))
        }
    }

    fn region_count(&mut self) -> usize {
        1
    }

    #[inline(always)]
    fn get_tile_range(&self, _: &Rect<DevicePixel>) -> TileRange {
        TileRange::new(0, 0, 0, 0)
    }

    fn instances(&mut self, _: usize) -> &mut Vec<RenderableId> {
        &mut self.renderables
    }

    fn build_and_process_tiles<F>(&mut self, _: usize, mut iteration_function: F)
                                  where F: for<'a> FnMut(Rect<DevicePixel>,
                                                         BuiltTile<'a>,
                                                         &'a mut Vec<RenderableId>) {
        // Upload vertices.
        gl::bind_buffer(gl::ARRAY_BUFFER, self.shared.transparent_vbo);
        gl::buffer_data(gl::ARRAY_BUFFER, &self.transparent_vertices[..], gl::DYNAMIC_DRAW);

        // Initialize framebuffers.
        let tiles_framebuffer_size = self.tiles_framebuffer_size();
        let extracted_tiles_framebuffer_size = self.extracted_tiles_framebuffer_size();
        let framebuffers = StencilRoutingFramebuffers::new(&tiles_framebuffer_size,
                                                           &extracted_tiles_framebuffer_size);

        // Clear.
        gl::bind_framebuffer(gl::FRAMEBUFFER, framebuffers.multisample_tiles_framebuffer);
        gl::viewport(0,
                     0,
                     tiles_framebuffer_size.width as GLint,
                     tiles_framebuffer_size.height as GLint);
        gl::clear_color(1.0, 1.0, 1.0, 1.0);
        gl::clear_stencil(0);
        gl::clear_depth(0.0);
        gl::clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT | gl::STENCIL_BUFFER_BIT);

        // Initialize the stencil buffer for routing.
        gl::use_program(self.shared.clear_program.0);
        gl::bind_vertex_array(self.shared.clear_vao);
        gl::bind_buffer(gl::ARRAY_BUFFER, self.shared.clear_vbo);
        gl::color_mask(false, false, false, false);
        gl::depth_mask(false);
        gl::disable(gl::DEPTH_TEST);
        gl::enable(gl::MULTISAMPLE);
        gl::enable(gl::SAMPLE_MASK);
        gl::enable(gl::STENCIL_TEST);
        for sample in 0..SAMPLES {
            gl::sample_mask_i(0, 1 << sample);
            gl::stencil_func(gl::ALWAYS, sample as GLint + 2, !0);
            gl::stencil_op(gl::KEEP, gl::KEEP, gl::REPLACE);
            gl::draw_arrays(gl::TRIANGLE_STRIP, 0, 4);
        }

        // TODO(pcwalton): Do a depth prepass for occlusion culling.

        // Perform transparent routed drawing.
        gl::use_program(self.shared.transparent_program.0);
        gl::bind_vertex_array(self.shared.transparent_vao);
        gl::bind_buffer(gl::ARRAY_BUFFER, self.shared.transparent_vbo);
        gl::color_mask(true, true, true, true);
        gl::disable(gl::DEPTH_TEST);
        gl::disable(gl::MULTISAMPLE);
        gl::disable(gl::SAMPLE_MASK);
        gl::enable(gl::STENCIL_TEST);
        gl::stencil_func(gl::EQUAL, 2, !0);
        gl::stencil_op(gl::DECR, gl::KEEP, gl::DECR);
        gl::uniform_2f(self.shared.transparent_framebuffer_size_uniform,
                       tiles_framebuffer_size.width as GLfloat,
                       tiles_framebuffer_size.height as GLfloat);
        gl::uniform_2f(self.shared.transparent_screen_size_uniform,
                       self.screen_size.width.0 as GLfloat,
                       self.screen_size.height.0 as GLfloat);
        gl::draw_arrays(gl::TRIANGLES, 0, self.transparent_vertices.len() as GLint);

        // Check for overflow.
        gl::use_program(self.shared.clear_program.0);
        gl::bind_vertex_array(self.shared.clear_vao);
        gl::bind_buffer(gl::ARRAY_BUFFER, self.shared.clear_vbo);
        gl::color_mask(false, false, false, false);
        gl::disable(gl::DEPTH_TEST);
        gl::enable(gl::MULTISAMPLE);
        gl::enable(gl::SAMPLE_MASK);
        gl::enable(gl::STENCIL_TEST);
        gl::sample_mask_i(0, 1 << (SAMPLES - 1));
        gl::stencil_func(gl::EQUAL, 0, !0);
        gl::stencil_op(gl::KEEP, gl::KEEP, gl::KEEP);
        gl::begin_query(gl::ANY_SAMPLES_PASSED, self.shared.samples_passed_query);
        gl::draw_arrays(gl::TRIANGLE_STRIP, 0, 4);
        gl::end_query(gl::ANY_SAMPLES_PASSED);

        // Extract the samples.
        gl::bind_framebuffer(gl::FRAMEBUFFER, framebuffers.extracted_tiles_framebuffer);
        gl::viewport(0,
                     0,
                     extracted_tiles_framebuffer_size.width as GLint,
                     extracted_tiles_framebuffer_size.height as GLint);
        gl::use_program(self.shared.extract_program.0);
        gl::bind_vertex_array(self.shared.extract_vao);
        gl::bind_buffer(gl::ARRAY_BUFFER, self.shared.extract_vbo);
        gl::color_mask(true, true, true, true);
        gl::disable(gl::DEPTH_TEST);
        gl::disable(gl::MULTISAMPLE);
        gl::disable(gl::SAMPLE_MASK);
        gl::disable(gl::STENCIL_TEST);
        gl::active_texture(gl::TEXTURE0);
        gl::bind_texture(gl::TEXTURE_2D_MULTISAMPLE, framebuffers.multisample_tiles_texture);
        gl::uniform_1i(self.shared.extract_texture_uniform, 0);
        gl::uniform_1i(self.shared.extract_samples_uniform, SAMPLES as GLint);
        gl::draw_arrays(gl::TRIANGLE_STRIP, 0, 4);
        gl::bind_texture(gl::TEXTURE_2D_MULTISAMPLE, 0);

        // Read back extracted data.
        self.renderables.clear();
        let mut indices = vec![];
        let extracted_sample_data =
            gl::read_pixels(0,
                            0,
                            extracted_tiles_framebuffer_size.width as GLint,
                            extracted_tiles_framebuffer_size.height as GLint,
                            gl::RGBA,
                            gl::UNSIGNED_INT_8_8_8_8_REV);
        for y in 0..tiles_framebuffer_size.height {
            for x in 0..tiles_framebuffer_size.width {
                for sample in 0..SAMPLES {
                    let index = ((y * extracted_tiles_framebuffer_size.width + x * SAMPLES +
                                  sample) * 4) as usize;
                    let sample_data = &extracted_sample_data[index..(index + 4)];
                    if sample_data == &[0xff, 0xff, 0xff, 0xff] {
                        break
                    }
                    let renderable_id = ((sample_data[1] as u32) << 8) | (sample_data[0] as u32);
                    /*println!("extracted sample data={:?} renderable id={:?}",
                             sample_data,
                             renderable_id);*/
                    let renderable_id = RenderableId(renderable_id);
                    let renderable_instance_id =
                        RenderableInstanceId(self.renderables.len() as u32);
                    indices.push(renderable_instance_id);
                    self.renderables.push(renderable_id);
                }
                let tile_size = Size2D::new(DevicePixel(TILE_SIZE as GLint),
                                            DevicePixel(TILE_SIZE as GLint));
                let tile_rect = Rect::new(Point2D::new(DevicePixel((x * TILE_SIZE) as GLint),
                                                       DevicePixel((y * TILE_SIZE) as GLint)),
                                          tile_size);
                indices.reverse();
                iteration_function(tile_rect,
                                   BuiltTile::Tile(&mut indices),
                                   &mut self.renderables);
                indices.clear();
            }
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
struct TileVertex {
    renderable_id: u32,
    position: Point3D<u16>,
}

