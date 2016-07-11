/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use euclid::{Point2D, Rect, Size2D};
use internal_types::DevicePixel;
use std::cmp;
use std::mem;
use std::ptr;
use tiling::{BuiltTile, RenderableId, RenderableInstanceId, RenderableList, TilingStrategy};
use util;

const FRAMEBUFFER_COUNT: u8 = 16;
const SAMPLES: u8 = 16;

// FIXME(pcwalton): Shouldn't be public.
pub const TILE_SIZE: u32 = 16;

pub struct SimdTilingStrategy {
    framebuffer_size: Size2D<u32>,
    framebuffers: [Vec<u16>; FRAMEBUFFER_COUNT as usize],
    stencil: Vec<u8>,
    vbos: Vec<Vec<u16>>,
    instances: Vec<RenderableId>,
}

impl SimdTilingStrategy {
    pub fn new() -> SimdTilingStrategy {
        SimdTilingStrategy {
            framebuffer_size: Size2D::zero(),
            framebuffers: [
                vec![], vec![], vec![], vec![],
                vec![], vec![], vec![], vec![],
                vec![], vec![], vec![], vec![],
                vec![], vec![], vec![], vec![],
            ],
            stencil: vec![],
            vbos: vec![vec![]; SAMPLES as usize],
            instances: vec![],
        }
    }

    fn tile_rect(&self, rect: &Rect<DevicePixel>) -> Rect<i32> {
        let framebuffer_width = self.framebuffer_size.width as i32;
        let framebuffer_height = self.framebuffer_size.height as i32;
        let tile_size = TILE_SIZE as i32;
        let left = util::clamp(rect.origin.x.0 / tile_size, 0, framebuffer_width - 1);
        let top = util::clamp(rect.origin.y.0 / tile_size, 0, framebuffer_height - 1);
        let right = util::clamp(rect.max_x().0 / tile_size, 0, framebuffer_width - 1);
        let bottom = util::clamp(rect.max_y().0 / tile_size, 0, framebuffer_height - 1);
        util::rect_from_points(left, top, right, bottom)
    }

    fn draw_opaque_scanline(&mut self, color: u16, left: i32, right: i32, y: i32) {
        if left > right {
            return
        }

        // Hoist out the `stencil_row` slicing here to avoid a bounds check in the hot loop below.
        let row_pixel_index = (y as usize) * (self.framebuffer_size.width as usize);
        let left_pixel_index = row_pixel_index + (left as usize);
        let right_pixel_index = row_pixel_index + (right as usize) + 1;
        let mut stencil_row = &mut self.stencil[left_pixel_index..right_pixel_index];
        for (i, stencil_ptr) in stencil_row.iter_mut().enumerate() {
            draw_opaque_pixel(stencil_ptr,
                              &mut self.framebuffers,
                              &self.framebuffer_size,
                              color,
                              left + (i as i32),
                              y)
        }
    }

    fn draw_transparent_scanline(&mut self, color: u16, left: i32, right: i32, y: i32) {
        if left > right {
            return
        }

        // Hoist out the `stencil_row` slicing here to avoid a bounds check in the hot loop below.
        let row_pixel_index = (y as usize) * (self.framebuffer_size.width as usize);
        let left_pixel_index = row_pixel_index + (left as usize);
        let right_pixel_index = row_pixel_index + (right as usize) + 1;
        let mut stencil_row = &mut self.stencil[left_pixel_index..right_pixel_index];
        for (i, stencil_ptr) in stencil_row.iter_mut().enumerate() {
            draw_transparent_pixel(stencil_ptr,
                                   &mut self.framebuffers,
                                   &self.framebuffer_size,
                                   color,
                                   left + (i as i32),
                                   y)
        }
    }

    fn draw_transparent_rect(&mut self, color: u16, rect: &Rect<DevicePixel>) {
        let tile_rect = self.tile_rect(rect);
        let (left, top) = (tile_rect.origin.x, tile_rect.origin.y);
        let (right, bottom) = (tile_rect.max_x(), tile_rect.max_y());
        /*println!("rect={:?} left={:?} top={:?} right={:?} bottom={:?}",
                rect,
                left,
                top,
                right,
                bottom);*/
        for y in top..(bottom + 1) {
            self.draw_transparent_scanline(color, left, right, y);
        }
    }

    fn draw_opaque_rect(&mut self, color: u16, rect: &Rect<DevicePixel>) {
        let framebuffer_width = self.framebuffer_size.width as i32;
        let tile_rect = self.tile_rect(rect);
        let tile_size = TILE_SIZE as i32;
        let (mut left, mut top) = (tile_rect.origin.x, tile_rect.origin.y);
        let (mut right, mut bottom) = (tile_rect.max_x(), tile_rect.max_y());

        let partially_overlaps_left_tile = rect.origin.x.0 % tile_size != 0;
        let partially_overlaps_top_tile = rect.origin.y.0 % tile_size != 0;
        let partially_overlaps_right_tile = rect.max_x().0 % tile_size != 0;
        let partially_overlaps_bottom_tile = rect.max_y().0 % tile_size != 0;
        let opaque_left = if partially_overlaps_left_tile {
            left + 1
        } else {
            left
        };
        let opaque_right = if partially_overlaps_right_tile {
            right - 1
        } else {
            right
        };
        let opaque_top = if partially_overlaps_top_tile {
            top + 1
        } else {
            top
        };
        let opaque_bottom = if partially_overlaps_bottom_tile {
            bottom - 1
        } else {
            bottom
        };

        if partially_overlaps_top_tile {
            self.draw_transparent_scanline(color, left, right, top);
        }
        for y in opaque_top..(opaque_bottom + 1) {
            let row_pixel_index = (y as usize) * (self.framebuffer_size.width as usize);
            if partially_overlaps_left_tile {
                draw_transparent_pixel(&mut self.stencil[row_pixel_index + left as usize],
                                       &mut self.framebuffers,
                                       &self.framebuffer_size,
                                       color,
                                       left,
                                       y)
            }
            self.draw_opaque_scanline(color, opaque_left, opaque_right, y);
            if partially_overlaps_right_tile && right != left {
                draw_transparent_pixel(&mut self.stencil[row_pixel_index + right as usize],
                                       &mut self.framebuffers,
                                       &self.framebuffer_size,
                                       color,
                                       right,
                                       y)
            }
        }
        if partially_overlaps_bottom_tile && bottom != top {
            self.draw_transparent_scanline(color, left, right, bottom)
        }
    }
}

impl TilingStrategy for SimdTilingStrategy {
    #[inline(never)]
    fn reset(&mut self, screen_rect: &Rect<DevicePixel>, device_pixel_ratio: f32) {
        self.framebuffer_size =
            Size2D::new(util::div_roundup(screen_rect.size.width.0 as u32, TILE_SIZE),
                        util::div_roundup(screen_rect.size.height.0 as u32, TILE_SIZE));
        let framebuffer_area = (self.framebuffer_size.width as usize) *
            (self.framebuffer_size.height as usize);

        for framebuffer in &mut self.framebuffers {
            framebuffer.resize(framebuffer_area * (SAMPLES as usize), 0xffff)
        }

        for element in &mut self.stencil {
            *element = 0
        }
        self.stencil.resize(framebuffer_area, 0);

        for vbo in &mut self.vbos {
            vbo.clear();
        }
        self.vbos.resize(SAMPLES as usize, vec![]);

        self.instances.clear()
    }

    #[inline(never)]
    fn add_renderables(&mut self, renderable_list: &RenderableList) {
        for (renderable_index, renderable) in renderable_list.renderables.iter().enumerate() {
            if renderable.is_opaque {
                self.draw_opaque_rect(renderable_index as u16, &renderable.bounding_rect)
            } else {
                self.draw_transparent_rect(renderable_index as u16, &renderable.bounding_rect)
            }
        }
    }

    fn region_count(&mut self) -> usize {
        1
    }

    fn instances(&mut self, region_index: usize) -> &mut Vec<RenderableId> {
        debug_assert!(region_index == 0);
        &mut self.instances
    }

    #[inline(never)]
    fn build_and_process_tiles<F>(&mut self, region_index: usize, mut iteration_function: F)
                                  where F: for<'a> FnMut(Rect<DevicePixel>,
                                                         BuiltTile<'a>,
                                                         &'a mut Vec<RenderableId>) {
        debug_assert!(region_index == 0);

        /*
        let mut vbo_end_ptrs = [ptr::null_mut(); SAMPLES as usize];
        for (i, vbo) in self.vbos.iter_mut().enumerate() {
            vbo.reserve((self.framebuffer_size.width as usize) *
                        (self.framebuffer_size.height as usize) *
                        (SAMPLES as usize));
            vbo_end_ptrs[i] = vbo.as_mut_ptr()
        }
        unsafe {
            webrender_create_vbos_avx2(self.framebuffer.as_mut_ptr(),
                                       self.stencil.as_ptr(),
                                       self.framebuffer_size.width as u64,
                                       self.framebuffer_size.height as u64,
                                       &mut vbo_end_ptrs[0]);
            //println!("vbos:");
            for (i, vbo_end_ptr) in vbo_end_ptrs.iter().enumerate() {
                let vbo_addr = self.vbos[i].as_ptr() as usize;
                self.vbos[i].set_len(((*vbo_end_ptr as usize) - vbo_addr as usize) /
                                     mem::size_of::<u16>());
                //println!("{}: {:?}", i, self.vbos[i]);
            }
        }*/


        let mut indices = vec![];

        /*
        for (i, vbo) in self.vbos.iter_mut().enumerate() {
            let mut vbo = vbo.iter();
            'outer: loop {
                let y = match vbo.next() {
                    Some(y) => *y as u32,
                    None => break,
                };
                let x_start = match vbo.next() {
                    Some(x_start) => *x_start as u32,
                    None => break,
                };
                let x_end = match vbo.next() {
                    Some(x_end) => *x_end as u32,
                    None => break,
                };

                indices.clear();
                for _ in 0..i {
                    let index = match vbo.next() {
                        Some(index) => *index as u32,
                        None => break 'outer,
                    };
                    indices.push(RenderableInstanceId(self.instances.len() as u32));
                    self.instances.push(RenderableId(index));
                }*/

        let mut iterator = TileIterator::new(&self.framebuffers,
                                             &self.stencil,
                                             &self.framebuffer_size);
        //print!("000: ");
        while let Some(Tile { y, x_start, x_end }) = iterator.next() {
            let renderables = &iterator.renderable_id_buffer[..];
            /*println!("y={:?} x={:?}-{:?} renderable length={:?}",
                     y,
                     x_start,
                     x_end,
                     renderables.len());*/

            /*
            for _ in x_start..x_end {
                let len = renderables.len();
                let ch = match len {
                    0...9 => (b'0' + (len as u8)) as char,
                    10...35 => (b'a' + ((len - 10) as u8)) as char,
                    _ => (b'A' + ((len - 36) as u8)) as char,
                };
                print!("{}", ch);
            }
            if x_end == self.framebuffer_size.width {
                print!("\n{:03}: ", y + 1);
            }
            */

            indices.clear();
            for &renderable_id in renderables {
                indices.push(RenderableInstanceId(self.instances.len() as u32));
                self.instances.push(renderable_id);
            }

            let rect =
                Rect::new(Point2D::new(DevicePixel((x_start * TILE_SIZE) as i32),
                                       DevicePixel((y * TILE_SIZE) as i32)),
                          Size2D::new(DevicePixel(((x_end - x_start) * TILE_SIZE) as i32),
                                      DevicePixel(TILE_SIZE as i32)));
            //println!("i={:?} rect={:?}", i, rect);
            indices.reverse();
            iteration_function(rect, BuiltTile::Tile(&mut indices), &mut self.instances)
        }
        //println!("");
    }
}

struct TileIterator<'a> {
    framebuffers: &'a [Vec<u16>],
    stencil: &'a [u8],
    framebuffer_size: Size2D<u32>,
    renderable_id_buffer: Vec<RenderableId>,
    short_renderable_id_buffer: Vec<u16>,
    position: Point2D<u32>,
}

impl<'a> TileIterator<'a> {
    #[inline]
    fn new<'b>(framebuffers: &'b [Vec<u16>], stencil: &'b [u8], framebuffer_size: &Size2D<u32>)
               -> TileIterator<'b> {
        TileIterator {
            framebuffers: framebuffers,
            stencil: stencil,
            framebuffer_size: *framebuffer_size,
            renderable_id_buffer: vec![],
            short_renderable_id_buffer: Vec::with_capacity(32),
            position: Point2D::zero(),
        }
    }
}

impl<'a> Iterator for TileIterator<'a> {
    type Item = Tile;

    #[inline]
    fn next(&mut self) -> Option<Tile> {
        if self.position.y == self.framebuffer_size.height {
            return None
        }

        let buffered_stencil_value = stencil_value(self.stencil,
                                                   &self.framebuffer_size,
                                                   &self.position);

        self.short_renderable_id_buffer.clear();
        for i in 0..util::div_roundup(buffered_stencil_value, SAMPLES as u8) {
            self.short_renderable_id_buffer
                .extend(framebuffer_samples(&self.framebuffers[..],
                                            i,
                                            buffered_stencil_value,
                                            &self.framebuffer_size,
                                            &self.position))
        }
        if buffered_stencil_value >= SAMPLES {
            //println!("short_renderable_id_buffer={:?}", self.short_renderable_id_buffer);
        }

        let (x_start, y) = (self.position.x, self.position.y);
        self.position.x += 1;
        'outer: while self.position.x < self.framebuffer_size.width {
            let stencil_value = stencil_value(self.stencil,
                                              &self.framebuffer_size,
                                              &self.position);
            if stencil_value != buffered_stencil_value {
                break
            }

            for i in 0..util::div_roundup(stencil_value, SAMPLES as u8) {
                let first_sample_index = (i as usize) * (SAMPLES as usize);
                let last_sample_index = cmp::min(((i as usize) + 1) * (SAMPLES as usize),
                                                 stencil_value as usize);
                let short_renderable_id_buffer =
                    &self.short_renderable_id_buffer[first_sample_index..last_sample_index];
                if framebuffer_samples(&self.framebuffers[..],
                                       i,
                                       stencil_value,
                                       &self.framebuffer_size,
                                       &self.position) != short_renderable_id_buffer {
                    break 'outer
                }
            }

            self.position.x += 1
        }

        let x_end = self.position.x;
        if x_end == self.framebuffer_size.width {
            self.position.x = 0;
            self.position.y += 1
        }

        self.renderable_id_buffer.clear();
        for &renderable_id in &self.short_renderable_id_buffer {
            self.renderable_id_buffer.push(RenderableId(renderable_id as u32))
        }

        Some(Tile {
            y: y,
            x_start: x_start,
            x_end: x_end,
        })
    }
}

fn stencil_value(stencil: &[u8],
                 framebuffer_size: &Size2D<u32>,
                 position: &Point2D<u32>)
                 -> u8 {
    stencil[(position.y as usize) * (framebuffer_size.width as usize) + (position.x as usize)]
}

fn framebuffer_samples<'a>(framebuffers: &'a [Vec<u16>],
                           framebuffer_index: u8,
                           stencil_value: u8,
                           framebuffer_size: &Size2D<u32>,
                           position: &Point2D<u32>)
                           -> &'a [u16] {
    let sample_index = cmp::min((framebuffer_index + 1) * SAMPLES, stencil_value) -
        framebuffer_index * SAMPLES;
    let first_framebuffer_sample_offset = ((position.y as usize) *
        (framebuffer_size.width as usize) + (position.x as usize)) * (SAMPLES as usize);
    let last_framebuffer_sample_offset = first_framebuffer_sample_offset +
        (sample_index as usize);
    let framebuffer_index = framebuffer_index as usize;
    &framebuffers[framebuffer_index][first_framebuffer_sample_offset..
                                     last_framebuffer_sample_offset]
}

fn framebuffer_index_and_sample_index_for_stencil_value(stencil_value: u8) -> (u8, u8) {
    let samples = SAMPLES as u8;
    let framebuffer_index = stencil_value / samples;
    let sample_index = stencil_value - framebuffer_index * samples;
    (framebuffer_index, sample_index)
}

struct Tile {
    y: u32,
    x_start: u32,
    x_end: u32,
}

fn draw_transparent_pixel(stencil_ptr: &mut u8,
                          framebuffers: &mut [Vec<u16>; FRAMEBUFFER_COUNT as usize],
                          framebuffer_size: &Size2D<u32>,
                          color: u16,
                          x: i32,
                          y: i32) {
    let stencil_value = *stencil_ptr;
    *stencil_ptr = stencil_value + 1;
    let (framebuffer_index, sample_index) =
        framebuffer_index_and_sample_index_for_stencil_value(stencil_value);
    let pixel_index = (y as usize) * (framebuffer_size.width as usize) + (x as usize);
    let framebuffer_pixel_index = pixel_index * (SAMPLES as usize) + (sample_index as usize);
    framebuffers[framebuffer_index as usize][framebuffer_pixel_index] = color
}

fn draw_opaque_pixel(stencil_ptr: &mut u8,
                     framebuffers: &mut [Vec<u16>; FRAMEBUFFER_COUNT as usize],
                     framebuffer_size: &Size2D<u32>,
                     color: u16,
                     x: i32,
                     y: i32) {
    *stencil_ptr = 1;
    let pixel_index = (y as usize) * (framebuffer_size.width as usize) + (x as usize);
    let framebuffer_pixel_index = pixel_index * (SAMPLES as usize);
    framebuffers[0][framebuffer_pixel_index] = color
}

