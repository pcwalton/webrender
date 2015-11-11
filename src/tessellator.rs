use euclid::{Point2D, Rect, Size2D};
use internal_types::{BasicRotationAngle, BorderCornerRect, RectPosUv};
use std::f32;
use util;

pub const RECT_COUNT: u32 = 16;

pub trait BorderCornerTessellation {
    fn tessellate_border_corner(&self,
                                outer_radius: &Size2D<f32>,
                                inner_radius: &Size2D<f32>,
                                rotation_angle: BasicRotationAngle)
                                -> Vec<BorderCornerRect>;
}

impl BorderCornerTessellation for Rect<f32> {
    fn tessellate_border_corner(&self,
                                outer_radius: &Size2D<f32>,
                                inner_radius: &Size2D<f32>,
                                rotation_angle: BasicRotationAngle)
                                -> Vec<BorderCornerRect> {
        let mut result = vec![];

        //println!("outer_radius={:?} inner_radius={:?}", outer_radius, inner_radius);

        let mut prev_x = 0.0;
        let mut prev_outer_y = outer_radius.height;
        let delta = outer_radius.width / (RECT_COUNT as f32);

        for rect_index in 0..RECT_COUNT {
            let next_x = prev_x + delta;

            let next_outer_radicand = 1.0 - (next_x / outer_radius.width) *
                (next_x / outer_radius.width);
            let next_outer_y = if next_outer_radicand < 0.0 {
                0.0
            } else {
                outer_radius.height * next_outer_radicand.sqrt()
            };

            let next_inner_radicand = 1.0 - (next_x / inner_radius.width) *
                (next_x / inner_radius.width);
            let next_inner_y = if next_inner_radicand < 0.0 {
                0.0
            } else {
                inner_radius.height * next_inner_radicand.sqrt()
            };

            /*println!("next_x={:?} next_outer_y={:?} next_inner_y={:?}",
                     next_x,
                     next_outer_y,
                     next_inner_y);*/

            let top_left = Point2D::new(prev_x, prev_outer_y);
            let bottom_right = Point2D::new(next_x, next_inner_y);

            let subrect = Rect::new(Point2D::new(top_left.x, bottom_right.y),
                                    Size2D::new(bottom_right.x - top_left.x,
                                                top_left.y - bottom_right.y));

            let subrect = match rotation_angle {
                BasicRotationAngle::Upright => {
                    Rect::new(Point2D::new(outer_radius.width - subrect.max_x(),
                                           outer_radius.height - subrect.max_y()),
                              subrect.size)
                }
                BasicRotationAngle::Clockwise90 => {
                    Rect::new(Point2D::new(subrect.origin.x,
                                           outer_radius.height - subrect.max_y()),
                              subrect.size)
                }
                BasicRotationAngle::Clockwise180 => {
                    subrect
                }
                BasicRotationAngle::Clockwise270 => {
                    Rect::new(Point2D::new(outer_radius.width - subrect.max_x(),
                                           subrect.origin.y),
                              subrect.size)
                }
            };

            //println!("angle={:?} subrect={:?}", rotation_angle, subrect);

            let subrect = subrect.translate(&self.origin);
            result.push(BorderCornerRect {
                rect: subrect,
                index: rect_index,
            });

            prev_x = next_x;
            prev_outer_y = next_outer_y;
        }

        result
    }
}

