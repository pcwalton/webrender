use euclid::{Point2D, Rect, Size2D};
use internal_types::BasicRotationAngle;

pub const RECT_COUNT: u32 = 4;

pub trait BorderCornerTessellation {
    fn tessellate_border_corner(&self,
                                outer_radius: &Size2D<f32>,
                                inner_radius: &Size2D<f32>,
                                rotation_angle: BasicRotationAngle,
                                index: u32)
                                -> Rect<f32>;
}

impl BorderCornerTessellation for Rect<f32> {
    fn tessellate_border_corner(&self,
                                outer_radius: &Size2D<f32>,
                                inner_radius: &Size2D<f32>,
                                rotation_angle: BasicRotationAngle,
                                index: u32)
                                -> Rect<f32> {
        let delta = outer_radius.width / (RECT_COUNT as f32);
        let prev_x = delta * (index as f32);
        let prev_outer_y = ellipse_y_coordinate(prev_x, outer_radius);

        let next_x = prev_x + delta;
        let next_inner_y = ellipse_y_coordinate(next_x, inner_radius);

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

        subrect.translate(&self.origin)
    }
}

fn ellipse_y_coordinate(x: f32, radius: &Size2D<f32>) -> f32 {
    let radicand = 1.0 - (x / radius.width) * (x / radius.width);
    if radicand < 0.0 {
        0.0
    } else {
        radius.height * radicand.sqrt()
    }
}

