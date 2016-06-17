/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use euclid::{Point2D, Rect};
use internal_types::DevicePixel;
use std::i32;
use std::mem;
use util::{rect_contains_rect, rect_from_points};

// TODO(gw): Support enum type for rect that can be axis-aligned or a (projected) polygon.

#[derive(Debug, Copy, Clone)]
struct BspNodeIndex(usize);

#[derive(Debug, Copy, Clone)]
struct BspItemIndex(usize);

#[derive(Debug, Copy, Clone, PartialEq)]
enum SplitKind {
    Horizontal,
    Vertical,
}

pub struct BspItem<T> {
    rect: Rect<DevicePixel>,
    pub data: T,
}

pub struct BspNode {
    rect: Rect<DevicePixel>,
    cover_items: Vec<BspItemIndex>,
    partial_items: Vec<BspItemIndex>,
    child_index: Option<BspNodeIndex>,
}

impl BspNode {
    fn new(rect: Rect<DevicePixel>) -> BspNode {
        BspNode {
            rect: rect,
            cover_items: Vec::new(),
            partial_items: Vec::new(),
            child_index: None,
        }
    }

    fn new_with_capacity(rect: Rect<DevicePixel>, capacity: usize) -> BspNode {
        BspNode {
            rect: rect,
            cover_items: Vec::with_capacity(capacity),
            partial_items: Vec::with_capacity(capacity),
            child_index: None,
        }
    }

    fn add(&mut self, rect: &Rect<DevicePixel>, item_index: BspItemIndex) {
        debug_assert!(self.rect.intersects(rect));
        if rect_contains_rect(rect, &self.rect) {
            self.cover_items.push(item_index);
        } else {
            self.partial_items.push(item_index);
        }
    }
}

pub struct BspTree<T> {
    nodes: Vec<BspNode>,
    pub items: Vec<BspItem<T>>,
}

impl<T: Copy> BspTree<T> {
    pub fn new(bounding_rect: Rect<DevicePixel>) -> BspTree<T> {
        let root = BspNode::new(bounding_rect);

        BspTree {
            nodes: vec![root],
            items: Vec::new(),
        }
    }

    #[inline(always)]
    fn node(&self, node_index: BspNodeIndex) -> &BspNode {
        let BspNodeIndex(node_index) = node_index;
        &self.nodes[node_index]
    }

    #[inline(always)]
    fn node_mut(&mut self, node_index: BspNodeIndex) -> &mut BspNode {
        let BspNodeIndex(node_index) = node_index;
        &mut self.nodes[node_index]
    }

    pub fn add(&mut self,
               rect: &Rect<DevicePixel>,
               data: T) {
        let item_index = BspItemIndex(self.items.len());
        self.items.push(BspItem {
            rect: *rect,
            data: data,
        });
        let root = self.node_mut(BspNodeIndex(0));
        root.add(rect, item_index);
    }

    fn split_internal<F>(&mut self,
                         node_index: BspNodeIndex,
                         mut split_kind: SplitKind,
                         level: i32,
                         current_cover_items: &Vec<T>,
                         cover_buffer: &mut Vec<T>,
                         partial_buffer: &mut Vec<T>,
                         device_pixel_ratio: f32,
                         f: &mut F) where F: FnMut(&Rect<DevicePixel>, &mut Vec<T>, &mut Vec<T>) {
        let mut partial_items = mem::replace(&mut self.node_mut(node_index)
                                                      .partial_items,
                                             Vec::new());

        // TODO(gw): Optimize this by making cover items a stack!
        let mut current_cover_items = current_cover_items.clone();
        for item_index in &self.node(node_index).cover_items {
            let BspItemIndex(item_index) = *item_index;
            let item = &self.items[item_index];
            current_cover_items.push(item.data);
        }

        let node_rect = self.node(node_index).rect;
        let mut do_split = true;

        if partial_items.is_empty() {
            do_split = false;
        }

        let min_size = 8;
        if node_rect.size.width.0 <= min_size &&
           node_rect.size.height.0 <= min_size {
            do_split = false;
        }

        if !do_split {
            let node = self.node(node_index);

            cover_buffer.clear();
            for item in &current_cover_items {
                cover_buffer.push(*item);
            }

            partial_buffer.clear();
            for item_index in &node.partial_items {
                let BspItemIndex(item_index) = *item_index;
                let item = &self.items[item_index];
                partial_buffer.push(item.data);
            }

            f(&node_rect, cover_buffer, partial_buffer);
            return;
        }

        // Sort by split kind, find median.
        // TODO(gw): Find a faster heuristic for splitting that gives good results?
        let nx0 = node_rect.origin.x;
        let ny0 = node_rect.origin.y;
        let nx1 = nx0 + node_rect.size.width;
        let ny1 = ny0 + node_rect.size.height;

        //let mut indent = String::new();
        //for _ in 0..level {
        //    indent.push_str("  ");
        //}

        if split_kind == SplitKind::Horizontal && partial_items.iter().all(|item_index| {
            let item = &self.items[item_index.0];
            (item.rect.origin.y <= ny0 || item.rect.origin.y >= ny1) &&
                (item.rect.max_y() <= ny0 || item.rect.max_y() >= ny1)
        }) {
            split_kind = SplitKind::Vertical
        } else if split_kind == SplitKind::Vertical && partial_items.iter().all(|item_index| {
            let item = &self.items[item_index.0];
            (item.rect.origin.x <= nx0 || item.rect.origin.x >= nx1) &&
                (item.rect.max_x() <= nx0 || item.rect.max_x() >= nx1)
        }) {
            split_kind = SplitKind::Horizontal
        }

        let mut split;
        match split_kind {
            SplitKind::Horizontal => {
                let center = DevicePixel::from_i32(((ny0.0 as f32 + ny1.0 as f32) * 0.5).round() as
                                                   i32);
                split = center;
                let mut best_distance = i32::MAX;

                for item_index in &partial_items {
                    let BspItemIndex(item_index) = *item_index;
                    let item = &self.items[item_index];

                    // Only a valid split point if it sits exactly on a device pixel!
                    // TODO(pcwalton): Are these conditionals necessary?
                    let y0 = item.rect.origin.y;
                    if y0 > ny0 && y0 < ny1 {
                        let d = (y0.0 - center.0).abs();
                        if d < best_distance {
                            best_distance = d;
                            split = y0;
                        }
                    }

                    let y1 = y0 + item.rect.size.height;
                    if y1 > ny0 && y1 < ny1 {
                        let d = (y1.0 - center.0).abs();
                        if d < best_distance {
                            best_distance = d;
                            split = y1;
                        }
                    }
                }
            }
            SplitKind::Vertical => {
                let center = DevicePixel::from_i32(((nx0.0 as f32 + nx1.0 as f32) * 0.5).round() as
                                                   i32);
                split = center;
                let mut best_distance = i32::MAX;

                for item_index in &partial_items {
                    let BspItemIndex(item_index) = *item_index;
                    let item = &self.items[item_index];

                    // Only a valid split point if it sits exactly on a device pixel!
                    // TODO(pcwalton): Are these conditionals necessary?
                    let x0 = item.rect.origin.x;
                    if x0 > nx0 && x0 < nx1 {
                        let d = (x0.0 - center.0).abs();
                        if d < best_distance {
                            best_distance = d;
                            split = x0;
                        }
                    }

                    let x1 = x0 + item.rect.size.height;
                    if x1 > nx0 && x1 < nx1 {
                        let d = (x1.0 - center.0).abs();
                        if d < best_distance {
                            best_distance = d;
                            split = x1;
                        }
                    }
                }
            }
        }

        let (c0, c1, next_split) = match split_kind {
            SplitKind::Horizontal => {
                let r0 = rect_from_points(nx0, ny0, nx1, split);
                let r1 = rect_from_points(nx0, split, nx1, ny1);
                let mut c0 = BspNode::new_with_capacity(r0, partial_items.len());
                let mut c1 = BspNode::new_with_capacity(r1, partial_items.len());
                for key in partial_items.drain(..) {
                    let BspItemIndex(i) = key;
                    let item_rect = &self.items[i].rect;
                    if item_rect.origin.y < split {
                        c0.add(item_rect, key);
                    }
                    if item_rect.origin.y + item_rect.size.height > split {
                        c1.add(item_rect, key);
                    }
                }
                (c0, c1, SplitKind::Vertical)
            }
            SplitKind::Vertical => {
                let r0 = rect_from_points(nx0, ny0, split, ny1);
                let r1 = rect_from_points(split, ny0, nx1, ny1);
                let mut c0 = BspNode::new_with_capacity(r0, partial_items.len());
                let mut c1 = BspNode::new_with_capacity(r1, partial_items.len());
                for key in partial_items.drain(..) {
                    let BspItemIndex(i) = key;
                    let item_rect = &self.items[i].rect;
                    if item_rect.origin.x < split {
                        c0.add(item_rect, key);
                    }
                    if item_rect.origin.x + item_rect.size.width > split {
                        c1.add(item_rect, key);
                    }
                }
                (c0, c1, SplitKind::Horizontal)
            }
        };

        let ci0 = BspNodeIndex(self.nodes.len() + 0);
        let ci1 = BspNodeIndex(self.nodes.len() + 1);
        self.node_mut(node_index).child_index = Some(ci0);
        self.nodes.push(c0);
        self.nodes.push(c1);
        self.split_internal(ci0,
                            next_split,
                            level+1,
                            &current_cover_items,
                            cover_buffer,
                            partial_buffer,
                            device_pixel_ratio,
                            f);
        self.split_internal(ci1,
                            next_split,
                            level+1,
                            &current_cover_items,
                            cover_buffer,
                            partial_buffer,
                            device_pixel_ratio,
                            f);
    }

    pub fn split<F>(&mut self,
                    device_pixel_ratio: f32,
                    f: &mut F) where F: FnMut(&Rect<DevicePixel>, &mut Vec<T>, &mut Vec<T>) {
        let mut cover_buffer = Vec::new();
        let mut partial_buffer = Vec::new();
        self.split_internal(BspNodeIndex(0),
                            SplitKind::Horizontal,
                            0,
                            &Vec::new(),
                            &mut cover_buffer,
                            &mut partial_buffer,
                            device_pixel_ratio,
                            f);
    }
}
