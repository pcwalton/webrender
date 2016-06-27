/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#![feature(step_by)]
//#![feature(mpsc_select)]

#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate log;

mod aabbtree;
mod batch;
mod batch_builder;
mod bsp_tiling_strategy;
mod bsptree;
mod debug_font_data;
mod debug_render;
mod device;
mod frame;
mod freelist;
mod geometry;
mod internal_types;
mod layer;
mod profiler;
mod quadtree;
mod render_backend;
mod resource_cache;
mod resource_list;
mod scene;
mod spring;
mod stencil_routing_tiling_strategy;
mod texture_cache;
mod tiling;
mod util;

mod platform {
    #[cfg(target_os="macos")]
    pub use platform::macos::font;
    #[cfg(any(target_os="linux", target_os="android", target_os = "windows"))]
    pub use platform::linux::font;

    #[cfg(target_os="macos")]
    pub mod macos {
        pub mod font;
    }
    #[cfg(any(target_os="linux", target_os="android", target_os = "windows"))]
    pub mod linux {
        pub mod font;
    }
}

pub mod renderer;

#[cfg(target_os="macos")]
extern crate core_graphics;
#[cfg(target_os="macos")]
extern crate core_text;

#[cfg(not(target_os="macos"))]
extern crate freetype;

extern crate app_units;
extern crate euclid;
extern crate fnv;
extern crate gleam;
//extern crate hprof;
extern crate ipc_channel;
extern crate num_traits;
//extern crate notify;
extern crate scoped_threadpool;
extern crate time;
extern crate webrender_traits;
extern crate offscreen_gl_context;
extern crate byteorder;
extern crate bit_set;

pub use renderer::{Renderer, RendererOptions};
