/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//======================================================================================
// Vertex shader attributes and uniforms
//======================================================================================
#ifdef WR_VERTEX_SHADER
    #define varying out

    // Uniform inputs
	uniform mat4 uTransform;       // Orthographic projection
    uniform float uDevicePixelRatio;

    // Attribute inputs
	in vec3 aPosition;
#endif

//======================================================================================
// Fragment shader attributes and uniforms
//======================================================================================
#ifdef WR_FRAGMENT_SHADER
    #define varying in

    // Uniform inputs
    uniform sampler2D sDiffuse;
    uniform sampler2D sMask;

    // Fragment shader outputs
    out vec4 oFragColor;
#endif

//======================================================================================
// Interpolator definitions
//======================================================================================

//======================================================================================
// VS only types and UBOs
//======================================================================================

//======================================================================================
// VS only functions
//======================================================================================
