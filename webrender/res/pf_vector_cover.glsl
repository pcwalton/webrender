/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include shared

#ifdef WR_VERTEX_SHADER

in ivec4 aBounds;

void main(void) {
    vec4 bounds = vec4(aBounds);
    vec2 position = bounds.xy + mix(vec2(0.0), bounds.zw, aPosition.xy);
    gl_Position = uTransform * vec4(position, aPosition.z, 1.0);
}

#endif

#ifdef WR_FRAGMENT_SHADER

void main(void) {
    oFragColor = TEXEL_FETCH(sColor0, ivec2(floor(gl_FragCoord.xy)), 0, ivec2(0)).rrrr;
}

#endif
