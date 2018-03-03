/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include shared

#ifdef WR_VERTEX_SHADER

in ivec4 aTargetRect;
in ivec2 aStencilOrigin;
in int aSubpixel;
in int aPad;

out vec2 vStencilUV;
flat out int vSubpixel;

void main(void) {
    vec4 targetRect = vec4(aTargetRect);
    vec2 stencilOrigin = vec2(aStencilOrigin);

    vec2 targetOffset = mix(vec2(0.0), targetRect.zw, aPosition.xy);
    vec2 targetPosition = targetRect.xy + targetOffset;
    vec2 stencilOffset = targetOffset * vec2(aSubpixel == 0 ? 1.0 : 3.0, 1.0);
    vec2 stencilPosition = stencilOrigin + stencilOffset;

    gl_Position = uTransform * vec4(targetPosition, aPosition.z, 1.0);
    vStencilUV = stencilPosition;
    vSubpixel = aSubpixel;
}

#endif

#ifdef WR_FRAGMENT_SHADER

in vec2 vStencilUV;
flat in int vSubpixel;

void main(void) {
    ivec2 stencilUV = ivec2(vStencilUV);
    if (vSubpixel == 0) {
        oFragColor = abs(TEXEL_FETCH(sColor0, stencilUV, 0, ivec2(0)).rrrr);
    } else {
        oFragColor = abs(vec4(TEXEL_FETCH(sColor0, stencilUV, 0, ivec2(0, 0)).r,
                              TEXEL_FETCH(sColor0, stencilUV, 0, ivec2(1, 0)).r,
                              TEXEL_FETCH(sColor0, stencilUV, 0, ivec2(2, 0)).r,
                              1.0));
    }
}

#endif
