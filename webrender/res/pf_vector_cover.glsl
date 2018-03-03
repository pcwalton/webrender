/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include shared

#ifdef WR_VERTEX_SHADER

in ivec4 aTargetRect;
in ivec2 aStencilOrigin;

out vec2 vStencilUV;

void main(void) {
    vec4 targetRect = vec4(aTargetRect);
    vec2 stencilOrigin = vec2(aStencilOrigin);

    vec2 offset = mix(vec2(0.0), targetRect.zw, aPosition.xy);
    vec2 targetPosition = targetRect.xy + offset, stencilPosition = stencilOrigin + offset;

    gl_Position = uTransform * vec4(targetPosition, aPosition.z, 1.0);
    vStencilUV = stencilPosition;
}

#endif

#ifdef WR_FRAGMENT_SHADER

in vec2 vStencilUV;

void main(void) {
    oFragColor = abs(TEXEL_FETCH(sColor0, ivec2(vStencilUV), 0, ivec2(0)).rrrr);
}

#endif
