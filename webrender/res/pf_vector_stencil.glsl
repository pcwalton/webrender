/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include shared

#ifdef WR_VERTEX_SHADER

in vec2 aFrom;
in vec2 aCtrl;
in vec2 aTo;
in int aPathID;

out float vWinding;

void main(void) {
    ivec2 pathAddress = ivec2(0.0, aPathID);
    mat2 transformLinear = mat2(TEXEL_FETCH(sColor0, pathAddress, 0, ivec2(0, 0)));
    vec2 transformTranslation = TEXEL_FETCH(sColor0, pathAddress, 0, ivec2(1, 0)).xy;
    float rectHeight = TEXEL_FETCH(sColor0, pathAddress, 0, ivec2(2, 0)).y;

    vec2 position;
    if (aPosition.x < 0.5)
        position.x = floor(min(aFrom.x, aTo.x));
    else
        position.x = ceil(max(aFrom.x, aTo.x));
    if (aPosition.y < 0.5)
        position.y = floor(min(aFrom.y, aTo.y));
    else
        position.y = rectHeight;

    position = transformLinear * position + transformTranslation;

    gl_Position = uTransform * vec4(position, aPosition.z, 1.0);
    vWinding = aFrom.x - aTo.x;
}

#endif

#ifdef WR_FRAGMENT_SHADER

in float vWinding;

void main(void) {
    oFragColor = vec4(sign(vWinding));
}

#endif
