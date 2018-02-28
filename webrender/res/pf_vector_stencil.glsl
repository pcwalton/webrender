/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include shared

#ifdef WR_VERTEX_SHADER

in vec2 aFrom;
in vec2 aCtrl;
in vec2 aTo;
in int aPathID;

void main(void) {
    vec2 position = mix(aFrom, aTo, aPosition.xy);

    ivec2 pathAddress = ivec2(0.0, aPathID);
    mat2 transformLinear = mat2(TEXEL_FETCH(sColor0, pathAddress, 0, ivec2(0, 0)));
    vec2 transformTranslation = TEXEL_FETCH(sColor0, pathAddress, 0, ivec2(1, 0)).xy;

    position = transformLinear * position + transformTranslation;

    gl_Position = uTransform * vec4(position, aPosition.z, 1.0);
}

#endif

#ifdef WR_FRAGMENT_SHADER

void main(void) {
    oFragColor = vec4(0.5);
}

#endif
