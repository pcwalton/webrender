/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include shared

#ifdef WR_VERTEX_SHADER

in vec2 aFromPosition;
in vec2 aCtrlPosition;
in vec2 aToPosition;
in vec2 aFromNormal;
in vec2 aCtrlNormal;
in vec2 aToNormal;
in int aPathID;
in int aPad;

out vec3 vUV;
out vec3 vXDist;

void main(void) {
    ivec2 pathAddress = ivec2(0.0, aPathID);
    mat2 transformLinear = mat2(TEXEL_FETCH(sColor0, pathAddress, 0, ivec2(0, 0)));
    vec2 transformTranslation = TEXEL_FETCH(sColor0, pathAddress, 0, ivec2(1, 0)).xy;

    vec4 miscInfo = TEXEL_FETCH(sColor0, pathAddress, 0, ivec2(2, 0));
    float rectHeight = miscInfo.x + transformTranslation.y;
    vec2 emboldenAmount = miscInfo.zw * 0.5;

    // Perform the transform.
    vec2 fromPosition = transformLinear * aFromPosition + transformTranslation;
    vec2 ctrlPosition = transformLinear * aCtrlPosition + transformTranslation;
    vec2 toPosition = transformLinear * aToPosition + transformTranslation;

    // Embolden as necessary.
    fromPosition -= aFromNormal * emboldenAmount;
    ctrlPosition -= aCtrlNormal * emboldenAmount;
    toPosition -= aToNormal * emboldenAmount;

    // Compute edge vectors.
    vec2 v02 = toPosition - fromPosition;
    vec2 v01 = ctrlPosition - fromPosition, v21 = ctrlPosition - toPosition;

    // Compute area of convex hull (w). Change from curve to line if appropriate.
    float w = determinant(mat2(v01, v02));
    float sqLen01 = dot(v01, v01), sqLen02 = dot(v02, v02), sqLen21 = dot(v21, v21);
    float minCtrlSqLen = dot(v02, v02) * 0.0001;
    float cosTheta = dot(v01, v21);
    if (sqLen01 < minCtrlSqLen || sqLen21 < minCtrlSqLen ||
        cosTheta * cosTheta >= 0.95 * sqLen01 * sqLen21) {
        w = 0.0;
        v01 = vec2(0.5, abs(v02.y) >= 0.01 ? 0.0 : 0.5) * v02.xx;
    }

    // Compute position and dilate. If too thin, discard to avoid artefacts.
    vec2 position;
    if (abs(v02.x) < 0.0001)
        position.x = 0.0;
    else if (aPosition.x < 0.5)
        position.x = floor(min(fromPosition.x, toPosition.x));
    else
        position.x = ceil(max(fromPosition.x, toPosition.x));
    if (aPosition.y < 0.5)
        position.y = floor(min(fromPosition.y, toPosition.y));
    else
        position.y = rectHeight;

    // Compute UV using Cramer's rule.
    // https://gamedev.stackexchange.com/a/63203
    vec2 v03 = position - fromPosition;
    vec3 uv = vec3(0.0, determinant(mat2(v01, v03)), sign(w));
    uv.x = uv.y + 0.5 * determinant(mat2(v03, v02));
    uv.xy /= determinant(mat2(v01, v02));

    // Compute X distances.
    vec3 xDist = position.x - vec3(fromPosition.x, ctrlPosition.x, toPosition.x);

    gl_Position = uTransform * vec4(position, aPosition.z, 1.0);
    vUV = uv;
    vXDist = xDist;
}

#endif

#ifdef WR_FRAGMENT_SHADER

in vec3 vUV;
in vec3 vXDist;

float fastSign(float x) {
    return x > 0.0 ? 1.0 : -1.0;
}

// Are we inside the convex hull of the curve? (This will always be false if this is a line.)
bool insideCurve(vec3 uv) {
    return uv.z != 0.0 && uv.x > 0.0 && uv.x < 1.0 && uv.y > 0.0 && uv.y < 1.0;
}

// Cubic approximation to the square area coverage, accurate to about 4%.
//
// FIXME(pcwalton): Unify with the existing area approximation in `shared.glsl`.
float estimateArea(float dist) {
    if (dist >= 0.707107)
        return 0.5;
    // Catch NaNs here.
    if (!(dist > -0.707107))
        return -0.5;
    return 1.14191 * dist - 0.83570 * dist * dist * dist;
}

float signedDistanceToCurve(vec2 uv, vec2 dUVDX, vec2 dUVDY, bool inCurve) {
    // u^2 - v for curves inside uv square; u - v otherwise.
    float g = uv.x;
    vec2 dG = vec2(dUVDX.x, dUVDY.x);
    if (inCurve) {
        g *= uv.x;
        dG *= 2.0 * uv.x;
    }
    g -= uv.y;
    dG -= vec2(dUVDX.y, dUVDY.y);
    return g / length(dG);
}

void main(void) {
    // Unpack.
    vec3 uv = vUV;
    vec2 dUVDX = dFdx(uv.xy), dUVDY = dFdy(uv.xy);
    vec3 xDist = vXDist;
    vec2 dXDistDX = dFdx(xDist.xz);

    // Calculate X distances between endpoints (x02, x10, and x21 respectively).
    vec3 vDist = xDist - xDist.zxy;

    // Compute winding number and convexity.
    bool inCurve = insideCurve(uv);
    float openWinding = fastSign(-vDist.x);
    float convex = uv.z != 0.0 ? uv.z : fastSign(vDist.x * dUVDY.y);

    // Compute open rect area.
    vec2 areas = clamp(xDist.xz / dXDistDX, -0.5, 0.5);
    float openRectArea = openWinding * (areas.y - areas.x);

    // Compute closed rect area and winding, if necessary.
    float closedRectArea = 0.0, closedWinding = 0.0;
    if (inCurve && vDist.y * vDist.z < 0.0) {
        closedRectArea = 0.5 - fastSign(vDist.y) * (vDist.x * vDist.y < 0.0 ? areas.y : areas.x);
        closedWinding = fastSign(vDist.y * dUVDY.y);
    }

    // Calculate approximate area of the curve covering this pixel square.
    float curveArea = estimateArea(signedDistanceToCurve(uv.xy, dUVDX, dUVDY, inCurve));

    // Calculate alpha.
    vec2 alpha = vec2(openWinding, closedWinding) * 0.5 + convex * curveArea;
    alpha *= vec2(openRectArea, closedRectArea);

    // Finish up.
    oFragColor = vec4(alpha.x + alpha.y);
}

#endif
