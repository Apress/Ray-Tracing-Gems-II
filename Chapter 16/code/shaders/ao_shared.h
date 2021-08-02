// Common GLSL file shared across ray tracing shaders.
// Copyright 2021 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
#ifndef VK_RTG2_RAY_TRACE_AO_SHARED_H
#define VK_RTG2_RAY_TRACE_AO_SHARED_H

#include "../common.h"

// Ray payload for both primary rays and ambient occlusion.
// Note that although this payload is flexible, we only need the isVisible
// field for ambient occlusion.
// In a more complex sample, we could have two miss shaders, for instance,
// one of which would use a payload of type {worldPosition, worldNormal} for
// primary rays, and another which would use a payload of type {isVisible} for
// ambient occlusion. This would involve an SBT with one ray generation shader,
// two miss shaders, and one closest hit shader.
struct RayPayload
{
  float hitSky;         // 0: occluded, 1: visible
  vec3  worldPosition;  // Position of intersection in world-space
  vec3  worldNormal;    // Normal at intersection in world-space
};

#endif  // #ifndef VK_RTG2_RAY_TRACE_AO_SHARED_H