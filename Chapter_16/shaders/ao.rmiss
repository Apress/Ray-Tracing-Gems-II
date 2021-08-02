// Copyright 2021 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#include "ao_shared.h"

layout(location = 0) rayPayloadInEXT RayPayload pay;

void main()
{
  pay.hitSky = 1.0f;
}