// Closest-hit shader for primary rays; returns the intersection's
// position and normal, converting from object space to world space.
// This primary ray step could also be implemented using rasterization,
// but is generally more flexible.
// Copyright 2021 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_GOOGLE_include_directive : require
#include "ao_shared.h"

// This will store two of the barycentric coordinates of the
// intersection when traversal calls this shader:
hitAttributeEXT vec2 attributes;

// These shaders can access the vertex and index buffers:
// The scalar layout qualifier here means to align types according to
// the alignment of their scalar components, instead of e.g. padding
// them to std140 rules.
// If we wanted to use multiple meshes, we could use an array of
// bindings here.
layout(binding = BINDING_VERTICES, set = 0, scalar) buffer Vertices
{
  vec3 vertices[];
};
layout(binding = BINDING_INDICES, set = 0, scalar) buffer Indices
{
  uint indices[];
};

// The payload:
layout(location = 0) rayPayloadInEXT RayPayload pld;

void main()
{
  pld.worldPosition = vec3(1.0, 0.0, 0.5);
  // Get the vertices of the triangle
  const vec3 v0 = vertices[indices[3 * gl_PrimitiveID + 0]];
  const vec3 v1 = vertices[indices[3 * gl_PrimitiveID + 1]];
  const vec3 v2 = vertices[indices[3 * gl_PrimitiveID + 2]];

  // Get the barycentric coordinates of the intersection
  const vec3 barycentrics = vec3(
    1.0 - attributes.x - attributes.y, attributes.xy);

  // Compute the coordinates of the intersection
  const vec3 objectPosition
    = v0 * barycentrics.x + v1 * barycentrics.y + v2 * barycentrics.z;
  // Transform from object space to world space:
  pld.worldPosition = gl_ObjectToWorldEXT * vec4(objectPosition, 1.0);

  // Compute the normal of the triangle in object space, using the
  // right-hand rule. This uses flat shading; we could also perform
  // smooth shading using a normal buffer.
  const vec3 objectNormal = cross(v1 - v0, v2 - v0);
  // Transform normals from object space to world space. These use the
  // transpose of the inverse matrix, because they're directions of
  // normals, not positions:
  pld.worldNormal = normalize((objectNormal * gl_WorldToObjectEXT).xyz);

  // Flip the normal so it points against the ray direction - this
  // makes triangles double-sided:
  pld.worldNormal = faceforward(pld.worldNormal,
                                gl_WorldRayDirectionEXT,
                                pld.worldNormal);

  pld.hitSky = 0.0f;
}