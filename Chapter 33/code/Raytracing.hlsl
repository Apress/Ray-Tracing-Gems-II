#ifndef RAYTRACING_HLSL
#define RAYTRACING_HLSL

#define HLSL
#include "engine/ShaderCompat.h"
#include "engine/RaytracingShaderHelper.hlsli"

static float gStep = 4.;
static float gMaxLenght = 600.;

//***************************************************************************
//*****------ Shader resources bound via root signatures -------*************
//***************************************************************************

// Scene wide resources.
//  g_* - bound via a global root signature.
//  l_* - bound via a local root signature.
RaytracingAccelerationStructure g_scene : register(t0, space0);
RWTexture2D<float4> g_renderTarget : register(u0);
ConstantBuffer<SceneConstantBuffer> g_sceneCB : register(b0);

// Procedural geometry resources
StructuredBuffer<InstanceBuffer> g_instanceBuffer : register(t3, space0);
ConstantBuffer<PrimitiveConstantBuffer> l_materialCB : register(b1);
ConstantBuffer<PrimitiveInstanceConstantBuffer> l_aabbCB: register(b2);

#include "Helpers.hlsli"

//***************************************************************************
//********************------ Ray gen shader.. -------************************
//***************************************************************************

[shader("raygeneration")]
void Raygen()
{
    // Generate a ray for a camera pixel corresponding to an index from the dispatched 2D grid.
    Ray ray = GenerateCameraRay(DispatchRaysIndex().xy, g_sceneCB.cameraPosition.xyz, g_sceneCB.projectionToWorld);
 
    // Cast a ray into the scene and retrieve a shaded color.
    uint currentRecursionDepth = 0;
    float4 color = TraceRadianceRay(ray, currentRecursionDepth);

    // Write the raytraced color to the output texture.
    g_renderTarget[DispatchRaysIndex().xy] = color;
}

void traceRaySegment(inout RayPayload payload)
{
    RayDesc rayDesc;
    rayDesc.Direction = WorldRayDirection();
    rayDesc.Origin = WorldRayOrigin() + (gStep - 0.000001) * rayDesc.Direction;
    rayDesc.TMin = 0.f;
    rayDesc.TMax = gStep;

   //update the ray
   //evalGraphRay(rayDesc.Origin, rayDesc.Direction, gStep-0.08); //Riemannian metric induced by a graph of a function
    
    TraceRay(g_scene,
        RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
        TraceRayParameters::InstanceMask,
        TraceRayParameters::HitGroup::Offset[RayType::Radiance],
        TraceRayParameters::HitGroup::GeometryStride,
        TraceRayParameters::MissShader::Offset[RayType::Radiance],
        rayDesc, payload);
}

//***************************************************************************
//**********************------ Miss shaders -------**************************
//***************************************************************************

[shader("miss")]
void Miss(inout RayPayload rayPayload)
{
    float4 backgroundColor = float4(BackgroundColor);
    rayPayload.color = backgroundColor;
    rayPayload.hit = false;
}

[shader("miss")]
void Miss_Shadow(inout ShadowRayPayload rayPayload)
{
   rayPayload.dist += gStep;
   rayPayload.hit = false;
}

#include "Pacman.hlsli"

#include "JuliaSets.hlsli"

#include "Mandelbulb.hlsli"

#include "Triangles.hlsli"

#endif // RAYTRACING_HLSL