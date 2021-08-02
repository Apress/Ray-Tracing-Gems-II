//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

#ifndef RAYTRACING_HLSL
#define RAYTRACING_HLSL

#define HLSL
#include "RaytracingHlslCompat.h"

// Subobjects definitions at library scope. 
GlobalRootSignature MyGlobalRootSignature =
{
    "DescriptorTable( UAV( u0 ) ),"                        // Output texture
    "SRV( t0 ),"                                           // Acceleration structure
    "CBV( b0 ),"                                           // Scene constants
    "DescriptorTable( SRV( t1, numDescriptors = unbounded ) ),"    // Static index and vertex buffers, and a texture.
    "StaticSampler(s0,"
        "filter = FILTER_MIN_MAG_MIP_LINEAR,"
        "addressU = TEXTURE_ADDRESS_WRAP,"
        "addressV = TEXTURE_ADDRESS_WRAP,"
        "addressW = TEXTURE_ADDRESS_WRAP,"
        "mipLODBias = 0.f,"
        "visibility = SHADER_VISIBILITY_ALL)"              // Static sampler
};

LocalRootSignature MyLocalRootSignature = 
{
    "RootConstants( num32BitConstants = 4, b1 )"           // Model constants        
};

TriangleHitGroup MyHitGroup =
{
    "",                     // AnyHit
    "MyClosestHitShader",   // ClosestHit
};

SubobjectToExportsAssociation  MyLocalRootSignatureAssociation =
{
    "MyLocalRootSignature",  // subobject name
    "MyHitGroup"             // export association 
};

RaytracingShaderConfig  MyShaderConfig =
{
    16, // max payload size
    8   // max attribute size
};

RaytracingPipelineConfig MyPipelineConfig =
{
    1 // max trace recursion depth
};


RWTexture2D<float4> RenderTarget : register(u0);
RaytracingAccelerationStructure Scene : register(t0);
ByteAddressBuffer Indices : register(t1);
StructuredBuffer<Vertex> Vertices : register(t2);
Texture2D<float4> Albedo[NUM_TEXTURES] : register(t3);

ConstantBuffer<SceneConstantBuffer> g_sceneCB : register(b0);
ConstantBuffer<ModelConstantBuffer> g_modelCB : register(b1);
sampler g_trilinear : register(s0);

// Load three 16 bit indices from a byte addressed buffer.
uint3 Load3x16BitIndices(uint offsetBytes)
{
    uint3 indices;

    // ByteAdressBuffer loads must be aligned at a 4 byte boundary.
    // Since we need to read three 16 bit indices: { 0, 1, 2 } 
    // aligned at a 4 byte boundary as: { 0 1 } { 2 0 } { 1 2 } { 0 1 } ...
    // we will load 8 bytes (~ 4 indices { a b | c d }) to handle two possible index triplet layouts,
    // based on first index's offsetBytes being aligned at the 4 byte boundary or not:
    //  Aligned:     { 0 1 | 2 - }
    //  Not aligned: { - 0 | 1 2 }
    const uint dwordAlignedOffset = offsetBytes & ~3;    
    const uint2 four16BitIndices = Indices.Load2(dwordAlignedOffset);
 
    // Aligned: { 0 1 | 2 - } => retrieve first three 16bit indices
    if (dwordAlignedOffset == offsetBytes)
    {
        indices.x = four16BitIndices.x & 0xffff;
        indices.y = (four16BitIndices.x >> 16) & 0xffff;
        indices.z = four16BitIndices.y & 0xffff;
    }
    else // Not aligned: { - 0 | 1 2 } => retrieve last three 16bit indices
    {
        indices.x = (four16BitIndices.x >> 16) & 0xffff;
        indices.y = four16BitIndices.y & 0xffff;
        indices.z = (four16BitIndices.y >> 16) & 0xffff;
    }

    return indices;
}

/////////// Begin ray cone functions ///////////
uint2 TexDims(Texture2D<float4> tex) { uint2 vSize; tex.GetDimensions(vSize.x, vSize.y); return vSize; }
uint2 TexDims(Texture2D<float3> tex) { uint2 vSize; tex.GetDimensions(vSize.x, vSize.y); return vSize; }
uint2 TexDims(Texture2D<float > tex) { uint2 vSize; tex.GetDimensions(vSize.x, vSize.y); return vSize; }

float2 UVAreaFromRayCone(float3 vRayDir,float3 vWorldNormal,float vRayConeWidth,float2 aUV[3],float3 aPos[3],float3x3 matWorld)
{
	float2 vUV10 = aUV[1]-aUV[0];
	float2 vUV20 = aUV[2]-aUV[0];
	float fTriUVArea = abs(vUV10.x*vUV20.y - vUV20.x*vUV10.y);

	// We need the area of the triangle, which is length(triangleNormal) in worldspace, and I
	// could not figure out a way with fewer than two 3x3 mtx multiplies for ray cones.
	float3 vEdge10 = mul(aPos[1]-aPos[0],matWorld);
	float3 vEdge20 = mul(aPos[2]-aPos[0],matWorld);

	float3 vFaceNrm = cross(vEdge10, vEdge20); // in world space, by design
	float fTriLODOffset = 0.5f * log2(fTriUVArea/length(vFaceNrm)); // Triangle-wide LOD offset value
	float fDistTerm = vRayConeWidth * vRayConeWidth;
	float fNormalTerm = dot(vRayDir, vWorldNormal);

	return float2(fTriLODOffset, fDistTerm/(fNormalTerm*fNormalTerm));
}

float UVAreaToTexLOD(uint2 vTexSize,float2 vUVAreaInfo)
{
	return vUVAreaInfo.x + 0.5f*log2(vTexSize.x * vTexSize.y * vUVAreaInfo.y);
}

float4 UVDerivsFromRayCone(float3 vRayDir,float3 vWorldNormal,float vRayConeWidth,float2 aUV[3],float3 aPos[3],float3x3 matWorld)
{
	float2 vUV10 = aUV[1]-aUV[0];
	float2 vUV20 = aUV[2]-aUV[0];
	float fQuadUVArea = abs(vUV10.x*vUV20.y - vUV20.x*vUV10.y);

	// Since the ray cone's width is in world-space, we need to compute the quad
	// area in world-space as well to enable proper ratio calculation
	float3 vEdge10 = mul(aPos[1]-aPos[0],matWorld);
	float3 vEdge20 = mul(aPos[2]-aPos[0],matWorld);
	float3 vFaceNrm = cross(vEdge10, vEdge20);
	float fQuadArea = length(vFaceNrm);

	float fDistTerm = abs(vRayConeWidth);
	float fNormalTerm = abs(dot(vRayDir,vWorldNormal));
	float fProjectedConeWidth = vRayConeWidth/fNormalTerm;
	float fVisibleAreaRatio = (fProjectedConeWidth*fProjectedConeWidth) / fQuadArea;

	float fVisibleUVArea = fQuadUVArea*fVisibleAreaRatio;
	float fULength = sqrt(fVisibleUVArea);
	return float4(fULength,0,0,fULength);
}
/////////// End ray cone functions ///////////


typedef BuiltInTriangleIntersectionAttributes MyAttributes;
struct RayPayload
{
    float4 color;
};

// Retrieve attribute at a hit position interpolated from vertex attributes using the hit's barycentrics.
float2 HitAttribute(float2 v0,float2 v1,float2 v2, BuiltInTriangleIntersectionAttributes attr)
{
    return v0 + attr.barycentrics.x * (v1-v0) + attr.barycentrics.y * (v2-v0);
}

float3 HitAttribute(float3 v0,float3 v1,float3 v2, BuiltInTriangleIntersectionAttributes attr)
{
    return v0 + attr.barycentrics.x * (v1-v0) + attr.barycentrics.y * (v2-v0);
}

// Generate a ray in world space for a camera pixel corresponding to an index from the dispatched 2D grid.
inline RayDesc GenerateCameraRay(uint2 index, float tmin,float tmax)
{
    float2 xy = index + 0.5f; // center in the middle of the pixel.
    float2 screenPos = xy / DispatchRaysDimensions().xy * 2.0 - 1.0;

    // Invert Y for DirectX-style coordinates.
    screenPos.y = -screenPos.y;

    // Unproject the pixel coordinate into a ray.
    float4 world = mul(float4(screenPos, 0, 1), g_sceneCB.projectionToWorld);

    world.xyz /= world.w;

    RayDesc ray;
    ray.Origin = g_sceneCB.cameraPosition.xyz;
    ray.Direction = normalize(world.xyz - ray.Origin);
    ray.TMin = tmin;
    ray.TMax = tmax;
    return ray;
}

[shader("raygeneration")]
void MyRaygenShader()
{
    RayDesc ray = GenerateCameraRay(DispatchRaysIndex().xy, 0, 10000.0);
    RayPayload payload = (RayPayload)0;
    TraceRay(Scene, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, ray, payload);

    // Write the raytraced color to the output texture.
    RenderTarget[DispatchRaysIndex().xy] = payload.color;
}

[shader("closesthit")]
void MyClosestHitShader(inout RayPayload payload, in MyAttributes attr)
{
    const uint3 indices = Load3x16BitIndices(PrimitiveIndex()*3*2); // 3 2-byte indices per triangle
    const Vertex verts[3] = { Vertices[indices[0]], Vertices[indices[1]], Vertices[indices[2]] };

    const float3 hitPosition = WorldRayOrigin() + RayTCurrent() * WorldRayDirection();
    const float2 uv = HitAttribute(verts[0].uv, verts[1].uv, verts[2].uv, attr);
    const float3 nrm = HitAttribute(verts[0].normal, verts[1].normal, verts[2].normal, attr); // in world space because model is in world space

    const float3 aPos[3] = { verts[0].position, verts[1].position, verts[2].position };
	const float2 aUVs[3] = { verts[0].uv, verts[1].uv, verts[2].uv };

	const float2 rayConeAtOrigin = float2(0,g_sceneCB.eyeToPixelConeSpreadAngle);
	const float2 rayConeAtHit = float2(
		// New cone width should increase by 2*RayLength*tan(SpreadAngle/2), but RayLength*SpreadAngle is a close approximation
		rayConeAtOrigin.x+rayConeAtOrigin.y*length(hitPosition-g_sceneCB.cameraPosition.xyz),
		rayConeAtOrigin.y+g_sceneCB.eyeToPixelConeSpreadAngle);

    const matrix matWorld = matrix(float4(1,0,0,0), float4(0,1,0,0), float4(0,0,1,0), float4(0,0,0,1));

    float4 color = float4(0,0,0,0);

    [branch]
    if (g_sceneCB.mipAlgorithm == 1) // Raytracing_Mip0
    {
        [loop]
        for (int i=0;i<NUM_TEXTURES;i++)
            color += Albedo[i].SampleLevel(g_trilinear,uv,0);
    }
    else if (g_sceneCB.mipAlgorithm == 2) // Raytracing_RayConesLevel
    {
	    float2 uvAreaFromCone = UVAreaFromRayCone(WorldRayDirection(),nrm,rayConeAtHit.x,aUVs,aPos,(float3x3)matWorld);

        [loop]
        for (int i=0;i<NUM_TEXTURES;i++)
        {
            uint2 vTexSize;
		    Albedo[i].GetDimensions(vTexSize.x,vTexSize.y);
            float texLOD = UVAreaToTexLOD(vTexSize,uvAreaFromCone);
            color += Albedo[i].SampleLevel(g_trilinear,uv,texLOD);
        }
    }
    else if (g_sceneCB.mipAlgorithm == 3) // Raytracing_RayConesGrad
    {
	    float4 uvDerivs = UVDerivsFromRayCone(WorldRayDirection(),nrm,rayConeAtHit.x,aUVs,aPos,(float3x3)matWorld);

        [loop]
        for (int i=0;i<NUM_TEXTURES;i++)
		    color += Albedo[i].SampleGrad(g_trilinear,uv,uvDerivs.xy,uvDerivs.zw);
    }
    payload.color = color/NUM_TEXTURES;
}

[shader("miss")]
void MyMissShader(inout RayPayload payload)
{
    payload.color = float4(0.0f, 0.2f, 0.4f, 1.0f);
}

#endif // RAYTRACING_HLSL