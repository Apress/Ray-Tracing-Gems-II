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

#define HLSL
#include "RaytracingStructs.hlsli"

// Library sub-objects
GlobalRootSignature MyGlobalRootSignature =
{
    "CBV(b0),"
    "CBV(b1),"
    "DescriptorTable( SRV(t0, numDescriptors=4) ),"
    "SRV(t4),"
    "SRV(t5),"
    "DescriptorTable( UAV(u0, numDescriptors=1) )," // No inline UAVs for textures, so need a descriptor table
    "DescriptorTable( SRV(t6, numDescriptors=unbounded) ),"
    "StaticSampler(s0,"
        "filter = FILTER_ANISOTROPIC,"
        "maxAnisotropy = 16,"
        "addressU = TEXTURE_ADDRESS_WRAP,"
        "addressV = TEXTURE_ADDRESS_WRAP,"
        "visibility = SHADER_VISIBILITY_ALL)"
};

LocalRootSignature MyLocalRootSignature = 
{
    "DescriptorTable( SRV(t0, numDescriptors=2, space=1) ),"
    "RootConstants( num32BitConstants=1, b0,space=1 )" // Must fit Material cbuffer below
};

TriangleHitGroup SingleDecalCHS_HitGroup = { "","DecalHit", }; // AnyHit, ClosestHit
TriangleHitGroup SingleDecalAnyHit_HitGroup = { "DecalAnyHit","", }; // AnyHit, ClosestHit
ProceduralPrimitiveHitGroup SingleDecalAABBIntersection_HitGroup = { "", "DecalAABBHit", "DecalAABBIntersection" }; // AnyHit, ClosestHit, Intersection
TriangleHitGroup MultiDecalCHS_HitGroup = { "","DecalMultiHit", }; // AnyHit, ClosestHit
ProceduralPrimitiveHitGroup MultiDecalAABBIntersection_HitGroup = { "DecalMultiAnyHit", "", "DecalAABBIntersection" }; // AnyHit, ClosestHit, Intersection
TriangleHitGroup DecalDebug_HitGroup = { "","DecalDebugHit", }; // AnyHit, ClosestHit
TriangleHitGroup PrimaryHitGroup = { "","PrimaryHit", }; // AnyHit, ClosestHit

RaytracingShaderConfig  MyShaderConfig = { 12,8 }; // max payload size, max attribute size
RaytracingPipelineConfig MyPipelineConfig = { 2 }; // max trace recursion depth (primary+shadow/decal)


// Global resource bindings
cbuffer LightingConstants : register(b0)
{
    float3 SunDirection, SunColor, AmbientColor;
}

cbuffer b1 : register(b1) { DynamicCB g_dynamic; };

StructuredBuffer<RayTraceMeshInfo> g_meshInfo : register(t0);
ByteAddressBuffer g_indices : register(t1);
ByteAddressBuffer g_attributes : register(t2);
StructuredBuffer<RayTraceDecalInfo> g_decalInfo : register(t3);
RaytracingAccelerationStructure g_accelScene : register(t4);
RaytracingAccelerationStructure g_accelDecals : register(t5);
Texture2D<float4> g_decalTextures[] : register(t6);
RWTexture2D<float4> g_screenOutput : register(u0);
SamplerState      g_s0 : register(s0);

// Local resource bindings (per mesh resources)
Texture2D<float4> g_localTexture : register(t0,space1);
Texture2D<float4> g_localNormal : register(t1,space1);
cbuffer Material : register(b0,space1) { uint MaterialID; }

// Constants
static const float FLT_MAX = asfloat(0x7F7FFFFF);
static const uint kDecalMissShaderIndex = 0;
static const uint kPrimaryMissShaderIndex = 1;
static const uint kModelInstanceMask = 0x1;
static const uint kDecalInstanceMask = 0x2;



// Functions
void GenerateCameraRay(uint2 index, out float3 origin, out float3 direction)
{
    float2 xy = index + 0.5; // center in the middle of the pixel
    float2 screenPos = xy / g_dynamic.resolution * 2.0 - 1.0;

    // Invert Y for DirectX-style coordinates
    screenPos.y = -screenPos.y;

    // Unproject into a ray
    float4 unprojected = mul(g_dynamic.cameraToWorld, float4(screenPos, 0, 1));
    float3 world = unprojected.xyz / unprojected.w;
    origin = g_dynamic.worldCameraPosition;
    direction = normalize(world - origin);
}

uint3 Load3x16BitIndices(uint offsetBytes)
{
    const uint dwordAlignedOffset = offsetBytes & ~3;
    const uint2 four16BitIndices = g_indices.Load2(dwordAlignedOffset);
    uint3 indices;
    if (dwordAlignedOffset == offsetBytes)
    {
        indices.x = four16BitIndices.x & 0xffff;
        indices.y = (four16BitIndices.x >> 16) & 0xffff;
        indices.z = four16BitIndices.y & 0xffff;
    }
    else
    {
        indices.x = (four16BitIndices.x >> 16) & 0xffff;
        indices.y = four16BitIndices.y & 0xffff;
        indices.z = (four16BitIndices.y >> 16) & 0xffff;
    }
    return indices;
}

float3 RayPlaneIntersection(float3 planeOrigin, float3 planeNormal, float3 rayOrigin, float3 rayDirection)
{
    float t = dot(-planeNormal, rayOrigin - planeOrigin) / dot(planeNormal, rayDirection);
    return rayOrigin + rayDirection * t;
}

// REF: https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
// From "Real-Time Collision Detection" by Christer Ericson
float3 BarycentricCoordinates(float3 pt, float3 v0, float3 v1, float3 v2)
{
    float3 e0 = v1 - v0;
    float3 e1 = v2 - v0;
    float3 e2 = pt - v0;
    float d00 = dot(e0, e0);
    float d01 = dot(e0, e1);
    float d11 = dot(e1, e1);
    float d20 = dot(e2, e0);
    float d21 = dot(e2, e1);
    float denom = 1.0 / (d00 * d11 - d01 * d01);
    float v = (d11 * d20 - d01 * d21) * denom;
    float w = (d00 * d21 - d01 * d20) * denom;
    float u = 1.0 - v - w;
    return float3(u, v, w);
}

void InterpolateAttributes(uint triangleID,RayTraceMeshInfo info,float3 worldPosition,float2 interpBary,
                           out float3 vsNormal,out float3 vsTangent,out float3 vsBitangent,
                           out float2 uv,out float2 ddxUV,out float2 ddyUV)
{
    const uint3 ii = Load3x16BitIndices(info.m_indexOffsetBytes + triangleID * 3 * 2);
    const float2 uv0 = asfloat(g_attributes.Load2(info.m_uvAttributeOffsetBytes + ii.x * info.m_attributeStrideBytes));
    const float2 uv1 = asfloat(g_attributes.Load2(info.m_uvAttributeOffsetBytes + ii.y * info.m_attributeStrideBytes));
    const float2 uv2 = asfloat(g_attributes.Load2(info.m_uvAttributeOffsetBytes + ii.z * info.m_attributeStrideBytes));

    float3 bary = float3(1.0 - interpBary.x - interpBary.y, interpBary.x, interpBary.y);
    uv = bary.x * uv0 + bary.y * uv1 + bary.z * uv2;

    const float3 normal0 = asfloat(g_attributes.Load3(info.m_normalAttributeOffsetBytes + ii.x * info.m_attributeStrideBytes));
    const float3 normal1 = asfloat(g_attributes.Load3(info.m_normalAttributeOffsetBytes + ii.y * info.m_attributeStrideBytes));
    const float3 normal2 = asfloat(g_attributes.Load3(info.m_normalAttributeOffsetBytes + ii.z * info.m_attributeStrideBytes));
    vsNormal = normalize(normal0 * bary.x + normal1 * bary.y + normal2 * bary.z);
    
    const float3 tangent0 = asfloat(g_attributes.Load3(info.m_tangentAttributeOffsetBytes + ii.x * info.m_attributeStrideBytes));
    const float3 tangent1 = asfloat(g_attributes.Load3(info.m_tangentAttributeOffsetBytes + ii.y * info.m_attributeStrideBytes));
    const float3 tangent2 = asfloat(g_attributes.Load3(info.m_tangentAttributeOffsetBytes + ii.z * info.m_attributeStrideBytes));
    vsTangent = normalize(tangent0 * bary.x + tangent1 * bary.y + tangent2 * bary.z);

    // Reintroduced the bitangent because we aren't storing the handedness of the tangent frame anywhere.  Assuming the space
    // is right-handed causes normal maps to invert for some surfaces.  The Sponza mesh has all three axes of the tangent frame.
    //float3 vsBitangent = normalize(cross(vsNormal, vsTangent)) * (isRightHanded ? 1.0 : -1.0);
    const float3 bitangent0 = asfloat(g_attributes.Load3(info.m_bitangentAttributeOffsetBytes + ii.x * info.m_attributeStrideBytes));
    const float3 bitangent1 = asfloat(g_attributes.Load3(info.m_bitangentAttributeOffsetBytes + ii.y * info.m_attributeStrideBytes));
    const float3 bitangent2 = asfloat(g_attributes.Load3(info.m_bitangentAttributeOffsetBytes + ii.z * info.m_attributeStrideBytes));
    vsBitangent = normalize(bitangent0 * bary.x + bitangent1 * bary.y + bitangent2 * bary.z);

    // TODO: Should just store uv partial derivatives in here rather than loading position and caculating it per pixel
    const float3 p0 = asfloat(g_attributes.Load3(info.m_positionAttributeOffsetBytes + ii.x * info.m_attributeStrideBytes));
    const float3 p1 = asfloat(g_attributes.Load3(info.m_positionAttributeOffsetBytes + ii.y * info.m_attributeStrideBytes));
    const float3 p2 = asfloat(g_attributes.Load3(info.m_positionAttributeOffsetBytes + ii.z * info.m_attributeStrideBytes));

    //---------------------------------------------------------------------------------------------
    // Compute partial derivatives of UV coordinates:
    //
    //  1) Construct a plane from the hit triangle
    //  2) Intersect two helper rays with the plane:  one to the right and one down
    //  3) Compute barycentric coordinates of the two hit points
    //  4) Reconstruct the UV coordinates at the hit points
    //  5) Take the difference in UV coordinates as the partial derivatives X and Y

    // Normal for plane
    float3 triangleNormal = normalize(cross(p2 - p0, p1 - p0));

    // Helper rays
    uint2 threadID = DispatchRaysIndex().xy;
    float3 ddxOrigin, ddxDir, ddyOrigin, ddyDir;
    GenerateCameraRay(uint2(threadID.x + 1, threadID.y), ddxOrigin, ddxDir);
    GenerateCameraRay(uint2(threadID.x, threadID.y + 1), ddyOrigin, ddyDir);

    // Intersect helper rays
    float3 xOffsetPoint = RayPlaneIntersection(worldPosition, triangleNormal, ddxOrigin, ddxDir);
    float3 yOffsetPoint = RayPlaneIntersection(worldPosition, triangleNormal, ddyOrigin, ddyDir);

    // Compute barycentrics 
    float3 baryX = BarycentricCoordinates(xOffsetPoint, p0, p1, p2);
    float3 baryY = BarycentricCoordinates(yOffsetPoint, p0, p1, p2);

    // Compute UVs and take the difference
    float3x2 uvMat = float3x2(uv0, uv1, uv2);
    ddxUV = mul(baryX, uvMat) - uv;
    ddyUV = mul(baryY, uvMat) - uv;
}

void AntiAliasSpecular(inout float3 texNormal, inout float gloss)
{
    float normalLenSq = dot(texNormal, texNormal);
    float invNormalLen = rsqrt(normalLenSq);
    texNormal *= invNormalLen;
    gloss = lerp(1, gloss, rcp(invNormalLen));
}

// Apply fresnel to modulate the specular albedo
void FSchlick(inout float3 specular, inout float3 diffuse, float3 lightDir, float3 halfVec)
{
    float fresnel = pow(1.0 - saturate(dot(lightDir, halfVec)), 5.0);
    specular = lerp(specular, 1, fresnel);
    diffuse = lerp(diffuse, 0, fresnel);
}

float3 ApplyLightCommon(
    float3    diffuseColor,    // Diffuse albedo
    float3    specularColor,    // Specular albedo
    float    specularMask,    // Where is it shiny or dingy?
    float    gloss,            // Specular power
    float3    normal,            // World-space normal
    float3    viewDir,        // World-space vector from eye to point
    float3    lightDir,        // World-space vector from point to light
    float3    lightColor        // Radiance of directional light
)
{
    float3 halfVec = normalize(lightDir - viewDir);
    float nDotH = saturate(dot(halfVec, normal));

    FSchlick(specularColor, diffuseColor, lightDir, halfVec);

    float specularFactor = specularMask * pow(nDotH, gloss) * (gloss + 2) / 8;
    float nDotL = saturate(dot(normal, lightDir));
    return nDotL * lightColor * (diffuseColor + specularFactor * specularColor);
}


struct DecalRayPayload
{
    float rayHitT;
    int decal;
};

// Make sure the payload size in MyShaderConfig above handles the count here
// Due to a bug in compiling DecalMultiAnyHit, this number cannot be changed without
// manually writing unrolled code. SortDecals is also hand-coded for 3
#define MAX_MULTI_AABB_DECALS 3

struct MultiDecalRayPayload
{
    int decal[MAX_MULTI_AABB_DECALS];
};

struct DecalAABBAttrib { };

float3 WorldPosToDecalUVW(float3x4 decalWorldToUnity,float3 worldPosition)
{
    return mul(decalWorldToUnity,float4(worldPosition,1)).xyz+0.5f;
}

bool IsInvalidDecalSampleLocation(float3 decalUVW)
{
    return any(decalUVW < float3(0,0,0)) || any(decalUVW > float3(1,1,1));
}

void SampleAndBlendDecal(inout float3 diffuse,inout float3 normal,float3 worldPos,int decalIndex)
{
    RayTraceDecalInfo decal = g_decalInfo[decalIndex];
    float2 decalUV = WorldPosToDecalUVW((float3x4)decal.worldToUnity,worldPos).xy;

    // The sampling code could either be embedded like here, or do a full callable shader invocation for "shader-graph" style materials
    if (decal.albedo >= 0)
    {
        float4 decalVal = g_decalTextures[decal.albedo].SampleLevel(g_s0,decalUV,0);
        diffuse = lerp(diffuse,decalVal.xyz,decalVal.w);
    }
    if (decal.normal >= 0)
    {
        float4 decalVal = g_decalTextures[decal.normal].SampleLevel(g_s0,decalUV,0);
        normal = lerp(normal,decalVal.xyz,decalVal.w);
    }
}

void SortDecals(inout int decal[MAX_MULTI_AABB_DECALS])
{
    int tmp;
    if ((uint)decal[2] < (uint)decal[0]) { tmp=decal[0]; decal[0]=decal[2]; decal[2]=tmp; }
    if ((uint)decal[1] < (uint)decal[0]) { tmp=decal[0]; decal[0]=decal[1]; decal[1]=tmp; }
    if ((uint)decal[2] < (uint)decal[1]) { tmp=decal[1]; decal[1]=decal[2]; decal[2]=tmp; }
}

void ApplyDecals(inout float3 diffuse,inout float3 normal,float3 worldPos)
{
    if ((g_dynamic.decalTraceMode == DTM_NoDecals) || (g_dynamic.decalTraceMode == DTM_DebugShowBoxes))
        return; // Do nothing

    RayDesc rayDesc = { worldPos, 0.0f, worldPos+normalize(float3(3,4,5))*g_dynamic.decalMaxDiagonal, 1.0f };

    if (g_dynamic.decalTraceMode == DTM_SingleBLASClosestHitLoop)
    {
        int decalIndex = -1;
        for (;;)
        {
            DecalRayPayload payload = { FLT_MAX, -1 };
            TraceRay(g_accelDecals, RAY_FLAG_CULL_FRONT_FACING_TRIANGLES, kDecalInstanceMask, 0,1,kDecalMissShaderIndex, rayDesc, payload);
            if (payload.decal == -1)
                break; // No hits
            if (payload.decal > -1)
            {
                decalIndex = payload.decal;
                break;
            }
            rayDesc.TMin = payload.rayHitT+0.0001f; // Continue searching
        }
        if (decalIndex > -1)
            SampleAndBlendDecal(diffuse,normal,worldPos,decalIndex);
    }
    else if (g_dynamic.decalTraceMode == DTM_SingleBLASAnyHit)
    {
        DecalRayPayload payload = { FLT_MAX, -1 };
        TraceRay(g_accelDecals, RAY_FLAG_CULL_FRONT_FACING_TRIANGLES|RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, kDecalInstanceMask, 0,1,kDecalMissShaderIndex, rayDesc, payload);
        if (payload.decal > -1)
            SampleAndBlendDecal(diffuse,normal,worldPos,payload.decal);
    }
    else if (g_dynamic.decalTraceMode == DTM_SingleAABBIntersection)
    {
        rayDesc.TMax = rayDesc.TMin+0.00001f; // Need a very tiny ray only
        DecalRayPayload payload = { FLT_MAX, -1 };
        TraceRay(g_accelDecals, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, kDecalInstanceMask, 0,1,kDecalMissShaderIndex, rayDesc, payload);
        if (payload.decal > -1)
            SampleAndBlendDecal(diffuse,normal,worldPos,payload.decal);
    }
    else if (g_dynamic.decalTraceMode == DTM_MultipleBLASClosestHitLoop)
    {
        // Flip the ray, start from its end and towards its origin
        rayDesc.Origin += rayDesc.Direction * rayDesc.TMax;
        rayDesc.Direction = -rayDesc.Direction;
        for (;;)
        {
            DecalRayPayload payload = { FLT_MAX, -1 };
            TraceRay(g_accelDecals, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, kDecalInstanceMask, 0,1,kDecalMissShaderIndex, rayDesc, payload);
            if (payload.decal == -1)
                break; // No hits
            if (payload.decal > -1)
                SampleAndBlendDecal(diffuse,normal,worldPos,payload.decal);
            rayDesc.TMin = payload.rayHitT+0.0001f; // Continue searching
        }
    }
    else if (g_dynamic.decalTraceMode == DTM_MultipleAABBIntersection)
    {
        rayDesc.TMax = rayDesc.TMin+0.00001f; // Need a very tiny ray only

        // Init the payload (all -1)
        MultiDecalRayPayload payload;
        [unroll]
        for (int i=0;i<MAX_MULTI_AABB_DECALS;i++)
            payload.decal[i] = -1;

        TraceRay(g_accelDecals, RAY_FLAG_SKIP_CLOSEST_HIT_SHADER|RAY_FLAG_FORCE_NON_OPAQUE, kDecalInstanceMask, 0,1,kDecalMissShaderIndex, rayDesc, payload);
        SortDecals(payload.decal);

        [unroll]
        for (i=0;i<MAX_MULTI_AABB_DECALS;i++)
        {
            if (payload.decal[i] > -1)
                SampleAndBlendDecal(diffuse,normal,worldPos,payload.decal[i]);
            else break;
        }
    }
}

// Shaders
struct RayPayload
{
    bool SkipShading;
    float RayHitT;
};

[shader("raygeneration")]
void PrimaryRayGen()
{
    float3 origin, direction;
    GenerateCameraRay(DispatchRaysIndex().xy, origin, direction);

    RayDesc rayDesc = { origin, 0.0f, direction, FLT_MAX };
    RayPayload payload;
    payload.SkipShading = false;
    payload.RayHitT = FLT_MAX;
    TraceRay(g_accelScene, RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
        kModelInstanceMask | (g_dynamic.decalTraceMode == DTM_DebugShowBoxes ? kDecalInstanceMask : 0),
        0,1,kPrimaryMissShaderIndex, rayDesc, payload);
}

[shader("closesthit")]
void DecalHit(inout DecalRayPayload payload, in BuiltInTriangleIntersectionAttributes)
{
    payload.rayHitT = RayTCurrent();
    payload.decal = IsInvalidDecalSampleLocation(WorldPosToDecalUVW(WorldToObject3x4(),WorldRayOrigin())) ? -2 : InstanceID();
}

[shader("anyhit")]
void DecalAnyHit(inout DecalRayPayload payload, in BuiltInTriangleIntersectionAttributes)
{
    if (IsInvalidDecalSampleLocation(WorldPosToDecalUVW(WorldToObject3x4(),WorldRayOrigin())))
        IgnoreHit();
    else
    {
        payload.decal = InstanceID();
        AcceptHitAndEndSearch();
    }
}

[shader("closesthit")]
void DecalAABBHit(inout DecalRayPayload payload, in DecalAABBAttrib)
{
    payload.decal = InstanceID();
}

[shader("closesthit")]
void DecalMultiHit(inout DecalRayPayload payload, in BuiltInTriangleIntersectionAttributes)
{
    payload.rayHitT = RayTCurrent();
    payload.decal = IsInvalidDecalSampleLocation(WorldPosToDecalUVW(WorldToObject3x4(),WorldRayOrigin()+WorldRayDirection())) ? -2 : InstanceID();
}

[shader("anyhit")]
void DecalMultiAnyHit(inout MultiDecalRayPayload payload, in DecalAABBAttrib)
{
    // BUG: This is not working. Manually unrolling it below works.
    //[unroll]
    //for (int i=0;i<MAX_MULTI_AABB_DECALS;i++)
    //{
    //    if (payload.decal[i] == -1)
    //    {
    //        payload.decal[i] = InstanceID();
    //        if (i == MAX_MULTI_AABB_DECALS-1)
    //            AcceptHitAndEndSearch(); // Max decals accumulated
    //        else IgnoreHit(); // Allow more
    //        return;
    //    }
    //}

    // Find slot to write to
    if (payload.decal[0] == -1)
    {
        payload.decal[0] = InstanceID();
        IgnoreHit(); // Allow more
    }
    else if (payload.decal[1] == -1)
    {
        payload.decal[1] = InstanceID();
        IgnoreHit(); // Allow more
    }
    else if (payload.decal[2] == -1)
    {
        payload.decal[2] = InstanceID();
        AcceptHitAndEndSearch(); // Max decals accumulated
    }
}

[shader("intersection")]
void DecalAABBIntersection()
{
    DecalAABBAttrib attr;
    ReportHit(RayTCurrent(), 0, attr);
}

[shader("closesthit")]
void DecalDebugHit(inout RayPayload, in BuiltInTriangleIntersectionAttributes)
{
    RayTraceDecalInfo info = g_decalInfo[InstanceID()];
    float3 decalPosition = ObjectRayOrigin() + ObjectRayDirection() * RayTCurrent();
    g_screenOutput[DispatchRaysIndex().xy] = float4(decalPosition+0.5f,1);
}

[shader("miss")]
void DecalMiss(inout DecalRayPayload) {}

[shader("closesthit")]
void PrimaryHit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    payload.RayHitT = RayTCurrent();
    if (payload.SkipShading)
        return;

    RayTraceMeshInfo info = g_meshInfo[MaterialID];
    float3 worldPosition = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();

    // Vertex fetch
    float2 uv,ddxUV,ddyUV;
    float3 vsNormal,vsTangent,vsBitangent;
    InterpolateAttributes(PrimitiveIndex(),info,worldPosition,attr.barycentrics,
        vsNormal,vsTangent,vsBitangent,
        uv,ddxUV,ddyUV);

    // Material fetch
    float3 diffuseColor = g_localTexture.SampleGrad(g_s0, uv, ddxUV, ddyUV).rgb;
    float gloss = 128.0;
    float3 normal = g_localNormal.SampleGrad(g_s0, uv, ddxUV, ddyUV).rgb * 2.0 - 1.0;

    ApplyDecals(diffuseColor,normal,worldPosition);

    {
        AntiAliasSpecular(normal, gloss);
        float3x3 tbn = float3x3(vsTangent, vsBitangent, vsNormal);
        normal = normalize(mul(normal, tbn));
    }
    float3 specularAlbedo = float3(0.56, 0.56, 0.56);
    float specularMask = 0; // TODO: read the texture
    
    // Shadow
    float shadow = 1.0;
    {
        float3 shadowDirection = SunDirection;
        float3 shadowOrigin = worldPosition;
        RayDesc rayDesc = { shadowOrigin,0.1f,shadowDirection,FLT_MAX };
        RayPayload shadowPayload;
        shadowPayload.SkipShading = true;
        shadowPayload.RayHitT = FLT_MAX;
        TraceRay(g_accelScene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,kModelInstanceMask,0,1,kPrimaryMissShaderIndex,rayDesc,shadowPayload);
        if (shadowPayload.RayHitT < FLT_MAX)
            shadow = 0.1;
    }
    
    // Shading
    const float3 viewDir = normalize(-WorldRayDirection());
    float3 outputColor = AmbientColor * diffuseColor + diffuseColor * shadow * ApplyLightCommon(
        diffuseColor,
        specularAlbedo,
        specularMask,
        gloss,
        normal,
        viewDir,
        SunDirection,
        SunColor);

    // Write to frame buffer
    g_screenOutput[DispatchRaysIndex().xy] = float4(outputColor, 1.0);
}

[shader("miss")]
void PrimaryMiss(inout RayPayload payload)
{
    if (!payload.SkipShading)
        g_screenOutput[DispatchRaysIndex().xy] = float4(0, 0, 0, 1);
}