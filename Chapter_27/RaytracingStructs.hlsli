#pragma once
struct RayTraceMeshInfo
{
    uint  m_indexOffsetBytes;
    uint  m_uvAttributeOffsetBytes;
    uint  m_normalAttributeOffsetBytes;
    uint  m_tangentAttributeOffsetBytes;
    uint  m_bitangentAttributeOffsetBytes;
    uint  m_positionAttributeOffsetBytes;
    uint  m_attributeStrideBytes;
    uint  m_materialInstanceId;
};

struct RayTraceDecalInfo
{
    float4x4 worldToUnity;
    int albedo,normal;
    uint index, padding;
};

// Volatile part (can be split into its own CBV). 
struct DynamicCB
{
    float4x4 cameraToWorld;
    float3   worldCameraPosition;
    float    decalMaxDiagonal;
    float2   resolution;
    uint     decalTraceMode;
};

enum DecalTraceMode
{
    DTM_NoDecals,
    DTM_SingleBLASClosestHitLoop,
    DTM_SingleBLASAnyHit,
    DTM_SingleAABBIntersection,
    DTM_MultipleBLASClosestHitLoop,
    DTM_MultipleAABBIntersection,
    DTM_DebugShowBoxes,
};