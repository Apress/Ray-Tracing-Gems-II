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

#define NOMINMAX

#include <d3d12.h>
#include <dxgi1_3.h>
#include "CameraController.h"
#include "BufferManager.h"
#include "Camera.h"
#include "Model.h"
#include "GpuBuffer.h"
#include "CommandContext.h"
#include "SystemTime.h"
#include "TextRenderer.h"
#include "ShadowCamera.h"
#include "PostEffects.h"
#include <atlbase.h>
#include "DXSampleHelper.h"
#include "DXDescriptorHeapStack.h"

#include "CompiledShaders/RaytracingShadersLib.h"
#include "HLSLCompat.hlsli"
#include "RaytracingStructs.hlsli"

using namespace GameCore;
using namespace Math;
using namespace Graphics;
const float M_PI = 3.14159265358979323846;
const float k_MaxDecalSize = 65.0f;//20.0f;
const float k_MinDecalSize = 35.0f;//15.0f;

// Global tweakables
ExpVar m_SunLightIntensity("Application/Lighting/Sun Light Intensity", 1.0f, 0.0f, 16.0f, 0.1f);
ExpVar m_AmbientIntensity("Application/Lighting/Ambient Intensity", 0.015f, -16.0f, 16.0f, 0.1f);
NumVar m_SunOrientation("Application/Lighting/Sun Orientation", -0.5f, -100.0f, 100.0f, 0.1f );
NumVar m_SunInclination("Application/Lighting/Sun Inclination", 0.75f, 0.0f, 1.0f, 0.01f );
const char* DecalTraceModeLabels[] = { "No Decals", "Single:Mesh", "Single:Mesh+AnyHit", "Single:AABB", "Multi:Mesh", "Multi:AABB", "Debug Boxes" };
EnumVar m_DecalTraceMode("Application/Decals/Trace Mode", DTM_MultipleAABBIntersection, _countof(DecalTraceModeLabels), DecalTraceModeLabels);
BoolVar m_DedicatedDecalsTLAS("Application/Decals/Use Dedicated TLAS", false);

__declspec(align(16)) struct LightConstants
{
    Vector3 sunDirection;
    Vector3 sunLight;
    Vector3 ambientLight;
};

struct MaterialRootConstant
{
    UINT MaterialID;
};

struct DispatchRayParams
{
    DispatchRayParams() {}
    DispatchRayParams(ID3D12StateObject *pPSO, void *pHitGroupShaderTable, UINT HitGroupStride, UINT HitGroupTableSize, LPCWSTR rayGenExportName, const std::vector<LPCWSTR>& missExportNames) :
        m_pPSO(pPSO)
    {
        const UINT shaderTableSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
        ID3D12StateObjectProperties* stateObjectProperties = nullptr;
        ThrowIfFailed(pPSO->QueryInterface(IID_PPV_ARGS(&stateObjectProperties)));
        void *pRayGenShaderData = stateObjectProperties->GetShaderIdentifier(rayGenExportName);

        m_HitGroupStride = HitGroupStride;

        // MiniEngine requires that all initial data be aligned to 16 bytes
        UINT alignment = 16;
        std::vector<BYTE> alignedShaderTableData(shaderTableSize + alignment - 1);
        BYTE *pAlignedShaderTableData = alignedShaderTableData.data() + ((UINT64)alignedShaderTableData.data() % alignment);
        memcpy(pAlignedShaderTableData, pRayGenShaderData, shaderTableSize);
        m_RayGenShaderTable.Create(L"Ray Gen Shader Table", 1, shaderTableSize, alignedShaderTableData.data());
        
        // Miss shaders
        alignedShaderTableData.resize(shaderTableSize*missExportNames.size() + alignment - 1);
        pAlignedShaderTableData = alignedShaderTableData.data() + ((UINT64)alignedShaderTableData.data() % alignment);
        int i=0;
        for (LPCWSTR missExportName : missExportNames)
        {
            void *pMissShaderData = stateObjectProperties->GetShaderIdentifier(missExportName);
            memcpy(pAlignedShaderTableData + shaderTableSize*(i++), pMissShaderData, shaderTableSize);
        }
        m_MissShaderTable.Create(L"Miss Shader Table", (uint32_t)missExportNames.size(), shaderTableSize, alignedShaderTableData.data());
        
        m_HitShaderTable.Create(L"Hit Shader Table", 1, HitGroupTableSize, pHitGroupShaderTable);
    }

    D3D12_DISPATCH_RAYS_DESC GetDispatchRayDesc(UINT DispatchWidth, UINT DispatchHeight)
    {
        D3D12_DISPATCH_RAYS_DESC dispatchRaysDesc = {};

        dispatchRaysDesc.RayGenerationShaderRecord.StartAddress = m_RayGenShaderTable.GetGpuVirtualAddress();
        dispatchRaysDesc.RayGenerationShaderRecord.SizeInBytes = m_RayGenShaderTable.GetBufferSize();
        dispatchRaysDesc.HitGroupTable.StartAddress = m_HitShaderTable.GetGpuVirtualAddress();
        dispatchRaysDesc.HitGroupTable.SizeInBytes = m_HitShaderTable.GetBufferSize();
        dispatchRaysDesc.HitGroupTable.StrideInBytes = m_HitGroupStride;
        dispatchRaysDesc.MissShaderTable.StartAddress = m_MissShaderTable.GetGpuVirtualAddress();
        dispatchRaysDesc.MissShaderTable.SizeInBytes = m_MissShaderTable.GetBufferSize();
        dispatchRaysDesc.MissShaderTable.StrideInBytes = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
        dispatchRaysDesc.Width = DispatchWidth;
        dispatchRaysDesc.Height = DispatchHeight;
        dispatchRaysDesc.Depth = 1;
        return dispatchRaysDesc;
    }

    UINT m_HitGroupStride;
    CComPtr<ID3D12StateObject> m_pPSO;
    ByteAddressBuffer   m_RayGenShaderTable;
    ByteAddressBuffer   m_MissShaderTable;
    ByteAddressBuffer   m_HitShaderTable;
};

class D3D12RaytracedDecalsSample : public GameCore::IGameApp
{
    // Scene management
    Camera m_Camera;
    std::auto_ptr<CameraController> m_CameraController;
    Matrix4 m_ViewProjMatrix;
    Vector3 m_SunDirection;
    ShadowCamera m_SunShadow;
    Model m_Model;
    std::vector<RayTraceDecalInfo> m_Decals;
    std::vector<const ManagedTexture*> m_DecalTextures;

    // Raytracing members
    struct ASINFO
    {
        D3D12_RAYTRACING_GEOMETRY_DESC desc;
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc;
        ByteAddressBuffer instancesBuffer;
        ByteAddressBuffer scratchBuffer;
        CComPtr<ID3D12Resource> asData;
    };

    const static UINT MaxRayRecursion = 2;
    CComPtr<ID3D12Device5> m_pRaytracingDevice;
    CComPtr<ID3D12RootSignature> m_GlobalRaytracingRootSignature;
    std::unique_ptr<DescriptorHeapStack> m_pRaytracingDescriptorHeap;

    StructuredBuffer m_UnityBoxAABBBuffer;
    StructuredBuffer m_UnityBoxVertexBuffer;
    ByteAddressBuffer m_UnityBoxIndexBuffer;
    std::vector<ASINFO> m_ModelBLASs;
    ASINFO m_BLAS_UnityBoxTris,m_BLAS_UnityBoxAABB;

    // TLASs for combined models+decals
    ASINFO m_TLAS_Combined_DebugShowBoxes;
    ASINFO m_TLAS_Combined_Single_BLAS_CHS;
    ASINFO m_TLAS_Combined_Single_BLAS_AnyHit;
    ASINFO m_TLAS_Combined_Single_AABB_Intersection;
    ASINFO m_TLAS_Combined_Multiple_BLAS_CHS;
    ASINFO m_TLAS_Combined_Multiple_AABB_Intersection;

    ASINFO m_TLAS_ModelsOnly;
    ASINFO m_TLAS_DecalsOnly_Single_BLAS_CHS;
    ASINFO m_TLAS_DecalsOnly_Single_BLAS_AnyHit;
    ASINFO m_TLAS_DecalsOnly_Single_AABB_Intersection;
    ASINFO m_TLAS_DecalsOnly_Multiple_BLAS_CHS;
    ASINFO m_TLAS_DecalsOnly_Multiple_AABB_Intersection;

    ByteAddressBuffer m_lightsConstantBuffer;
    ByteAddressBuffer m_dynamicConstantBuffer;
    StructuredBuffer m_MeshInfoBuffer, m_DecalsInfoBuffer;

    D3D12_GPU_DESCRIPTOR_HANDLE m_GpuSceneMaterialSrvs[27];
    D3D12_CPU_DESCRIPTOR_HANDLE m_SceneMeshInfo, m_SceneDecalInfo, m_SceneIndices;
    D3D12_GPU_DESCRIPTOR_HANDLE m_OutputUAV, m_SceneSrvs, m_DecalTextureSrvStart;

    DispatchRayParams m_DispatchRayParams;

    void GenerateDecals(const Model& model,float minSize,float maxSize)
    {
        const float kPrincipalDimMinRatio = 2.0f;
        const float kDensity = 100.0f; // Higher value means less decals
        const int kMinDecalsPerMesh = 0;

        m_Decals.clear();
        m_Decals.reserve(model.m_Header.meshCount);
        std::srand(10);

        auto Random01 = []() -> float
        {
            int rnd = std::rand();
            return ((rnd&0xFFFF + (rnd>>16)) & 0xFFFF) / (float)(0xFFFF);
        };
        auto RandomUnitVector = [&Random01]() -> Vector3
        {
            float r = Random01()*2*M_PI;
            float t = Random01()*M_PI;
            float sr = sinf(r);
            float st = sinf(t);
            float cr = cosf(r);
            float ct = cosf(t);
            return Vector3(sr*ct,sr*st,cr);
        };
        auto GetVertexVec3 = [](uint16_t index,const uint8_t *verts,const uint32_t vertexStride)
        {
            const float *p = (const float*)(verts+vertexStride*index);
            return Vector3(p[0],p[1],p[2]);
        };

        auto PushRandomDecal = [this,&Random01](Vector3 from,Vector3 to,float size)
        {
            // The vector between 'from' and 'to' determines the decal projection direction
            // We randomize a rotation around that vector
            Vector3 forward = Math::Normalize(to-from);
            Vector3 tempRight = Vector3(1,0,0);
            if (fabsf(Math::Dot(forward,tempRight)) > 0.98f)
                tempRight = Vector3(0,1,0);
            Vector3 up = Math::Normalize(Math::Cross(forward,tempRight));
            Vector3 right = Math::Cross(forward,up);
            Matrix4 randomRot(Math::XMMatrixRotationAxis(forward,Random01()*M_PI*2.0f));
            up = Vector3(Math::XMVector3TransformNormal(up,randomRot));
            right = Vector3(Math::XMVector3TransformNormal(right,randomRot));
            Matrix4 unityToWorld = Matrix4(right*size,up*size,to-from,from);
            unityToWorld = Matrix4(right*size,up*size,to-from,Math::Lerp(from,to,Vector3(0.5f)));
            Matrix4 worldToUnity = Math::Invert(unityToWorld);
            worldToUnity = Math::Transpose(worldToUnity); // Transpose for HLSL matrix storage

            RayTraceDecalInfo d = RayTraceDecalInfo();
            d.index = (int)m_Decals.size();
            memcpy(&d.worldToUnity, &worldToUnity, sizeof(d.worldToUnity));
            d.albedo = ((uint32_t)std::rand())%m_DecalTextures.size();
            d.normal = -1;
            m_Decals.push_back(d);
        };

        // Boring decal placement code
        for (int i=0;i<(int)model.m_Header.meshCount;i++)
        {
            const Model::Mesh& mesh = model.m_pMesh[i];

            const float bbMin[3] = { mesh.boundingBox.min.GetX(), mesh.boundingBox.min.GetY(), mesh.boundingBox.min.GetZ() };
            const float bbMax[3] = { mesh.boundingBox.max.GetX(), mesh.boundingBox.max.GetY(), mesh.boundingBox.max.GetZ() };
            const float dims[3] = {
                max(0.001f,fabsf(bbMax[0]-bbMin[0])),
                max(0.001f,fabsf(bbMax[1]-bbMin[1])),
                max(0.001f,fabsf(bbMax[2]-bbMin[2])) };
            float maxDim = max(dims[0],dims[1]);
            maxDim = max(maxDim,dims[2]);
            float minDim = min(dims[0],dims[1]);
            minDim = min(minDim,dims[2]);
            const Vector3 majorNrm = Math::Normalize(Vector3(dims[0],dims[1],dims[2]));

            if (minDim < minSize)
                continue; // Too small to cast on

            int count = (int)((dims[0]*dims[1]*dims[2]) / (maxSize*maxSize*maxSize*kDensity));
            count = std::max(count,kMinDecalsPerMesh);

            const uint32_t vertexCount = mesh.vertexCount;
            const uint32_t triCount = mesh.indexCount/3;
            const uint16_t *indices = (const uint16_t*)(model.m_pIndexData+mesh.indexDataByteOffset);
            const uint8_t *verts = (const uint8_t*)(model.m_pVertexData+mesh.vertexDataByteOffset);
            const uint32_t vertexStride = mesh.vertexStride;
            const uint32_t nrmOffset = mesh.attrib[Model::attrib_normal].offset;

            // Triangle-based placement
            int triAttempts = count;
            while (triAttempts-- > 0)
            {
                int rndTri = std::rand() % triCount;
                Vector3 tri[] =
                {
                    GetVertexVec3(indices[rndTri*3+0],verts,vertexStride),
                    GetVertexVec3(indices[rndTri*3+1],verts,vertexStride),
                    GetVertexVec3(indices[rndTri*3+2],verts,vertexStride)
                };
                const float triArea = Math::Length(tri[1]-tri[0])*Math::Length(tri[2]-tri[0])*0.5f;
                if (triArea < maxSize)
                    continue;

                float bary0 = Random01(), bary1 = Random01();
                while (bary1+bary0 > 1.0f)
                    bary1 = Random01();

                Vector3 pos = tri[0]*bary0+tri[1]*bary1+tri[2]*(1-bary0-bary1);
                Vector3 nrm = Math::Normalize(Math::Cross(tri[1]-tri[0],tri[2]-tri[0]));
                if (Math::Dot(majorNrm,Math::Abs(nrm)) < 0.8f)
                    continue; // Prefer triangles with world-aligned normals (easier on the eyes)

                float size = Lerp(minSize,min(maxSize,triArea),Random01());
                PushRandomDecal(pos-nrm*size,pos+nrm*size,size);
                count--;
            }

            // Vertex-based placement
            while (count > 0)
            {
                int rndVtx = std::rand() % vertexCount;
                Vector3 pos = GetVertexVec3(rndVtx,verts,vertexStride);
                Vector3 nrm = GetVertexVec3(rndVtx,verts+nrmOffset,vertexStride);
                if (fabsf(nrm.GetX()) > 10.0f)
                    continue; // Bad vertex in model? >:(
                if (Math::Dot(majorNrm,Math::Abs(nrm)) < 0.8f)
                    continue; // Prefer triangles with world-aligned normals (easier on the eyes)
                float size = Lerp(minSize,maxSize,Random01());
                PushRandomDecal(pos-nrm*size*0.5f,pos+nrm*size*0.5f,size);
                count--;
            }
        }
    }

    static void RemoveMeshFromModel(Model& model,const char *texDiffuseName)
    {
        for (int i=0; i<(int)model.m_Header.meshCount; ++i)
        {
            const Model::Mesh& mesh = model.m_pMesh[i];
            const Model::Material& mat = model.m_pMaterial[mesh.materialIndex];
            if (std::string(mat.texDiffusePath).find(texDiffuseName) != std::string::npos)
            {
                int meshesAfter = model.m_Header.meshCount-i-1;
                if (meshesAfter)
                    memcpy(model.m_pMesh+i,model.m_pMesh+i+1,sizeof(Model::Mesh)*meshesAfter);
                model.m_Header.meshCount--;
                i--; 
            }
        }
    }

    void InitializeSceneInfo(const Model& model)
    {
        //
        // Mesh info
        //
        std::vector<RayTraceMeshInfo>   meshInfoData(model.m_Header.meshCount);
        for (UINT i=0; i < model.m_Header.meshCount; ++i)
        {
            meshInfoData[i].m_indexOffsetBytes = model.m_pMesh[i].indexDataByteOffset;
            meshInfoData[i].m_uvAttributeOffsetBytes = model.m_pMesh[i].vertexDataByteOffset + model.m_pMesh[i].attrib[Model::attrib_texcoord0].offset;
            meshInfoData[i].m_normalAttributeOffsetBytes = model.m_pMesh[i].vertexDataByteOffset + model.m_pMesh[i].attrib[Model::attrib_normal].offset;
            meshInfoData[i].m_positionAttributeOffsetBytes = model.m_pMesh[i].vertexDataByteOffset + model.m_pMesh[i].attrib[Model::attrib_position].offset;
            meshInfoData[i].m_tangentAttributeOffsetBytes = model.m_pMesh[i].vertexDataByteOffset + model.m_pMesh[i].attrib[Model::attrib_tangent].offset;
            meshInfoData[i].m_bitangentAttributeOffsetBytes = model.m_pMesh[i].vertexDataByteOffset + model.m_pMesh[i].attrib[Model::attrib_bitangent].offset;
            meshInfoData[i].m_attributeStrideBytes = model.m_pMesh[i].vertexStride;
            meshInfoData[i].m_materialInstanceId = model.m_pMesh[i].materialIndex;
            ASSERT(meshInfoData[i].m_materialInstanceId < 27);
        }

        m_MeshInfoBuffer.Create(L"RayTraceMeshInfo",
            (UINT)meshInfoData.size(),
            sizeof(meshInfoData[0]),
            meshInfoData.data());

        m_DecalsInfoBuffer.Create(L"RayTraceDecalInfo",
            (UINT)m_Decals.size(),
            sizeof(m_Decals[0]),
            m_Decals.data());

        m_SceneIndices = model.m_IndexBuffer.GetSRV();
        m_SceneMeshInfo = m_MeshInfoBuffer.GetSRV();
        m_SceneDecalInfo = m_DecalsInfoBuffer.GetSRV();
    }

    void InitializeViews(const Model& model)
    {
        D3D12_CPU_DESCRIPTOR_HANDLE uavHandle;
        UINT uavDescriptorIndex;
        m_pRaytracingDescriptorHeap->AllocateDescriptor(uavHandle, uavDescriptorIndex);
        Graphics::g_Device->CopyDescriptorsSimple(1, uavHandle, g_SceneColorBuffer.GetUAV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        m_OutputUAV = m_pRaytracingDescriptorHeap->GetGpuHandle(uavDescriptorIndex);

        // Scene SRVs are 4 consequetive ones bound in one descriptor table
        // g_meshInfo,g_indices,g_attributes,g_decalInfo
        // So the three allocations below must be called in that exact order

        // g_meshInfo
        D3D12_CPU_DESCRIPTOR_HANDLE srvHandle;
        UINT srvDescriptorIndex;
        m_pRaytracingDescriptorHeap->AllocateDescriptor(srvHandle, srvDescriptorIndex);
        Graphics::g_Device->CopyDescriptorsSimple(1, srvHandle, m_SceneMeshInfo, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        m_SceneSrvs = m_pRaytracingDescriptorHeap->GetGpuHandle(srvDescriptorIndex);

        // g_indices
        UINT unused;
        m_pRaytracingDescriptorHeap->AllocateDescriptor(srvHandle, unused);
        Graphics::g_Device->CopyDescriptorsSimple(1, srvHandle, m_SceneIndices, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        // g_attributes
        m_pRaytracingDescriptorHeap->AllocateBufferSrv(*const_cast<ID3D12Resource*>(model.m_VertexBuffer.GetResource()));

        // g_decalInfo
        m_pRaytracingDescriptorHeap->AllocateDescriptor(srvHandle, unused);
        Graphics::g_Device->CopyDescriptorsSimple(1, srvHandle, m_SceneDecalInfo, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        // Decal textures
        UINT slot;
        m_pRaytracingDescriptorHeap->AllocateDescriptor(srvHandle, slot);
        m_DecalTextureSrvStart = m_pRaytracingDescriptorHeap->GetGpuHandle(slot);
        for (size_t i=1;i<m_DecalTextures.size();i++)
        {
            D3D12_CPU_DESCRIPTOR_HANDLE srvUnused;
            m_pRaytracingDescriptorHeap->AllocateDescriptor(srvUnused, unused);
        }
        Graphics::g_Device->CopyDescriptorsSimple((UINT)m_DecalTextures.size(), srvHandle, m_DecalTextures.front()->GetSRV(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        // An array of descriptors that will be references by each mesh's local root sig
        // Those don't need to be in any particular order
        for (UINT i = 0; i < model.m_Header.materialCount; i++)
        {
            UINT slot;
            m_pRaytracingDescriptorHeap->AllocateDescriptor(srvHandle, slot);
            Graphics::g_Device->CopyDescriptorsSimple(1, srvHandle, *model.GetSRVs(i), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            m_pRaytracingDescriptorHeap->AllocateDescriptor(srvHandle, unused);
            Graphics::g_Device->CopyDescriptorsSimple(1, srvHandle, model.GetSRVs(i)[3], D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            
            m_GpuSceneMaterialSrvs[i] = m_pRaytracingDescriptorHeap->GetGpuHandle(slot);
        }
    }

    static void CreateBLAS(ID3D12Device5 *device,D3D12_GPU_VIRTUAL_ADDRESS vb,D3D12_GPU_VIRTUAL_ADDRESS ib,
        UINT vertexCount,UINT vbStride,UINT indexCount,D3D12_RAYTRACING_GEOMETRY_FLAGS geomFlags,ASINFO& blas)
    {
        const D3D12_HEAP_PROPERTIES defaultHeapDesc = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlag = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

        blas.desc = D3D12_RAYTRACING_GEOMETRY_DESC();
        blas.desc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        blas.desc.Flags = geomFlags;

        D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC &trianglesDesc = blas.desc.Triangles;
        trianglesDesc.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
        trianglesDesc.VertexCount = vertexCount;
        trianglesDesc.VertexBuffer.StartAddress = vb;
        trianglesDesc.IndexBuffer = ib;
        trianglesDesc.VertexBuffer.StrideInBytes = vbStride;
        trianglesDesc.IndexCount = indexCount;
        trianglesDesc.IndexFormat = DXGI_FORMAT_R16_UINT;
        trianglesDesc.Transform3x4 = 0;

        blas.buildDesc.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        blas.buildDesc.Inputs.NumDescs = 1;
        blas.buildDesc.Inputs.pGeometryDescs = &blas.desc;
        blas.buildDesc.Inputs.Flags = buildFlag;
        blas.buildDesc.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildInfo;
        device->GetRaytracingAccelerationStructurePrebuildInfo(&blas.buildDesc.Inputs, &prebuildInfo);

        blas.scratchBuffer.Create(L"BLAS_Scratch", (UINT)prebuildInfo.ScratchDataSizeInBytes, 1);

        auto blasResDesc = CD3DX12_RESOURCE_DESC::Buffer(prebuildInfo.ResultDataMaxSizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        device->CreateCommittedResource(
            &defaultHeapDesc,
            D3D12_HEAP_FLAG_NONE, 
            &blasResDesc, 
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            nullptr, 
            IID_PPV_ARGS(&blas.asData));

        blas.buildDesc.DestAccelerationStructureData = blas.asData->GetGPUVirtualAddress();
        blas.buildDesc.ScratchAccelerationStructureData = blas.scratchBuffer.GetGpuVirtualAddress();
    }

    static void CreateModelBLAS(ID3D12Device5 *device,const Model& model,std::vector<ASINFO>& meshesBLASs)
    {
        meshesBLASs.resize(model.m_Header.meshCount);
        for (UINT i=0; i<model.m_Header.meshCount; i++)
        {
            auto &mesh = model.m_pMesh[i];
            CreateBLAS(
                device,
                model.m_VertexBuffer.GetGpuVirtualAddress() + (mesh.vertexDataByteOffset + mesh.attrib[Model::attrib_position].offset),
                model.m_IndexBuffer.GetGpuVirtualAddress() + mesh.indexDataByteOffset,
                mesh.vertexCount,mesh.vertexStride,mesh.indexCount,D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,meshesBLASs[i]);
        }
    }

    void CreateUnityBoxes()
    {
        const float verts[8*3] =
        {
            -0.5f,-0.5f,-0.5f, // 0=LBB
             0.5f,-0.5f,-0.5f, // 1=RBB
            -0.5f, 0.5f,-0.5f, // 2=LTB
             0.5f, 0.5f,-0.5f, // 3=RTB
            -0.5f,-0.5f, 0.5f, // 4=LBF
             0.5f,-0.5f, 0.5f, // 5=RBF
            -0.5f, 0.5f, 0.5f, // 6=LTF
             0.5f, 0.5f, 0.5f, // 7=RTF
        };
        const UINT16 indices[6*2*3] =
        {
            4,5,7, 6,4,7, // Front
            3,1,0, 3,0,2, // Back
            6,2,0, 0,4,6, // Left
            1,3,7, 7,5,1, // Right
            3,2,6, 6,7,3, // Top
            4,0,1, 1,5,4, // Bottom
        };
        m_UnityBoxVertexBuffer.Create(L"UnityBoxVerts",8,sizeof(float)*3,verts);
        m_UnityBoxIndexBuffer.Create(L"UnityBoxIndices",36,sizeof(UINT16),indices);
        CreateBLAS(m_pRaytracingDevice,m_UnityBoxVertexBuffer.GetGpuVirtualAddress(),m_UnityBoxIndexBuffer.GetGpuVirtualAddress(),8,sizeof(float)*3,36,D3D12_RAYTRACING_GEOMETRY_FLAG_NONE,m_BLAS_UnityBoxTris);

        alignas(64) D3D12_RAYTRACING_AABB aabb = D3D12_RAYTRACING_AABB();
        aabb.MinX = aabb.MinY = aabb.MinZ = -0.5f;
        aabb.MaxX = aabb.MaxY = aabb.MaxZ =  0.5f;
        m_UnityBoxAABBBuffer.Create(L"UnityBoxAABB",1,sizeof(D3D12_RAYTRACING_AABB),&aabb);

        const D3D12_HEAP_PROPERTIES defaultHeapDesc = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlag = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

        D3D12_RAYTRACING_AABB_BYTE_ALIGNMENT;
        m_BLAS_UnityBoxAABB.desc = D3D12_RAYTRACING_GEOMETRY_DESC();
        m_BLAS_UnityBoxAABB.desc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
        m_BLAS_UnityBoxAABB.desc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION;

        D3D12_RAYTRACING_GEOMETRY_AABBS_DESC &aabbDesc = m_BLAS_UnityBoxAABB.desc.AABBs;
        aabbDesc.AABBCount = 1;
        aabbDesc.AABBs.StartAddress = m_UnityBoxAABBBuffer.GetGpuVirtualAddress();
        aabbDesc.AABBs.StrideInBytes = sizeof(D3D12_RAYTRACING_AABB);

        m_BLAS_UnityBoxAABB.buildDesc.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        m_BLAS_UnityBoxAABB.buildDesc.Inputs.NumDescs = 1;
        m_BLAS_UnityBoxAABB.buildDesc.Inputs.pGeometryDescs = &m_BLAS_UnityBoxAABB.desc;
        m_BLAS_UnityBoxAABB.buildDesc.Inputs.Flags = buildFlag;
        m_BLAS_UnityBoxAABB.buildDesc.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildInfo;
        m_pRaytracingDevice->GetRaytracingAccelerationStructurePrebuildInfo(&m_BLAS_UnityBoxAABB.buildDesc.Inputs, &prebuildInfo);

        m_BLAS_UnityBoxAABB.scratchBuffer.Create(L"AABB_Scratch", (UINT)prebuildInfo.ScratchDataSizeInBytes, 1);

        auto blasResDesc = CD3DX12_RESOURCE_DESC::Buffer(prebuildInfo.ResultDataMaxSizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        m_pRaytracingDevice->CreateCommittedResource(
            &defaultHeapDesc,
            D3D12_HEAP_FLAG_NONE, 
            &blasResDesc, 
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            nullptr, 
            IID_PPV_ARGS(&m_BLAS_UnityBoxAABB.asData));

        m_BLAS_UnityBoxAABB.buildDesc.DestAccelerationStructureData = m_BLAS_UnityBoxAABB.asData->GetGPUVirtualAddress();
        m_BLAS_UnityBoxAABB.buildDesc.ScratchAccelerationStructureData = m_BLAS_UnityBoxAABB.scratchBuffer.GetGpuVirtualAddress();
    }

    static void CreateSceneTLAS(ID3D12Device5 *device,DecalTraceMode decalTraceMode,const std::vector<ASINFO> *meshesBLASs,const std::vector<RayTraceDecalInfo> *decals,ASINFO *decalBLAS,ASINFO& tlas)
    {
        const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlag = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
        D3D12_HEAP_PROPERTIES defaultHeapDesc = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);

        const UINT numInstances = (UINT)((meshesBLASs ? meshesBLASs->size() : 0) + (decals ? decals->size() : 0));

        tlas = ASINFO();
        tlas.buildDesc.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        tlas.buildDesc.Inputs.NumDescs = numInstances;
        tlas.buildDesc.Inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
        tlas.buildDesc.Inputs.pGeometryDescs = nullptr;
        tlas.buildDesc.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;

        // Those match the order used in InitializeRaytracingPSO
        const UINT DecalHitShaderTableRecord = 0;
        const UINT DecalAnyHitShaderTableRecord = 1;
        const UINT DecalAABBIntersectShaderTableRecord = 2;
        const UINT MultiDecalHitShaderTableRecord = 3;
        const UINT MultiDecalAABBIntersectShaderTableRecord = 4;
        const UINT DecalDebugHitShaderTableRecord = 5;
        const UINT FirstMeshShaderTableRecord = 6; // Mesh shader table records come after decal records

        std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instanceDescs;
        instanceDescs.reserve(numInstances);
        if (meshesBLASs)
        {
            for (UINT i=0; i<meshesBLASs->size(); i++)
            {
                D3D12_RAYTRACING_INSTANCE_DESC instanceDesc = D3D12_RAYTRACING_INSTANCE_DESC();
        
                // Identity matrix
                ZeroMemory(instanceDesc.Transform, sizeof(instanceDesc.Transform));
                instanceDesc.Transform[0][0] = 1.0f;
                instanceDesc.Transform[1][1] = 1.0f;
                instanceDesc.Transform[2][2] = 1.0f;
        
                instanceDesc.AccelerationStructure = (*meshesBLASs)[i].asData->GetGPUVirtualAddress();
                instanceDesc.Flags = 0;
                instanceDesc.InstanceID = i;
                instanceDesc.InstanceMask = 1;
                instanceDesc.InstanceContributionToHitGroupIndex = FirstMeshShaderTableRecord+i;

                instanceDescs.push_back(instanceDesc);
            }
        }
        if (decals && decalBLAS)
        {
            UINT decalHitGroupIndex;
            switch (decalTraceMode)
            {
            case DTM_SingleBLASClosestHitLoop:
                decalHitGroupIndex = DecalHitShaderTableRecord;
                break;
            case DTM_SingleBLASAnyHit:
                decalHitGroupIndex = DecalAnyHitShaderTableRecord;
                break;
            case DTM_SingleAABBIntersection:
                decalHitGroupIndex = DecalAABBIntersectShaderTableRecord;
                break;
            case DTM_DebugShowBoxes:
                decalHitGroupIndex = DecalDebugHitShaderTableRecord;
                break;
            case DTM_MultipleBLASClosestHitLoop:
                decalHitGroupIndex = MultiDecalHitShaderTableRecord;
                break;
            case DTM_MultipleAABBIntersection:
                decalHitGroupIndex = MultiDecalAABBIntersectShaderTableRecord;
                break;
            default:
                assert(false); // Should not be here
            }

            for (UINT i=0; i<decals->size(); i++)
            {
                Math::Matrix4 matDecal;
                memcpy(&matDecal,&(*decals)[i].worldToUnity,sizeof(Math::Matrix4));
                matDecal = Math::Invert(Math::Transpose(matDecal));
        
                D3D12_RAYTRACING_INSTANCE_DESC instanceDesc = D3D12_RAYTRACING_INSTANCE_DESC();
                instanceDesc.Transform[0][0] = matDecal.GetX().GetX();
                instanceDesc.Transform[0][1] = matDecal.GetY().GetX();
                instanceDesc.Transform[0][2] = matDecal.GetZ().GetX();
                instanceDesc.Transform[0][3] = matDecal.GetW().GetX();

                instanceDesc.Transform[1][0] = matDecal.GetX().GetY();
                instanceDesc.Transform[1][1] = matDecal.GetY().GetY();
                instanceDesc.Transform[1][2] = matDecal.GetZ().GetY();
                instanceDesc.Transform[1][3] = matDecal.GetW().GetY();

                instanceDesc.Transform[2][0] = matDecal.GetX().GetZ();
                instanceDesc.Transform[2][1] = matDecal.GetY().GetZ();
                instanceDesc.Transform[2][2] = matDecal.GetZ().GetZ();
                instanceDesc.Transform[2][3] = matDecal.GetW().GetZ();
        
                instanceDesc.AccelerationStructure = decalBLAS->asData->GetGPUVirtualAddress();
                instanceDesc.Flags = 0;
                instanceDesc.InstanceID = i;
                instanceDesc.InstanceMask = 2;
                instanceDesc.InstanceContributionToHitGroupIndex = decalHitGroupIndex;

                instanceDescs.push_back(instanceDesc);
            }
        }

        tlas.instancesBuffer.Create(L"Instances", numInstances, sizeof(D3D12_RAYTRACING_INSTANCE_DESC), instanceDescs.data());

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildInfo;
        device->GetRaytracingAccelerationStructurePrebuildInfo(&tlas.buildDesc.Inputs, &prebuildInfo);
    
        tlas.scratchBuffer.Create(L"TLAS_Scratch", (UINT)prebuildInfo.ScratchDataSizeInBytes, 1);

        auto tlasResDesc = CD3DX12_RESOURCE_DESC::Buffer(prebuildInfo.ResultDataMaxSizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        device->CreateCommittedResource(
            &defaultHeapDesc,
            D3D12_HEAP_FLAG_NONE,
            &tlasResDesc,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            nullptr,
            IID_PPV_ARGS(&tlas.asData));

        tlas.buildDesc.DestAccelerationStructureData = tlas.asData->GetGPUVirtualAddress();
        tlas.buildDesc.ScratchAccelerationStructureData = tlas.scratchBuffer.GetGpuVirtualAddress();
        tlas.buildDesc.Inputs.InstanceDescs = tlas.instancesBuffer.GetGpuVirtualAddress();
    }

    void SetPipelineStateStackSize(LPCWSTR raygen, LPCWSTR closestHit, LPCWSTR miss, UINT maxRecursion, ID3D12StateObject *pStateObject)
    {
        ID3D12StateObjectProperties* stateObjectProperties = nullptr;
        ThrowIfFailed(pStateObject->QueryInterface(IID_PPV_ARGS(&stateObjectProperties)));
        UINT64 closestHitStackSize = stateObjectProperties->GetShaderStackSize(closestHit);
        UINT64 missStackSize = stateObjectProperties->GetShaderStackSize(miss);
        UINT64 raygenStackSize = stateObjectProperties->GetShaderStackSize(raygen);

        UINT64 totalStackSize = raygenStackSize + std::max(missStackSize, closestHitStackSize) * maxRecursion;
        stateObjectProperties->SetPipelineStackSize(totalStackSize);
    }

    void InitializeRaytracingPSO(const Model &model, UINT numMeshes)
    {
        std::vector<D3D12_STATE_SUBOBJECT> subObjects;

        D3D12_STATE_SUBOBJECT dxilLibSubObject = D3D12_STATE_SUBOBJECT();
        D3D12_DXIL_LIBRARY_DESC dxilLibDesc = D3D12_DXIL_LIBRARY_DESC();
        dxilLibDesc.DXILLibrary.pShaderBytecode = g_pRaytracingShadersLib;
        dxilLibDesc.DXILLibrary.BytecodeLength = ARRAYSIZE(g_pRaytracingShadersLib);
        dxilLibSubObject.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
        dxilLibSubObject.pDesc = &dxilLibDesc;
        subObjects.push_back(dxilLibSubObject);

        D3D12_STATE_OBJECT_DESC stateObject;
        stateObject.NumSubobjects = (UINT)subObjects.size();
        stateObject.pSubobjects = subObjects.data();
        stateObject.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE;

        const UINT shaderIdentifierSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
        #define ALIGN(alignment, num) ((((num) + alignment - 1) / alignment) * alignment)
        const UINT offsetToDescriptorHandle = ALIGN(sizeof(D3D12_GPU_DESCRIPTOR_HANDLE), shaderIdentifierSize);
        const UINT offsetToMaterialConstants = ALIGN(sizeof(UINT32), offsetToDescriptorHandle + sizeof(D3D12_GPU_DESCRIPTOR_HANDLE));
        const UINT shaderRecordSizeInBytes = ALIGN(D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT, offsetToMaterialConstants + sizeof(MaterialRootConstant));
    
        const LPCWSTR rayGenShaderExportName = L"PrimaryRayGen";
        const LPCWSTR decal_SingleCHS_HitGroupExportName = L"SingleDecalCHS_HitGroup";
        const LPCWSTR decal_SingleAnyHit_HitGroupExportName = L"SingleDecalAnyHit_HitGroup";
        const LPCWSTR decal_SingleAABB_HitGroupExportName = L"SingleDecalAABBIntersection_HitGroup";
        const LPCWSTR decal_MultiCHS_HitGroupExportName = L"MultiDecalCHS_HitGroup";
        const LPCWSTR decal_MultiAABB_HitGroupExportName = L"MultiDecalAABBIntersection_HitGroup";
        const LPCWSTR decal_Debug_HitGroupExportName = L"DecalDebug_HitGroup";
        const LPCWSTR meshHitGroupExportName = L"PrimaryHitGroup";
        const LPCWSTR meshMissExportName = L"PrimaryMiss";
        const LPCWSTR decalMissExportName = L"DecalMiss";

        std::vector<byte> pHitShaderTable(shaderRecordSizeInBytes*(numMeshes+6));

        CComPtr<ID3D12StateObject> pso;
        CComPtr<ID3D12StateObjectProperties> stateObjectProperties;
        m_pRaytracingDevice->CreateStateObject(&stateObject, IID_PPV_ARGS(&pso));
        ThrowIfFailed(pso->QueryInterface(IID_PPV_ARGS(&stateObjectProperties)));

        const void *pDecal_SingleCHS_HitGroupID = stateObjectProperties->GetShaderIdentifier(decal_SingleCHS_HitGroupExportName);
        const void *pDecal_SingleAnyHit_GroupID = stateObjectProperties->GetShaderIdentifier(decal_SingleAnyHit_HitGroupExportName);
        const void *pDecal_SingleAABBIntersect_GroupID = stateObjectProperties->GetShaderIdentifier(decal_SingleAABB_HitGroupExportName);
        const void *pDecal_MultiCHS_HitGroupID = stateObjectProperties->GetShaderIdentifier(decal_MultiCHS_HitGroupExportName);
        const void *pDecal_MultiAABBIntersect_GroupID = stateObjectProperties->GetShaderIdentifier(decal_MultiAABB_HitGroupExportName);
        const void *pDecal_Debug_HitGroupID = stateObjectProperties->GetShaderIdentifier(decal_Debug_HitGroupExportName);
        const void *pMeshHitGroupID = stateObjectProperties->GetShaderIdentifier(meshHitGroupExportName);
        byte *pShaderTable = pHitShaderTable.data();

        // First entry is the decal hit group
        memset(pShaderTable,0,shaderRecordSizeInBytes);
        memcpy(pShaderTable, pDecal_SingleCHS_HitGroupID, shaderIdentifierSize);
        pShaderTable += shaderRecordSizeInBytes; // Move past the decal hit group record
        memcpy(pShaderTable, pDecal_SingleAnyHit_GroupID, shaderIdentifierSize);
        pShaderTable += shaderRecordSizeInBytes; // Move past the decal any-hit group record
        memcpy(pShaderTable, pDecal_SingleAABBIntersect_GroupID, shaderIdentifierSize);
        pShaderTable += shaderRecordSizeInBytes; // Move past the decal AABB-intersection group record
        memcpy(pShaderTable, pDecal_MultiCHS_HitGroupID, shaderIdentifierSize);
        pShaderTable += shaderRecordSizeInBytes; // Move past the multi decal hit group record
        memcpy(pShaderTable, pDecal_MultiAABBIntersect_GroupID, shaderIdentifierSize);
        pShaderTable += shaderRecordSizeInBytes; // Move past the multi decal AABB-intersection group record
        memcpy(pShaderTable, pDecal_Debug_HitGroupID, shaderIdentifierSize);
        pShaderTable += shaderRecordSizeInBytes; // Move past the decal debug hit group record

        // Fill model hit groups
        for (UINT i = 0; i < numMeshes; i++)
        {
            byte *pShaderRecord = i * shaderRecordSizeInBytes + pShaderTable;
            memcpy(pShaderRecord, pMeshHitGroupID, shaderIdentifierSize);

            UINT materialIndex = model.m_pMesh[i].materialIndex;
            memcpy(pShaderRecord + offsetToDescriptorHandle, &m_GpuSceneMaterialSrvs[materialIndex].ptr, sizeof(m_GpuSceneMaterialSrvs[materialIndex].ptr));

            MaterialRootConstant material;
            material.MaterialID = i;
            memcpy(pShaderRecord + offsetToMaterialConstants, &material, sizeof(material));
        }

        std::vector<LPCWSTR> missShaders({ decalMissExportName, meshMissExportName });
        m_DispatchRayParams = DispatchRayParams(pso, pHitShaderTable.data(), shaderRecordSizeInBytes, (UINT)pHitShaderTable.size(), rayGenShaderExportName, missShaders);

        m_pRaytracingDevice->CreateRootSignature(0, g_pRaytracingShadersLib, ARRAYSIZE(g_pRaytracingShadersLib), IID_PPV_ARGS(&m_GlobalRaytracingRootSignature));

	    //WCHAR hitGroupExportNameClosestHitType[64];
	    //swprintf_s(hitGroupExportNameClosestHitType, L"%s::closesthit", meshHitGroupExportName);
	    //SetPipelineStateStackSize(rayGenShaderExportName, hitGroupExportNameClosestHitType, missExportName, MaxRayRecursion, m_DispatchRayParams.m_pPSO);
    }

    void Raytrace(GraphicsContext& context, const Math::Camera& camera, ColorBuffer& colorTarget, const ASINFO& sceneTLAS,const ASINFO& decalsTLAS)
    {
        ScopedTimer _p0(L"Raytracing", context);

        // Prepare constants
        DynamicCB inputs = DynamicCB();
        auto m0 = camera.GetViewProjMatrix();
        auto m1 = Transpose(Invert(m0));
        memcpy(&inputs.cameraToWorld, &m1, sizeof(inputs.cameraToWorld));
        memcpy(&inputs.worldCameraPosition, &camera.GetPosition(), sizeof(inputs.worldCameraPosition));
        inputs.decalMaxDiagonal = Math::Length(Vector3(k_MaxDecalSize,k_MaxDecalSize,k_MaxDecalSize));
        inputs.resolution.x = (float)colorTarget.GetWidth();
        inputs.resolution.y = (float)colorTarget.GetHeight();
        inputs.decalTraceMode = m_DecalTraceMode;

        LightConstants lightConstants = {};
        lightConstants.sunDirection = m_SunDirection;
        lightConstants.sunLight = Vector3(1.0f, 1.0f, 1.0f) * m_SunLightIntensity;
        lightConstants.ambientLight = Vector3(1.0f, 1.0f, 1.0f) * m_AmbientIntensity;
        context.WriteBuffer(m_lightsConstantBuffer, 0, &lightConstants, sizeof(lightConstants));
        context.WriteBuffer(m_dynamicConstantBuffer, 0, &inputs, sizeof(inputs));

        context.TransitionResource(m_dynamicConstantBuffer, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
        context.TransitionResource(m_lightsConstantBuffer, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
        context.TransitionResource(colorTarget, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        context.FlushResourceBarriers();

        CComPtr<ID3D12GraphicsCommandList4> pCL;
        context.GetCommandList()->QueryInterface(IID_PPV_ARGS(&pCL));

        ID3D12DescriptorHeap *pDescriptorHeaps[] = { &m_pRaytracingDescriptorHeap->GetDescriptorHeap() };
        pCL->SetDescriptorHeaps(ARRAYSIZE(pDescriptorHeaps), pDescriptorHeaps);

        pCL->SetComputeRootSignature(m_GlobalRaytracingRootSignature);
        pCL->SetComputeRootConstantBufferView(0, m_lightsConstantBuffer.GetGpuVirtualAddress());
        pCL->SetComputeRootConstantBufferView(1, m_dynamicConstantBuffer.GetGpuVirtualAddress());
        pCL->SetComputeRootDescriptorTable(2, m_SceneSrvs);
        pCL->SetComputeRootShaderResourceView(3, sceneTLAS.asData->GetGPUVirtualAddress());
        pCL->SetComputeRootShaderResourceView(4, decalsTLAS.asData->GetGPUVirtualAddress());
        pCL->SetComputeRootDescriptorTable(5, m_OutputUAV);
        pCL->SetComputeRootDescriptorTable(6, m_DecalTextureSrvStart);

        D3D12_DISPATCH_RAYS_DESC dispatchRaysDesc = m_DispatchRayParams.GetDispatchRayDesc(colorTarget.GetWidth(), colorTarget.GetHeight());
        pCL->SetPipelineState1(m_DispatchRayParams.m_pPSO);
        pCL->DispatchRays(&dispatchRaysDesc);
    }

public:
    virtual void Startup(void) override
    {
        ThrowIfFailed(g_Device->QueryInterface(IID_PPV_ARGS(&m_pRaytracingDevice)), L"Couldn't get DirectX Raytracing interface for the device.\n");

        m_pRaytracingDescriptorHeap = std::unique_ptr<DescriptorHeapStack>(
            new DescriptorHeapStack(*g_Device, 300, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 0));
        
        GraphicsContext& gfxContext = GraphicsContext::Begin(L"Create Acceleration Structure");
        CComPtr<ID3D12GraphicsCommandList4> pCL;
        gfxContext.GetCommandList()->QueryInterface(IID_PPV_ARGS(&pCL));

        ID3D12DescriptorHeap *descriptorHeaps[] = { &m_pRaytracingDescriptorHeap->GetDescriptorHeap() };
        pCL->SetDescriptorHeaps(ARRAYSIZE(descriptorHeaps), descriptorHeaps);


        // Scene loading
        TextureManager::Initialize(L"Assets/Textures/");

        // Decal textures
        m_DecalTextures.push_back(TextureManager::LoadFromFile("Decals/boom",true));
        m_DecalTextures.push_back(TextureManager::LoadFromFile("Decals/bored",true));
        m_DecalTextures.push_back(TextureManager::LoadFromFile("Decals/happy",true));
        m_DecalTextures.push_back(TextureManager::LoadFromFile("Decals/lol",true));
        m_DecalTextures.push_back(TextureManager::LoadFromFile("Decals/mask",true));
        m_DecalTextures.push_back(TextureManager::LoadFromFile("Decals/scared",true));
        ASSERT(m_DecalTextures.size() == 6, "Failed to load decal textures");

        bool bModelLoadSuccess = m_Model.Load("Assets/Models/sponza.h3d", true);
        ASSERT(bModelLoadSuccess, "Failed to load model");
        ASSERT(m_Model.m_Header.meshCount > 0, "Model contains no meshes");

        RemoveMeshFromModel(m_Model,"gi_flag"); // Remove the annoying gi flag mesh
        GenerateDecals(m_Model,k_MinDecalSize,k_MaxDecalSize);

        // Buffers and descriptors and views
        m_lightsConstantBuffer.Create(L"Hit Constant Buffer", 1, sizeof(LightConstants));
        m_dynamicConstantBuffer.Create(L"Dynamic Constant Buffer", 1, sizeof(DynamicCB));

        InitializeSceneInfo(m_Model);
        InitializeViews(m_Model);
        const UINT numMeshes = m_Model.m_Header.meshCount;

        CreateUnityBoxes();

        // Generate acceleration structures
        CreateModelBLAS(m_pRaytracingDevice,m_Model,m_ModelBLASs);

        // Combined TLASs
        CreateSceneTLAS(m_pRaytracingDevice,DTM_DebugShowBoxes,             &m_ModelBLASs,&m_Decals,&m_BLAS_UnityBoxTris,   m_TLAS_Combined_DebugShowBoxes);
        CreateSceneTLAS(m_pRaytracingDevice,DTM_SingleBLASClosestHitLoop,   &m_ModelBLASs,&m_Decals,&m_BLAS_UnityBoxTris,   m_TLAS_Combined_Single_BLAS_CHS);
        CreateSceneTLAS(m_pRaytracingDevice,DTM_SingleBLASAnyHit,           &m_ModelBLASs,&m_Decals,&m_BLAS_UnityBoxTris,   m_TLAS_Combined_Single_BLAS_AnyHit);
        CreateSceneTLAS(m_pRaytracingDevice,DTM_SingleAABBIntersection,     &m_ModelBLASs,&m_Decals,&m_BLAS_UnityBoxAABB,   m_TLAS_Combined_Single_AABB_Intersection);
        CreateSceneTLAS(m_pRaytracingDevice,DTM_MultipleBLASClosestHitLoop, &m_ModelBLASs,&m_Decals,&m_BLAS_UnityBoxTris,   m_TLAS_Combined_Multiple_BLAS_CHS);
        CreateSceneTLAS(m_pRaytracingDevice,DTM_MultipleAABBIntersection,   &m_ModelBLASs,&m_Decals,&m_BLAS_UnityBoxAABB,   m_TLAS_Combined_Multiple_AABB_Intersection);

        // Separated TLASs
        CreateSceneTLAS(m_pRaytracingDevice,DTM_DebugShowBoxes,             &m_ModelBLASs,nullptr,nullptr,             m_TLAS_ModelsOnly);
        CreateSceneTLAS(m_pRaytracingDevice,DTM_SingleBLASClosestHitLoop,   nullptr,&m_Decals,&m_BLAS_UnityBoxTris,    m_TLAS_DecalsOnly_Single_BLAS_CHS);
        CreateSceneTLAS(m_pRaytracingDevice,DTM_SingleBLASAnyHit,           nullptr,&m_Decals,&m_BLAS_UnityBoxTris,    m_TLAS_DecalsOnly_Single_BLAS_AnyHit);
        CreateSceneTLAS(m_pRaytracingDevice,DTM_SingleAABBIntersection,     nullptr,&m_Decals,&m_BLAS_UnityBoxAABB,    m_TLAS_DecalsOnly_Single_AABB_Intersection);
        CreateSceneTLAS(m_pRaytracingDevice,DTM_MultipleBLASClosestHitLoop, nullptr,&m_Decals,&m_BLAS_UnityBoxTris,    m_TLAS_DecalsOnly_Multiple_BLAS_CHS);
        CreateSceneTLAS(m_pRaytracingDevice,DTM_MultipleAABBIntersection,   nullptr,&m_Decals,&m_BLAS_UnityBoxAABB,    m_TLAS_DecalsOnly_Multiple_AABB_Intersection);

        // Build all model BLASs
        pCL->BuildRaytracingAccelerationStructure(&m_BLAS_UnityBoxTris.buildDesc,0,nullptr);
        pCL->BuildRaytracingAccelerationStructure(&m_BLAS_UnityBoxAABB.buildDesc,0,nullptr);
        for (UINT i=0; i<m_ModelBLASs.size(); i++)
            pCL->BuildRaytracingAccelerationStructure(&m_ModelBLASs[i].buildDesc,0,nullptr);

        // Fence GPU to wait until all BLASs are finished before building the TLASs for them
        auto uavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
        pCL->ResourceBarrier(1, &uavBarrier);
        pCL->BuildRaytracingAccelerationStructure(&m_TLAS_Combined_DebugShowBoxes.buildDesc,0,nullptr);
        pCL->BuildRaytracingAccelerationStructure(&m_TLAS_Combined_Single_BLAS_CHS.buildDesc,0,nullptr);
        pCL->BuildRaytracingAccelerationStructure(&m_TLAS_Combined_Single_BLAS_AnyHit.buildDesc,0,nullptr);
        pCL->BuildRaytracingAccelerationStructure(&m_TLAS_Combined_Single_AABB_Intersection.buildDesc,0,nullptr);
        pCL->BuildRaytracingAccelerationStructure(&m_TLAS_Combined_Multiple_BLAS_CHS.buildDesc,0,nullptr);
        pCL->BuildRaytracingAccelerationStructure(&m_TLAS_Combined_Multiple_AABB_Intersection.buildDesc,0,nullptr);
        pCL->BuildRaytracingAccelerationStructure(&m_TLAS_ModelsOnly.buildDesc,0,nullptr);
        pCL->BuildRaytracingAccelerationStructure(&m_TLAS_DecalsOnly_Single_BLAS_CHS.buildDesc,0,nullptr);
        pCL->BuildRaytracingAccelerationStructure(&m_TLAS_DecalsOnly_Single_BLAS_AnyHit.buildDesc,0,nullptr);
        pCL->BuildRaytracingAccelerationStructure(&m_TLAS_DecalsOnly_Single_AABB_Intersection.buildDesc,0,nullptr);
        pCL->BuildRaytracingAccelerationStructure(&m_TLAS_DecalsOnly_Multiple_BLAS_CHS.buildDesc,0,nullptr);
        pCL->BuildRaytracingAccelerationStructure(&m_TLAS_DecalsOnly_Multiple_AABB_Intersection.buildDesc,0,nullptr);
        gfxContext.Finish(true);

        InitializeRaytracingPSO(m_Model, numMeshes);

        float modelRadius = Length(m_Model.m_Header.boundingBox.max - m_Model.m_Header.boundingBox.min) * .5f;
        const Vector3 eye = (m_Model.m_Header.boundingBox.min + m_Model.m_Header.boundingBox.max) * .5f + Vector3(modelRadius * .25f, 0.0f, 0.0f);
        const Vector3 at = Vector3(0,eye.GetY(),0);
        m_Camera.SetEyeAtUp( eye, at, Vector3(kYUnitVector) );
        m_Camera.SetZRange( 1.0f, 10000.0f );
        m_CameraController.reset(new CameraController(m_Camera, Vector3(kYUnitVector)));

        //MotionBlur::Enable = false;//true;
        //TemporalEffects::EnableTAA = false;//true;
        //FXAA::Enable = false;
        //PostEffects::EnableHDR = true;
        PostEffects::Exposure = 10.0f;
        PostEffects::EnableAdaptation = false;//true;
        PostEffects::BloomEnable = false;//true;
        //SSAO::Enable = true;
    }

    virtual void Cleanup(void) override
    {
        m_Model.Clear();
    }

    virtual void Update(float deltaT) override
    {
        ScopedTimer _prof(L"Update State");

        m_CameraController->Update(deltaT);

        m_ViewProjMatrix = m_Camera.GetViewProjMatrix();

        float costheta = cosf(m_SunOrientation);
        float sintheta = sinf(m_SunOrientation);
        float cosphi = cosf(m_SunInclination * 3.14159f * 0.5f);
        float sinphi = sinf(m_SunInclination * 3.14159f * 0.5f);
        m_SunDirection = Normalize(Vector3( costheta * cosphi, sinphi, sintheta * cosphi ));
    }

    virtual void RenderScene(void) override
    {
        // Determine the ray tracing TLASs to use for this frame
        ASINFO *sceneTLAS=nullptr,*decalsTLAS=nullptr;
        if (m_DecalTraceMode == DTM_NoDecals)
        {
            decalsTLAS = sceneTLAS = &m_TLAS_ModelsOnly;
        }
        else if (m_DecalTraceMode == DTM_DebugShowBoxes)
        {
            decalsTLAS = sceneTLAS = &m_TLAS_Combined_DebugShowBoxes;
        }
        else if (m_DedicatedDecalsTLAS)
        {
            sceneTLAS = &m_TLAS_ModelsOnly;
            switch (m_DecalTraceMode)
            {
            case DTM_SingleBLASClosestHitLoop:
                decalsTLAS = &m_TLAS_DecalsOnly_Single_BLAS_CHS;
                break;
            case DTM_SingleBLASAnyHit:
                decalsTLAS = &m_TLAS_DecalsOnly_Single_BLAS_AnyHit;
                break;
            case DTM_SingleAABBIntersection:
                decalsTLAS = &m_TLAS_DecalsOnly_Single_AABB_Intersection;
                break;
            case DTM_MultipleBLASClosestHitLoop:
                decalsTLAS = &m_TLAS_DecalsOnly_Multiple_BLAS_CHS;
                break;
            case DTM_MultipleAABBIntersection:
                decalsTLAS = &m_TLAS_DecalsOnly_Multiple_AABB_Intersection;
                break;
            }
        }
        else
        {
            switch (m_DecalTraceMode)
            {
            case DTM_SingleBLASClosestHitLoop:
                decalsTLAS = sceneTLAS = &m_TLAS_Combined_Single_BLAS_CHS;
                break;
            case DTM_SingleBLASAnyHit:
                decalsTLAS = sceneTLAS = &m_TLAS_Combined_Single_BLAS_AnyHit;
                break;
            case DTM_SingleAABBIntersection:
                decalsTLAS = sceneTLAS = &m_TLAS_Combined_Single_AABB_Intersection;
                break;
            case DTM_MultipleBLASClosestHitLoop:
                decalsTLAS = sceneTLAS = &m_TLAS_Combined_Multiple_BLAS_CHS;
                break;
            case DTM_MultipleAABBIntersection:
                decalsTLAS = sceneTLAS = &m_TLAS_Combined_Multiple_AABB_Intersection;
                break;
            }
        }


        GraphicsContext& gfxContext = GraphicsContext::Begin(L"Scene Render");
        {
            ScopedTimer _prof(L"Raytrace", gfxContext);

            Raytrace(gfxContext, m_Camera, g_SceneColorBuffer, *sceneTLAS, *decalsTLAS);

            // Clear the gfxContext's descriptor heap since ray tracing changes this underneath the sheets
            gfxContext.SetDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, nullptr);
        }

        gfxContext.Finish();
    }

    virtual void RenderUI(GraphicsContext& gfxContext) override
    {
        const UINT framesToAverage = 20;
        static float frameRates[framesToAverage] = {};
        frameRates[Graphics::GetFrameCount() % framesToAverage] = Graphics::GetFrameRate();
        float rollingAverageFrameRate = 0.0;
        for (auto frameRate : frameRates)
            rollingAverageFrameRate += frameRate / framesToAverage;

        float primaryRaysPerSec = g_SceneColorBuffer.GetWidth() * g_SceneColorBuffer.GetHeight() * rollingAverageFrameRate / (1000000.0f);
        int decalCount = (int)m_Decals.size();
        TextContext text(gfxContext);
        text.Begin();
        text.DrawFormattedString("\n[Esc] exit. [Backspace] tuning menu. %d Decals. Million Primary Rays/s: %7.3f.", decalCount, primaryRaysPerSec);
        text.End();
    }
};

int wmain(int argc, wchar_t** argv)
{
    #if _DEBUG
    CComPtr<ID3D12Debug> debugInterface;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugInterface))))
    {
        debugInterface->EnableDebugLayer();
        debugInterface = nullptr;
    }
    #endif

    CComPtr<IDXGIFactory2> pFactory;
    CreateDXGIFactory2(0, IID_PPV_ARGS(&pFactory));
    CComPtr<IDXGIAdapter1> pAdapter;
    bool validDeviceFound = false;
    for (uint32_t Idx = 0; !validDeviceFound && DXGI_ERROR_NOT_FOUND != pFactory->EnumAdapters1(Idx, &pAdapter); ++Idx)
    {
        DXGI_ADAPTER_DESC1 desc;
        pAdapter->GetDesc1(&desc);

        ComPtr<ID3D12Device> testDevice;
        D3D12_FEATURE_DATA_D3D12_OPTIONS5 featureSupportData = {};

        validDeviceFound = SUCCEEDED(D3D12CreateDevice(pAdapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&testDevice)))
            && SUCCEEDED(testDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &featureSupportData, sizeof(featureSupportData)))
            && featureSupportData.RaytracingTier != D3D12_RAYTRACING_TIER_NOT_SUPPORTED;
        pAdapter = nullptr;
    }
    pFactory = nullptr;

    s_EnableVSync.Decrement();
    TargetResolution = k720p;
    g_DisplayWidth = 1280;
    g_DisplayHeight = 720;
    GameCore::RunApplication(D3D12RaytracedDecalsSample(), L"D3D12RaytracedDecalsSample"); 
    return 0;
}