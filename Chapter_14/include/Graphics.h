/* Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "Structures.h"
#include "Gui.h"
#include "Input.h"

static const D3D12_HEAP_PROPERTIES UploadHeapProperties =
{
	D3D12_HEAP_TYPE_UPLOAD,
	D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
	D3D12_MEMORY_POOL_UNKNOWN,
	0, 0
};

static const D3D12_HEAP_PROPERTIES DefaultHeapProperties =
{
	D3D12_HEAP_TYPE_DEFAULT,
	D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
	D3D12_MEMORY_POOL_UNKNOWN,
	0, 0
};

static const D3D12_HEAP_PROPERTIES ReadbackHeapProperties =
{
	D3D12_HEAP_TYPE_READBACK,
	D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
	D3D12_MEMORY_POOL_UNKNOWN,
	0, 0
};

namespace D3DResources
{
	void CreateBuffer(D3D12Global &d3d, D3D12BufferCreateInfo &info, ID3D12Resource** ppResource, std::wstring debugName = L"");
	void CreateGeometryBuffers(D3D12Global &d3d, D3D12Resources &resources, Scene &scene);
	void CreateTextures(D3D12Global& d3d, D3D12Resources& resources, const Scene& scene);
	void CreateBackBufferRTV(D3D12Global &d3d, D3D12Resources &resources);
	void CreateRaytracingDataCB(D3D12Global &d3d, D3D12Resources &resources);
	void CreateDescriptorHeaps(D3D12Global &d3d, D3D12Resources &resources);
	void UploadToGPU(D3D12Global& d3d, const void* bufferData, ID3D12Resource* targetBuffer, D3D12_RESOURCE_STATES targetState, std::vector<D3D12_SUBRESOURCE_DATA>* subresourcesData = nullptr);
	void UploadToGPU(D3D12Global& d3d, const Texture* texture, ID3D12Resource* targetBuffer, D3D12_RESOURCE_STATES targetState);
	void ReleaseTemporaryBuffers(D3D12Global& d3d, D3D12Resources& resources);

	void UpdateRaytracingDataCB(D3D12Global &d3d, DXRGlobal& dxr, D3D12Resources &resources, float elapsedTime);

	void Destroy(D3D12Resources &resources);
}

namespace D3DShaders
{
	void InitShaderCompiler(D3D12ShaderCompilerInfo &shaderCompiler);
	void CompileShader(D3D12ShaderCompilerInfo &compilerInfo, RtProgram &program);
	void CompileShader(D3D12ShaderCompilerInfo &compilerInfo, D3D12ShaderInfo &info, IDxcBlob** blob);
	void Destroy(D3D12ShaderCompilerInfo &shaderCompiler);
}

namespace D3D12
{	
	void CreateDevice(D3D12Global &d3d);
	void CreateCommandList(D3D12Global &d3d);
	void CreateCommandQueue(D3D12Global &d3d);
	void CreateCommandAllocator(D3D12Global &d3d);
	void CreateCommandList(D3D12Global &d3d);
	void CreateFence(D3D12Global &d3d);
	void CreateSwapChain(D3D12Global &d3d, HWND &window);

	ID3D12RootSignature* CreateRootSignature(D3D12Global &d3d, const D3D12_ROOT_SIGNATURE_DESC &desc, std::wstring debugName = L"");

	void ResetCommandList(D3D12Global &d3d);
	void SubmitCommandList(D3D12Global &d3d);
	void Present(D3D12Global &d3d);
	void WaitForGPU(D3D12Global &d3d);
	void MoveToNextFrame(D3D12Global &d3d);
	bool ScreenCapture(D3D12Global& d3d, std::string filename);

	void Destroy(D3D12Global &d3d);
}

namespace DXR
{	
	void CreateBottomLevelAS(D3D12Global &d3d, DXRGlobal &dxr, D3D12Resources &resources, Scene& scene);
	void CreateTopLevelAS(D3D12Global &d3d, DXRGlobal &dxr, D3D12Resources &resources, Scene& scene);
	void CreateRTProgram(D3D12Global &d3d, DXRGlobal &dxr, D3D12ShaderCompilerInfo &shaderCompiler);
	void CreatePipelineStateObject(D3D12Global &d3d, DXRGlobal &dxr);
	void CreateShaderTable(D3D12Global &d3d, DXRGlobal &dxr, D3D12Resources &resources);
	void CreateDescriptorHeaps(D3D12Global& d3d, DXRGlobal& dxr, D3D12Resources& resources);
	void CreateDXRResources(D3D12Global& d3d, DXRGlobal& dxr, D3D12Resources& resources);
	void FillDescriptorHeaps(D3D12Global &d3d, DXRGlobal &dxr, D3D12Resources &resources);

	void ReloadShaders(D3D12Global& d3d, DXRGlobal& dxr, D3D12Resources& resources, D3D12ShaderCompilerInfo& shaderCompiler);

	void BuildCommandList(D3D12Global &d3d, DXRGlobal &dxr, D3D12Resources &resources, Gui* gui, InputInfo* input);

	void Destroy(DXRGlobal &dxr);
}

/**
* This enum specifies a layout of resources in ray tracing shaders
*/
enum class DescriptorHeapConstants {

	// List of resources declared in the shader, as they appear in the descriptors heap
	RtDataCB = 0,
	RTOutput,
	AccumulationBuffer,
	SceneTLAS,
	MaterialsBuffer,
	IndexBuffers,
	VertexBuffers = IndexBuffers + MAX_INSTANCES_COUNT,
	Textures = VertexBuffers + MAX_INSTANCES_COUNT,
	Total = Textures + MAX_TEXTURES_COUNT,

	// Constant buffer range
	CBStart = RtDataCB,
	CBEnd = RtDataCB,
	CBTotal = CBEnd - CBStart + 1,

	// UAV space 0 range
	UAV0Start = RTOutput,
	UAV0End = AccumulationBuffer,
	UAV0Total = UAV0End - UAV0Start + 1,

	// SRV space 0 range
	SRV0Start = SceneTLAS,
	SRV0End = MaterialsBuffer,
	SRV0Total = SRV0End - SRV0Start + 1,

	// SRV space 1 range
	SRV1Start = IndexBuffers,
	SRV1End = VertexBuffers - 1,
	SRV1Total = SRV1End - SRV1Start + 1,

	// SRV space 2 range
	SRV2Start = VertexBuffers,
	SRV2End = Textures - 1,
	SRV2Total = SRV2End - SRV2Start + 1,

	// SRV space 3 range
	SRV3Start = Textures,
	SRV3End = Total - 1,
	SRV3Total = SRV3End - SRV3Start + 1,
};