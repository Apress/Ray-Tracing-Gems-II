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

#include "Common.h"
#include "..\shaders\shared.h"
#include "Scene.h"

//--------------------------------------------------------------------------------------
// Global Structures
//--------------------------------------------------------------------------------------

struct ConfigInfo 
{
	int				width = 1280;
	int				height = 720;
	bool			vsync = false;
	HINSTANCE		instance = NULL;
	std::string		scenePath = "";
	std::string		sceneFile = "";
};

//--------------------------------------------------------------------------------------
// D3D12
//--------------------------------------------------------------------------------------

struct D3D12BufferCreateInfo
{
	UINT64 size = 0;
	UINT64 alignment = 0;
	D3D12_HEAP_TYPE heapType = D3D12_HEAP_TYPE_DEFAULT;
	D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE;
	D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;

	D3D12BufferCreateInfo() {}

	D3D12BufferCreateInfo(UINT64 InSize, D3D12_RESOURCE_FLAGS InFlags) : size(InSize), flags(InFlags) {}

	D3D12BufferCreateInfo(UINT64 InSize, D3D12_HEAP_TYPE InHeapType, D3D12_RESOURCE_STATES InState) :
		size(InSize),
		heapType(InHeapType),
		state(InState) {}

	D3D12BufferCreateInfo(UINT64 InSize, D3D12_RESOURCE_FLAGS InFlags, D3D12_RESOURCE_STATES InState) :
		size(InSize),
		flags(InFlags),
		state(InState) {}

	D3D12BufferCreateInfo(UINT64 InSize, UINT64 InAlignment, D3D12_HEAP_TYPE InHeapType, D3D12_RESOURCE_FLAGS InFlags, D3D12_RESOURCE_STATES InState) :
		size(InSize),
		alignment(InAlignment),
		heapType(InHeapType),
		flags(InFlags),
		state(InState) {}
};

struct D3D12ShaderCompilerInfo 
{
	dxc::DxcDllSupport		DxcDllHelper;
	IDxcCompiler*			compiler = nullptr;
	IDxcLibrary*			library = nullptr;
};

struct D3D12ShaderInfo 
{
	LPCWSTR		filename = nullptr;
	LPCWSTR		entryPoint = nullptr;
	LPCWSTR		targetProfile = nullptr;
	DxcDefine*	defines = nullptr;
	UINT32		defineCount = 0;

	D3D12ShaderInfo() {}
	D3D12ShaderInfo(LPCWSTR inFilename, LPCWSTR inEntryPoint, LPCWSTR inProfile)
	{
		filename = inFilename;
		entryPoint = inEntryPoint;
		targetProfile = inProfile;
	}
};

struct D3D12Resources 
{
	ID3D12Resource*									DXROutput;
	ID3D12Resource*									accumulationBuffer;
	ID3D12Resource*									materialsBuffer;

	std::vector<ID3D12Resource*>					sceneVBs;
	std::vector<ID3D12Resource*>					sceneIBs;
	std::vector<D3D12_VERTEX_BUFFER_VIEW>			sceneVBViews;
	std::vector<D3D12_INDEX_BUFFER_VIEW>			sceneIBViews;
	std::vector<ID3D12Resource*>					sceneTextures;

	ID3D12Resource*									raytracingDataCB = nullptr;
	ID3D12Resource*									raytracingDataCBUpload = nullptr;
	RaytracingData									raytracingData;
	UINT											raytracingDataCBSize = 0;

	ID3D12DescriptorHeap*							rtvHeap = nullptr;
	ID3D12DescriptorHeap*							descriptorHeap = nullptr;

	UINT											rtvDescSize = 0;
	UINT											cbvSrvUavDescSize = 0;

	Scene											scene;
};

struct D3D12Global
{
	IDXGIFactory6*									factory = nullptr;
	IDXGIAdapter1*									adapter = nullptr;
	ID3D12Device5*									device = nullptr;
	ID3D12GraphicsCommandList4*						cmdList = nullptr;
	ID3D12CommandQueue*								cmdQueue = nullptr;
	ID3D12CommandAllocator*							cmdAlloc[2] = { nullptr, nullptr };

	IDXGISwapChain3*								swapChain = nullptr;
	ID3D12Resource*									backBuffer[2] = { nullptr, nullptr };

	ID3D12Fence*									fence = nullptr;
	UINT64											fenceValues[2] = { 0, 0 };
	HANDLE											fenceEvent;
	UINT											frameIndex = 0;
	bool											isTearingSupport = false;
	ID3D12Debug*									debugController = nullptr;

	int												width = 1280;
	int												height = 720;
	bool											vsync = false;

	bool											renderGui = true;
	std::vector<ID3D12Resource*>					temporaryBuffers;
};

class LinearGPUAllocator {
public:

	void Initialize(D3D12Global& d3d, UINT64 initialSize, UINT64 alignment = 0, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE heapType = D3D12_HEAP_TYPE_DEFAULT);
	D3D12_GPU_VIRTUAL_ADDRESS Allocate(D3D12Global& d3d, UINT64 size, ID3D12Resource*& resource, UINT64& offset);
	D3D12_GPU_VIRTUAL_ADDRESS Allocate(D3D12Global& d3d, UINT64 size);
	void Reset();
	void Destroy();

private:

	void AllocateInternal(D3D12Global& d3d, UINT64 size);
	bool isInitialized = false;
	D3D12BufferCreateInfo bufferInfo;
	ID3D12Resource* currentBlock = nullptr;
	UINT64 currentOffset = 0;
	std::vector<ID3D12Resource*> obsoleteBlocks;
};

//--------------------------------------------------------------------------------------
//  DXR
//--------------------------------------------------------------------------------------

struct AccelerationStructureBuffer
{
	ID3D12Resource* pResult = nullptr;
	ID3D12Resource* pInstanceDesc = nullptr;	// only used in top-level AS

	void Release()
	{
		SAFE_RELEASE(pResult);
		SAFE_RELEASE(pInstanceDesc);
	}
};

struct RtProgram
{
	D3D12ShaderInfo			info = {};
	IDxcBlob*				blob = nullptr;

	RtProgram() {}

	RtProgram(D3D12ShaderInfo shaderInfo)
	{
		info = shaderInfo;
	}
};

struct DXRGlobal
{
	Camera											camera;
	float											cameraSpeedAdjustment = 1.0f;

	AccelerationStructureBuffer						TLAS;
	std::vector<AccelerationStructureBuffer>		BLASes;
	uint64_t										tlasSize;
	LinearGPUAllocator								scratchBuffersCache;

	ID3D12RootSignature*							globalRootSignature = nullptr;

	ID3D12Resource*									shaderTable = nullptr;
	uint32_t										shaderTableRecordSize = 0;
	std::vector<uint8_t>							shaderTableData;

	RtProgram										rtProgram;
	
	ID3D12StateObject*								rtpso = nullptr;
	ID3D12StateObjectProperties*					rtpsoInfo = nullptr;

	uint32_t										frameNumber = 0;
	uint32_t										accumulatedFrames = 0;
	int												maxBounces = 8;
	bool											enableAntiAliasing = true; 
	float											exposureAdjustment = 0.8f;
	float											skyIntensity = 3.0f;
	bool											enableAccumulation = true;
	bool											enableDirectLighting = true;
	DirectX::XMMATRIX								lastView;
	bool											enableSun = true;
	bool											enableHeadlight = false;
	float											sunIntensity = 1.0f;
	float											headlightIntensity = 10.0f;
	float											focusDistance = 10.0f;
	float											apertureSize = 0.0f;
	bool											forceAccumulationReset = false;
	float											sunAzimuth = 295.0f;
	float											sunElevation = 78.0f;
};
