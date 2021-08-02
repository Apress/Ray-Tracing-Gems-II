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

#include <wrl.h>
#include <atlcomcli.h>
#include <chrono>
#include <wincodec.h>
#include <dxgidebug.h>

#include "Graphics.h"
#include "Utils.h"
#include "thirdparty/d3dx12/d3dx12.h"
#include "thirdparty/directxtk/ScreenGrab12.h"

using namespace std;
using namespace DirectX;
using Microsoft::WRL::ComPtr;

//--------------------------------------------------------------------------------------
// Resource Functions
//--------------------------------------------------------------------------------------

namespace D3DResources
{

/**
* Create a GPU buffer resource.
*/
void CreateBuffer(D3D12Global &d3d, D3D12BufferCreateInfo& info, ID3D12Resource** ppResource, std::wstring debugName)
{
	D3D12_HEAP_PROPERTIES heapDesc = {};
	heapDesc.Type = info.heapType;
	heapDesc.CreationNodeMask = 1;
	heapDesc.VisibleNodeMask = 1;

	D3D12_RESOURCE_DESC resourceDesc = {};
	resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
	resourceDesc.Alignment = info.alignment;
	resourceDesc.Height = 1;
	resourceDesc.DepthOrArraySize = 1;
	resourceDesc.MipLevels = 1;
	resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
	resourceDesc.SampleDesc.Count = 1;
	resourceDesc.SampleDesc.Quality = 0;
	resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
	resourceDesc.Width = info.size;
	resourceDesc.Flags = info.flags;

	// Create the GPU resource
	HRESULT hr = d3d.device->CreateCommittedResource(&heapDesc, D3D12_HEAP_FLAG_NONE, &resourceDesc, info.state, nullptr, IID_PPV_ARGS(ppResource));
	Utils::Validate(hr, L"Error: failed to create buffer resource!");

#if NAME_D3D_RESOURCES
	(*ppResource)->SetName(debugName.c_str());
#endif
}

/**
* Uploads data into target buffer in the device memory using upload heap
*/
void UploadToGPU(D3D12Global& d3d, const void* bufferData, ID3D12Resource* targetBuffer, D3D12_RESOURCE_STATES targetState, std::vector<D3D12_SUBRESOURCE_DATA>* subresourcesData) {

	// Figure out size required for the upload buffer
	const bool subresourcesSpecified = (subresourcesData != nullptr);
	UINT subresourceCount = (subresourcesSpecified ? UINT(subresourcesData->size()) : 1);
	UINT64 requiredSize = GetRequiredIntermediateSize(targetBuffer, 0, subresourceCount);

	// Align buffer for texture uploads (this may not be necessary for other types of buffers)
	UINT64 uploadBufferSize = ALIGN(D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT, requiredSize);

	// Allocate temporary buffer on the upload heap
	ID3D12Resource* uploadBuffer = nullptr;
	D3D12BufferCreateInfo infoUpload(uploadBufferSize, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
	D3DResources::CreateBuffer(d3d, infoUpload, &uploadBuffer, L"Upload Buffer " + std::to_wstring(d3d.temporaryBuffers.size()));

	// Remember this buffer so we can deallocate it after use
	d3d.temporaryBuffers.push_back(uploadBuffer);

	// Build description of subresource data or use the one provided
	D3D12_SUBRESOURCE_DATA* bufferDataDescPtr = nullptr;

	if (!subresourcesSpecified) {
		D3D12_SUBRESOURCE_DATA bufferDataDesc = {};
		bufferDataDesc.pData = bufferData;
		bufferDataDesc.RowPitch = uploadBufferSize;
		bufferDataDesc.SlicePitch = bufferDataDesc.RowPitch;
		bufferDataDescPtr = &bufferDataDesc;
	} else {
		bufferDataDescPtr = subresourcesData->data();
	}

	// Schedule a copy from the upload heap to the device memory
	UINT64 uploadedBytes = UpdateSubresources(d3d.cmdList, targetBuffer, uploadBuffer, 0, 0, subresourceCount, bufferDataDescPtr);
	HRESULT hr = (requiredSize == uploadedBytes ? S_OK : E_FAIL);
	Utils::Validate(hr, L"Error: failed to upload data via upload heap!");

	// Transition resource to target state
	D3D12_RESOURCE_BARRIER barrier = {};
	barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barrier.Transition.pResource = targetBuffer;
	barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
	barrier.Transition.StateAfter = targetState;
	barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	d3d.cmdList->ResourceBarrier(1, &barrier);
}

/**
* Uploads texture with mips into target buffer in the device memory using upload heap
*/
void UploadToGPU(D3D12Global& d3d, const Texture* texture, ID3D12Resource* targetBuffer, D3D12_RESOURCE_STATES targetState) {

	// Construct subresource data for all mips
	std::vector<D3D12_SUBRESOURCE_DATA> subresourcesData;
	for (UINT i = 0; i < UINT(texture->mips.size()); i++) {
		D3D12_SUBRESOURCE_DATA textureData = {};
		textureData.pData = texture->mips[i].texels;
		textureData.RowPitch = texture->mips[i].rowPitch;
		textureData.SlicePitch = texture->mips[i].texelBytes;
		subresourcesData.push_back(textureData);
	}

	UploadToGPU(d3d, nullptr, targetBuffer, targetState, &subresourcesData);
}

/*
* Deallocates temporary buffers used for uploading to GPU memory
*/
void ReleaseTemporaryBuffers(D3D12Global& d3d, D3D12Resources& resources) {
	for (ID3D12Resource* resource : d3d.temporaryBuffers) {
		SAFE_RELEASE(resource);
	}
	d3d.temporaryBuffers.clear();
}

/*
* Create geometry vertex and index buffers
*/
void CreateGeometryBuffers(D3D12Global &d3d, D3D12Resources &resources, Scene &scene) 
{
	resources.sceneVBs.resize(scene.numGeometries);
	resources.sceneVBViews.resize(scene.numGeometries);
	resources.sceneIBs.resize(scene.numGeometries);
	resources.sceneIBViews.resize(scene.numGeometries);

	for (size_t meshIndex = 0; meshIndex < scene.meshes.size(); meshIndex++)
	{
		// Get the mesh
		const Mesh mesh = scene.meshes[meshIndex];
		for (size_t primitiveIndex = 0; primitiveIndex < mesh.primitives.size(); primitiveIndex++)
		{
			// Get the mesh primitive
			const MeshPrimitive primitive = mesh.primitives[primitiveIndex];
			Utils::Validate(primitive.index >= MAX_INSTANCES_COUNT ? E_FAIL : S_OK, L"Error: too many meshes!");

			// Create a string name for this mesh
			std::wstring meshName;
			meshName.append(mesh.name.empty() ? L"<unnamed>" : std::wstring(mesh.name.begin(), mesh.name.end()));
			meshName.append(L", ID: ");
			meshName.append(std::to_wstring(meshIndex));
			meshName.append(L", Primitive: ");
			meshName.append(std::to_wstring(primitiveIndex));

			// Create the vertex buffer resource
			{
				// Allocate the vertex buffer and upload data to GPU
				D3D12BufferCreateInfo info(((UINT)primitive.vertices.size() * sizeof(Vertex)), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
				CreateBuffer(d3d, info, &resources.sceneVBs[primitive.index], L"VB: " + meshName);
				UploadToGPU(d3d, primitive.vertices.data(), resources.sceneVBs[primitive.index], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

				// Initialize the vertex buffer view
				resources.sceneVBViews[primitive.index].BufferLocation = resources.sceneVBs[primitive.index]->GetGPUVirtualAddress();
				resources.sceneVBViews[primitive.index].StrideInBytes = sizeof(Vertex);
				resources.sceneVBViews[primitive.index].SizeInBytes = static_cast<UINT>(info.size);

				// Create the vertex buffer SRV
				D3D12_SHADER_RESOURCE_VIEW_DESC vertexSRVDesc;
				vertexSRVDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
				vertexSRVDesc.Format = DXGI_FORMAT_R32_TYPELESS;
				vertexSRVDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
				vertexSRVDesc.Buffer.StructureByteStride = 0;
				vertexSRVDesc.Buffer.FirstElement = 0;
				vertexSRVDesc.Buffer.NumElements = (static_cast<UINT>(primitive.vertices.size()) * sizeof(Vertex)) / sizeof(float);
				vertexSRVDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

				D3D12_CPU_DESCRIPTOR_HANDLE handle = resources.descriptorHeap->GetCPUDescriptorHandleForHeapStart();
				handle.ptr += resources.cbvSrvUavDescSize * (SIZE_T(DescriptorHeapConstants::VertexBuffers) + primitive.index);
				d3d.device->CreateShaderResourceView(resources.sceneVBs[primitive.index], &vertexSRVDesc, handle);
			}

			// Create the index buffer resource
			{
				// Allocate the index buffer and upload data to GPU
				D3D12BufferCreateInfo info(((UINT)primitive.indices.size() * sizeof(UINT)), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
				CreateBuffer(d3d, info, &resources.sceneIBs[primitive.index], L"IB: " + meshName);
				UploadToGPU(d3d, primitive.indices.data(), resources.sceneIBs[primitive.index], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

				// Initialize the index buffer view
				resources.sceneIBViews[primitive.index].BufferLocation = resources.sceneIBs[primitive.index]->GetGPUVirtualAddress();
				resources.sceneIBViews[primitive.index].SizeInBytes = static_cast<UINT>(info.size);
				resources.sceneIBViews[primitive.index].Format = DXGI_FORMAT_R32_UINT;

				// Create the index buffer SRV
				D3D12_SHADER_RESOURCE_VIEW_DESC indexSRVDesc;
				indexSRVDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
				indexSRVDesc.Format = DXGI_FORMAT_R32_TYPELESS;
				indexSRVDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
				indexSRVDesc.Buffer.StructureByteStride = 0;
				indexSRVDesc.Buffer.FirstElement = 0;
				indexSRVDesc.Buffer.NumElements = (static_cast<UINT>(primitive.indices.size()) * sizeof(UINT)) / sizeof(float);
				indexSRVDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

				D3D12_CPU_DESCRIPTOR_HANDLE handle = resources.descriptorHeap->GetCPUDescriptorHandleForHeapStart();
				handle.ptr += resources.cbvSrvUavDescSize * (SIZE_T(DescriptorHeapConstants::IndexBuffers) + primitive.index);
				d3d.device->CreateShaderResourceView(resources.sceneIBs[primitive.index], &indexSRVDesc, handle);
			}
		}
	}

	// Create the material buffer
	{
		Utils::Validate(scene.materials.size() > MAX_MATERIALS_COUNT ? E_FAIL : S_OK, L"Error: too many materials!");

		// Prepare a linear buffer of all materials
		std::vector<MaterialData> materials;
		materials.resize(scene.materials.size());

		for (size_t materialIndex = 0; materialIndex < scene.materials.size(); materialIndex++)
		{
			materials[materialIndex] = scene.materials[materialIndex].data;
		}

		// Allocate the material buffer and upload data to GPU
		D3D12BufferCreateInfo materialsBufferInfo(scene.materials.size() * sizeof(MaterialData), D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
		CreateBuffer(d3d, materialsBufferInfo, &resources.materialsBuffer, L"Materials Buffer");
		UploadToGPU(d3d, materials.data(), resources.materialsBuffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

		// Create the material buffer SRV
		D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
		srvDesc.Format = DXGI_FORMAT_UNKNOWN;
		srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		srvDesc.Buffer.FirstElement = 0;
		srvDesc.Buffer.NumElements = static_cast<UINT>(scene.materials.size());
		srvDesc.Buffer.StructureByteStride = sizeof(MaterialData);
		srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
		srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

		D3D12_CPU_DESCRIPTOR_HANDLE handle = resources.descriptorHeap->GetCPUDescriptorHandleForHeapStart();
		handle.ptr += resources.cbvSrvUavDescSize * SIZE_T(DescriptorHeapConstants::MaterialsBuffer);
		d3d.device->CreateShaderResourceView(resources.materialsBuffer, &srvDesc, handle);
	}
}

/**
* Create the textures.
*/
void CreateTextures(D3D12Global& d3d, D3D12Resources& resources, const Scene& scene)
{
	// Early out if there are no scene textures
	if (scene.textures.size() == 0) return;
	Utils::Validate(scene.textures.size() > MAX_TEXTURES_COUNT ? E_FAIL : S_OK, L"Error: too many textures!");

	// Prepare a handle where slots for texture views start in the descriptor heap
	D3D12_CPU_DESCRIPTOR_HANDLE handle = resources.descriptorHeap->GetCPUDescriptorHandleForHeapStart();
	handle.ptr += (resources.cbvSrvUavDescSize * SIZE_T(DescriptorHeapConstants::Textures));

	// Create the texture resources
	resources.sceneTextures.resize(scene.textures.size());
	for (UINT textureIndex = 0; textureIndex < scene.textures.size(); textureIndex++)
	{
		// Get the texture
		const Texture texture = scene.textures[textureIndex];
		DXGI_FORMAT textureFormat = DXGI_FORMAT_UNKNOWN;

		if (texture.mips.empty()) continue;

		// Create a string name for this texture
		std::wstring name = L"Texture: ";
		name.append(texture.name.empty() ? L"<unnamed>" : std::wstring(texture.name.begin(), texture.name.end()));
		name.append(L", ID: ");
		name.append(std::to_wstring(textureIndex));

		if (texture.isBC7Compressed) 
			textureFormat = texture.doSrgb ? DXGI_FORMAT_BC7_UNORM_SRGB : DXGI_FORMAT_BC7_UNORM;
		else
			textureFormat = texture.doSrgb ? DXGI_FORMAT_R8G8B8A8_UNORM_SRGB : DXGI_FORMAT_R8G8B8A8_UNORM;

		// Create the texture resource and upload texture to GPU
		D3D12_RESOURCE_DESC desc = {};
		desc.Format = textureFormat;
		desc.Width = texture.mips[0].width;
		desc.Height = texture.mips[0].height;
		desc.MipLevels = UINT16(texture.mips.size());
		desc.DepthOrArraySize = 1;
		desc.SampleDesc.Count = 1;
		desc.SampleDesc.Quality = 0;
		desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
		desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
		desc.Flags = D3D12_RESOURCE_FLAG_NONE;

		HRESULT hr = d3d.device->CreateCommittedResource(&DefaultHeapProperties, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&resources.sceneTextures[textureIndex]));
		Utils::Validate(hr, L"Error: failed to create texture!");
		UploadToGPU(d3d, &texture, resources.sceneTextures[textureIndex], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

#if NAME_D3D_RESOURCES
		resources.sceneTextures[textureIndex]->SetName(name.c_str());
#endif

		// Create an SRV for the texture
		D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
		srvDesc.Format = textureFormat;
		srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
		srvDesc.Texture2D.MipLevels = UINT(texture.mips.size());
		srvDesc.Texture2D.MostDetailedMip = 0;
		srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		d3d.device->CreateShaderResourceView(resources.sceneTextures[textureIndex], &srvDesc, handle);

		// Increment the slot on the descriptor heap
		handle.ptr += resources.cbvSrvUavDescSize;
	}
}

/**
* Create the back buffer's RTV view.
*/
void CreateBackBufferRTV(D3D12Global &d3d, D3D12Resources &resources)
{
	HRESULT hr;
	D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle;

	rtvHandle = resources.rtvHeap->GetCPUDescriptorHandleForHeapStart();

	// Create a RTV for each back buffer
	for (UINT n = 0; n < 2; n++)
	{
		hr = d3d.swapChain->GetBuffer(n, IID_PPV_ARGS(&d3d.backBuffer[n]));
		Utils::Validate(hr, L"Error: failed to get swap chain buffer!");

		d3d.device->CreateRenderTargetView(d3d.backBuffer[n], nullptr, rtvHandle);

#if NAME_D3D_RESOURCES
		d3d.backBuffer[n]->SetName((L"Back Buffer " + std::to_wstring(n)).c_str());
#endif

		rtvHandle.ptr += resources.rtvDescSize;
	}
}

/**
* Create and initialize the constant buffer for raytracing data.
*/
void CreateRaytracingDataCB(D3D12Global &d3d, D3D12Resources &resources) 
{
	resources.raytracingDataCBSize = ALIGN(D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, sizeof(resources.raytracingData));

	// Allocate the buffer and upload data to GPU
	D3D12BufferCreateInfo raytracingDataCBBufferInfo(resources.raytracingDataCBSize, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	CreateBuffer(d3d, raytracingDataCBBufferInfo, &resources.raytracingDataCB);

	// Figure out size required for the upload buffer
	UINT64 uploadBufferSize = GetRequiredIntermediateSize(resources.raytracingDataCB, 0, 1);
	uploadBufferSize = ALIGN(D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT, uploadBufferSize);

	// Allocate buffer on the upload heap
	D3D12BufferCreateInfo infoUpload(uploadBufferSize, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
	D3DResources::CreateBuffer(d3d, infoUpload, &resources.raytracingDataCBUpload);

#if NAME_D3D_RESOURCES
	resources.raytracingDataCB->SetName(L"Raytracing Data Constant Buffer");
	resources.raytracingDataCBUpload->SetName(L"Raytracing Data Constant Upload Buffer");
#endif
}

/**
* Create the RTV descriptor heap.
*/
void CreateDescriptorHeaps(D3D12Global &d3d, D3D12Resources &resources)
{
	// Describe the RTV descriptor heap
	D3D12_DESCRIPTOR_HEAP_DESC rtvDesc = {};
	rtvDesc.NumDescriptors = 2;
	rtvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
	rtvDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

	// Create the RTV heap
	HRESULT hr = d3d.device->CreateDescriptorHeap(&rtvDesc, IID_PPV_ARGS(&resources.rtvHeap));
	Utils::Validate(hr, L"Error: failed to create RTV descriptor heap!");
#if NAME_D3D_RESOURCES
	resources.rtvHeap->SetName(L"RTV Descriptor Heap");
#endif

	resources.rtvDescSize = d3d.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
}

/**
* Update the constant buffer with raytracing data.
*/
void UpdateRaytracingDataCB(D3D12Global& d3d, DXRGlobal& dxr, D3D12Resources& resources, float elapsedTime)
{
	XMMATRIX view, invView;
	XMFLOAT3 eye, focus, up;

	// Construct the view matrix using camera data
	eye = dxr.camera.position;
	up = dxr.camera.up;
	XMStoreFloat3(&focus, XMVectorAdd(XMLoadFloat3(&dxr.camera.position), XMLoadFloat3(&dxr.camera.forward)));

	float fovRadians = XMConvertToRadians(dxr.camera.fov);
	view = XMMatrixLookAtLH(XMLoadFloat3(&eye), XMLoadFloat3(&focus), XMLoadFloat3(&up));
	invView = XMMatrixInverse(NULL, view);

	// Note: near and far plane settings have no effect in ray tracing, thanks to the way we construct primary rays
	float nearPlane = 0.001f;
	float farPlane = 100.00f;
	XMMATRIX projection = XMMatrixPerspectiveFovLH(fovRadians, dxr.camera.aspect, nearPlane, farPlane);
	
	// Set camera matrices
	resources.raytracingData.view = XMMatrixTranspose(invView);
	resources.raytracingData.proj = projection;
	
	// Prepare and set lights into constant buffer (headlight and sun)
	int lightCount = 0;
	if (dxr.enableHeadlight) {
		Light headLight;
		headLight.position = eye;
		headLight.position.y += 1.0f;
		headLight.intensity = XMFLOAT3(dxr.headlightIntensity, dxr.headlightIntensity, dxr.headlightIntensity);
		headLight.type = POINT_LIGHT;

		resources.raytracingData.lights[lightCount++] = headLight;
	}
	
	if (dxr.enableSun) {
		Light sun;

		XMFLOAT3 zAxis = { 0, 0, 1 };
		XMFLOAT3 yAxis = { 0, 1, 0 };
		sun.position = { 1, 0, 0 }; //< Store direction to sun in 'position' field
		
		XMStoreFloat3(&sun.position, XMVector3Rotate(XMLoadFloat3(&sun.position), XMQuaternionRotationAxis(XMLoadFloat3(&zAxis), XMConvertToRadians(dxr.sunElevation))));
		XMStoreFloat3(&sun.position, XMVector3Rotate(XMLoadFloat3(&sun.position), XMQuaternionRotationAxis(XMLoadFloat3(&yAxis), XMConvertToRadians(dxr.sunAzimuth))));

		sun.intensity = XMFLOAT3(dxr.sunIntensity, dxr.sunIntensity, dxr.sunIntensity);
		sun.type = DIRECTIONAL_LIGHT;

		resources.raytracingData.lights[lightCount++] = sun;
	}
	resources.raytracingData.lightCount = dxr.enableDirectLighting ? lightCount : 0;

	// Set other ray tracing variables
	resources.raytracingData.frameNumber = dxr.frameNumber;
	resources.raytracingData.maxBounces = dxr.maxBounces;
	resources.raytracingData.exposureAdjustment = dxr.exposureAdjustment;
	resources.raytracingData.skyIntensity = dxr.skyIntensity;
	resources.raytracingData.enableAntiAliasing = dxr.enableAntiAliasing;

	bool resetAccumulation = dxr.forceAccumulationReset || !dxr.enableAccumulation || memcmp(&dxr.lastView, &resources.raytracingData.view, sizeof(DirectX::XMMATRIX));
	dxr.lastView = resources.raytracingData.view;
	dxr.forceAccumulationReset = false;

	if (resetAccumulation) dxr.accumulatedFrames = 0;
	dxr.accumulatedFrames++;
	resources.raytracingData.enableAccumulation = !resetAccumulation && dxr.enableAccumulation;
	resources.raytracingData.accumulatedFrames = dxr.enableAccumulation ? dxr.accumulatedFrames : 1u;
	
	resources.raytracingData.focusDistance = dxr.focusDistance;
	resources.raytracingData.apertureSize = dxr.apertureSize;

	// Upload data to the constant buffer in GPU memory
	D3D12_SUBRESOURCE_DATA bufferDataDesc = {};
	bufferDataDesc.pData = &resources.raytracingData;
	bufferDataDesc.RowPitch = resources.raytracingDataCBSize;
	bufferDataDesc.SlicePitch = bufferDataDesc.RowPitch;

	// Transition resource for update
	D3D12_RESOURCE_BARRIER barrier = {};
	barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barrier.Transition.pResource = resources.raytracingDataCB;
	barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
	barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
	barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	d3d.cmdList->ResourceBarrier(1, &barrier);

	// Schedule a copy from the upload heap to the device memory
	UINT64 uploadedBytes = UpdateSubresources(d3d.cmdList, resources.raytracingDataCB, resources.raytracingDataCBUpload, 0, 0, 1, &bufferDataDesc);
	HRESULT hr = (resources.raytracingDataCBSize == uploadedBytes ? S_OK : E_FAIL);
	Utils::Validate(hr, L"Error: failed to update constant buffer!");

	// Transition resource for shader use
	barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
	barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
	d3d.cmdList->ResourceBarrier(1, &barrier);

	// Increment frame counter
	dxr.frameNumber++;
}

/**
 * Release the resources.
 */
void Destroy(D3D12Resources &resources)
{
	SAFE_RELEASE(resources.DXROutput);
	SAFE_RELEASE(resources.accumulationBuffer);
	SAFE_RELEASE(resources.raytracingDataCB);
	SAFE_RELEASE(resources.raytracingDataCBUpload);
	SAFE_RELEASE(resources.rtvHeap);
	SAFE_RELEASE(resources.descriptorHeap);
	SAFE_RELEASE(resources.materialsBuffer);
	
	// Release scene geometry
	size_t resourceIndex;
	for (resourceIndex = 0; resourceIndex < resources.sceneVBs.size(); resourceIndex++)
	{
		SAFE_RELEASE(resources.sceneVBs[resourceIndex]);
		SAFE_RELEASE(resources.sceneIBs[resourceIndex]);
	}
	resources.sceneVBs.clear();
	resources.sceneIBs.clear();

	// Release scene textures
	for (resourceIndex = 0; resourceIndex < resources.sceneTextures.size(); resourceIndex++)
	{
		SAFE_RELEASE(resources.sceneTextures[resourceIndex]);
	}
	resources.sceneTextures.clear();
}

}

//--------------------------------------------------------------------------------------
// D3D12 Shader Functions
//--------------------------------------------------------------------------------------

namespace D3DShaders
{

/**
* Compile an HLSL shader using dxcompiler.
*/
void CompileShader(D3D12ShaderCompilerInfo &compilerInfo, D3D12ShaderInfo &info, IDxcBlob** blob) 
{
	HRESULT hr;
	UINT32 code = 0;
	IDxcBlobEncoding* pShaderText = nullptr;
	IDxcOperationResult* result = nullptr;
	bool retryCompile = true;

	while (retryCompile) {

		// Load and encode the shader file
		hr = compilerInfo.library->CreateBlobFromFile(info.filename, &code, &pShaderText);
		Utils::Validate(hr, L"Error: failed to create blob from shader file!");
		if (FAILED(hr) || pShaderText == nullptr) return;

		// Create the compiler include handler
		CComPtr<IDxcIncludeHandler> dxcIncludeHandler;
		hr = compilerInfo.library->CreateIncludeHandler(&dxcIncludeHandler);
		Utils::Validate(hr, L"Error: failed to create include handler");

		// Additional compiler flags (always present)
		std::vector<std::wstring> compilerFlags;
		compilerFlags.push_back(L"/enable-16bit-types");

		// Process compiler flags to an array of char* pointers
		std::vector<LPCWSTR> arguments;
		for (size_t i = 0; i < compilerFlags.size(); i++) arguments.push_back(compilerFlags[i].c_str());

		// Compile the shader
		hr = compilerInfo.compiler->Compile(
			pShaderText,
			info.filename,
			info.entryPoint,
			info.targetProfile,
			arguments.data(),
			UINT(compilerFlags.size()),
			info.defines,
			info.defineCount,
			dxcIncludeHandler,
			&result);

		Utils::Validate(hr, L"Error: failed to compile shader!");

		// Verify the result
		result->GetStatus(&hr);
		if (FAILED(hr))
		{
			IDxcBlobEncoding* error;
			hr = result->GetErrorBuffer(&error);
			Utils::Validate(hr, L"Error: failed to get shader compiler error buffer!");

			// Convert error blob to a string
			vector<char> infoLog(error->GetBufferSize() + 1);
			memcpy(infoLog.data(), error->GetBufferPointer(), error->GetBufferSize());
			infoLog[error->GetBufferSize()] = 0;

			string errorMsg = "Shader Compiler Error:\n";
			errorMsg.append(infoLog.data());

			if (MessageBoxA(nullptr, errorMsg.c_str(), "Error!", MB_RETRYCANCEL) == IDRETRY) {
				// Another retry
				continue;
			} else {
				// User canceled
				return;
			}
		}

		// Successful compilation
		retryCompile = false;
	}

	hr = result->GetResult(blob);
	Utils::Validate(hr, L"Error: failed to get shader blob result!");
}

/**
* Compile an HLSL ray tracing shader using dxcompiler.
*/
void CompileShader(D3D12ShaderCompilerInfo &compilerInfo, RtProgram &program) 
{
	CompileShader(compilerInfo, program.info, &program.blob);	
}

/**
* Initialize the shader compiler.
*/
void InitShaderCompiler(D3D12ShaderCompilerInfo &shaderCompiler) 
{
	HRESULT hr = shaderCompiler.DxcDllHelper.Initialize();
	Utils::Validate(hr, L"Failed to initialize DxCDllSupport!");

	hr = shaderCompiler.DxcDllHelper.CreateInstance(CLSID_DxcCompiler, &shaderCompiler.compiler);
	Utils::Validate(hr, L"Failed to create DxcCompiler!");

	hr = shaderCompiler.DxcDllHelper.CreateInstance(CLSID_DxcLibrary, &shaderCompiler.library);
	Utils::Validate(hr, L"Failed to create DxcLibrary!");
}

/**
 * Release shader compiler resources.
 */
void Destroy(D3D12ShaderCompilerInfo &shaderCompiler)
{
	SAFE_RELEASE(shaderCompiler.compiler);
	SAFE_RELEASE(shaderCompiler.library);
	shaderCompiler.DxcDllHelper.Cleanup();
}

}

//--------------------------------------------------------------------------------------
// D3D12 Functions
//--------------------------------------------------------------------------------------

namespace D3D12
{

/**
* Create the device.
*/
void CreateDevice(D3D12Global &d3d)
{

#if defined(_DEBUG)
	// Enable the D3D12 debug layer.
	{
		if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&d3d.debugController))))
		{
			d3d.debugController->EnableDebugLayer();
		}
	}
#endif

	// Create a DXGI Factory
	HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&d3d.factory));
	Utils::Validate(hr, L"Error: failed to create DXGI factory!");

	// Create the device
	d3d.adapter = nullptr;
	for (UINT adapterIndex = 0; DXGI_ERROR_NOT_FOUND != d3d.factory->EnumAdapters1(adapterIndex, &d3d.adapter); ++adapterIndex)
	{
		DXGI_ADAPTER_DESC1 adapterDesc;
		d3d.adapter->GetDesc1(&adapterDesc);

		if (adapterDesc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
		{			
			continue;	// Don't select the Basic Render Driver adapter.
		}

		if (SUCCEEDED(D3D12CreateDevice(d3d.adapter, D3D_FEATURE_LEVEL_12_1, _uuidof(ID3D12Device5), (void**)&d3d.device)))
		{
			// Check if the device supports ray tracing.
			D3D12_FEATURE_DATA_D3D12_OPTIONS5 features = {};
			HRESULT hr = d3d.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &features, sizeof(features));
			if (FAILED(hr) || features.RaytracingTier < D3D12_RAYTRACING_TIER_1_0)
			{
				SAFE_RELEASE(d3d.device);
				d3d.device = nullptr;
				continue;
			}

#if NAME_D3D_RESOURCES
			d3d.device->SetName(L"DXR Device");
#endif
			printf("Running on DXGI Adapter %S\n", adapterDesc.Description);
			break;
		}

		if (d3d.device == nullptr)
		{
			// Didn't find a device that supports ray tracing.
			Utils::Validate(E_FAIL, L"Error: failed to create ray tracing device!");
		}
	}
}

/**
* Create the command queue.
*/
void CreateCommandQueue(D3D12Global &d3d) 
{
	D3D12_COMMAND_QUEUE_DESC desc = {};
	desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
	desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

	HRESULT hr = d3d.device->CreateCommandQueue(&desc, IID_PPV_ARGS(&d3d.cmdQueue));
	Utils::Validate(hr, L"Error: failed to create command queue!");

#if NAME_D3D_RESOURCES
	d3d.cmdQueue->SetName(L"D3D12 Command Queue");
#endif
}

/**
* Create the command allocator for each frame.
*/
void CreateCommandAllocator(D3D12Global &d3d) 
{
	for (UINT n = 0; n < 2; n++)
	{
		HRESULT hr = d3d.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&d3d.cmdAlloc[n]));
		Utils::Validate(hr, L"Error: failed to create the command allocator!");

#if NAME_D3D_RESOURCES
		d3d.cmdAlloc[n]->SetName((L"D3D12 Command Allocator " + std::to_wstring(n)).c_str());
#endif
	}
}

/**
* Create the command list.
*/
void CreateCommandList(D3D12Global &d3d) 
{
	HRESULT hr = d3d.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, d3d.cmdAlloc[d3d.frameIndex], nullptr, IID_PPV_ARGS(&d3d.cmdList));
	hr = d3d.cmdList->Close();
	Utils::Validate(hr, L"Error: failed to create the command list!");

#if NAME_D3D_RESOURCES
	d3d.cmdList->SetName(L"D3D12 Command List");
#endif
}

/**
* Create a fence.
*/
void CreateFence(D3D12Global &d3d) 
{
	HRESULT hr = d3d.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&d3d.fence));
	Utils::Validate(hr, L"Error: failed to create fence!");

#if NAME_D3D_RESOURCES
	d3d.fence->SetName(L"D3D12 Fence");
#endif

	d3d.fenceValues[d3d.frameIndex]++;

	// Create an event handle to use for frame synchronization
	d3d.fenceEvent = CreateEventEx(nullptr, FALSE, FALSE, EVENT_ALL_ACCESS);
	if (d3d.fenceEvent == nullptr) 
	{
		hr = HRESULT_FROM_WIN32(GetLastError());
		Utils::Validate(hr, L"Error: failed to create fence event!");
	}
}

/**
* Create the swap chain.
*/
void CreateSwapChain(D3D12Global &d3d, HWND &window) 
{
	// Check for tearing support
	BOOL allowTearing = FALSE;
	HRESULT hr = d3d.factory->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allowTearing, sizeof(allowTearing));
	Utils::Validate(hr, L"Error: failed to check tearing support!");
	d3d.isTearingSupport = allowTearing;

	// Describe the swap chain
	DXGI_SWAP_CHAIN_DESC1 desc = {};
	desc.BufferCount = 2;
	desc.Width = d3d.width;
	desc.Height = d3d.height;
	desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	desc.SampleDesc.Count = 1;
	desc.Flags = d3d.isTearingSupport ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;

	// Create the swap chain
	IDXGISwapChain1* swapChain;
	hr = d3d.factory->CreateSwapChainForHwnd(d3d.cmdQueue, window, &desc, nullptr, nullptr, &swapChain);
	Utils::Validate(hr, L"Error: failed to create swap chain!");

	// Associate the swap chain with a window
	hr = d3d.factory->MakeWindowAssociation(window, DXGI_MWA_NO_ALT_ENTER);
	Utils::Validate(hr, L"Error: failed to make window association!");

	// Get the swap chain interface
	hr = swapChain->QueryInterface(__uuidof(IDXGISwapChain3), reinterpret_cast<void**>(&d3d.swapChain));
	Utils::Validate(hr, L"Error: failed to cast swap chain!");

	SAFE_RELEASE(swapChain);
	d3d.frameIndex = d3d.swapChain->GetCurrentBackBufferIndex();
}

/**
* Create a root signature.
*/
ID3D12RootSignature* CreateRootSignature(D3D12Global &d3d, const D3D12_ROOT_SIGNATURE_DESC &desc, std::wstring debugName)
{
	ID3DBlob* sig;
	ID3DBlob* error;
	HRESULT hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &error);
	Utils::Validate(hr, L"Error: failed to serialize root signature!");

	ID3D12RootSignature* pRootSig;
	hr = d3d.device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&pRootSig));
	Utils::Validate(hr, L"Error: failed to create root signature!");

#if NAME_D3D_RESOURCES
	pRootSig->SetName(debugName.c_str());
#endif

	SAFE_RELEASE(sig);
	SAFE_RELEASE(error);
	return pRootSig;
}

/**
* Reset the command list.
*/
void ResetCommandList(D3D12Global &d3d) 
{
	// Reset the command allocator for the current frame
	HRESULT hr = d3d.cmdAlloc[d3d.frameIndex]->Reset();
	Utils::Validate(hr, L"Error: failed to reset command allocator!");

	// Reset the command list for the current frame
	hr = d3d.cmdList->Reset(d3d.cmdAlloc[d3d.frameIndex], nullptr);
	Utils::Validate(hr, L"Error: failed to reset command list!");
}

/*
* Submit the command list.
*/
void SubmitCommandList(D3D12Global &d3d) 
{
	d3d.cmdList->Close();

	ID3D12CommandList* pGraphicsList = { d3d.cmdList };
	d3d.cmdQueue->ExecuteCommandLists(1, &pGraphicsList);
	d3d.fenceValues[d3d.frameIndex]++;
	HRESULT hr = d3d.cmdQueue->Signal(d3d.fence, d3d.fenceValues[d3d.frameIndex]);
	Utils::Validate(hr, L"Error: failed to signal fence!");
}

/**
 * Swap the buffers.
 */
void Present(D3D12Global &d3d) 
{
	HRESULT hr = d3d.swapChain->Present(d3d.vsync, (!d3d.vsync && d3d.isTearingSupport) ? DXGI_PRESENT_ALLOW_TEARING : 0);
	if (FAILED(hr))
	{
		hr = d3d.device->GetDeviceRemovedReason();
		Utils::Validate(hr, L"Error: failed to present!");
	}
}

/*
* Wait for pending GPU work to complete.
*/
void WaitForGPU(D3D12Global &d3d) 
{
	// Schedule a signal command in the queue
	HRESULT hr = d3d.cmdQueue->Signal(d3d.fence, d3d.fenceValues[d3d.frameIndex]);
	Utils::Validate(hr, L"Error: failed to signal fence!");

	// Wait until the fence has been processed
	hr = d3d.fence->SetEventOnCompletion(d3d.fenceValues[d3d.frameIndex], d3d.fenceEvent);
	Utils::Validate(hr, L"Error: failed to set fence event!");

	WaitForSingleObjectEx(d3d.fenceEvent, INFINITE, FALSE);

	// Increment the fence value for the current frame
	d3d.fenceValues[d3d.frameIndex]++;
}

/**
* Prepare to render the next frame.
*/
void MoveToNextFrame(D3D12Global &d3d) 
{
	// Schedule a Signal command in the queue
	const UINT64 currentFenceValue = d3d.fenceValues[d3d.frameIndex];
	HRESULT hr = d3d.cmdQueue->Signal(d3d.fence, currentFenceValue);
	Utils::Validate(hr, L"Error: failed to signal command queue!");

	// Update the frame index
	d3d.frameIndex = d3d.swapChain->GetCurrentBackBufferIndex();

	// If the next frame is not ready to be rendered yet, wait until it is
	if (d3d.fence->GetCompletedValue() < d3d.fenceValues[d3d.frameIndex])
	{
		hr = d3d.fence->SetEventOnCompletion(d3d.fenceValues[d3d.frameIndex], d3d.fenceEvent);
		Utils::Validate(hr, L"Error: failed to set fence value!");

		WaitForSingleObjectEx(d3d.fenceEvent, INFINITE, FALSE);
	}

	// Set the fence value for the next frame
	d3d.fenceValues[d3d.frameIndex] = currentFenceValue + 1;
}

/*
* Save the back buffer to disk using DirectxTk library.
*/
bool ScreenCapture(D3D12Global& d3d, std::string filename)
{
	HRESULT hr = CoInitialize(NULL);
	Utils::Validate(hr, L"Error: failed to initialize screen capture!");

	hr = SaveWICTextureToFile(d3d.cmdQueue, d3d.backBuffer[d3d.frameIndex], GUID_ContainerFormatPng, (std::wstring(filename.begin(), filename.end()) + L".png").c_str(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_PRESENT, nullptr, nullptr, false);
	if (FAILED(hr)) return false;
	
	return true;
}

/**
 * Release D3D12 resources.
 */
void Destroy(D3D12Global &d3d)
{
	SAFE_RELEASE(d3d.fence);
	SAFE_RELEASE(d3d.backBuffer[1]);
	SAFE_RELEASE(d3d.backBuffer[0]);
	SAFE_RELEASE(d3d.swapChain);
	SAFE_RELEASE(d3d.cmdAlloc[0]);
	SAFE_RELEASE(d3d.cmdAlloc[1]);
	SAFE_RELEASE(d3d.cmdQueue);
	SAFE_RELEASE(d3d.cmdList);
	SAFE_RELEASE(d3d.device);
	SAFE_RELEASE(d3d.adapter);
	SAFE_RELEASE(d3d.factory);
	SAFE_RELEASE(d3d.debugController);

#if defined(_DEBUG)
	{
		IDXGIDebug1* dxgiDebug;
		if (SUCCEEDED(DXGIGetDebugInterface1(0, IID_PPV_ARGS(&dxgiDebug))))
		{
			dxgiDebug->ReportLiveObjects(DXGI_DEBUG_ALL, DXGI_DEBUG_RLO_FLAGS(DXGI_DEBUG_RLO_SUMMARY | DXGI_DEBUG_RLO_IGNORE_INTERNAL));
		}
	}
#endif
}

}

//--------------------------------------------------------------------------------------
// DXR Functions
//--------------------------------------------------------------------------------------

namespace DXR
{

// Names of these shaders must match the ones in HLSL
const std::wstring missShaderEntryPoint = L"Miss";
const std::wstring missShadowShaderEntryPoint = L"MissShadow";
const std::wstring closestHitShaderEntryPoint = L"ClosestHit";
const std::wstring anyHitShaderEntryPoint = L"AnyHit";
const std::wstring anyHitShadowShaderEntryPoint = L"AnyHitShadow";
const std::wstring raygenShaderEntryPoint = L"RayGen";

// Hitgroup names can be arbitrary
const std::wstring standardHitGroup = L"HitGroup";
const std::wstring shadowHitGroup = L"HitGroupShadow";

/**
* Create the bottom level acceleration structures and their associated buffers.
*/
void CreateBottomLevelAS(D3D12Global& d3d, DXRGlobal& dxr, D3D12Resources& resources, Scene& scene)
{
	D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

	// Describe the BLAS geometries. Each mesh primitive populates a BLAS.
	dxr.BLASes.resize(scene.numGeometries);
	for (size_t meshIndex = 0; meshIndex < scene.meshes.size(); meshIndex++)
	{
		// Get the mesh
		const Mesh mesh = scene.meshes[meshIndex];
		for (size_t primitiveIndex = 0; primitiveIndex < mesh.primitives.size(); primitiveIndex++)
		{
			// Get the mesh primitive
			const MeshPrimitive primitive = mesh.primitives[primitiveIndex];

			// Create a string name for BLAS
			std::wstring name = L"DXR BLASes: ";
			name.append(mesh.name.empty() ? L"<unnamed>" : std::wstring(mesh.name.begin(), mesh.name.end()));
			name.append(L", ID: ");
			name.append(std::to_wstring(meshIndex));
			name.append(L", Primitive: ");
			name.append(std::to_wstring(primitiveIndex));

			// Describe the geometry
			D3D12_RAYTRACING_GEOMETRY_DESC desc = {};
			desc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
			desc.Triangles.VertexBuffer.StartAddress = resources.sceneVBs[primitive.index]->GetGPUVirtualAddress();
			desc.Triangles.VertexBuffer.StrideInBytes = resources.sceneVBViews[primitive.index].StrideInBytes;
			desc.Triangles.VertexCount = static_cast<UINT>(primitive.vertices.size());
			desc.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
			desc.Triangles.IndexBuffer = resources.sceneIBs[primitive.index]->GetGPUVirtualAddress();
			desc.Triangles.IndexFormat = resources.sceneIBViews[primitive.index].Format;
			desc.Triangles.IndexCount = static_cast<UINT>(primitive.indices.size());

			// Mark geometry as opaque if there's no transparent material (this will prevent any hit shaders being executed for this geometry)
			if (primitive.opaque) desc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE; 

			// Describe the acceleration structure inputs
			D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS ASInputs = {};
			ASInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
			ASInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
			ASInputs.pGeometryDescs = &desc;
			ASInputs.NumDescs = 1;
			ASInputs.Flags = buildFlags;

			// Get the size requirements for the BLAS buffer
			D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO ASPreBuildInfo = {};
			d3d.device->GetRaytracingAccelerationStructurePrebuildInfo(&ASInputs, &ASPreBuildInfo);

			ASPreBuildInfo.ScratchDataSizeInBytes = ALIGN(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT, ASPreBuildInfo.ScratchDataSizeInBytes);
			ASPreBuildInfo.ResultDataMaxSizeInBytes = ALIGN(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT, ASPreBuildInfo.ResultDataMaxSizeInBytes);

			// Create the BLAS buffer
			D3D12BufferCreateInfo bufferInfo(ASPreBuildInfo.ResultDataMaxSizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
			bufferInfo.alignment = max(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT, D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);
			D3DResources::CreateBuffer(d3d, bufferInfo, &dxr.BLASes[primitive.index].pResult, name);

			// Describe and build the BLAS
			D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {};
			buildDesc.Inputs = ASInputs;
			buildDesc.ScratchAccelerationStructureData = dxr.scratchBuffersCache.Allocate(d3d, ASPreBuildInfo.ScratchDataSizeInBytes);
			buildDesc.DestAccelerationStructureData = dxr.BLASes[primitive.index].pResult->GetGPUVirtualAddress();

			d3d.cmdList->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);

			// Wait for the BLAS build to complete
			D3D12_RESOURCE_BARRIER uavBarrier;
			uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
			uavBarrier.UAV.pResource = dxr.BLASes[primitive.index].pResult;
			uavBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
			d3d.cmdList->ResourceBarrier(1, &uavBarrier);
		}
	}
}

/**
* Create the top level acceleration structure and its associated buffers.
*/
void CreateTopLevelAS(D3D12Global &d3d, DXRGlobal &dxr, D3D12Resources &resources, Scene& scene)
{
	// Describe the TLAS instance(s)
	std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instances;
	for (size_t instanceIndex = 0; instanceIndex < scene.instances.size(); instanceIndex++)
	{
		const Instance instance = scene.instances[instanceIndex];
		const Mesh mesh = scene.meshes[instance.mesh];
		for (size_t primitiveIndex = 0; primitiveIndex < mesh.primitives.size(); primitiveIndex++)
		{
			const MeshPrimitive primitive = mesh.primitives[primitiveIndex];

			// Describe the instance
			D3D12_RAYTRACING_INSTANCE_DESC desc = {};

			// Encode material and primitive ID into InstanceID
			desc.InstanceID = packInstanceID(primitive.material, primitive.index);
			desc.InstanceContributionToHitGroupIndex = 0;
			desc.InstanceMask = 0xFF;
			desc.AccelerationStructure = dxr.BLASes[primitive.index].pResult->GetGPUVirtualAddress();
			desc.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_FRONT_COUNTERCLOCKWISE;

			// Disable front or back face culling for meshes with double sided materials
			if (scene.materials[primitive.material].data.doubleSided)
			{
				desc.Flags |= D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE;
			}

			// Write the instance transform
			memcpy(desc.Transform, instance.transform, sizeof(XMFLOAT4) * 3);

			instances.push_back(desc);
		}
	}

	// Create the TLAS instance buffer
	D3D12BufferCreateInfo instanceBufferInfo;
	instanceBufferInfo.size = sizeof(D3D12_RAYTRACING_INSTANCE_DESC) * instances.size();
	instanceBufferInfo.heapType = D3D12_HEAP_TYPE_DEFAULT;
	instanceBufferInfo.flags = D3D12_RESOURCE_FLAG_NONE;
	instanceBufferInfo.state = D3D12_RESOURCE_STATE_COPY_DEST;
	D3DResources::CreateBuffer(d3d, instanceBufferInfo, &dxr.TLAS.pInstanceDesc, L"DXR TLAS Instance Descriptors");
	D3DResources::UploadToGPU(d3d, instances.data(), dxr.TLAS.pInstanceDesc, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

	D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

	// Get the size requirements for the TLAS buffers
	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS ASInputs = {};
	ASInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
	ASInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
	ASInputs.InstanceDescs = dxr.TLAS.pInstanceDesc->GetGPUVirtualAddress();
	ASInputs.NumDescs = (UINT)instances.size();
	ASInputs.Flags = buildFlags;

	// Get the size requirements for the TLAS buffers
	D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO ASPreBuildInfo = {};
	d3d.device->GetRaytracingAccelerationStructurePrebuildInfo(&ASInputs, &ASPreBuildInfo);

	ASPreBuildInfo.ResultDataMaxSizeInBytes = ALIGN(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT, ASPreBuildInfo.ResultDataMaxSizeInBytes);
	ASPreBuildInfo.ScratchDataSizeInBytes = ALIGN(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT, ASPreBuildInfo.ScratchDataSizeInBytes);

	// Set TLAS size
	dxr.tlasSize = ASPreBuildInfo.ResultDataMaxSizeInBytes;

	// Create the TLAS buffer
	D3D12BufferCreateInfo bufferInfo(ASPreBuildInfo.ResultDataMaxSizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
	bufferInfo.alignment = max(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT, D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);
	D3DResources::CreateBuffer(d3d, bufferInfo, &dxr.TLAS.pResult, L"DXR TLAS");

	// Describe and build the TLAS
	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {};
	buildDesc.Inputs = ASInputs;
	buildDesc.ScratchAccelerationStructureData = dxr.scratchBuffersCache.Allocate(d3d, ASPreBuildInfo.ScratchDataSizeInBytes);
	buildDesc.DestAccelerationStructureData = dxr.TLAS.pResult->GetGPUVirtualAddress();

	d3d.cmdList->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);

	// Wait for the TLAS build to complete
	D3D12_RESOURCE_BARRIER barrier = {};
	barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
	barrier.UAV.pResource = dxr.TLAS.pResult;

	d3d.cmdList->ResourceBarrier(1, &barrier);
}

/**
* Load and create the DXR Ray Generation program and global root signature.
*/
void CreateRTProgram(D3D12Global &d3d, DXRGlobal &dxr, D3D12ShaderCompilerInfo &shaderCompiler)
{
	// Load and compile the ray generation shader
	dxr.rtProgram = RtProgram(D3D12ShaderInfo(L"shaders\\PathTracer.hlsl", L"", L"lib_6_3"));
	D3DShaders::CompileShader(shaderCompiler, dxr.rtProgram);

	// Describe the global root signature
	D3D12_DESCRIPTOR_RANGE ranges[6];

	ranges[0].BaseShaderRegister = 0;
	ranges[0].NumDescriptors = UINT(DescriptorHeapConstants::CBTotal);
	ranges[0].RegisterSpace = 0;
	ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
	ranges[0].OffsetInDescriptorsFromTableStart = UINT(DescriptorHeapConstants::CBStart);

	ranges[1].BaseShaderRegister = 0;
	ranges[1].NumDescriptors = UINT(DescriptorHeapConstants::UAV0Total);
	ranges[1].RegisterSpace = 0;
	ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
	ranges[1].OffsetInDescriptorsFromTableStart = UINT(DescriptorHeapConstants::UAV0Start);

	ranges[2].BaseShaderRegister = 0;
	ranges[2].NumDescriptors = UINT(DescriptorHeapConstants::SRV0Total);
	ranges[2].RegisterSpace = 0;
	ranges[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
	ranges[2].OffsetInDescriptorsFromTableStart = UINT(DescriptorHeapConstants::SRV0Start);

	ranges[3].BaseShaderRegister = 0;
	ranges[3].NumDescriptors = UINT(DescriptorHeapConstants::SRV1Total);
	ranges[3].RegisterSpace = 1;
	ranges[3].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
	ranges[3].OffsetInDescriptorsFromTableStart = UINT(DescriptorHeapConstants::SRV1Start);

	ranges[4].BaseShaderRegister = 0;
	ranges[4].NumDescriptors = UINT(DescriptorHeapConstants::SRV2Total);
	ranges[4].RegisterSpace = 2;
	ranges[4].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
	ranges[4].OffsetInDescriptorsFromTableStart = UINT(DescriptorHeapConstants::SRV2Start);

	ranges[5].BaseShaderRegister = 0;
	ranges[5].NumDescriptors = UINT(DescriptorHeapConstants::SRV3Total);
	ranges[5].RegisterSpace = 3;
	ranges[5].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
	ranges[5].OffsetInDescriptorsFromTableStart = UINT(DescriptorHeapConstants::SRV3Start);

	D3D12_ROOT_PARAMETER param0 = {};
	param0.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
	param0.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
	param0.DescriptorTable.NumDescriptorRanges = _countof(ranges);
	param0.DescriptorTable.pDescriptorRanges = ranges;

	D3D12_ROOT_PARAMETER rootParams[1] = { param0 };

	D3D12_STATIC_SAMPLER_DESC staticSampler = {};
	staticSampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
	staticSampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
	staticSampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
	staticSampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
	staticSampler.MipLODBias = 0.f;
	staticSampler.MaxAnisotropy = 1;
	staticSampler.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
	staticSampler.BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
	staticSampler.MinLOD = 0.f;
	staticSampler.MaxLOD = D3D12_FLOAT32_MAX;
	staticSampler.ShaderRegister = 0;
	staticSampler.RegisterSpace = 0;
	staticSampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

	D3D12_ROOT_SIGNATURE_DESC rootDesc = {};
	rootDesc.NumParameters = _countof(rootParams);
	rootDesc.pParameters = rootParams;
	rootDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
	rootDesc.NumStaticSamplers = 1;
	rootDesc.pStaticSamplers = &staticSampler;

	// Create the root signature
	dxr.globalRootSignature = D3D12::CreateRootSignature(d3d, rootDesc, L"DXR Global Root Signature");
}

/**
* Create the DXR pipeline state object.
*/
void CreatePipelineStateObject(D3D12Global &d3d, DXRGlobal &dxr)
{
	// Need 12 subobjects:
	// 1 for RGS program
	// 2 for Miss programs
	// 1 for CHS program
	// 2 for AHS program
	// 2 for Hit Groups
	// 2 for Shader Config (config and association)
	// 1 for Global Root Signature
	// 1 for Pipeline Config	
	D3D12_STATE_SUBOBJECT subobjects[12];
	size_t index = 0;

	// Add state subobject for the RGS
	D3D12_EXPORT_DESC rgsExportDesc = {};
	rgsExportDesc.Name = raygenShaderEntryPoint.c_str();
	rgsExportDesc.ExportToRename = raygenShaderEntryPoint.c_str();
	rgsExportDesc.Flags = D3D12_EXPORT_FLAG_NONE;

	D3D12_DXIL_LIBRARY_DESC	rgsLibDesc = {};
	rgsLibDesc.DXILLibrary.BytecodeLength = dxr.rtProgram.blob->GetBufferSize();
	rgsLibDesc.DXILLibrary.pShaderBytecode = dxr.rtProgram.blob->GetBufferPointer();
	rgsLibDesc.NumExports = 1;
	rgsLibDesc.pExports = &rgsExportDesc;

	D3D12_STATE_SUBOBJECT rgs = {};
	rgs.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
	rgs.pDesc = &rgsLibDesc;

	subobjects[index++] = rgs;

	// Add state subobject for the Miss shader (standard rays)
	D3D12_EXPORT_DESC msExportDesc = {};
	msExportDesc.Name = missShaderEntryPoint.c_str();
	msExportDesc.ExportToRename = missShaderEntryPoint.c_str();
	msExportDesc.Flags = D3D12_EXPORT_FLAG_NONE;

	D3D12_DXIL_LIBRARY_DESC	msLibDesc = {};
	msLibDesc.DXILLibrary.BytecodeLength = dxr.rtProgram.blob->GetBufferSize();
	msLibDesc.DXILLibrary.pShaderBytecode = dxr.rtProgram.blob->GetBufferPointer();
	msLibDesc.NumExports = 1;
	msLibDesc.pExports = &msExportDesc;

	D3D12_STATE_SUBOBJECT ms = {};
	ms.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
	ms.pDesc = &msLibDesc;

	subobjects[index++] = ms;

	// Add state subobject for the Miss shader (shadow rays)
	D3D12_EXPORT_DESC msShadowExportDesc = {};
	msShadowExportDesc.Name = missShadowShaderEntryPoint.c_str();
	msShadowExportDesc.ExportToRename = missShadowShaderEntryPoint.c_str();
	msShadowExportDesc.Flags = D3D12_EXPORT_FLAG_NONE;

	D3D12_DXIL_LIBRARY_DESC	msShadowLibDesc = {};
	msShadowLibDesc.DXILLibrary.BytecodeLength = dxr.rtProgram.blob->GetBufferSize();
	msShadowLibDesc.DXILLibrary.pShaderBytecode = dxr.rtProgram.blob->GetBufferPointer();
	msShadowLibDesc.NumExports = 1;
	msShadowLibDesc.pExports = &msShadowExportDesc;

	D3D12_STATE_SUBOBJECT msShadow = {};
	msShadow.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
	msShadow.pDesc = &msShadowLibDesc;

	subobjects[index++] = msShadow;

	// Add state subobject for the Closest Hit shader
	D3D12_EXPORT_DESC chsExportDesc = {};
	chsExportDesc.Name = closestHitShaderEntryPoint.c_str();
	chsExportDesc.ExportToRename = closestHitShaderEntryPoint.c_str();
	chsExportDesc.Flags = D3D12_EXPORT_FLAG_NONE;

	D3D12_DXIL_LIBRARY_DESC	chsLibDesc = {};
	chsLibDesc.DXILLibrary.BytecodeLength = dxr.rtProgram.blob->GetBufferSize();
	chsLibDesc.DXILLibrary.pShaderBytecode = dxr.rtProgram.blob->GetBufferPointer();
	chsLibDesc.NumExports = 1;
	chsLibDesc.pExports = &chsExportDesc;

	D3D12_STATE_SUBOBJECT chs = {};
	chs.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
	chs.pDesc = &chsLibDesc;

	subobjects[index++] = chs;

	// Add state subobject for the Any Hit shader
	D3D12_EXPORT_DESC ahsExportDesc = {};
	ahsExportDesc.Name = anyHitShaderEntryPoint.c_str();
	ahsExportDesc.ExportToRename = anyHitShaderEntryPoint.c_str();
	ahsExportDesc.Flags = D3D12_EXPORT_FLAG_NONE;

	D3D12_DXIL_LIBRARY_DESC	ahsLibDesc = {};
	ahsLibDesc.DXILLibrary.BytecodeLength = dxr.rtProgram.blob->GetBufferSize();
	ahsLibDesc.DXILLibrary.pShaderBytecode = dxr.rtProgram.blob->GetBufferPointer();
	ahsLibDesc.NumExports = 1;
	ahsLibDesc.pExports = &ahsExportDesc;

	D3D12_STATE_SUBOBJECT ahs = {};
	ahs.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
	ahs.pDesc = &ahsLibDesc;

	subobjects[index++] = ahs;

	// Add state subobject for the Any Hit shader
	D3D12_EXPORT_DESC ahsShadowExportDesc = {};
	ahsShadowExportDesc.Name = anyHitShadowShaderEntryPoint.c_str();
	ahsShadowExportDesc.ExportToRename = anyHitShadowShaderEntryPoint.c_str();
	ahsShadowExportDesc.Flags = D3D12_EXPORT_FLAG_NONE;

	D3D12_DXIL_LIBRARY_DESC	ahsShadowLibDesc = {};
	ahsShadowLibDesc.DXILLibrary.BytecodeLength = dxr.rtProgram.blob->GetBufferSize();
	ahsShadowLibDesc.DXILLibrary.pShaderBytecode = dxr.rtProgram.blob->GetBufferPointer();
	ahsShadowLibDesc.NumExports = 1;
	ahsShadowLibDesc.pExports = &ahsShadowExportDesc;

	D3D12_STATE_SUBOBJECT ahsShadow = {};
	ahsShadow.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
	ahsShadow.pDesc = &ahsShadowLibDesc;

	subobjects[index++] = ahsShadow;

	// Add a state subobject for the standard hit group
	D3D12_HIT_GROUP_DESC hitGroupDesc = {};
	hitGroupDesc.ClosestHitShaderImport = closestHitShaderEntryPoint.c_str();
	hitGroupDesc.AnyHitShaderImport = anyHitShaderEntryPoint.c_str();
	hitGroupDesc.HitGroupExport = standardHitGroup.c_str();

	D3D12_STATE_SUBOBJECT hitGroup = {};
	hitGroup.Type = D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP;
	hitGroup.pDesc = &hitGroupDesc;

	subobjects[index++] = hitGroup;

	// Add a state subobject for the hit group for shadow rays
	D3D12_HIT_GROUP_DESC hitGroupShadowDesc = {};
	hitGroupShadowDesc.AnyHitShaderImport = anyHitShadowShaderEntryPoint.c_str();
	hitGroupShadowDesc.HitGroupExport = shadowHitGroup.c_str();

	D3D12_STATE_SUBOBJECT hitGroupShadow = {};
	hitGroupShadow.Type = D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP;
	hitGroupShadow.pDesc = &hitGroupShadowDesc;

	subobjects[index++] = hitGroupShadow;

	// Add a state subobject for the shader payload configuration
	D3D12_RAYTRACING_SHADER_CONFIG shaderDesc = {};
	shaderDesc.MaxPayloadSizeInBytes = max(sizeof(HitInfo), sizeof(ShadowHitInfo));
	shaderDesc.MaxAttributeSizeInBytes = D3D12_RAYTRACING_MAX_ATTRIBUTE_SIZE_IN_BYTES;

	D3D12_STATE_SUBOBJECT shaderConfigObject = {};
	shaderConfigObject.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG;
	shaderConfigObject.pDesc = &shaderDesc;

	subobjects[index++] = shaderConfigObject;

	// Create a list of the shader export names that use the payload
	const WCHAR* shaderExports[] = { raygenShaderEntryPoint.c_str(), missShaderEntryPoint.c_str(), standardHitGroup.c_str(), missShadowShaderEntryPoint.c_str(), shadowHitGroup.c_str() };

	// Add a state subobject for the association between shaders and the payload
	D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION shaderPayloadAssociation = {};
	shaderPayloadAssociation.NumExports = _countof(shaderExports);
	shaderPayloadAssociation.pExports = shaderExports;
	shaderPayloadAssociation.pSubobjectToAssociate = &subobjects[index - 1];

	D3D12_STATE_SUBOBJECT shaderPayloadAssociationObject = {};
	shaderPayloadAssociationObject.Type = D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION;
	shaderPayloadAssociationObject.pDesc = &shaderPayloadAssociation;

	subobjects[index++] = shaderPayloadAssociationObject;

	D3D12_STATE_SUBOBJECT globalRootSig;
	globalRootSig.Type = D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE;
	globalRootSig.pDesc = &dxr.globalRootSignature;

	subobjects[index++] = globalRootSig;

	// Add a state subobject for the ray tracing pipeline config
	D3D12_RAYTRACING_PIPELINE_CONFIG pipelineConfig = {};
	pipelineConfig.MaxTraceRecursionDepth = 1;

	D3D12_STATE_SUBOBJECT pipelineConfigObject = {};
	pipelineConfigObject.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG;
	pipelineConfigObject.pDesc = &pipelineConfig;

	subobjects[index++] = pipelineConfigObject;

	// Describe the Ray Tracing Pipeline State Object
	D3D12_STATE_OBJECT_DESC pipelineDesc = {};
	pipelineDesc.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE;
	pipelineDesc.NumSubobjects = static_cast<UINT>(_countof(subobjects));
	pipelineDesc.pSubobjects = subobjects;

	// Create the RT Pipeline State Object (RTPSO)
	HRESULT hr = d3d.device->CreateStateObject(&pipelineDesc, IID_PPV_ARGS(&dxr.rtpso));
	Utils::Validate(hr, L"Error: failed to create state object!");
#if NAME_D3D_RESOURCES
	dxr.rtpso->SetName(L"DXR Pipeline State Object");
#endif

	// Get the RTPSO properties
	hr = dxr.rtpso->QueryInterface(IID_PPV_ARGS(&dxr.rtpsoInfo));
	Utils::Validate(hr, L"Error: failed to get RTPSO info object!");
}

/**
* Create the DXR shader table.
*/
void CreateShaderTable(D3D12Global &d3d, DXRGlobal &dxr, D3D12Resources &resources) 
{
	/*
	The Shader Table layout is as follows:
		Entry 0 - Miss shader for standard rays
		Entry 1 - Miss shader for shadow rays
		Entry 2 - Hit Group for standard rays
		Entry 3 - Hit Group for shadow rays
		Entry 4 - Ray Generation program
	All shader records in the Shader Table must have the same size, so shader record size will be based on the largest required entry.
	The entry size must be aligned up to D3D12_RAYTRACING_SHADER_BINDING_TABLE_RECORD_BYTE_ALIGNMENT
	Address of the first shader of any kind (miss/hit/raygen) must be aligned to 64 bytes (D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT), as 
	it is provided as StartAddress the to dispatch rays call.
	*/

	uint32_t shaderIdSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
	dxr.shaderTableRecordSize = shaderIdSize;
	dxr.shaderTableRecordSize = ALIGN(D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT, dxr.shaderTableRecordSize);

	uint32_t shaderTableSize = (dxr.shaderTableRecordSize * 5);		// 5 shader records in the table
	shaderTableSize = ALIGN(D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT, shaderTableSize);

	// Create the shader table buffer
	D3D12BufferCreateInfo bufferInfo(shaderTableSize, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
	D3DResources::CreateBuffer(d3d, bufferInfo, &dxr.shaderTable, L"DXR Shader Table");

	// Build shader table data on CPU
	dxr.shaderTableData.reserve(shaderTableSize);
	uint8_t* pData = dxr.shaderTableData.data();

	// Shader Record 0 - Miss program for standard rays
	memcpy(pData, dxr.rtpsoInfo->GetShaderIdentifier(missShaderEntryPoint.c_str()), shaderIdSize);

	// Shader Record 1 - Miss program for shadow rays
	pData += dxr.shaderTableRecordSize;
	memcpy(pData, dxr.rtpsoInfo->GetShaderIdentifier(missShadowShaderEntryPoint.c_str()), shaderIdSize);

	// Shader Record 2 - Hit Group for standard rays
	pData += dxr.shaderTableRecordSize;
	memcpy(pData, dxr.rtpsoInfo->GetShaderIdentifier(standardHitGroup.c_str()), shaderIdSize);

	// Shader Record 3 - Hit Group for shadow rays
	pData += dxr.shaderTableRecordSize;
	memcpy(pData, dxr.rtpsoInfo->GetShaderIdentifier(shadowHitGroup.c_str()), shaderIdSize);

	// Shader Record 4 - Ray Generation program
	pData += dxr.shaderTableRecordSize;
	memcpy(pData, dxr.rtpsoInfo->GetShaderIdentifier(raygenShaderEntryPoint.c_str()), shaderIdSize);

	// Upload shader table to GPU
	D3DResources::UploadToGPU(d3d, dxr.shaderTableData.data(), dxr.shaderTable, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
}

/**
* Create the DXR descriptor heap for CBVs, SRVs, and the output UAV.
*/
void CreateDescriptorHeaps(D3D12Global& d3d, DXRGlobal& dxr, D3D12Resources& resources)
{
	// Describe the CBV/SRV/UAV heap
	D3D12_DESCRIPTOR_HEAP_DESC desc = {};
	desc.NumDescriptors = UINT(DescriptorHeapConstants::Total);
	desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

	// Create the descriptor heap
	HRESULT hr = d3d.device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&resources.descriptorHeap));
	Utils::Validate(hr, L"Error: failed to create DXR CBV/SRV/UAV descriptor heap!");

	// Get the descriptor heap handle and increment size
	D3D12_CPU_DESCRIPTOR_HANDLE handle = resources.descriptorHeap->GetCPUDescriptorHandleForHeapStart();
	resources.cbvSrvUavDescSize = d3d.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

#if NAME_D3D_RESOURCES
	resources.descriptorHeap->SetName(L"DXR Descriptor Heap");
#endif
}

/**
* Release the shaders and RTPSO. Reload and compile the shaders, recreate the RTPSO, and rewrite the shader table entries.
*/
void ReloadShaders(D3D12Global& d3d, DXRGlobal& dxr, D3D12Resources& resources, D3D12ShaderCompilerInfo& shaderCompiler)
{
	SAFE_RELEASE(dxr.rtProgram.blob);
	SAFE_RELEASE(dxr.rtpso);
	SAFE_RELEASE(dxr.rtpsoInfo);
	SAFE_RELEASE(dxr.globalRootSignature);
	SAFE_RELEASE(dxr.shaderTable);

	DXR::CreateRTProgram(d3d, dxr, shaderCompiler);
	DXR::CreatePipelineStateObject(d3d, dxr);
	DXR::CreateShaderTable(d3d, dxr, resources);

	dxr.forceAccumulationReset = true;
}

/**
* Create the DXR descriptor heap for CBVs, SRVs, and the output UAV.
*/
void FillDescriptorHeaps(D3D12Global &d3d, DXRGlobal &dxr, D3D12Resources &resources)
{
	// Get the descriptor heap handle and increment size
	D3D12_CPU_DESCRIPTOR_HANDLE handle = resources.descriptorHeap->GetCPUDescriptorHandleForHeapStart();

	// Create the RaytracingData CBV
	D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc = {};
	cbvDesc.SizeInBytes = ALIGN(D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, sizeof(resources.raytracingData));
	cbvDesc.BufferLocation = resources.raytracingDataCB->GetGPUVirtualAddress();
	d3d.device->CreateConstantBufferView(&cbvDesc, handle);

	// Create the DXR output buffer UAV
	D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
	uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;

	handle.ptr += resources.cbvSrvUavDescSize;
	d3d.device->CreateUnorderedAccessView(resources.DXROutput, nullptr, &uavDesc, handle);

	handle.ptr += resources.cbvSrvUavDescSize;
	d3d.device->CreateUnorderedAccessView(resources.accumulationBuffer, nullptr, &uavDesc, handle);

	// Create the DXR Top Level Acceleration Structure SRV
	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc;
	srvDesc.Format = DXGI_FORMAT_UNKNOWN;
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srvDesc.RaytracingAccelerationStructure.Location = dxr.TLAS.pResult->GetGPUVirtualAddress();

	handle.ptr += resources.cbvSrvUavDescSize;
	d3d.device->CreateShaderResourceView(nullptr, &srvDesc, handle);
}

/**
* Create the DXR output buffer.
*/
void CreateDXRResources(D3D12Global &d3d, DXRGlobal& dxr, D3D12Resources &resources)
{
	// Describe the DXR output resource (texture)
	// Dimensions and format should match the swapchain
	// Initialize as a copy source, since we will copy this buffer's contents to the swapchain
	D3D12_RESOURCE_DESC desc = {};
	desc.DepthOrArraySize = 1;
	desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
	desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
	desc.Width = d3d.width;
	desc.Height = d3d.height;
	desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
	desc.MipLevels = 1;
	desc.SampleDesc.Count = 1;
	desc.SampleDesc.Quality = 0;

	// Create the buffer resource for RT output
	HRESULT hr = d3d.device->CreateCommittedResource(&DefaultHeapProperties, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_SOURCE, nullptr, IID_PPV_ARGS(&resources.DXROutput));
	Utils::Validate(hr, L"Error: failed to create DXR output buffer!");
#if NAME_D3D_RESOURCES
	resources.DXROutput->SetName(L"DXR Output Buffer");
#endif

	// Create the buffer resource for accumulation
	desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	hr = d3d.device->CreateCommittedResource(&DefaultHeapProperties, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_SOURCE, nullptr, IID_PPV_ARGS(&resources.accumulationBuffer));
	Utils::Validate(hr, L"Error: failed to create DXR accumulation buffer!");
#if NAME_D3D_RESOURCES
	resources.accumulationBuffer->SetName(L"DXR Accumulation Buffer");
#endif

	// Initialize linear allocator for scratch buffers
	dxr.scratchBuffersCache.Initialize(d3d,
		4 * 1024 * 1024 /* 4MB initially */,
		max(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT, D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT),
		D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}

/**
* Builds the frame's DXR command list.
*/
void BuildCommandList(D3D12Global &d3d, DXRGlobal &dxr, D3D12Resources &resources, Gui* gui, InputInfo* input)
{
	D3D12_RESOURCE_BARRIER Barriers[3] = {};

	// Transition the back buffer to a copy destination
	Barriers[0].Transition.pResource = d3d.backBuffer[d3d.frameIndex];
	Barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
	Barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
	Barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

	// Transition the DXR output buffer to a copy source
	Barriers[1].Transition.pResource = resources.DXROutput;
	Barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
	Barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
	Barriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

	// Transition the DXR output buffer to a copy source
	Barriers[2].UAV.pResource = resources.accumulationBuffer;
	Barriers[2].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
	Barriers[2].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;

	// Wait for the transitions to complete
	d3d.cmdList->ResourceBarrier(_countof(Barriers), Barriers);

	// Set the UAV/SRV/CBV heap
	ID3D12DescriptorHeap* ppHeaps[] = { resources.descriptorHeap };
	d3d.cmdList->SetDescriptorHeaps(_countof(ppHeaps), ppHeaps);
	d3d.cmdList->SetComputeRootSignature(dxr.globalRootSignature);
	d3d.cmdList->SetComputeRootDescriptorTable(0, resources.descriptorHeap->GetGPUDescriptorHandleForHeapStart());

	// Dispatch rays
	D3D12_DISPATCH_RAYS_DESC desc = {};
	desc.MissShaderTable.StartAddress = dxr.shaderTable->GetGPUVirtualAddress();
	desc.MissShaderTable.SizeInBytes = UINT64(dxr.shaderTableRecordSize) * 2;	//< Two Miss programs (normal & shadows)
	desc.MissShaderTable.StrideInBytes = dxr.shaderTableRecordSize;

	desc.HitGroupTable.StartAddress = dxr.shaderTable->GetGPUVirtualAddress() + (UINT64(dxr.shaderTableRecordSize) * 2); //< Higroups start after 2 other records (miss shaders)
	desc.HitGroupTable.SizeInBytes = UINT64(dxr.shaderTableRecordSize) * 2;		//< Two hit groups (normal & shadows)
	desc.HitGroupTable.StrideInBytes = dxr.shaderTableRecordSize;

	desc.RayGenerationShaderRecord.StartAddress = dxr.shaderTable->GetGPUVirtualAddress() + (UINT64(dxr.shaderTableRecordSize) * 4);//< Raygen starts after 4 other records
	desc.RayGenerationShaderRecord.SizeInBytes = dxr.shaderTableRecordSize;	    //< One raygen shader

	desc.Width = d3d.width;
	desc.Height = d3d.height;
	desc.Depth = 1;
	
	d3d.cmdList->SetPipelineState1(dxr.rtpso);
	d3d.cmdList->DispatchRays(&desc);

	// Transition DXR output to a copy source
	Barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
	Barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
	
	// Wait for the transitions to complete
	d3d.cmdList->ResourceBarrier(1, &Barriers[1]);

	// Copy the DXR output to the back buffer
	d3d.cmdList->CopyResource(d3d.backBuffer[d3d.frameIndex], resources.DXROutput);

	// Transition back buffer to present
	Barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
	Barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
	
	// Wait for the transitions to complete
	d3d.cmdList->ResourceBarrier(1, &Barriers[0]);

	if (d3d.renderGui) {
		// Setup the GUI
		gui->Indent(16.0f);
		if (gui->Button("[F1]Screenshot", 110.f, 20.f)) input->captureScreenshot = true;
		gui->SameLine();
		if (gui->Button("[F2]Hide GUI", 100.f, 20.f)) input->toggleGui = true;
		gui->SameLine();
		if (gui->Button("[F5]Reload Shaders", 140.f, 20.f)) input->reloadShaders = true;
		dxr.forceAccumulationReset |= gui->DragFloat("Camera Speed", &dxr.cameraSpeedAdjustment, 0.01f, 100.0f);
		gui->Indent(-16.0f);
		gui->Separator();
		gui->Text("Path Tracing:");
		gui->Indent(16.0f);
		dxr.forceAccumulationReset |= gui->DragInt("Max. Bounces", &dxr.maxBounces, 0, 16);
		gui->Checkbox("Accumulate", &dxr.enableAccumulation);
		gui->Separator();
		gui->Indent(-16.0f);
		gui->Text("Camera:");
		gui->Indent(16.0f);
		dxr.forceAccumulationReset |= gui->DragFloat("FOV", &dxr.camera.fov, 8.0f, 80.0f);
		dxr.forceAccumulationReset |= gui->DragFloat("Focus Distance", &dxr.focusDistance, 0.0f, 10.0f);
		dxr.forceAccumulationReset |= gui->DragFloat("Aperture Size", &dxr.apertureSize, 0.0f, 0.5f);
		dxr.forceAccumulationReset |= gui->Checkbox("Antialiasing", &dxr.enableAntiAliasing);
		gui->DragFloat("Exposure Adjustment", &dxr.exposureAdjustment, 0.0f, 10.0f);
		gui->Separator();
		gui->Indent(-16.0f);
		gui->Text("Lighting:");
		gui->Indent(16.0f);
		dxr.forceAccumulationReset |= gui->DragFloat("Sky Intensity", &dxr.skyIntensity, 0.0f, 100.0f);
		dxr.forceAccumulationReset |= gui->Checkbox("Direct Lighting", &dxr.enableDirectLighting);
		if (dxr.enableDirectLighting) {
			dxr.forceAccumulationReset |= gui->Checkbox("Sun", &dxr.enableSun);
			dxr.forceAccumulationReset |= gui->DragFloat("Sun Intensity", &dxr.sunIntensity, 0.0f, 100.0f);
			dxr.forceAccumulationReset |= gui->DragFloat("Sun Azimuth", &dxr.sunAzimuth, 0.0f, 360.0f);
			dxr.forceAccumulationReset |= gui->DragFloat("Sun Elevation", &dxr.sunElevation, 0.0f, 90.0f);
			dxr.forceAccumulationReset |= gui->Checkbox("Headlight", &dxr.enableHeadlight);
			dxr.forceAccumulationReset |= gui->DragFloat("Headlight Intensity", &dxr.headlightIntensity, 0.0f, 1000.0f);
		}
		gui->Indent(-16.0f);
	}

	// Submit the command list and wait for the GPU to idle
	D3D12::SubmitCommandList(d3d);
	D3D12::WaitForGPU(d3d);
}

/**
 * Release DXR resources.
 */
void Destroy(DXRGlobal &dxr)
{
	// Release acceleration structures
	dxr.TLAS.Release();

	for (size_t i = 0; i < dxr.BLASes.size(); i++)
	{
		dxr.BLASes[i].Release();
	}

	SAFE_RELEASE(dxr.shaderTable);
	SAFE_RELEASE(dxr.rtProgram.blob);
	SAFE_RELEASE(dxr.rtpso);
	SAFE_RELEASE(dxr.rtpsoInfo);
	SAFE_RELEASE(dxr.globalRootSignature);

	dxr.scratchBuffersCache.Destroy();
}

}

// ========================================================================================================
//	Linear GPU Allocator definitions
// ========================================================================================================

/**
 * Initialize linear allocator - allocate block of memory and remember underlying buffer parameters
 */
void LinearGPUAllocator::Initialize(D3D12Global& d3d, UINT64 initialSize, UINT64 alignment, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state, D3D12_HEAP_TYPE heapType) {

	bufferInfo.alignment = alignment;
	bufferInfo.flags = flags;
	bufferInfo.heapType = heapType;
	bufferInfo.state = state;

	AllocateInternal(d3d, initialSize);

	isInitialized = true;
}

/**
 * Allocates memory of requested size from pre-allocated block
 */
D3D12_GPU_VIRTUAL_ADDRESS LinearGPUAllocator::Allocate(D3D12Global &d3d, UINT64 size, ID3D12Resource* &resource, UINT64& offset) {

	if (!isInitialized) Utils::Validate(E_FAIL, L"Couldn't allocate memory from LinearAllocator. Forgot to call Initialize()?");

	// If there's not enough space in the block, allocate new one
	if ((currentOffset + size) > bufferInfo.size) {
		AllocateInternal(d3d, Utils::NextPowerOfTwo((currentOffset + size) * 2));
	}

	// Resource and offset where allocation happened can be useful for binding buffers
	offset = currentOffset;
	resource = currentBlock;

	// Return address pointing at free block of necessary size
	D3D12_GPU_VIRTUAL_ADDRESS result = currentBlock->GetGPUVirtualAddress() + currentOffset;
	currentOffset += size;

	return result;
}

/**
 * Allocates memory of requested size from pre-allocated block
 */
D3D12_GPU_VIRTUAL_ADDRESS LinearGPUAllocator::Allocate(D3D12Global &d3d, UINT64 size) {
	UINT64 offset;
	ID3D12Resource* resource;
	return Allocate(d3d, size, resource, offset);
}

/**
 * Releases obsolete memory blocks and reset pointer to current memory block
 */
void LinearGPUAllocator::Reset() {
	currentOffset = 0;
	for (auto block : obsoleteBlocks) block->Release();
	obsoleteBlocks.clear();
}

/**
 * Release the resources
 */
void LinearGPUAllocator::Destroy() {
	Reset();
	SAFE_RELEASE(currentBlock);
	bufferInfo.size = 0;
	isInitialized = false;
}

/**
 * Allocates underlying memory block for fast allocations
 */
void LinearGPUAllocator::AllocateInternal(D3D12Global &d3d, UINT64 size) {

	// Set current block aside for deallocation once the frame is finished
	if (currentBlock != nullptr) {
		obsoleteBlocks.push_back(currentBlock);
		currentBlock = nullptr;
	}
	currentOffset = 0;

	// Update buffer size and allocate new block
	bufferInfo.size = size;
	D3DResources::CreateBuffer(d3d, bufferInfo, &currentBlock, L"Linear GPU Allocator Block");
}