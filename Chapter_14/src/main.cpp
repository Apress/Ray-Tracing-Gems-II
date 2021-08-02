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

#include "Window.h"
#include "Graphics.h"
#include "Utils.h"
#include "GLTF.h"
#include <chrono>

#include "Input.h"

 // Windows DPI Scaling
#include <ShellScalingApi.h>
#pragma comment(lib, "shcore.lib")

#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif

/**
 * Your ray tracing application!
 */
class DXRApplication
{
public:

	void Init(ConfigInfo &config) 
	{		
		// Create a new window
		HRESULT hr = Window::Create(config.width, config.height, config.instance, window, L"Reference Path Tracer Sample", &gui);
		Utils::Validate(hr, L"Error: failed to create window!");

		d3d.width = config.width;
		d3d.height = config.height;
		d3d.vsync = config.vsync;

		// Default to the windmill scene if nothing else was specified
		if (config.sceneFile.empty()) {
			Utils::Validate(E_FAIL, L"Error: you have to specify a GLTF scene to load (via '-scene' cmd. line parameter)!");
		}

		// Extract scene root path from sceneFile if it wasn't specified
		if (config.scenePath.empty()) {
			config.scenePath = Utils::ExtractPath(config.sceneFile);

			// Set scene file to file name only
			config.sceneFile = std::string(config.sceneFile.begin() + config.scenePath.length(), config.sceneFile.end());
		}

		// Load a model
		hr = GLTF::Load(config, resources.scene) ? S_OK : E_FAIL;
		Utils::Validate(hr, L"Error: failed to load GLTF assets!");

		// Initialize camera
		dxr.camera = resources.scene.cameras.empty() ? Camera() : resources.scene.cameras[0];

		// Initialize the shader compiler
		D3DShaders::InitShaderCompiler(shaderCompiler);

		// Initialize D3D12
		D3D12::CreateDevice(d3d);
		D3D12::CreateCommandQueue(d3d);
		D3D12::CreateCommandAllocator(d3d);
		D3D12::CreateFence(d3d);		
		D3D12::CreateSwapChain(d3d, window);
		D3D12::CreateCommandList(d3d);
		D3D12::ResetCommandList(d3d);

		// Initialize GUI
		gui.Init(d3d, window);
		gui.SetDpiScaling(Utils::GetDpiScale(window));

		// Create common resources
		D3DResources::CreateDescriptorHeaps(d3d, resources);
		DXR::CreateDescriptorHeaps(d3d, dxr, resources);
		D3DResources::CreateBackBufferRTV(d3d, resources);
		D3DResources::CreateGeometryBuffers(d3d, resources, resources.scene);
		D3DResources::CreateRaytracingDataCB(d3d, resources);
		
		// Create DXR specific resources
		DXR::CreateDXRResources(d3d, dxr, resources);
		DXR::CreateBottomLevelAS(d3d, dxr, resources, resources.scene);
		DXR::CreateTopLevelAS(d3d, dxr, resources, resources.scene);
		DXR::FillDescriptorHeaps(d3d, dxr, resources);
		D3DResources::CreateTextures(d3d, resources, resources.scene);
		DXR::CreateRTProgram(d3d, dxr, shaderCompiler);
		DXR::CreatePipelineStateObject(d3d, dxr);
		DXR::CreateShaderTable(d3d, dxr, resources);

		// Execute command list to upload GPU resources
		d3d.cmdList->Close();
		ID3D12CommandList* pGraphicsList = { d3d.cmdList };
		d3d.cmdQueue->ExecuteCommandLists(1, &pGraphicsList);

		D3D12::WaitForGPU(d3d);
		D3D12::ResetCommandList(d3d);
		
		// Release temporary resources once upload to GPU is finished
		D3DResources::ReleaseTemporaryBuffers(d3d, resources);

		// Release GLTF data once it was uploaded to GPU
		GLTF::Cleanup(resources.scene);

		lastFrameTime = std::chrono::steady_clock::now();
	}

	void Update() 
	{
		// Calculate frame time
		float elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - lastFrameTime).count() * 0.001f;
		lastFrameTime = std::chrono::steady_clock::now();

		// Process input from mouse and keyboard
		input.width = d3d.width;
		input.height = d3d.height;
		dxr.camera.aspect = float(d3d.width) / float(d3d.height);

		bool wasInput = false;
		wasInput |= Input::KeyHandler(input, dxr.camera, dxr.cameraSpeedAdjustment, elapsedTime);

		if (!gui.WantCaptureMouse()) {
			wasInput |= Input::MouseHandler(input, dxr.camera, elapsedTime);
		}

		if (input.toggleGui) d3d.renderGui = !d3d.renderGui;
		input.toggleGui = false;

		// Update GUI
		gui.Update(d3d, elapsedTime);

		// Reload shaders on request
		if (input.reloadShaders) DXR::ReloadShaders(d3d, dxr, resources, shaderCompiler);
		input.reloadShaders = false;

		// Update ray tracing data constant buffer
		D3DResources::UpdateRaytracingDataCB(d3d, dxr, resources, elapsedTime);
	}

	void Render() 
	{		
		// Run ray tracing
		DXR::BuildCommandList(d3d, dxr, resources, &gui, &input);

		// Render GUI
		gui.Render(d3d, resources);

		// End the frame and reste command list
		D3D12::Present(d3d);

		if (input.captureScreenshot) D3D12::ScreenCapture(d3d, "screenshot");
		input.captureScreenshot = false;

		D3D12::MoveToNextFrame(d3d);
		D3D12::ResetCommandList(d3d);

		// Cleanup temporary resources
		D3DResources::ReleaseTemporaryBuffers(d3d, resources);
		dxr.scratchBuffersCache.Reset();
	}

	void Cleanup() 
	{
		D3D12::WaitForGPU(d3d);
		CloseHandle(d3d.fenceEvent);

		gui.Destroy();

		DXR::Destroy(dxr);
		D3DResources::Destroy(resources);		
		D3DShaders::Destroy(shaderCompiler);
		D3D12::Destroy(d3d);

		DestroyWindow(window);
	}
	
private:
	HWND window = {};

	InputInfo input = {};

	DXRGlobal dxr = {};
	D3D12Global d3d = {};
	D3D12Resources resources = {};
	D3D12ShaderCompilerInfo shaderCompiler = {};
	Gui gui = {};
	std::chrono::steady_clock::time_point lastFrameTime = {};
};

/**
 * Program entry point.
 */
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow) 
{	
	UNREFERENCED_PARAMETER(hInstance);
	UNREFERENCED_PARAMETER(hPrevInstance);

	// Tell Windows that we're DPI aware (we handle scaling ourselves, e.g. the scaling of GUI)
	SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE);

	HRESULT hr = EXIT_SUCCESS;
	{
		MSG msg = { 0 };

		// Get the application configuration
		ConfigInfo config;
		hr = Utils::ParseCommandLine(lpCmdLine, config);
		if (hr != EXIT_SUCCESS) return hr;

		// Initialize
		DXRApplication app;
		app.Init(config);

		// Main loop
		while (WM_QUIT != msg.message) 
		{
			if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) 
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}

			app.Update();
			app.Render();
		}

		app.Cleanup();
	}

#if defined _CRTDBG_MAP_ALLOC
	_CrtDumpMemoryLeaks();
#endif

	return hr;
}