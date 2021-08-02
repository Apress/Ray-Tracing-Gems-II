/* Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "Gui.h"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_win32.h"
#include "imgui/imgui_impl_dx12.h"

IMGUI_IMPL_API LRESULT  ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

void Gui::Init(D3D12Global &d3d, HWND window) {

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;

	ImGui::StyleColorsDark();

	{
		D3D12_DESCRIPTOR_HEAP_DESC desc = {};
		desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
		desc.NumDescriptors = 1;
		desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
		if (d3d.device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&g_pd3dSrvDescHeap)) != S_OK)
			return;
	}

	// Setup Platform/Renderer bindings
	ImGui_ImplWin32_Init((void*)window);
	ImGui_ImplDX12_Init(d3d.device, NUM_FRAMES_IN_FLIGHT,
		DXGI_FORMAT_R8G8B8A8_UNORM, g_pd3dSrvDescHeap,
		g_pd3dSrvDescHeap->GetCPUDescriptorHandleForHeapStart(),
		g_pd3dSrvDescHeap->GetGPUDescriptorHandleForHeapStart());
}

void Gui::Update(D3D12Global& d3d, float elapsedTime)
{
	if (!d3d.renderGui) return;

	// Start the Dear ImGui frame
	ImGui_ImplDX12_NewFrame();
	ImGui_ImplWin32_NewFrame();
	ImGui::NewFrame();

	// Size the debug window based on the application height
	ImGui::SetNextWindowSize(ImVec2(ImGui::GetWindowWidth() * dpiScaling, d3d.height - 40.f));
	ImGui::Begin("Path Tracer", NULL, ImGuiWindowFlags_AlwaysAutoResize);

	// We must select font scale inside of Begin/End
	ImGui::SetWindowFontScale(dpiScaling);
	
	Text("Frame Time: %.02fms", elapsedTime);
}

void Gui::Render(D3D12Global &d3d, D3D12Resources &resources) {
	if (!d3d.renderGui) return;

	ImGui::End();

	UINT backBufferIdx = d3d.swapChain->GetCurrentBackBufferIndex();
	d3d.cmdAlloc[d3d.frameIndex]->Reset();

	D3D12_RESOURCE_BARRIER barrier = {};
	barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
	barrier.Transition.pResource = d3d.backBuffer[d3d.frameIndex];
	barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
	barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;

	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

	D3D12_CPU_DESCRIPTOR_HANDLE renderTargetViewHandle = resources.rtvHeap->GetCPUDescriptorHandleForHeapStart();
	UINT renderTargetViewDescriptorSize = resources.rtvDescSize;
	renderTargetViewHandle.ptr += (SIZE_T(renderTargetViewDescriptorSize) * d3d.frameIndex);

	d3d.cmdList->Reset(d3d.cmdAlloc[d3d.frameIndex], NULL);
	d3d.cmdList->ResourceBarrier(1, &barrier);
	d3d.cmdList->OMSetRenderTargets(1, &renderTargetViewHandle, FALSE, NULL);
	d3d.cmdList->SetDescriptorHeaps(1, &g_pd3dSrvDescHeap);
	ImGui::Render();
	ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), d3d.cmdList);
	barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
	barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
	d3d.cmdList->ResourceBarrier(1, &barrier);
	d3d.cmdList->Close();

	d3d.cmdQueue->ExecuteCommandLists(1, (ID3D12CommandList* const*)&d3d.cmdList);
}

void Gui::SetDpiScaling(float newDpiScaling) {
	dpiScaling = newDpiScaling;
}

bool Gui::CallWndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
	return ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam);
}

bool Gui::WantCaptureMouse()
{
	ImGuiIO& io = ImGui::GetIO();
	return io.WantCaptureMouse;
}

void Gui::Text(const char* text) {
	ImGui::Text(text);
}

void Gui::Text(const char* text, double x) {
	ImGui::Text(text, x);
}

bool Gui::SliderFloat(const char* label, float* v, float min, float max) {
	return ImGui::SliderFloat(label, v, min, max);
}

bool Gui::SliderInt(const char* label, int* v, int min, int max) {
	return ImGui::SliderInt(label, v, min, max);
}

bool Gui::DragInt(const char* label, int* v, int min, int max) {
	return ImGui::DragInt(label, v, 1, min, max);
}

bool Gui::DragFloat(const char* label, float* v, float min, float max) {
	return ImGui::DragFloat(label, v, (max - min) * 0.01f, min, max);
}
bool Gui::Combo(const char* label, int* currentItem, const char* options) {
	return ImGui::Combo(label, currentItem, options);
}

bool Gui::Button(const char* label, float width, float height)
{
	return ImGui::Button(label, ImVec2(width * dpiScaling, height * dpiScaling));
}

bool Gui::Checkbox(const char* label, bool* v) {
	return ImGui::Checkbox(label, v);
}

void Gui::Separator() {
	ImGui::Separator();
}

void Gui::SameLine() {
	ImGui::SameLine();
}

void Gui::Indent(float v) {
	ImGui::Indent(v);
}

void Gui::Destroy() {
	ImGui_ImplDX12_Shutdown();
	ImGui_ImplWin32_Shutdown();
	ImGui::DestroyContext();

	g_pd3dSrvDescHeap->Release();
	g_pd3dSrvDescHeap = nullptr;
}
