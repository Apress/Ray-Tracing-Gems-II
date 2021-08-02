#pragma once

#include "Structures.h"
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

class Gui {
public:

	void Init(D3D12Global &d3d, HWND window);
	void Update(D3D12Global& d3d, float elapsedTime);
	void Render(D3D12Global &d3d, D3D12Resources &resources);

	bool CallWndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
	void SetDpiScaling(float newDpiScaling);

	void Destroy();
	bool WantCaptureMouse();

	void Text(const char* text);
	void Text(const char* text, double x);
	bool SliderFloat(const char* label, float* v, float min, float max);
	bool SliderInt(const char* label, int* v, int min, int max);
	bool DragFloat(const char* label, float* v, float min, float max);
	bool DragInt(const char* label, int* v, int min, int max);
	bool Combo(const char* label, int* currentItem, const char* options);
	bool Checkbox(const char* label, bool* v);
	bool Button(const char* label, float width, float height);
	void Separator();
	void SameLine();
	void Indent(float v);
private:

	struct FrameContext
	{
		ID3D12CommandAllocator* CommandAllocator;
		UINT64                  FenceValue;
	};

	float						 dpiScaling = 1.0f;

	UINT                         g_frameIndex = 0;
	HANDLE                       g_hSwapChainWaitableObject = NULL;
	static int const			 NUM_FRAMES_IN_FLIGHT = 3;
	FrameContext                 g_frameContext[NUM_FRAMES_IN_FLIGHT] = {};

	ID3D12DescriptorHeap*        g_pd3dSrvDescHeap = NULL;
};