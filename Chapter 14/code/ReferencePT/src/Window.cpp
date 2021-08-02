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

#include "Utils.h"
#include "directxtk/Keyboard.h"
#include "directxtk/Mouse.h"

#include <iostream>

using namespace DirectX;

/**
 * Windows message loop.
 */
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) 
{
	if (Window::gui != nullptr) {
		if (Window::gui->CallWndProcHandler(hWnd, message, wParam, lParam)) return true;
	}

	PAINTSTRUCT ps;
    switch( message ) 
	{
        case WM_PAINT:
            BeginPaint( hWnd, &ps );
            EndPaint( hWnd, &ps );
            break;
		case WM_ACTIVATEAPP:
			Keyboard::ProcessMessage(message, wParam, lParam);
			Mouse::ProcessMessage(message, wParam, lParam);
			break;
		case WM_DPICHANGED:
			if (Window::gui != nullptr) {
				Window::gui->SetDpiScaling(Utils::GetDpiScale(hWnd));
			}
			break;
		case WM_INPUT:
		case WM_MOUSEMOVE:
		case WM_LBUTTONDOWN:
		case WM_LBUTTONUP:
		case WM_RBUTTONDOWN:
		case WM_RBUTTONUP:
		case WM_MBUTTONDOWN:
		case WM_MBUTTONUP:
		case WM_MOUSEWHEEL:
		case WM_XBUTTONDOWN:
		case WM_XBUTTONUP:
		case WM_MOUSEHOVER:
			Mouse::ProcessMessage(message, wParam, lParam);
			break;
		case WM_KEYDOWN:
		case WM_SYSKEYDOWN:
		case WM_KEYUP:
		case WM_SYSKEYUP:
			Keyboard::ProcessMessage(message, wParam, lParam);
			break;
		case WM_DESTROY:
			PostQuitMessage(0);
			break;
        default:
            return DefWindowProc( hWnd, message, wParam, lParam );
    }
    return 0;
}

namespace Window
{

/**
 * Create a new window.
*/
HRESULT Create(LONG width, LONG height, HINSTANCE& instance, HWND &window, LPCWSTR title, Gui* gui)
{
	Window::gui = gui;

	// Register the window class
	WNDCLASSEX wcex;
	wcex.cbSize = sizeof(WNDCLASSEX);
	wcex.style = CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc = WndProc;
	wcex.cbClsExtra = 0;
	wcex.cbWndExtra = 0;
	wcex.hInstance = instance;
	wcex.hCursor = LoadCursor(NULL, IDC_ARROW);
	wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	wcex.lpszMenuName = NULL;
	wcex.lpszClassName = L"WindowClass";
	wcex.hIcon = nullptr;
	wcex.hIconSm = nullptr;

	if (!RegisterClassEx(&wcex)) 
	{
		throw std::runtime_error("Failed to register window!");
	}

	// Get the desktop resolution
	RECT desktop;
	const HWND hDesktop = GetDesktopWindow();
	GetWindowRect(hDesktop, &desktop);

	int x = (desktop.right - width) / 2;

	// Create the window
	RECT rc = { 0, 0, width, height };
	AdjustWindowRect( &rc, WS_OVERLAPPEDWINDOW, FALSE );
	window = CreateWindow(wcex.lpszClassName, title, (WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX), x, 0, (rc.right - rc.left), (rc.bottom - rc.top), NULL, NULL, instance, NULL);
	if (!window) return E_FAIL;

	// Set the window icon
	HANDLE hIcon = LoadImageA(GetModuleHandle(NULL), "nvidia.ico", IMAGE_ICON, 0, 0, LR_DEFAULTSIZE | LR_LOADFROMFILE);
	SendMessage(window, WM_SETICON, ICON_BIG, (LPARAM)hIcon);
	
	// Show the window
	ShowWindow(window, SW_SHOWDEFAULT);
	UpdateWindow(window);

	return S_OK;
}

}