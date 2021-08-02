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

#include "Utils.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include <fstream>
#include <shellapi.h>
#include <unordered_map>

using namespace std;

namespace Utils
{

//--------------------------------------------------------------------------------------
// Command Line Parser
//--------------------------------------------------------------------------------------

HRESULT ParseCommandLine(LPWSTR lpCmdLine, ConfigInfo &config)
{
	LPWSTR* argv = NULL;
	int argc = 0;

	argv = CommandLineToArgvW(GetCommandLine(), &argc);
	if (argv == NULL)
	{
		MessageBox(NULL, L"Unable to parse command line!", L"Error", MB_OK);
		return E_FAIL;
	}

	if (argc > 1)
	{
		char str[1024];
		int i = 1;
		while (i < argc)
		{
			wcstombs(str, argv[i], 1024);

			if (strcmp(str, "-width") == 0)
			{
				i++;
				wcstombs(str, argv[i], 1024);
				config.width = atoi(str);
				i++;
				continue;
			}

			if (strcmp(str, "-height") == 0)
			{
				i++;
				wcstombs(str, argv[i], 1024);
				config.height = atoi(str);
				i++;
				continue;
			}

			if (strcmp(str, "-vsync") == 0)
			{
				i++;
				wcstombs(str, argv[i], 1024);
				config.vsync = (atoi(str) > 0);
				i++;
				continue;
			}

			if (strcmp(str, "-scenePath") == 0)
			{
				i++;
				wcstombs(str, argv[i], 1024);
				config.scenePath = str;
				i++;
				continue;
			}

			if (strcmp(str, "-scene") == 0)
			{
				i++;
				wcstombs(str, argv[i], 1024);
				config.sceneFile = str;
				i++;
				continue;
			}

			i++;
		}
	}
	else 
	{
		MessageBox(NULL, L"Incorrect command line usage!", L"Error", MB_OK);
		return E_FAIL;
	}

	LocalFree(argv);
	return S_OK;
}

//--------------------------------------------------------------------------------------
// Error Messaging
//--------------------------------------------------------------------------------------

void Validate(HRESULT hr, LPWSTR msg)
{
	if (FAILED(hr))
	{
		MessageBox(NULL, msg, L"Error", MB_OK);
		PostQuitMessage(EXIT_FAILURE);
	}
}

//--------------------------------------------------------------------------------------
// Misc
//--------------------------------------------------------------------------------------

UINT64 NextPowerOfTwo(UINT64 v) {
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v |= v >> 32;
	v++;

	return v;
}

float GetDpiScale(HWND window) {

	unsigned int dpi = GetDpiForWindow(window);

	const float defaultDpi = 96.0f; //< Default monitor DPI of the yesteryear
	float dpiScale = dpi / defaultDpi;

	return dpiScale;
}

std::wstring ExtractPath(std::wstring filePath) {

	auto lastSlash = wcsrchr(filePath.c_str(), '\\');
	auto lastForwardSlash = wcsrchr(filePath.c_str(), '/');

	if (lastForwardSlash && (!lastSlash || lastSlash < lastForwardSlash))
		lastSlash = lastForwardSlash;

	if (!lastSlash) return L".\\";

	return std::wstring(filePath.c_str(), lastSlash + 1);
}

std::string ExtractPath(std::string filePath) {

	auto lastSlash = strrchr(filePath.c_str(), '\\');
	auto lastForwardSlash = strrchr(filePath.c_str(), '/');

	if (lastForwardSlash && (!lastSlash || lastSlash < lastForwardSlash))
		lastSlash = lastForwardSlash;

	if (!lastSlash) return ".\\";

	return std::string(filePath.c_str(), lastSlash + 1);
}

}
