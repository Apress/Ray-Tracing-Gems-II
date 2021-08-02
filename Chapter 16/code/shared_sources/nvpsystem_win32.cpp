/*-----------------------------------------------------------------------
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
*/ //--------------------------------------------------------------------

#include "nvpsystem.hpp"

#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <commdlg.h>
#include <windows.h>
#include <windowsx.h>

#include "resources.h"

#include <vector>
#include <algorithm>
#include <io.h>
#include <stdio.h>
#include <string>

#ifdef USESOCKETS
#include "socketSampleMessages.h"
#endif

// from https://docs.microsoft.com/en-us/windows/desktop/gdi/capturing-an-image

static int CaptureAnImage(HWND hWnd, const char* filename)
{
  HDC     hdcWindow;
  HDC     hdcMemDC  = NULL;
  HBITMAP hbmScreen = NULL;
  BITMAP  bmpScreen;

  // Retrieve the handle to a display device context for the client
  // area of the window.
  hdcWindow = GetDC(hWnd);

  // Create a compatible DC which is used in a BitBlt from the window DC
  hdcMemDC = CreateCompatibleDC(hdcWindow);

  if(!hdcMemDC)
  {
    LOGE("CreateCompatibleDC has failed\n");
    goto done;
  }

  // Get the client area for size calculation
  RECT rcClient;
  GetClientRect(hWnd, &rcClient);

  // Create a compatible bitmap from the Window DC
  hbmScreen = CreateCompatibleBitmap(hdcWindow, rcClient.right - rcClient.left, rcClient.bottom - rcClient.top);

  if(!hbmScreen)
  {
    LOGE("CreateCompatibleBitmap Failed\n");
    goto done;
  }

  // Select the compatible bitmap into the compatible memory DC.
  SelectObject(hdcMemDC, hbmScreen);

  // Bit block transfer into our compatible memory DC.
  if(!BitBlt(hdcMemDC, 0, 0, rcClient.right - rcClient.left, rcClient.bottom - rcClient.top, hdcWindow, 0, 0, SRCCOPY))
  {
    LOGE("BitBlt has failed\n");
    goto done;
  }

  // Get the BITMAP from the HBITMAP
  GetObject(hbmScreen, sizeof(BITMAP), &bmpScreen);

  BITMAPFILEHEADER bmfHeader;
  BITMAPINFOHEADER bi;

  bi.biSize          = sizeof(BITMAPINFOHEADER);
  bi.biWidth         = bmpScreen.bmWidth;
  bi.biHeight        = bmpScreen.bmHeight;
  bi.biPlanes        = 1;
  bi.biBitCount      = 32;
  bi.biCompression   = BI_RGB;
  bi.biSizeImage     = 0;
  bi.biXPelsPerMeter = 0;
  bi.biYPelsPerMeter = 0;
  bi.biClrUsed       = 0;
  bi.biClrImportant  = 0;

  DWORD dwBmpSize = ((bmpScreen.bmWidth * bi.biBitCount + 31) / 32) * 4 * bmpScreen.bmHeight;

  // Starting with 32-bit Windows, GlobalAlloc and LocalAlloc are implemented as wrapper functions that
  // call HeapAlloc using a handle to the process's default heap. Therefore, GlobalAlloc and LocalAlloc
  // have greater overhead than HeapAlloc.
  HANDLE hDIB     = GlobalAlloc(GHND, dwBmpSize);
  char*  lpbitmap = (char*)GlobalLock(hDIB);

  // Gets the "bits" from the bitmap and copies them into a buffer
  // which is pointed to by lpbitmap.
  GetDIBits(hdcWindow, hbmScreen, 0, (UINT)bmpScreen.bmHeight, lpbitmap, (BITMAPINFO*)&bi, DIB_RGB_COLORS);

  // A file is created, this is where we will save the screen capture.
  HANDLE hFile = CreateFileA(filename, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

  // Add the size of the headers to the size of the bitmap to get the total file size
  DWORD dwSizeofDIB = dwBmpSize + sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);

  //Offset to where the actual bitmap bits start.
  bmfHeader.bfOffBits = (DWORD)sizeof(BITMAPFILEHEADER) + (DWORD)sizeof(BITMAPINFOHEADER);

  //Size of the file
  bmfHeader.bfSize = dwSizeofDIB;

  //bfType must always be BM for Bitmaps
  bmfHeader.bfType = 0x4D42;  //BM

  DWORD dwBytesWritten = 0;
  WriteFile(hFile, (LPSTR)&bmfHeader, sizeof(BITMAPFILEHEADER), &dwBytesWritten, NULL);
  WriteFile(hFile, (LPSTR)&bi, sizeof(BITMAPINFOHEADER), &dwBytesWritten, NULL);
  WriteFile(hFile, (LPSTR)lpbitmap, dwBmpSize, &dwBytesWritten, NULL);
#ifdef USESOCKETS
  // TODO!!
  unsigned char* data = NULL;
  size_t         sz   = 0;
  int            w = 0, h = 0;
  ::postScreenshot(data, sz, w, h);
#endif

  //Unlock and Free the DIB from the heap
  GlobalUnlock(hDIB);
  GlobalFree(hDIB);

  //Close the handle for the file that was created
  CloseHandle(hFile);

  //Clean up
done:
  DeleteObject(hbmScreen);
  DeleteObject(hdcMemDC);
  ReleaseDC(hWnd, hdcWindow);

  return 0;
}

void NVPSystem::windowScreenshot(GLFWwindow* glfwin, const char* filename)
{
  CaptureAnImage(glfwGetWin32Window(glfwin), filename);
}

void NVPSystem::windowClear(GLFWwindow* glfwin, uint32_t r, uint32_t g, uint32_t b)
{
  HWND hwnd = glfwGetWin32Window(glfwin);

  HDC hdcWindow = GetDC(hwnd);

  RECT rcClient;
  GetClientRect(hwnd, &rcClient);
  HBRUSH hbr = CreateSolidBrush(RGB(r, g, b));

  FillRect(hdcWindow, &rcClient, hbr);

  ReleaseDC(hwnd, hdcWindow);
  DeleteBrush(hbr);
}

std::string NVPSystem::windowOpenFileDialog(GLFWwindow* glfwin, const char* title, const char* exts)
{
  HWND hwnd = glfwGetWin32Window(glfwin);

  std::string       filename;
  std::vector<char> extsfixed;
  for(size_t i = 0; i < strlen(exts); i++)
  {
    if(exts[i] == '|')
    {
      extsfixed.push_back(0);
    }
    else
    {
      extsfixed.push_back(exts[i]);
    }
  }
  extsfixed.push_back(0);
  extsfixed.push_back(0);

  OPENFILENAME ofn;           // common dialog box structure
  char         szFile[1024];  // buffer for file name

  // Initialize OPENFILENAME
  ZeroMemory(&ofn, sizeof(ofn));
  ofn.lStructSize = sizeof(ofn);
  ofn.hwndOwner   = hwnd;
  ofn.lpstrFile   = szFile;
  // Set lpstrFile[0] to '\0' so that GetOpenFileName does not
  // use the contents of szFile to initialize itself.
  ofn.lpstrFile[0]    = '\0';
  ofn.nMaxFile        = sizeof(szFile);
  ofn.lpstrFilter     = extsfixed.data();
  ofn.nFilterIndex    = 1;
  ofn.lpstrFileTitle  = NULL;
  ofn.nMaxFileTitle   = 0;
  ofn.lpstrInitialDir = NULL;
  ofn.Flags           = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
  ofn.lpstrTitle      = title;

  // Display the Open dialog box.

  if(GetOpenFileNameA(&ofn) == TRUE)
  {
    return ofn.lpstrFile;
  }
  return filename;
}

void NVPSystem::sleep(double seconds)
{
  ::Sleep(DWORD(seconds * 1000.0));
}

void NVPSystem::platformInit()
{
#ifdef MEMORY_LEAKS_CHECK
  _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
  _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG | _CRTDBG_MODE_WNDW);
#endif
}

void NVPSystem::platformDeinit()
{
#ifdef MEMORY_LEAKS_CHECK
  _CrtDumpMemoryLeaks();
#endif
}
