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

#ifndef __NVPSYSTEM_H__
#define __NVPSYSTEM_H__
#pragma warning(disable : 4996)  // preventing snprintf >> _snprintf_s
//#pragma message("---------- >including nvpwindow.hpp")

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>

#ifdef WIN32
#ifdef MEMORY_LEAKS_CHECK
#pragma message("build will Check for Memory Leaks!")
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
#include <stdlib.h>
inline void* operator new(size_t size, const char* file, int line)
{
  return ::operator new(size, 1, file, line);
}

inline void __cdecl operator delete(void* ptr, const char* file, int line)
{
  ::operator delete(ptr, _NORMAL_BLOCK, file, line);
}

#define DEBUG_NEW new(__FILE__, __LINE__)
#define MALLOC_DBG(x) _malloc_dbg(x, 1, __FILE__, __LINE__);
#define malloc(x) MALLOC_DBG(x)
#define new DEBUG_NEW
#endif

#endif  // WIN32


#include <nvh/nvprint.hpp>

class NVPSystem
{
public:
  ////////////////////////////////////////////////////////////////////////
  // system related

  // exeFileName is typically argv[0]
  static void init(const char* exeFileNameWPath, const char* projectName);
  static void deinit();

  static void pollEvents();
  static void waitEvents();
  static void postTiming(float ms, int fps, const char* details = NULL);

  static double getTime();  // in seconds
  static void   sleep(double seconds);

  static std::string exePath();

  static bool isInited();

  // uses operating system specific code for sake of debugging/automated testing
  static void        windowScreenshot(GLFWwindow* glfwin, const char* filename);
  static void        windowClear(GLFWwindow* glfwin, uint32_t r, uint32_t g, uint32_t b);
  static std::string windowOpenFileDialog(GLFWwindow* glfwin, const char* title, const char* exts);

  // simple helper class, put it into your main function
  NVPSystem(const char* exeFileNameWPath, const char* projectName) { init(exeFileNameWPath, projectName); }
  ~NVPSystem() { deinit(); }

private:
  static void platformInit();
  static void platformDeinit();
};

#endif