/* Copyright (c) 2013-2018, NVIDIA CORPORATION. All rights reserved.
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
#pragma once

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <stdint.h>
#include <string>

namespace nvgl
{

/**
  # struct nvgl::ContextWindowCreateInfo
  
  Set up the context properties for a OpenGL ContextWindow.
  e.g. version, core/compatibiltiy etc.
*/

  struct ContextWindowCreateInfo {
    int         major;
    int         minor;
    int         device;

    int         MSAA;
    int         depth;
    int         stencil;
    bool        debug;
    bool        robust;
    bool        core;
    bool        forward;
    bool        stereo;
    class ContextWindow*  share;

    ContextWindowCreateInfo(int _major = 4, int _minor = 3,
      bool _core = false, int _MSAA = 0, int _depth = 24, int _stencil = 8,
      bool _debug = false, bool _robust = false,
      bool _forward = false, bool _stereo = false, class ContextWindow* _share = 0)
    {
      major = _major;
      minor = _minor;
      MSAA = _MSAA;
      depth = _depth;
      stencil = _stencil;
      core = _core;
      debug = _debug;
      robust = _robust;
      forward = _forward;
      stereo = _stereo;
      share = _share;
      device = 0;
    }
  };

/**
  # class nvgl::ContextWindow
  
  Sets up an OpenGL context from a provided `GLFWwindow`.
  Makes use of `glDebugMessageCallback` to hook up an error callback
  and loads all extensions provided by `extensions_gl.hpp`
*/

  class ContextWindow
  {
  public:
    struct ContextWindowInternalGL*  m_internal = nullptr;

    uint32_t      m_debugFilter;
    std::string   m_debugTitle;
    std::string   m_deviceName;

    ContextWindow(ContextWindow const&) = delete;
    ContextWindow& operator=(ContextWindow const&) = delete;

    ContextWindow();

    bool init(const ContextWindowCreateInfo* cflags, GLFWwindow* window, const char* dbgTitle = "test");
    void deinit();

    void swapInterval(int i);
    void swapBuffers();

    int   extensionSupported(const char* name);
    void* getProcAddress(const char* name);

    void makeContextCurrent();
    void makeContextNonCurrent();

    void screenshot(const char* filename, int x, int y, int width, int height, unsigned char* data);

    // TODO: check if this is really necessary : local method getProcAddressGL could be enough
    static void* sysGetProcAddress(const char* name);
  };

}
