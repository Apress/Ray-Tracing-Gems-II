/* Copyright (c) 2014-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef NV_WINDOWPROFILER_GL_INCLUDED
#define NV_WINDOWPROFILER_GL_INCLUDED

#include <nvh/appwindowprofiler.hpp>
#include "profiler_gl.hpp"
#include "contextwindow_gl.hpp"

//////////////////////////////////////////////////////////////////////////
/**
  # class nvgl::AppWindowProfilerGL

  AppWindowProfilerGL derives from nvh::AppWindowProfiler
  and overrides the context and swapbuffer functions.
  
  To influence the context creation modify
  `m_contextInfo` prior running AppWindowProfiler::run,
  which triggers window, and context creation etc.

  The class comes with a nvgl::ProfilerGL instance that references the 
  AppWindowProfiler::m_profiler's data.
*/

namespace nvgl
{

  #define NV_PROFILE_GL_SECTION(name)  nvgl::ProfilerGL::Section _tempTimer(m_profilerGL, name)
  #define NV_PROFILE_GL_SPLIT()        m_profilerGL.accumulationSplit()

  class AppWindowProfilerGL : public nvh::AppWindowProfiler {
  public:
    
    AppWindowProfilerGL(bool singleThreaded = true, bool doSwap = true)
        : nvh::AppWindowProfiler(singleThreaded, doSwap)
        , m_profilerGL(&m_profiler)
    {
      m_contextInfo.robust = false;
      m_contextInfo.core   = false;
#ifdef NDEBUG
      m_contextInfo.debug  = false;
#else
      m_contextInfo.debug  = true;
#endif
      m_contextInfo.share  = NULL;
      m_contextInfo.major = 4;
      m_contextInfo.minor = 5;
    }

    nvgl::ContextWindowCreateInfo   m_contextInfo;
    ContextWindow        m_contextWindow;
    
    nvgl::ProfilerGL       m_profilerGL;


    int run(const std::string& name, int argc, const char** argv, int width, int height)
    {
      return AppWindowProfiler::run(name, argc, argv, width, height, true);
    }

    virtual void  contextInit() override;
    virtual void  contextDeinit() override;

    virtual void swapResize(int width, int height) override
    {
      m_windowState.m_swapSize[0] = width;
      m_windowState.m_swapSize[1] = height;
    }
    virtual void swapPrepare() override { }
    virtual void swapBuffers() override { m_contextWindow.swapBuffers(); }
    virtual void swapVsync(bool state) override { m_contextWindow.swapInterval(state ? 1 : 0); }
    virtual const char* contextGetDeviceName() override { return m_contextWindow.m_deviceName.c_str(); }
  };
}


#endif


