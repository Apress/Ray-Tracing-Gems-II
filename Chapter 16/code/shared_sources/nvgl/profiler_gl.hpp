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

#ifndef NV_PROFILERGL_INCLUDED
#define NV_PROFILERGL_INCLUDED

#include "extensions_gl.hpp"
#include "../nvh/profiler.hpp"

namespace nvgl {

  //////////////////////////////////////////////////////////////////////////
  /**
    # class nvgl::ProfilerGL

    ProfilerGL extends Profiler and uses `glQueryCounter(... GL_TIMESTAMP)`
    to compute the GPU time of a section.
    `glPushDebugGroup` and `glPopDebugGroup` are used within each timed
    section, so that the section names can show up in NSightGraphics,
    renderdoc or comparable tools.

  */

class ProfilerGL : public nvh::Profiler
{
public:
  // utility class to call begin/end within local scope
  class Section
  {
  public:
    Section(ProfilerGL& profiler, const char* name, bool singleShot = false)
        : m_profiler(profiler)
    {
      m_id = profiler.beginSection(name, singleShot);
    }
    ~Section() { m_profiler.endSection(m_id); }

  private:
    SectionID   m_id;
    ProfilerGL& m_profiler;
  };

  // recurring, must be within beginFrame/endFrame
  Section timeRecurring(const char* name) { return Section(*this, name, false); }

  // singleShot, results are available after FRAME_DELAY many endFrame
  Section timeSingle(const char* name) { return Section(*this, name, true); }

  //////////////////////////////////////////////////////////////////////////

  ProfilerGL(nvh::Profiler* masterProfiler = nullptr)
      : Profiler(masterProfiler)
  {
  }

  void init();
  void deinit();

  SectionID beginSection(const char* name, bool singleShot = false);
  void      endSection(SectionID slot);

private:
  void resizeQueries();

  std::vector<GLuint> m_queries;
};
}  // namespace nvgl

#endif
