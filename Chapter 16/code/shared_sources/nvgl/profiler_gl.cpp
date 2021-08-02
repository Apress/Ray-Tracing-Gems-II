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

#include "profiler_gl.hpp"
#include <assert.h>


//////////////////////////////////////////////////////////////////////////

namespace nvgl {

void ProfilerGL::resizeQueries()
{
  uint32_t timers = getRequiredTimers();
  uint32_t old    = (uint32_t)m_queries.size();

  if(old == timers)
  {
    return;
  }

  m_queries.resize(timers, 0);
  uint32_t add = timers - old;
  glGenQueries(add, &m_queries[old]);
}

void ProfilerGL::init()
{
  resizeQueries();
}

void ProfilerGL::deinit()
{
  if(m_queries.empty())
    return;

  glDeleteQueries((GLuint)m_queries.size(), m_queries.data());
  m_queries.clear();
}


nvh::Profiler::SectionID ProfilerGL::beginSection(const char* name, bool singleShot)
{
  nvh::Profiler::gpuTimeProvider_fn fnProvider = [&](SectionID i, uint32_t queryFrame, double& gpuTime) 
  { 
    uint32_t idxBegin = getTimerIdx(i, queryFrame, true);
    uint32_t idxEnd   = getTimerIdx(i, queryFrame, false);

    GLint available = 0;
    glGetQueryObjectiv(m_queries[idxEnd], GL_QUERY_RESULT_AVAILABLE, &available);

    if(available)
    {
      GLuint64 beginTime;
      GLuint64 endTime;
      glGetQueryObjectui64v(m_queries[idxBegin], GL_QUERY_RESULT, &beginTime);
      glGetQueryObjectui64v(m_queries[idxEnd], GL_QUERY_RESULT, &endTime);

      gpuTime = double(endTime - beginTime) / double(1000);

      return true;
    }
    else
    {
      return false;
    }
  };

  SectionID slot = Profiler::beginSection(name, "GL ", fnProvider, singleShot);

  if(m_queries.size() != getRequiredTimers())
  {
    resizeQueries();
  }

  glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, name);

  uint32_t idx  = getTimerIdx(slot, getSubFrame(slot), true);
  glQueryCounter(m_queries[idx], GL_TIMESTAMP);

  return slot;
}

void ProfilerGL::endSection(SectionID slot)
{
  uint32_t idx = getTimerIdx(slot, getSubFrame(slot), false);
  glQueryCounter(m_queries[idx], GL_TIMESTAMP);
  glPopDebugGroup();
  Profiler::endSection(slot);
}


}  // namespace nvgl
