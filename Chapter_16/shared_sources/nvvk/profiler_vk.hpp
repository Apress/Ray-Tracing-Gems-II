/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "nvh/profiler.hpp"
#include <string>
#include <vulkan/vulkan_core.h>

namespace nvvk {

//////////////////////////////////////////////////////////////////////////
/**
  # class nvvk::ProfilerVK

  ProfilerVK derives from nvh::Profiler and uses vkCmdWriteTimestamp
  to measure the gpu time within a section.

  If profiler.setLabelUsage(true) was used then it will make use
  of vkCmdDebugMarkerBeginEXT and vkCmdDebugMarkerEndEXT for each
  section so that it shows up in tools like NsightGraphics and renderdoc.
  
  Currently the commandbuffers must support vkCmdResetQueryPool as well.
  
  When multiple queues are used there could be problems with the "nesting"
  of sections. In that case multiple profilers, one per queue, are most
  likely better.


  Example:

  ``` c++
  nvvk::ProfilerVK profiler;
  std::string     profilerStats;

  profiler.init(device, physicalDevice, queueFamilyIndex);
  profiler.setLabelUsage(true); // depends on VK_EXT_debug_utils
  
  while(true)
  {
    profiler.beginFrame();

    ... setup frame ...


    {
      // use the Section class to time the scope
      auto sec = profiler.timeRecurring("draw", cmd);

      vkCmdDraw(cmd, ...);
    }

    ... submit cmd buffer ...

    profiler.endFrame();

    // generic print to string
    profiler.print(profilerStats);

    // or access data directly
    nvh::Profiler::TimerInfo info;
    if( profiler.getTimerInfo("draw", info)) {
      // do some updates
      updateProfilerUi("draw", info.gpu.average);
    }
  }

  ```
*/

class ProfilerVK : public nvh::Profiler
{
public:
  // hostReset usage depends on VK_EXT_host_query_reset
  // mandatory for transfer-only queues

  //////////////////////////////////////////////////////////////////////////

  // utility class to call begin/end within local scope
  class Section
  {
  public:
    Section(ProfilerVK& profiler, const char* name, VkCommandBuffer cmd, bool singleShot = false, bool hostReset = false)
        : m_profiler(profiler)
    {
      m_id  = profiler.beginSection(name, cmd, singleShot, hostReset);
      m_cmd = cmd;
    }
    ~Section() { m_profiler.endSection(m_id, m_cmd); }

  private:
    SectionID       m_id;
    VkCommandBuffer m_cmd;
    ProfilerVK&     m_profiler;
  };

  // recurring, must be within beginFrame/endFrame
  Section timeRecurring(const char* name, VkCommandBuffer cmd, bool hostReset = false)
  {
    return Section(*this, name, cmd, false, hostReset);
  }

  // singleShot, results are available after FRAME_DELAY many endFrame
  Section timeSingle(const char* name, VkCommandBuffer cmd, bool hostReset = false)
  {
    return Section(*this, name, cmd, true, hostReset);
  }

  //////////////////////////////////////////////////////////////////////////

  ProfilerVK(nvh::Profiler* masterProfiler = nullptr)
      : Profiler(masterProfiler)
  {
    m_debugName = "nvvk::ProfilerVK:" + std::to_string((uint64_t)this);
  }

  ProfilerVK(VkDevice device, VkPhysicalDevice physicalDevice, nvh::Profiler* masterProfiler = nullptr)
      : Profiler(masterProfiler)
  {
    init(device, physicalDevice);
  }

  ~ProfilerVK() { deinit(); }

  void init(VkDevice device, VkPhysicalDevice physicalDevice, int queueFamilyIndex = 0);
  void deinit();
  void setDebugName(const std::string& name) { m_debugName = name; }

  // enable debug label per section, requires VK_EXT_debug_utils
  void setLabelUsage(bool state);

  SectionID beginSection(const char* name, VkCommandBuffer cmd, bool singleShot = false, bool hostReset = false);
  void      endSection(SectionID slot, VkCommandBuffer cmd);

  bool getSectionTime(SectionID i, uint32_t queryFrame, double& gpuTime);

private:
  void resize();
  bool m_useLabels = false;
#if 0
  bool m_useCoreHostReset = false;
#endif

  VkDevice    m_device          = VK_NULL_HANDLE;
  VkQueryPool m_queryPool       = VK_NULL_HANDLE;
  uint32_t    m_queryPoolSize   = 0;
  float       m_frequency       = 1.0f;
  uint64_t    m_queueFamilyMask = ~0;
  std::string m_debugName;
};
}  // namespace nvvk
