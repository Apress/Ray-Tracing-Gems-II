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

#include "profiler_vk.hpp"
#include "debug_util_vk.hpp"
#include "error_vk.hpp"
#include <assert.h>


//////////////////////////////////////////////////////////////////////////

namespace nvvk {

void ProfilerVK::init(VkDevice device, VkPhysicalDevice physicalDevice, int queueFamilyIndex)
{
  assert(!m_device);
  m_device = device;
#if 0
  m_useCoreHostReset = supportsCoreHostReset;
#endif

  VkPhysicalDeviceProperties properties;
  vkGetPhysicalDeviceProperties(physicalDevice, &properties);

  m_frequency = properties.limits.timestampPeriod;

  std::vector<VkQueueFamilyProperties> queueProperties;
  uint32_t                             queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
  queueProperties.resize(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueProperties.data());

  uint32_t validBits = queueProperties[queueFamilyIndex].timestampValidBits;

  m_queueFamilyMask  = validBits == 64 ? uint64_t(-1) : ((uint64_t(1) << validBits) - uint64_t(1));


  resize();
}

void ProfilerVK::deinit()
{
  if(m_queryPool)
  {
    vkDestroyQueryPool(m_device, m_queryPool, nullptr);
    m_queryPool = VK_NULL_HANDLE;
  }
  m_device = VK_NULL_HANDLE;
}

void ProfilerVK::setLabelUsage(bool state)
{
  m_useLabels = state;
}

void ProfilerVK::resize()
{
  if(getRequiredTimers() < m_queryPoolSize)
    return;

  if(m_queryPool)
  {
    // FIXME we may loose results this way
    // not exactly efficient, but when timers changed a lot, we have a slow frame anyway
    // cleaner would be allocating more pools
    VkResult result = vkDeviceWaitIdle(m_device);
    if(nvvk::checkResult(result, __FILE__, __LINE__))
    {
      exit(-1);
    }
    vkDestroyQueryPool(m_device, m_queryPool, nullptr);
  }

  VkQueryPoolCreateInfo createInfo = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
  createInfo.queryType             = VK_QUERY_TYPE_TIMESTAMP;
  createInfo.queryCount            = getRequiredTimers();
  m_queryPoolSize                  = createInfo.queryCount;

  VkResult res = vkCreateQueryPool(m_device, &createInfo, nullptr, &m_queryPool);
  assert(res == VK_SUCCESS);

  nvvk::DebugUtil(m_device).setObjectName(m_queryPool, m_debugName);
}

nvh::Profiler::SectionID ProfilerVK::beginSection(const char* name, VkCommandBuffer cmd, bool singleShot, bool useHostReset)
{
  nvh::Profiler::gpuTimeProvider_fn fnProvider = [&](SectionID i, uint32_t queryFrame, double& gpuTime) {
    return getSectionTime(i, queryFrame, gpuTime);
  };


  SectionID slot = Profiler::beginSection(name, "VK ", fnProvider, singleShot);
  if(getRequiredTimers() > m_queryPoolSize)
  {
    resize();
  }
  if(m_useLabels)
  {
    VkDebugUtilsLabelEXT label = {VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
    label.pLabelName           = name;
    label.color[1]             = 1.0f;
    vkCmdBeginDebugUtilsLabelEXT(cmd, &label);
  }
#if 0
  else if(m_useMarkers)
  {
    VkDebugMarkerMarkerInfoEXT marker = {VK_STRUCTURE_TYPE_DEBUG_MARKER_MARKER_INFO_EXT};
    marker.pMarkerName                = name;
    vkCmdDebugMarkerBeginEXT(cmd, &marker);
  }
#endif

  uint32_t idx = getTimerIdx(slot, getSubFrame(slot), true);

  if(useHostReset)
  {
#if 0
    if(m_useCoreHostReset)
    {
      vkResetQueryPool(m_device, m_queryPool, idx, 2);
    }
    else
#endif
    {
      vkResetQueryPoolEXT(m_device, m_queryPool, idx, 2);
    }
  }
  else
  {
    // not ideal to do this per query
    vkCmdResetQueryPool(cmd, m_queryPool, idx, 2);
  }
  // log timestamp
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_queryPool, idx);

  return slot;
}

void ProfilerVK::endSection(SectionID slot, VkCommandBuffer cmd)
{
  uint32_t idx = getTimerIdx(slot, getSubFrame(slot), false);
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_queryPool, idx);
  if(m_useLabels)
  {
    vkCmdEndDebugUtilsLabelEXT(cmd);
  }
#if 0
  else if(m_useMarkers)
  {
    vkCmdDebugMarkerEndEXT(cmd);
  }
#endif
  Profiler::endSection(slot);
}


bool ProfilerVK::getSectionTime(SectionID i, uint32_t queryFrame, double& gpuTime)
{
  bool     isRecurring = isSectionRecurring(i);
  uint32_t idxBegin    = getTimerIdx(i, queryFrame, true);
  uint32_t idxEnd      = getTimerIdx(i, queryFrame, false);

  uint64_t times[2];
  VkResult result = vkGetQueryPoolResults(m_device, m_queryPool, idxBegin, 2, sizeof(uint64_t) * 2, times, sizeof(uint64_t),
                                          VK_QUERY_RESULT_64_BIT | (isRecurring ? VK_QUERY_RESULT_WAIT_BIT : 0));
  // validation layer bug, complains if VK_QUERY_RESULT_WAIT_BIT is not provided, even if we wait
  // through another fence for the buffer containing the problem
  // fixed in VK SDK fixed with 1.1.126, but we keep old logic still here

  if(result == VK_SUCCESS)
  {
    uint64_t mask = m_queueFamilyMask;
    gpuTime       = (double((times[1] & mask) - (times[0] & mask)) * double(m_frequency)) / double(1000);
    return true;
  }
  else
  {
    return false;
  }
}

}  // namespace nvvk
