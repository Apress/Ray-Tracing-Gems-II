/* Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

// This is a companion utility to add debug information to an application
// See https://vulkan.lunarg.com/doc/sdk/1.1.114.0/windows/chunked_spec/chap39.html
// - User defined name to objects
// - Logically annotate region of command buffers
// - Scoped command buffer label to make thing simpler

#pragma once

#include <string>
#include <vulkan/vulkan_core.h>

namespace nvvk {

class DebugUtil
{
public:
  DebugUtil() = default;
  DebugUtil(VkDevice device)
      : m_device(device)
  {
  }

  static void setEnabled(bool state) { s_enabled = state; }

  void setup(VkDevice device) { m_device = device; }

  void setObjectName(const uint64_t object, const std::string& name, VkObjectType t)
  {
    if(s_enabled)
    {
      VkDebugUtilsObjectNameInfoEXT s{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT, nullptr, t, object, name.c_str()};
      vkSetDebugUtilsObjectNameEXT(m_device, &s);
    }
  }

  // clang-format off
  void setObjectName(VkBuffer object, const std::string& name)                  { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_BUFFER); }
  void setObjectName(VkBufferView object, const std::string& name)              { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_BUFFER_VIEW); }
  void setObjectName(VkCommandBuffer object, const std::string& name)           { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_COMMAND_BUFFER ); }
  void setObjectName(VkCommandPool object, const std::string& name)             { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_COMMAND_POOL ); }
  void setObjectName(VkDescriptorPool object, const std::string& name)          { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_DESCRIPTOR_POOL); }
  void setObjectName(VkDescriptorSet object, const std::string& name)           { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_DESCRIPTOR_SET); }
  void setObjectName(VkDescriptorSetLayout object, const std::string& name)     { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT); }
  void setObjectName(VkDeviceMemory object, const std::string& name)            { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_DEVICE_MEMORY); }
  void setObjectName(VkFramebuffer object, const std::string& name)             { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_FRAMEBUFFER); }
  void setObjectName(VkImage object, const std::string& name)                   { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_IMAGE); }
  void setObjectName(VkImageView object, const std::string& name)               { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_IMAGE_VIEW); }
  void setObjectName(VkPipeline object, const std::string& name)                { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_PIPELINE); }
  void setObjectName(VkPipelineLayout object, const std::string& name)          { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_PIPELINE_LAYOUT); }
  void setObjectName(VkQueryPool object, const std::string& name)               { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_QUERY_POOL); }
  void setObjectName(VkQueue object, const std::string& name)                   { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_QUEUE); }
  void setObjectName(VkRenderPass object, const std::string& name)              { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_RENDER_PASS); }
  void setObjectName(VkSampler object, const std::string& name)                 { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_SAMPLER); }
  void setObjectName(VkSemaphore object, const std::string& name)               { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_SEMAPHORE); }
  void setObjectName(VkShaderModule object, const std::string& name)            { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_SHADER_MODULE); }
  void setObjectName(VkSwapchainKHR object, const std::string& name)            { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_SWAPCHAIN_KHR); }

#if VK_NV_ray_tracing
  void setObjectName(VkAccelerationStructureNV object, const std::string& name)  { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_NV); }
#endif
#if VK_KHR_acceleration_structure
  void setObjectName(VkAccelerationStructureKHR object, const std::string& name) { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR); }
#endif
  // clang-format on

  //
  //---------------------------------------------------------------------------
  //
  void beginLabel(VkCommandBuffer cmdBuf, const std::string& label)
  {
    if(s_enabled)
    {
      VkDebugUtilsLabelEXT s{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT, nullptr, label.c_str(), {1.0f, 1.0f, 1.0f, 1.0f}};
      vkCmdBeginDebugUtilsLabelEXT(cmdBuf, &s);
    }
  }
  void endLabel(VkCommandBuffer cmdBuf)
  {
    if(s_enabled)
    {
      vkCmdEndDebugUtilsLabelEXT(cmdBuf);
    }
  }
  void insertLabel(VkCommandBuffer cmdBuf, const std::string& label)
  {
    if(s_enabled)
    {
      VkDebugUtilsLabelEXT s{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT, nullptr, label.c_str(), {1.0f, 1.0f, 1.0f, 1.0f}};
      vkCmdInsertDebugUtilsLabelEXT(cmdBuf, &s);
    }
  }
  //
  // Begin and End Command Label MUST be balanced, this helps as it will always close the opened label
  //
  struct ScopedCmdLabel
  {
    ScopedCmdLabel(VkCommandBuffer cmdBuf, const std::string& label)
        : m_cmdBuf(cmdBuf)
    {
      if(s_enabled)
      {
        VkDebugUtilsLabelEXT s{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT, nullptr, label.c_str(), {1.0f, 1.0f, 1.0f, 1.0f}};
        vkCmdBeginDebugUtilsLabelEXT(cmdBuf, &s);
      }
    }
    ~ScopedCmdLabel()
    {
      if(s_enabled)
      {
        vkCmdEndDebugUtilsLabelEXT(m_cmdBuf);
      }
    }
    void setLabel(const std::string& label)
    {
      if(s_enabled)
      {
        VkDebugUtilsLabelEXT s{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT, nullptr, label.c_str(), {1.0f, 1.0f, 1.0f, 1.0f}};
        vkCmdInsertDebugUtilsLabelEXT(m_cmdBuf, &s);
      }
    }

  private:
    VkCommandBuffer m_cmdBuf;
  };

  ScopedCmdLabel scopeLabel(VkCommandBuffer cmdBuf, const std::string& label) { return ScopedCmdLabel(cmdBuf, label); }

private:
  VkDevice    m_device{VK_NULL_HANDLE};
  static bool s_enabled;
};

}  // namespace nvvk
