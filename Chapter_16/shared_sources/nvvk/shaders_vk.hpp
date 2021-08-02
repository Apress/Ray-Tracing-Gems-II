/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include <assert.h>
#include <string>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace nvvk {
/**
# function in nvvk

- createShaderModule : create the shader module from various binary code inputs
- createShaderStageInfo: create the shader module and setup the stage from the incoming binary code
*/
inline VkShaderModule createShaderModule(VkDevice device, const uint32_t* binarycode, size_t sizeInBytes)
{
  VkShaderModuleCreateInfo createInfo = {};
  createInfo.sType                    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize                 = sizeInBytes;
  createInfo.pCode                    = binarycode;

  VkShaderModule shaderModule = VK_NULL_HANDLE;
  if(vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
  {
    assert(0 && "failed to create shader module!");
  }

  return shaderModule;
}

inline VkShaderModule createShaderModule(VkDevice device, const char* binarycode, size_t numInt32)
{
  return createShaderModule(device, (const uint32_t*)binarycode, numInt32 * 4);
}

inline VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code)
{
  return createShaderModule(device, (const uint32_t*)code.data(), code.size());
}

inline VkShaderModule createShaderModule(VkDevice device, const std::vector<uint32_t>& code)
{
  return createShaderModule(device, code.data(), 4 * code.size());
}

inline VkShaderModule createShaderModule(VkDevice device, const std::string& code)
{
  return createShaderModule(device, (const uint32_t*)code.data(), code.size());
}

template <typename T>
inline VkPipelineShaderStageCreateInfo createShaderStageInfo(VkDevice              device,
                                                             const std::vector<T>& code,
                                                             VkShaderStageFlagBits stage,
                                                             const char*           entryPoint = "main")
{
  VkPipelineShaderStageCreateInfo shaderStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  shaderStage.stage  = stage;
  shaderStage.module = createShaderModule(device, code);
  shaderStage.pName  = entryPoint;
  return shaderStage;
}

inline VkPipelineShaderStageCreateInfo createShaderStageInfo(VkDevice              device,
                                                             const std::string&    code,
                                                             VkShaderStageFlagBits stage,
                                                             const char*           entryPoint = "main")
{
  VkPipelineShaderStageCreateInfo shaderStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  shaderStage.stage  = stage;
  shaderStage.module = createShaderModule(device, code);
  shaderStage.pName  = entryPoint;
  return shaderStage;
}
}  // namespace nvvk
