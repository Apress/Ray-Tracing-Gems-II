/* Copyright (c) 2014-2019, NVIDIA CORPORATION. All rights reserved.
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

#include <platform.h>

#include <vector>
#include <vulkan/vulkan_core.h>

namespace nvvk {

//////////////////////////////////////////////////////////////////////////
/**
  The utilities in this file provide a more direct approach, we encourage to use
  higher-level mechanisms also provided in the allocator / memorymanagement classes.

  # functions in nvvk

  - makeBufferCreateInfo : wraps setup of VkBufferCreateInfo (implicitly sets VK_BUFFER_USAGE_TRANSFER_DST_BIT)
  - makeBufferViewCreateInfo : wraps setup of VkBufferViewCreateInfo
  - createBuffer : wraps vkCreateBuffer
  - createBufferView : wraps vkCreateBufferView
  - getBufferDeviceAddressKHR : wraps vkGetBufferDeviceAddressKHR
  - getBufferDeviceAddress : wraps vkGetBufferDeviceAddress

  ~~~ C++
  VkBufferCreateInfo bufferCreate = makeBufferCreateInfo (size, VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT);
  VkBuffer buffer                 = createBuffer(device, bufferCreate);
  VkBufferView bufferView         = createBufferView(device, makeBufferViewCreateInfo(buffer, VK_FORMAT_R8G8B8A8_UNORM, size));
  ~~~
*/

// implicitly sets VK_BUFFER_USAGE_TRANSFER_DST_BIT
inline VkBufferCreateInfo makeBufferCreateInfo(VkDeviceSize size, VkBufferUsageFlags usage, VkBufferCreateFlags flags = 0)
{
  VkBufferCreateInfo createInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  createInfo.size               = size;
  createInfo.usage              = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  createInfo.flags              = flags;
  
  return createInfo;
}

inline VkBufferViewCreateInfo makeBufferViewCreateInfo(VkBuffer                buffer,
                                                       VkFormat                format,
                                                       VkDeviceSize            range,
                                                       VkDeviceSize            offset = 0,
                                                       VkBufferViewCreateFlags flags  = 0)
{
  VkBufferViewCreateInfo createInfo = {VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO};
  createInfo.buffer                 = buffer;
  createInfo.offset                 = offset;
  createInfo.range                  = range;
  createInfo.flags                  = flags;
  createInfo.format                 = format;

  return createInfo;
}

inline VkBufferViewCreateInfo makeBufferViewCreateInfo(const VkDescriptorBufferInfo& descrInfo,
                                                       VkFormat                      fmt,
                                                       VkBufferViewCreateFlags       flags = 0)
{
  VkBufferViewCreateInfo createInfo = {VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO};
  createInfo.buffer                 = descrInfo.buffer;
  createInfo.offset                 = descrInfo.offset;
  createInfo.range                  = descrInfo.range;
  createInfo.flags                  = flags;
  createInfo.format                 = fmt;

  return createInfo;
}


inline VkDeviceAddress getBufferDeviceAddressKHR(VkDevice device, VkBuffer buffer)
{
  VkBufferDeviceAddressInfo info = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR};
  info.buffer = buffer;
  return vkGetBufferDeviceAddressKHR(device, &info);
}

inline VkDeviceAddress getBufferDeviceAddress(VkDevice device, VkBuffer buffer)
{
  VkBufferDeviceAddressInfo info = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  info.buffer = buffer;
  return vkGetBufferDeviceAddress(device, &info);
}

//////////////////////////////////////////////////////////////////////////
// these use pass by value so one can easily chain createBuffer(device, makeBufferCreateInfo(...));

inline VkBuffer createBuffer(VkDevice device, VkBufferCreateInfo info)
{
  VkBuffer buffer;
  VkResult result = vkCreateBuffer(device, &info, nullptr, &buffer);
  assert(result == VK_SUCCESS);
  return buffer;
}

inline VkBufferView createBufferView(VkDevice device, VkBufferViewCreateInfo info)
{
  VkBufferView bufferView;
  VkResult result = vkCreateBufferView(device, &info, nullptr, &bufferView);
  assert(result == VK_SUCCESS);
  return bufferView;
}


}  // namespace nvvk
