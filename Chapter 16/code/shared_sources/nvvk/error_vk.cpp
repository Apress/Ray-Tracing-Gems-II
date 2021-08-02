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

#include "error_vk.hpp"

#include <nvh/nvprint.hpp>

namespace nvvk {

const char* getResultString(VkResult result)
{
  const char* resultString = "unknown";

#define STR(a)                                                                                                         \
  case a:                                                                                                              \
    resultString = #a;                                                                                                 \
    break;

  switch(result)
  {
    STR(VK_SUCCESS);
    STR(VK_NOT_READY);
    STR(VK_TIMEOUT);
    STR(VK_EVENT_SET);
    STR(VK_EVENT_RESET);
    STR(VK_INCOMPLETE);
    STR(VK_ERROR_OUT_OF_HOST_MEMORY);
    STR(VK_ERROR_OUT_OF_DEVICE_MEMORY);
    STR(VK_ERROR_INITIALIZATION_FAILED);
    STR(VK_ERROR_DEVICE_LOST);
    STR(VK_ERROR_MEMORY_MAP_FAILED);
    STR(VK_ERROR_LAYER_NOT_PRESENT);
    STR(VK_ERROR_EXTENSION_NOT_PRESENT);
    STR(VK_ERROR_FEATURE_NOT_PRESENT);
    STR(VK_ERROR_INCOMPATIBLE_DRIVER);
    STR(VK_ERROR_TOO_MANY_OBJECTS);
    STR(VK_ERROR_FORMAT_NOT_SUPPORTED);
    STR(VK_ERROR_FRAGMENTED_POOL);
    STR(VK_ERROR_OUT_OF_POOL_MEMORY);
    STR(VK_ERROR_INVALID_EXTERNAL_HANDLE);
    STR(VK_ERROR_SURFACE_LOST_KHR);
    STR(VK_ERROR_NATIVE_WINDOW_IN_USE_KHR);
    STR(VK_SUBOPTIMAL_KHR);
    STR(VK_ERROR_OUT_OF_DATE_KHR);
    STR(VK_ERROR_INCOMPATIBLE_DISPLAY_KHR);
    STR(VK_ERROR_VALIDATION_FAILED_EXT);
    STR(VK_ERROR_INVALID_SHADER_NV);
    STR(VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT);
    STR(VK_ERROR_FRAGMENTATION_EXT);
    STR(VK_ERROR_NOT_PERMITTED_EXT);
    STR(VK_ERROR_INVALID_DEVICE_ADDRESS_EXT);
    STR(VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT);
  }
#undef STR
  return resultString;
}

bool checkResult(VkResult result, const char* message)
{
  if(result == VK_SUCCESS)
  {
    return false;
  }

  if(result < 0)
  {
    if(message)
    {
      LOGE("VkResult %d - %s - %s\n", result, getResultString(result), message);
    }
    else
    {
      LOGE("VkResult %d - %s\n", result, getResultString(result));
    }
    assert(!"Critical Vulkan Error");
    return true;
  }

  return false;
}

//--------------------------------------------------------------------------------------------------
// Check the result of Vulkan and in case of error, provide a string about what happened
//
bool checkResult(VkResult result, const char* file, int32_t line)
{
  if(result == VK_SUCCESS)
  {
    return false;
  }

  if(result < 0)
  {
    LOGE("%s(%d): Vulkan Error : %s\n", file, line, getResultString(result));
    assert(!"Critical Vulkan Error");

    return true;
  }
  
  return false;
}
}  // namespace nvvk
