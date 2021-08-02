/* Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "samplers_vk.hpp"

namespace nvvk {
//////////////////////////////////////////////////////////////////////////

void SamplerPool::deinit()
{
  if(!m_device)
    return;

  for(auto it : m_entries)
  {
    if(it.sampler)
    {
      vkDestroySampler(m_device, it.sampler, nullptr);
    }
  }

  m_freeIndex = ~0;
  m_entries.clear();
  m_samplerMap.clear();
  m_stateMap.clear();
  m_device = nullptr;
}

VkSampler SamplerPool::acquireSampler(const VkSamplerCreateInfo& createInfo)
{
  SamplerState state;
  state.createInfo = createInfo;

  const Chain* ext = (const Chain*)createInfo.pNext;
  while(ext)
  {
    switch(ext->sType)
    {
      case VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO:
        state.reduction = *(const VkSamplerReductionModeCreateInfo*)ext;
        break;
      case VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO:
        state.ycbr = *(const VkSamplerYcbcrConversionCreateInfo*)ext;
        break;
      default:
        assert(0 && "unsupported sampler create");
    }
    ext = ext->pNext;
  }
  // always remove pointers for comparison lookup
  state.createInfo.pNext = nullptr;
  state.reduction.pNext  = nullptr;
  state.ycbr.pNext       = nullptr;

  auto it = m_stateMap.find(state);
  if(it == m_stateMap.end())
  {
    uint32_t index = 0;
    if(m_freeIndex != ~0)
    {
      index       = m_freeIndex;
      m_freeIndex = m_entries[index].nextFreeIndex;
    }
    else
    {
      index = (uint32_t)m_entries.size();
      m_entries.resize(m_entries.size() + 1);
    }

    VkSampler sampler;
    VkResult  result = vkCreateSampler(m_device, &createInfo, nullptr, &sampler);
    assert(result == VK_SUCCESS);

    m_entries[index].refCount = 1;
    m_entries[index].sampler  = sampler;
    m_entries[index].state    = state;

    m_stateMap.insert({state, index});
    m_samplerMap.insert({sampler, index});

    return sampler;
  }
  else
  {
    m_entries[it->second].refCount++;
    return m_entries[it->second].sampler;
  }
}

void SamplerPool::releaseSampler(VkSampler sampler)
{
  auto it = m_samplerMap.find(sampler);
  assert(it != m_samplerMap.end());

  uint32_t index = it->second;
  Entry&   entry = m_entries[index];

  assert(entry.sampler == sampler);
  assert(entry.refCount);

  entry.refCount--;

  if(!entry.refCount)
  {
    vkDestroySampler(m_device, sampler, nullptr);
    entry.sampler       = nullptr;
    entry.nextFreeIndex = m_freeIndex;
    m_freeIndex         = index;

    m_stateMap.erase(entry.state);
    m_samplerMap.erase(sampler);
  }
}

VkSamplerCreateInfo makeSamplerCreateInfo(VkFilter             magFilter,
                                          VkFilter             minFilter,
                                          VkSamplerAddressMode addressModeU,
                                          VkSamplerAddressMode addressModeV,
                                          VkSamplerAddressMode addressModeW,
                                          VkBool32             anisotropyEnable,
                                          float                maxAnisotropy,
                                          VkSamplerMipmapMode  mipmapMode,
                                          float                minLod,
                                          float                maxLod,
                                          float                mipLodBias,
                                          VkBool32             compareEnable,
                                          VkCompareOp          compareOp,
                                          VkBorderColor        borderColor,
                                          VkBool32             unnormalizedCoordinates)
{
  VkSamplerCreateInfo samplerInfo     = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  samplerInfo.flags                   = 0;
  samplerInfo.pNext                   = nullptr;
  samplerInfo.magFilter               = magFilter;
  samplerInfo.minFilter               = minFilter;
  samplerInfo.mipmapMode              = mipmapMode;
  samplerInfo.addressModeU            = addressModeU;
  samplerInfo.addressModeV            = addressModeV;
  samplerInfo.addressModeW            = addressModeW;
  samplerInfo.anisotropyEnable        = anisotropyEnable;
  samplerInfo.maxAnisotropy           = maxAnisotropy;
  samplerInfo.borderColor             = borderColor;
  samplerInfo.unnormalizedCoordinates = unnormalizedCoordinates;
  samplerInfo.compareEnable           = compareEnable;
  samplerInfo.compareOp               = compareOp;
  return samplerInfo;
}

}  // namespace nvvk
