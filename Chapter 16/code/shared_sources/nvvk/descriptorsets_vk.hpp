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

#include <assert.h>
#include <platform.h>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace nvvk {


// utility for additional feature support
enum class DescriptorSupport : uint32_t
{
  CORE_1_0     = 0,  // VK Version 1.0
  CORE_1_2     = 1,  // VK Version 1.2 (adds descriptor_indexing)
  INDEXING_EXT = 2,  // VK_EXT_descriptor_indexing
};
using DescriptorSupport_t = std::underlying_type_t<DescriptorSupport>;
inline DescriptorSupport operator|(DescriptorSupport lhs, DescriptorSupport rhs)
{
  return static_cast<DescriptorSupport>(static_cast<DescriptorSupport_t>(lhs) | static_cast<DescriptorSupport_t>(rhs));
}
inline DescriptorSupport operator&(DescriptorSupport lhs, DescriptorSupport rhs)
{
  return static_cast<DescriptorSupport>(static_cast<DescriptorSupport_t>(lhs) & static_cast<DescriptorSupport_t>(rhs));
}
inline bool isSet(DescriptorSupport test, DescriptorSupport query)
{
  return (test & query) == query;
}
inline bool isAnySet(DescriptorSupport test, DescriptorSupport query)
{
  return (test & query) != DescriptorSupport::CORE_1_0;
}

/**
  # functions in nvvk

  - createDescriptorPool : wrappers for vkCreateDescriptorPool
  - allocateDescriptorSet : allocates a single VkDescriptorSet
  - allocateDescriptorSets : allocates multiple VkDescriptorSets

*/

inline VkDescriptorPool createDescriptorPool(VkDevice device, size_t poolSizeCount, const VkDescriptorPoolSize* poolSizes, uint32_t maxSets)
{
  VkResult result;

  VkDescriptorPool           descrPool;
  VkDescriptorPoolCreateInfo descrPoolInfo = {};
  descrPoolInfo.sType                      = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  descrPoolInfo.pNext                      = nullptr;
  descrPoolInfo.maxSets                    = maxSets;
  descrPoolInfo.poolSizeCount              = uint32_t(poolSizeCount);
  descrPoolInfo.pPoolSizes                 = poolSizes;

  // scene pool
  result = vkCreateDescriptorPool(device, &descrPoolInfo, nullptr, &descrPool);
  assert(result == VK_SUCCESS);
  return descrPool;
}

inline VkDescriptorPool createDescriptorPool(VkDevice device, const std::vector<VkDescriptorPoolSize>& poolSizes, uint32_t maxSets)
{
  return createDescriptorPool(device, poolSizes.size(), poolSizes.data(), maxSets);
}

#ifdef VULKAN_HPP
inline VkDescriptorPool createDescriptorPool(vk::Device device, const std::vector<vk::DescriptorPoolSize>& poolSizes, uint32_t maxSets)
{
  return createDescriptorPool(device, poolSizes.size(), reinterpret_cast<const VkDescriptorPoolSize*>(poolSizes.data()), maxSets);
}
#endif

inline VkDescriptorSet allocateDescriptorSet(VkDevice device, VkDescriptorPool pool, VkDescriptorSetLayout layout)
{
  VkResult                    result;
  VkDescriptorSetAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  allocInfo.descriptorPool              = pool;
  allocInfo.descriptorSetCount          = 1;
  allocInfo.pSetLayouts                 = &layout;

  VkDescriptorSet set;
  result = vkAllocateDescriptorSets(device, &allocInfo, &set);
  assert(result == VK_SUCCESS);
  return set;
}

inline void allocateDescriptorSets(VkDevice                      device,
                                   VkDescriptorPool              pool,
                                   VkDescriptorSetLayout         layout,
                                   uint32_t                      count,
                                   std::vector<VkDescriptorSet>& sets)
{
  sets.resize(count);
  std::vector<VkDescriptorSetLayout> layouts(count, layout);

  VkResult                    result;
  VkDescriptorSetAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  allocInfo.descriptorPool              = pool;
  allocInfo.descriptorSetCount          = count;
  allocInfo.pSetLayouts                 = layouts.data();

  result = vkAllocateDescriptorSets(device, &allocInfo, sets.data());
  assert(result == VK_SUCCESS);
}
#ifdef VULKAN_HPP
inline void allocateDescriptorSets(vk::Device                      device,
                                   vk::DescriptorPool              pool,
                                   vk::DescriptorSetLayout         layout,
                                   uint32_t                        count,
                                   std::vector<vk::DescriptorSet>& sets)
{
  allocateDescriptorSets(device, pool, layout, count, reinterpret_cast<std::vector<VkDescriptorSet>&>(sets));
}
#endif

/////////////////////////////////////////////////////////////////////////////
/**
  # class nvvk::DescriptorSetBindings

  Helper class that keeps a vector of `VkDescriptorSetLayoutBinding` for a single
  `VkDescriptorSetLayout`. Provides helper functions to create `VkDescriptorSetLayout`
  as well as `VkDescriptorPool` based on this information, as well as utilities
  to fill the `VkWriteDescriptorSet` structure with binding information stored
  within the class.

  The class comes with the convenience functionality that when you make a
  VkWriteDescriptorSet you provide the binding slot, rather than the
  index of the binding's storage within this class. This results in a small
  linear search, but makes it easy to change the content/order of bindings
  at creation time.

  Example :
  ~~~C++
  DescriptorSetBindings binds;

  binds.addBinding( VIEW_BINDING, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT);
  binds.addBinding(XFORM_BINDING, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT);

  VkDescriptorSetLayout layout = binds.createLayout(device);

  #if SINGLE_LAYOUT_POOL
    // let's create a pool with 2 sets
    VkDescriptorPool      pool   = binds.createPool(device, 2);
  #else
    // if you want to combine multiple layouts into a common pool
    std::vector<VkDescriptorPoolSize> poolSizes;
    bindsA.addRequiredPoolSizes(poolSizes, numSetsA);
    bindsB.addRequiredPoolSizes(poolSizes, numSetsB);
    VkDescriptorPool      pool   = nvvk::createDescriptorPool(device, poolSizes,
                                                              numSetsA + numSetsB);
  #endif

  // fill them
  std::vector<VkWriteDescriptorSet> updates;

  updates.push_back(binds.makeWrite(0, VIEW_BINDING, &view0BufferInfo));
  updates.push_back(binds.makeWrite(1, VIEW_BINDING, &view1BufferInfo));
  updates.push_back(binds.makeWrite(0, XFORM_BINDING, &xform0BufferInfo));
  updates.push_back(binds.makeWrite(1, XFORM_BINDING, &xform1BufferInfo));

  vkUpdateDescriptorSets(device, updates.size(), updates.data(), 0, nullptr);
  ~~~
*/

class DescriptorSetBindings
{
public:
  DescriptorSetBindings() = default;
  DescriptorSetBindings(const std::vector<VkDescriptorSetLayoutBinding>& bindings)
      : m_bindings(bindings)
  {
  }

  // Add a binding to the descriptor set
  void addBinding(uint32_t binding,          // Slot to which the descriptor will be bound, corresponding to the layout
                                             // binding index in the shader
                  VkDescriptorType   type,   // Type of the bound descriptor(s)
                  uint32_t           count,  // Number of descriptors
                  VkShaderStageFlags stageFlags,  // Shader stages at which the bound resources will be available
                  const VkSampler*   pImmutableSampler = nullptr  // Corresponding sampler, in case of textures
  )
  {
    m_bindings.push_back({binding, type, count, stageFlags, pImmutableSampler});
  }

  void addBinding(const VkDescriptorSetLayoutBinding& layoutBinding) { m_bindings.emplace_back(layoutBinding); }

  void setBindings(const std::vector<VkDescriptorSetLayoutBinding>& bindings) { m_bindings = bindings; }

  // requires use of SUPPORT_INDEXING_EXT/SUPPORT_INDEXING_V1_2 on createLayout
  void setBindingFlags(uint32_t binding, VkDescriptorBindingFlags bindingFlags);

  void clear()
  {
    m_bindings.clear();
    m_bindingFlags.clear();
  }
  bool                                empty() const { return m_bindings.empty(); }
  size_t                              size() const { return m_bindings.size(); }
  const VkDescriptorSetLayoutBinding* data() const { return m_bindings.data(); }

  VkDescriptorType getType(uint32_t binding) const;
  uint32_t         getCount(uint32_t binding) const;


  // Once the bindings have been added, this generates the descriptor layout corresponding to the
  // bound resources.
  VkDescriptorSetLayout createLayout(VkDevice                         device,
                                     VkDescriptorSetLayoutCreateFlags flags        = 0,
                                     DescriptorSupport                supportFlags = DescriptorSupport::CORE_1_0) const;

  // Once the bindings have been added, this generates the descriptor pool with enough space to
  // handle all the bound resources and allocate up to maxSets descriptor sets
  VkDescriptorPool createPool(VkDevice device, uint32_t maxSets = 1) const;

  // appends the required poolsizes for N sets
  void addRequiredPoolSizes(std::vector<VkDescriptorPoolSize>& poolSizes, uint32_t numSets) const;

  // provide single element
  VkWriteDescriptorSet makeWrite(VkDescriptorSet dstSet, uint32_t dstBinding, uint32_t arrayElement = 0) const;
  VkWriteDescriptorSet makeWrite(VkDescriptorSet              dstSet,
                                 uint32_t                     dstBinding,
                                 const VkDescriptorImageInfo* pImageInfo,
                                 uint32_t                     arrayElement = 0) const;
  VkWriteDescriptorSet makeWrite(VkDescriptorSet               dstSet,
                                 uint32_t                      dstBinding,
                                 const VkDescriptorBufferInfo* pBufferInfo,
                                 uint32_t                      arrayElement = 0) const;
  VkWriteDescriptorSet makeWrite(VkDescriptorSet     dstSet,
                                 uint32_t            dstBinding,
                                 const VkBufferView* pTexelBufferView,
                                 uint32_t            arrayElement = 0) const;
#if VK_NV_ray_tracing
  VkWriteDescriptorSet makeWrite(VkDescriptorSet                                    dstSet,
                                 uint32_t                                           dstBinding,
                                 const VkWriteDescriptorSetAccelerationStructureNV* pAccel,
                                 uint32_t                                           arrayElement = 0) const;
#endif
#if VK_KHR_acceleration_structure
  VkWriteDescriptorSet makeWrite(VkDescriptorSet                                     dstSet,
                                 uint32_t                                            dstBinding,
                                 const VkWriteDescriptorSetAccelerationStructureKHR* pAccel,
                                 uint32_t                                            arrayElement = 0) const;
#endif
#if VK_EXT_inline_uniform_block
  VkWriteDescriptorSet makeWrite(VkDescriptorSet                                  dstSet,
                                 uint32_t                                         dstBinding,
                                 const VkWriteDescriptorSetInlineUniformBlockEXT* pInlineUniform,
                                 uint32_t                                         arrayElement = 0) const;
#endif
  // provide full array
  VkWriteDescriptorSet makeWriteArray(VkDescriptorSet dstSet, uint32_t dstBinding) const;
  VkWriteDescriptorSet makeWriteArray(VkDescriptorSet dstSet, uint32_t dstBinding, const VkDescriptorImageInfo* pImageInfo) const;
  VkWriteDescriptorSet makeWriteArray(VkDescriptorSet dstSet, uint32_t dstBinding, const VkDescriptorBufferInfo* pBufferInfo) const;
  VkWriteDescriptorSet makeWriteArray(VkDescriptorSet dstSet, uint32_t dstBinding, const VkBufferView* pTexelBufferView) const;
#if VK_NV_ray_tracing
  VkWriteDescriptorSet makeWriteArray(VkDescriptorSet                                    dstSet,
                                      uint32_t                                           dstBinding,
                                      const VkWriteDescriptorSetAccelerationStructureNV* pAccel) const;
#endif
#if VK_KHR_acceleration_structure
  VkWriteDescriptorSet makeWriteArray(VkDescriptorSet                                     dstSet,
                                      uint32_t                                            dstBinding,
                                      const VkWriteDescriptorSetAccelerationStructureKHR* pAccel) const;
#endif
#if VK_EXT_inline_uniform_block
  VkWriteDescriptorSet makeWriteArray(VkDescriptorSet                                  dstSet,
                                      uint32_t                                         dstBinding,
                                      const VkWriteDescriptorSetInlineUniformBlockEXT* pInline) const;
#endif
#ifdef VULKAN_HPP
  void addBinding(uint32_t binding,           // Slot to which the descriptor will be bound, corresponding to the layout
                                              // binding index in the shader
                  vk::DescriptorType   type,  // Type of the bound descriptor(s)
                  uint32_t             count,       // Number of descriptors
                  vk::ShaderStageFlags stageFlags,  // Shader stages at which the bound resources will be available
                  const vk::Sampler*   pImmutableSampler = nullptr  // Corresponding sampler, in case of textures
  )
  {
    m_bindings.push_back({binding, static_cast<VkDescriptorType>(type), count, static_cast<VkShaderStageFlags>(stageFlags),
                          reinterpret_cast<const VkSampler*>(pImmutableSampler)});
  }
  void setBindings(const std::vector<vk::DescriptorSetLayoutBinding>& bindings)
  {
    setBindings(reinterpret_cast<const std::vector<VkDescriptorSetLayoutBinding>&>(bindings));
  }

  void setBindingFlags(uint32_t binding, vk::DescriptorBindingFlags bindingFlags)
  {
    setBindingFlags(binding, static_cast<VkDescriptorBindingFlags>(bindingFlags));
  }

  void addRequiredPoolSizes(std::vector<vk::DescriptorPoolSize>& poolSizes, uint32_t numSets) const
  {
    addRequiredPoolSizes(reinterpret_cast<std::vector<VkDescriptorPoolSize>&>(poolSizes), numSets);
  }

  vk::WriteDescriptorSet makeWrite(vk::DescriptorSet              dstSet,
                                   uint32_t                       dstBinding,
                                   const vk::DescriptorImageInfo* pImageInfo,
                                   uint32_t                       arrayElement = 0) const
  {
    return makeWrite(dstSet, dstBinding, reinterpret_cast<const VkDescriptorImageInfo*>(pImageInfo), arrayElement);
  }
  vk::WriteDescriptorSet makeWrite(vk::DescriptorSet               dstSet,
                                   uint32_t                        dstBinding,
                                   const vk::DescriptorBufferInfo* pBufferInfo,
                                   uint32_t                        arrayElement = 0) const
  {
    return makeWrite(dstSet, dstBinding, reinterpret_cast<const VkDescriptorBufferInfo*>(pBufferInfo), arrayElement);
  }
  vk::WriteDescriptorSet makeWrite(vk::DescriptorSet     dstSet,
                                   uint32_t              dstBinding,
                                   const vk::BufferView* pTexelBufferView,
                                   uint32_t              arrayElement = 0) const
  {
    return makeWrite(dstSet, dstBinding, reinterpret_cast<const VkBufferView*>(pTexelBufferView), arrayElement);
  }
#if VK_NV_ray_tracing
  vk::WriteDescriptorSet makeWrite(vk::DescriptorSet                                    dstSet,
                                   uint32_t                                             dstBinding,
                                   const vk::WriteDescriptorSetAccelerationStructureNV* pAccel,
                                   uint32_t                                             arrayElement = 0) const
  {
    return makeWrite(dstSet, dstBinding, reinterpret_cast<const VkWriteDescriptorSetAccelerationStructureNV*>(pAccel), arrayElement);
  }
#endif
#if VK_KHR_acceleration_structure
  vk::WriteDescriptorSet makeWrite(vk::DescriptorSet                                     dstSet,
                                   uint32_t                                              dstBinding,
                                   const vk::WriteDescriptorSetAccelerationStructureKHR* pAccel,
                                   uint32_t                                              arrayElement = 0) const
  {
    return makeWrite(dstSet, dstBinding, reinterpret_cast<const VkWriteDescriptorSetAccelerationStructureKHR*>(pAccel), arrayElement);
  }
#endif
#if VK_EXT_inline_uniform_block
  vk::WriteDescriptorSet makeWrite(vk::DescriptorSet                                  dstSet,
                                   uint32_t                                           dstBinding,
                                   const vk::WriteDescriptorSetInlineUniformBlockEXT* pInlineUniform,
                                   uint32_t                                           arrayElement = 0) const
  {
    return makeWrite(dstSet, dstBinding,
                     reinterpret_cast<const VkWriteDescriptorSetInlineUniformBlockEXT*>(pInlineUniform), arrayElement);
  }
#endif
  vk::WriteDescriptorSet makeWriteArray(vk::DescriptorSet dstSet, uint32_t dstBinding, const vk::DescriptorImageInfo* pImageInfo) const
  {
    return makeWriteArray(dstSet, dstBinding, reinterpret_cast<const VkDescriptorImageInfo*>(pImageInfo));
  }
  vk::WriteDescriptorSet makeWriteArray(vk::DescriptorSet dstSet, uint32_t dstBinding, const vk::DescriptorBufferInfo* pBufferInfo) const
  {
    return makeWriteArray(dstSet, dstBinding, reinterpret_cast<const VkDescriptorBufferInfo*>(pBufferInfo));
  }
  vk::WriteDescriptorSet makeWriteArray(vk::DescriptorSet dstSet, uint32_t dstBinding, const vk::BufferView* pTexelBufferView) const
  {
    return makeWriteArray(dstSet, dstBinding, reinterpret_cast<const VkBufferView*>(pTexelBufferView));
  }
#if VK_NV_ray_tracing
  vk::WriteDescriptorSet makeWriteArray(vk::DescriptorSet                                    dstSet,
                                        uint32_t                                             dstBinding,
                                        const vk::WriteDescriptorSetAccelerationStructureNV* pAccel) const
  {
    return makeWriteArray(dstSet, dstBinding, reinterpret_cast<const VkWriteDescriptorSetAccelerationStructureNV*>(pAccel));
  }
#endif
#if VK_KHR_acceleration_structure
  vk::WriteDescriptorSet makeWriteArray(vk::DescriptorSet                                     dstSet,
                                        uint32_t                                              dstBinding,
                                        const vk::WriteDescriptorSetAccelerationStructureKHR* pAccel) const
  {
    return makeWriteArray(dstSet, dstBinding, reinterpret_cast<const VkWriteDescriptorSetAccelerationStructureKHR*>(pAccel));
  }
#endif
#if VK_EXT_inline_uniform_block
  vk::WriteDescriptorSet makeWriteArray(vk::DescriptorSet                                  dstSet,
                                        uint32_t                                           dstBinding,
                                        const vk::WriteDescriptorSetInlineUniformBlockEXT* pInline) const
  {
    return makeWriteArray(dstSet, dstBinding, reinterpret_cast<const VkWriteDescriptorSetInlineUniformBlockEXT*>(pInline));
  }
#endif
#endif

private:
  std::vector<VkDescriptorSetLayoutBinding> m_bindings;
  std::vector<VkDescriptorBindingFlags>     m_bindingFlags;
};

/////////////////////////////////////////////////////////////
/**
# class nvvk::DescriptorSetContainer

Container class that stores allocated DescriptorSets
as well as reflection, layout and pool for a single
VkDescripterSetLayout.

Example:
~~~ C++
    container.init(device, allocator);

    // setup dset layouts
    container.addBinding(0, UBO...)
    container.addBinding(1, SSBO...)
    container.initLayout();

    // allocate descriptorsets
    container.initPool(17);

    // update descriptorsets
    writeUpdates.push_back( container.makeWrite(0, 0, &..) );
    writeUpdates.push_back( container.makeWrite(0, 1, &..) );
    writeUpdates.push_back( container.makeWrite(1, 0, &..) );
    writeUpdates.push_back( container.makeWrite(1, 1, &..) );
    writeUpdates.push_back( container.makeWrite(2, 0, &..) );
    writeUpdates.push_back( container.makeWrite(2, 1, &..) );
    ...

    // at render time

    vkCmdBindDescriptorSets(cmd, GRAPHICS, pipeLayout, 1, 1, container.at(7).getSets());
~~~

*/
class DescriptorSetContainer
{
public:
  DescriptorSetContainer(DescriptorSetContainer const&) = delete;
  DescriptorSetContainer& operator=(DescriptorSetContainer const&) = delete;

  DescriptorSetContainer() {}
  DescriptorSetContainer(VkDevice device) { init(device); }
  void init(VkDevice device);

  ~DescriptorSetContainer() { deinit(); }

  void setBindings(const std::vector<VkDescriptorSetLayoutBinding>& bindings);
  void addBinding(VkDescriptorSetLayoutBinding layoutBinding);
  void addBinding(uint32_t           binding,
                  VkDescriptorType   descriptorType,
                  uint32_t           descriptorCount,
                  VkShaderStageFlags stageFlags,
                  const VkSampler*   pImmutableSamplers = nullptr);

  // requires use of SUPPORT_INDEXING_EXT/SUPPORT_INDEXING_V1_2 on initLayout
  void setBindingFlags(uint32_t binding, VkDescriptorBindingFlags bindingFlags);

  VkDescriptorSetLayout initLayout(VkDescriptorSetLayoutCreateFlags flags        = 0,
                                   DescriptorSupport                supportFlags = DescriptorSupport::CORE_1_0);

  // inits pool and immediately allocates all numSets-many DescriptorSets
  VkDescriptorPool initPool(uint32_t numAllocatedSets);

  // optionally generates a pipelinelayout for the descriptorsetlayout
  VkPipelineLayout initPipeLayout(uint32_t                    numRanges = 0,
                                  const VkPushConstantRange*  ranges    = nullptr,
                                  VkPipelineLayoutCreateFlags flags     = 0);

  void deinitPool();
  void deinitLayout();
  void deinit();

  //////////////////////////////////////////////////////////////////////////

  VkDescriptorSet        getSet(uint32_t dstSetIdx = 0) const { return m_descriptorSets[dstSetIdx]; }
  const VkDescriptorSet* getSets(uint32_t dstSetIdx = 0) const { return m_descriptorSets.data() + dstSetIdx; }
  uint32_t               getSetsCount() const { return static_cast<uint32_t>(m_descriptorSets.size()); }

  VkDescriptorSetLayout        getLayout() const { return m_layout; }
  VkPipelineLayout             getPipeLayout() const { return m_pipelineLayout; }
  const DescriptorSetBindings& getBindings() const { return m_bindings; }
  VkDevice                     getDevice() const { return m_device; }

  //////////////////////////////////////////////////////////////////////////

  // provide single element
  VkWriteDescriptorSet makeWrite(uint32_t dstSetIdx, uint32_t dstBinding, const VkDescriptorImageInfo* pImageInfo, uint32_t arrayElement = 0) const
  {
    return m_bindings.makeWrite(getSet(dstSetIdx), dstBinding, pImageInfo, arrayElement);
  }
  VkWriteDescriptorSet makeWrite(uint32_t dstSetIdx, uint32_t dstBinding, const VkDescriptorBufferInfo* pBufferInfo, uint32_t arrayElement = 0) const
  {
    return m_bindings.makeWrite(getSet(dstSetIdx), dstBinding, pBufferInfo, arrayElement);
  }
  VkWriteDescriptorSet makeWrite(uint32_t dstSetIdx, uint32_t dstBinding, const VkBufferView* pTexelBufferView, uint32_t arrayElement = 0) const
  {
    return m_bindings.makeWrite(getSet(dstSetIdx), dstBinding, pTexelBufferView, arrayElement);
  }
#if VK_NV_ray_tracing
  VkWriteDescriptorSet makeWrite(uint32_t                                           dstSetIdx,
                                 uint32_t                                           dstBinding,
                                 const VkWriteDescriptorSetAccelerationStructureNV* pAccel,
                                 uint32_t                                           arrayElement = 0) const
  {
    return m_bindings.makeWrite(getSet(dstSetIdx), dstBinding, pAccel, arrayElement);
  }
#endif
#if VK_KHR_acceleration_structure
  VkWriteDescriptorSet makeWrite(uint32_t                                            dstSetIdx,
                                 uint32_t                                            dstBinding,
                                 const VkWriteDescriptorSetAccelerationStructureKHR* pAccel,
                                 uint32_t                                            arrayElement = 0) const
  {
    return m_bindings.makeWrite(getSet(dstSetIdx), dstBinding, pAccel, arrayElement);
  }
#endif
#if VK_EXT_inline_uniform_block
  VkWriteDescriptorSet makeWrite(uint32_t                                         dstSetIdx,
                                 uint32_t                                         dstBinding,
                                 const VkWriteDescriptorSetInlineUniformBlockEXT* pInline,
                                 uint32_t                                         arrayElement = 0) const
  {
    return m_bindings.makeWrite(getSet(dstSetIdx), dstBinding, pInline, arrayElement);
  }
#endif
  // provide full array
  VkWriteDescriptorSet makeWriteArray(uint32_t dstSetIdx, uint32_t dstBinding, const VkDescriptorImageInfo* pImageInfo) const
  {
    return m_bindings.makeWriteArray(getSet(dstSetIdx), dstBinding, pImageInfo);
  }
  VkWriteDescriptorSet makeWriteArray(uint32_t dstSetIdx, uint32_t dstBinding, const VkDescriptorBufferInfo* pBufferInfo) const
  {
    return m_bindings.makeWriteArray(getSet(dstSetIdx), dstBinding, pBufferInfo);
  }
  VkWriteDescriptorSet makeWriteArray(uint32_t dstSetIdx, uint32_t dstBinding, const VkBufferView* pTexelBufferView) const
  {
    return m_bindings.makeWriteArray(getSet(dstSetIdx), dstBinding, pTexelBufferView);
  }
#if VK_NV_ray_tracing
  VkWriteDescriptorSet makeWriteArray(uint32_t dstSetIdx, uint32_t dstBinding, const VkWriteDescriptorSetAccelerationStructureNV* pAccel) const
  {
    return m_bindings.makeWriteArray(getSet(dstSetIdx), dstBinding, pAccel);
  }
#endif
#if VK_KHR_acceleration_structure
  VkWriteDescriptorSet makeWriteArray(uint32_t dstSetIdx, uint32_t dstBinding, const VkWriteDescriptorSetAccelerationStructureKHR* pAccel) const
  {
    return m_bindings.makeWriteArray(getSet(dstSetIdx), dstBinding, pAccel);
  }
#endif
#if VK_EXT_inline_uniform_block
  VkWriteDescriptorSet makeWriteArray(uint32_t dstSetIdx, uint32_t dstBinding, const VkWriteDescriptorSetInlineUniformBlockEXT* pInline) const
  {
    return m_bindings.makeWriteArray(getSet(dstSetIdx), dstBinding, pInline);
  }
#endif
#ifdef VULKAN_HPP
  void addBinding(uint32_t binding,           // Slot to which the descriptor will be bound, corresponding to the layout
                                              // binding index in the shader
                  vk::DescriptorType   type,  // Type of the bound descriptor(s)
                  uint32_t             count,       // Number of descriptors
                  vk::ShaderStageFlags stageFlags,  // Shader stages at which the bound resources will be available
                  const vk::Sampler*   pImmutableSampler = nullptr  // Corresponding sampler, in case of textures
  )
  {
    m_bindings.addBinding({binding, static_cast<VkDescriptorType>(type), count, static_cast<VkShaderStageFlags>(stageFlags),
                           reinterpret_cast<const VkSampler*>(pImmutableSampler)});
  }
  void setBindings(const std::vector<vk::DescriptorSetLayoutBinding>& bindings)
  {
    m_bindings.setBindings(reinterpret_cast<const std::vector<VkDescriptorSetLayoutBinding>&>(bindings));
  }
  void setBindingFlags(uint32_t binding, vk::DescriptorBindingFlags bindingFlags)
  {
    m_bindings.setBindingFlags(binding, static_cast<VkDescriptorBindingFlags>(bindingFlags));
  }

  vk::WriteDescriptorSet makeWrite(uint32_t dstSetIdx, uint32_t dstBinding, const vk::DescriptorImageInfo* pImageInfo, uint32_t arrayElement = 0) const
  {
    return m_bindings.makeWrite(getSet(dstSetIdx), dstBinding, reinterpret_cast<const VkDescriptorImageInfo*>(pImageInfo), arrayElement);
  }
  vk::WriteDescriptorSet makeWrite(uint32_t                        dstSetIdx,
                                   uint32_t                        dstBinding,
                                   const vk::DescriptorBufferInfo* pBufferInfo,
                                   uint32_t                        arrayElement = 0) const
  {
    return m_bindings.makeWrite(getSet(dstSetIdx), dstBinding,
                                reinterpret_cast<const VkDescriptorBufferInfo*>(pBufferInfo), arrayElement);
  }
  vk::WriteDescriptorSet makeWrite(uint32_t dstSetIdx, uint32_t dstBinding, const vk::BufferView* pTexelBufferView, uint32_t arrayElement = 0) const
  {
    return m_bindings.makeWrite(getSet(dstSetIdx), dstBinding, reinterpret_cast<const VkBufferView*>(pTexelBufferView), arrayElement);
  }
#if VK_NV_ray_tracing
  vk::WriteDescriptorSet makeWrite(uint32_t                                             dstSetIdx,
                                   uint32_t                                             dstBinding,
                                   const vk::WriteDescriptorSetAccelerationStructureNV* pAccel,
                                   uint32_t                                             arrayElement = 0) const
  {
    return m_bindings.makeWrite(getSet(dstSetIdx), dstBinding,
                                reinterpret_cast<const VkWriteDescriptorSetAccelerationStructureNV*>(pAccel), arrayElement);
  }
#endif
#if VK_KHR_acceleration_structure
  vk::WriteDescriptorSet makeWrite(uint32_t                                              dstSetIdx,
                                   uint32_t                                              dstBinding,
                                   const vk::WriteDescriptorSetAccelerationStructureKHR* pAccel,
                                   uint32_t                                              arrayElement = 0) const
  {
    return m_bindings.makeWrite(getSet(dstSetIdx), dstBinding,
                                reinterpret_cast<const VkWriteDescriptorSetAccelerationStructureKHR*>(pAccel), arrayElement);
  }
#endif
  vk::WriteDescriptorSet makeWrite(uint32_t                                           dstSetIdx,
                                   uint32_t                                           dstBinding,
                                   const vk::WriteDescriptorSetInlineUniformBlockEXT* pInlineUniform,
                                   uint32_t                                           arrayElement = 0) const
  {
    return m_bindings.makeWrite(getSet(dstSetIdx), dstBinding,
                                reinterpret_cast<const VkWriteDescriptorSetInlineUniformBlockEXT*>(pInlineUniform), arrayElement);
  }
  vk::WriteDescriptorSet makeWriteArray(uint32_t dstSetIdx, uint32_t dstBinding, const vk::DescriptorImageInfo* pImageInfo) const
  {
    return m_bindings.makeWriteArray(getSet(dstSetIdx), dstBinding, reinterpret_cast<const VkDescriptorImageInfo*>(pImageInfo));
  }
  vk::WriteDescriptorSet makeWriteArray(uint32_t dstSetIdx, uint32_t dstBinding, const vk::DescriptorBufferInfo* pBufferInfo) const
  {
    return m_bindings.makeWriteArray(getSet(dstSetIdx), dstBinding, reinterpret_cast<const VkDescriptorBufferInfo*>(pBufferInfo));
  }
  vk::WriteDescriptorSet makeWriteArray(uint32_t dstSetIdx, uint32_t dstBinding, const vk::BufferView* pTexelBufferView) const
  {
    return m_bindings.makeWriteArray(getSet(dstSetIdx), dstBinding, reinterpret_cast<const VkBufferView*>(pTexelBufferView));
  }
#if VK_NV_ray_tracing
  vk::WriteDescriptorSet makeWriteArray(uint32_t dstSetIdx, uint32_t dstBinding, const vk::WriteDescriptorSetAccelerationStructureNV* pAccel) const
  {
    return m_bindings.makeWriteArray(getSet(dstSetIdx), dstBinding,
                                     reinterpret_cast<const VkWriteDescriptorSetAccelerationStructureNV*>(pAccel));
  }
#endif
#if VK_KHR_acceleration_structure
  vk::WriteDescriptorSet makeWriteArray(uint32_t                                              dstSetIdx,
                                        uint32_t                                              dstBinding,
                                        const vk::WriteDescriptorSetAccelerationStructureKHR* pAccel) const
  {
    return m_bindings.makeWriteArray(getSet(dstSetIdx), dstBinding,
                                     reinterpret_cast<const VkWriteDescriptorSetAccelerationStructureKHR*>(pAccel));
  }
#endif
#if VK_EXT_inline_uniform_block
  vk::WriteDescriptorSet makeWriteArray(uint32_t dstSetIdx, uint32_t dstBinding, const vk::WriteDescriptorSetInlineUniformBlockEXT* pInline) const
  {
    return m_bindings.makeWriteArray(getSet(dstSetIdx), dstBinding,
                                     reinterpret_cast<const VkWriteDescriptorSetInlineUniformBlockEXT*>(pInline));
  }
#endif
#endif
protected:
  VkDevice                     m_device         = VK_NULL_HANDLE;
  VkDescriptorSetLayout        m_layout         = VK_NULL_HANDLE;
  VkDescriptorPool             m_pool           = VK_NULL_HANDLE;
  VkPipelineLayout             m_pipelineLayout = VK_NULL_HANDLE;
  std::vector<VkDescriptorSet> m_descriptorSets = {};
  DescriptorSetBindings        m_bindings       = {};
};

//////////////////////////////////////////////////////////////////////////
/**
# class nvvk::TDescriptorSetContainer<SETS,PIPES=1>

Templated version of DescriptorSetContainer :

- SETS  - many DescriptorSetContainers
- PIPES - many VkPipelineLayouts

The pipeline layouts are stored separately, the class does
not use the pipeline layouts of the embedded DescriptorSetContainers.

Example :

~~~ C++
Usage, e.g.SETS = 2, PIPES = 2

container.init(device, allocator);

// setup dset layouts
container.at(0).addBinding(0, UBO...)
container.at(0).addBinding(1, SSBO...)
container.at(0).initLayout();
container.at(1).addBinding(0, COMBINED_SAMPLER...)
container.at(1).initLayout();

// pipe 0 uses set 0 alone
container.initPipeLayout(0, 1);
// pipe 1 uses sets 0, 1
container.initPipeLayout(1, 2);

// allocate descriptorsets
container.at(0).initPool(1);
container.at(1).initPool(16);

// update descriptorsets

writeUpdates.push_back(container.at(0).makeWrite(0, 0, &..));
writeUpdates.push_back(container.at(0).makeWrite(0, 1, &..));
writeUpdates.push_back(container.at(1).makeWrite(0, 0, &..));
writeUpdates.push_back(container.at(1).makeWrite(1, 0, &..));
writeUpdates.push_back(container.at(1).makeWrite(2, 0, &..));
...

// at render time

vkCmdBindDescriptorSets(cmd, GRAPHICS, container.getPipeLayout(0), 0, 1, container.at(0).getSets());
..
vkCmdBindDescriptorSets(cmd, GRAPHICS, container.getPipeLayout(1), 1, 1, container.at(1).getSets(7));
~~~
*/
template <int SETS, int PIPES = 1>
class TDescriptorSetContainer
{
public:
  TDescriptorSetContainer() {}
  TDescriptorSetContainer(VkDevice device) { init(device); }
  ~TDescriptorSetContainer() { deinit(); }

  void init(VkDevice device);
  void deinit();
  void deinitLayouts();
  void deinitPools();

  // pipelayout uses range of m_sets[0.. first null or SETS[
  VkPipelineLayout initPipeLayout(uint32_t                    pipe,
                                  uint32_t                    numRanges = 0,
                                  const VkPushConstantRange*  ranges    = nullptr,
                                  VkPipelineLayoutCreateFlags flags     = 0);

  // pipelayout uses range of m_sets[0..numDsets[
  VkPipelineLayout initPipeLayout(uint32_t                    pipe,
                                  uint32_t                    numDsets,
                                  uint32_t                    numRanges = 0,
                                  const VkPushConstantRange*  ranges    = nullptr,
                                  VkPipelineLayoutCreateFlags flags     = 0);

  DescriptorSetContainer&       at(uint32_t set) { return m_sets[set]; }
  const DescriptorSetContainer& at(uint32_t set) const { return m_sets[set]; }
  DescriptorSetContainer&       operator[](uint32_t set) { return m_sets[set]; }
  const DescriptorSetContainer& operator[](uint32_t set) const { return m_sets[set]; }
  VkPipelineLayout              getPipeLayout(uint32_t pipe = 0) const
  {
    assert(pipe <= PIPES);
    return m_pipelayouts[pipe];
  }

protected:
  VkPipelineLayout       m_pipelayouts[PIPES] = {};
  DescriptorSetContainer m_sets[SETS];
};

//////////////////////////////////////////////////////////////////////////

template <int SETS, int PIPES>
VkPipelineLayout TDescriptorSetContainer<SETS, PIPES>::initPipeLayout(uint32_t                    pipe,
                                                                      uint32_t                    numDsets,
                                                                      uint32_t                    numRanges /*= 0*/,
                                                                      const VkPushConstantRange*  ranges /*= nullptr*/,
                                                                      VkPipelineLayoutCreateFlags flags /*= 0*/)
{
  assert(pipe <= uint32_t(PIPES));
  assert(numDsets <= uint32_t(SETS));
  assert(m_pipelayouts[pipe] == VK_NULL_HANDLE);

  VkDevice device = m_sets[0].getDevice();

  VkDescriptorSetLayout setLayouts[SETS];
  for(uint32_t d = 0; d < numDsets; d++)
  {
    setLayouts[d] = m_sets[d].getLayout();
    assert(setLayouts[d]);
  }

  VkResult                   result;
  VkPipelineLayoutCreateInfo layoutCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layoutCreateInfo.setLayoutCount             = numDsets;
  layoutCreateInfo.pSetLayouts                = setLayouts;
  layoutCreateInfo.pushConstantRangeCount     = numRanges;
  layoutCreateInfo.pPushConstantRanges        = ranges;
  layoutCreateInfo.flags                      = flags;

  result = vkCreatePipelineLayout(device, &layoutCreateInfo, nullptr, &m_pipelayouts[pipe]);
  assert(result == VK_SUCCESS);
  return m_pipelayouts[pipe];
}

template <int SETS, int PIPES>
VkPipelineLayout TDescriptorSetContainer<SETS, PIPES>::initPipeLayout(uint32_t                    pipe,
                                                                      uint32_t                    numRanges /*= 0*/,
                                                                      const VkPushConstantRange*  ranges /*= nullptr*/,
                                                                      VkPipelineLayoutCreateFlags flags /*= 0*/)
{
  assert(pipe <= uint32_t(PIPES));
  assert(m_pipelayouts[pipe] == VK_NULL_HANDLE);

  VkDevice device = m_sets[0].getDevice();

  VkDescriptorSetLayout setLayouts[SETS];
  int                   used;
  for(used = 0; used < SETS; used++)
  {
    setLayouts[used] = m_sets[used].getLayout();
    if(!setLayouts[used])
      break;
  }

  VkResult                   result;
  VkPipelineLayoutCreateInfo layoutCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layoutCreateInfo.setLayoutCount             = uint32_t(used);
  layoutCreateInfo.pSetLayouts                = setLayouts;
  layoutCreateInfo.pushConstantRangeCount     = numRanges;
  layoutCreateInfo.pPushConstantRanges        = ranges;
  layoutCreateInfo.flags                      = flags;

  result = vkCreatePipelineLayout(device, &layoutCreateInfo, nullptr, &m_pipelayouts[pipe]);
  assert(result == VK_SUCCESS);
  return m_pipelayouts[pipe];
}

template <int SETS, int PIPES>
void TDescriptorSetContainer<SETS, PIPES>::deinitPools()
{
  for(int d = 0; d < SETS; d++)
  {
    m_sets[d].deinitPool();
  }
}

template <int SETS, int PIPES>
void TDescriptorSetContainer<SETS, PIPES>::deinitLayouts()
{
  VkDevice device = m_sets[0].getDevice();

  for(int p = 0; p < PIPES; p++)
  {
    if(m_pipelayouts[p])
    {
      vkDestroyPipelineLayout(device, m_pipelayouts[p], nullptr);
      m_pipelayouts[p] = VK_NULL_HANDLE;
    }
  }
  for(int d = 0; d < SETS; d++)
  {
    m_sets[d].deinitLayout();
  }
}

template <int SETS, int PIPES>
void TDescriptorSetContainer<SETS, PIPES>::deinit()
{
  deinitPools();
  deinitLayouts();
}

template <int SETS, int PIPES>
void TDescriptorSetContainer<SETS, PIPES>::init(VkDevice device)
{
  for(int d = 0; d < SETS; d++)
  {
    m_sets[d].init(device);
  }
}


}  // namespace nvvk
