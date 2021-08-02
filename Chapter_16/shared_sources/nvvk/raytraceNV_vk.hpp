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
#pragma once

/**

# class nvvk::RaytracingBuilderNV

Base functionality of raytracing

This class does not implement all what you need to do raytracing, but
helps creating the BLAS and TLAS, which then can be used by different
raytracing usage.

# Setup and Usage
~~~~ C++
m_rtBuilder.setup(device, memoryAllocator, queueIndex);
// Create array of VkGeometryNV
m_rtBuilder.buildBlas(allBlas);
// Create array of RaytracingBuilder::instance
m_rtBuilder.buildTlas(instances);
// Retrieve the acceleration structure
const VkAccelerationStructureNV& tlas = m.rtBuilder.getAccelerationStructure()
~~~~
*/


#include <mutex>
#include <vulkan/vulkan.h>

#include "allocator_vk.hpp"
#include "commands_vk.hpp"
#include "debug_util_vk.hpp"
#include "nvh/nvprint.hpp"
#include "nvmath/nvmath.h"

#if VK_NV_ray_tracing

// See https://www.khronos.org/registry/vulkan/specs/1.1-extensions/html/chap33.html#acceleration-structure
struct VkGeometryInstanceNV
{
  /// Transform matrix, containing only the top 3 rows
  float transform[12];
  /// Instance index
  uint32_t instanceId : 24;
  /// Visibility mask
  uint32_t mask : 8;
  /// Index of the hit group which will be invoked when a ray hits the instance
  uint32_t hitGroupId : 24;
  /// Instance flags, such as culling
  uint32_t flags : 8;
  /// Opaque handle of the bottom-level acceleration structure
  uint64_t accelerationStructureHandle;
};

namespace nvvk {
struct RaytracingBuilderNV
{
  RaytracingBuilderNV(RaytracingBuilderNV const&) = delete;
  RaytracingBuilderNV& operator=(RaytracingBuilderNV const&) = delete;

  RaytracingBuilderNV() = default;

  //--------------------------------------------------------------------------------------------------
  // Initializing the allocator and querying the raytracing properties
  //
  void setup(VkDevice device, nvvk::Allocator* allocator, uint32_t queueIndex)
  {
    m_device     = device;
    m_queueIndex = queueIndex;
    m_debug.setup(device);
    m_alloc = allocator;
  }

  // This is an instance of a BLAS
  struct Instance
  {
    uint32_t                  blasId{0};      // Index of the BLAS in m_blas
    uint32_t                  instanceId{0};  // Instance Index (gl_InstanceID)
    uint32_t                  hitGroupId{0};  // Hit group index in the SBT
    uint32_t                  mask{0xFF};     // Visibility mask, will be AND-ed with ray mask
    VkGeometryInstanceFlagsNV flags = VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV;
    nvmath::mat4f             transform{nvmath::mat4f(1)};  // Identity
  };


  //--------------------------------------------------------------------------------------------------
  // Destroying all allocations
  //
  void destroy()
  {
    for(auto& b : m_blas)
    {
      m_alloc->destroy(b.as);
    }
    m_alloc->destroy(m_tlas.as);
    m_alloc->destroy(m_instBuffer);
  }

  // Returning the constructed top-level acceleration structure
  VkAccelerationStructureNV getAccelerationStructure() const { return m_tlas.as.accel; }

  //--------------------------------------------------------------------------------------------------
  // Create all the BLAS from the vector of vectors of VkGeometryNV
  // - There will be one BLAS per vector of VkGeometryNV
  // - There will be as many BLAS there are items in the geoms vector
  // - The resulting BLAS are stored in m_blas
  //
  void buildBlas(const std::vector<std::vector<VkGeometryNV>>& geoms,
                 VkBuildAccelerationStructureFlagsNV flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_NV)
  {
    m_blas.resize(geoms.size());

    VkDeviceSize maxScratch{0};

    // Is compaction requested?
    bool doCompaction = (flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_NV)
                        == VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_NV;
    std::vector<VkDeviceSize> originalSizes;
    originalSizes.resize(m_blas.size());


    // Iterate over the groups of geometries, creating one BLAS for each group
    for(size_t i = 0; i < geoms.size(); i++)
    {
      Blas& blas{m_blas[i]};

      // Set the geometries that will be part of the BLAS
      blas.asInfo.geometryCount = static_cast<uint32_t>(geoms[i].size());
      blas.asInfo.pGeometries   = geoms[i].data();
      blas.asInfo.flags         = flags;
      VkAccelerationStructureCreateInfoNV createinfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV};
      createinfo.info = blas.asInfo;

      // Create an acceleration structure identifier and allocate memory to store the
      // resulting structure data
      blas.as = m_alloc->createAcceleration(createinfo);
      m_debug.setObjectName(blas.as.accel, (std::string("Blas" + std::to_string(i)).c_str()));

      // Estimate the amount of scratch memory required to build the BLAS, and update the
      // size of the scratch buffer that will be allocated to sequentially build all BLASes
      VkAccelerationStructureMemoryRequirementsInfoNV memoryRequirementsInfo{
          VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV};
      memoryRequirementsInfo.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_BUILD_SCRATCH_NV;
      memoryRequirementsInfo.accelerationStructure = blas.as.accel;


      VkMemoryRequirements2 reqMem{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
      vkGetAccelerationStructureMemoryRequirementsNV(m_device, &memoryRequirementsInfo, &reqMem);
      VkDeviceSize scratchSize = reqMem.memoryRequirements.size;


      // Original size
      memoryRequirementsInfo.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_NV;
      vkGetAccelerationStructureMemoryRequirementsNV(m_device, &memoryRequirementsInfo, &reqMem);
      originalSizes[i] = reqMem.memoryRequirements.size;

      maxScratch = std::max(maxScratch, scratchSize);
    }

    // Allocate the scratch buffers holding the temporary data of the acceleration structure builder
    nvvk::Buffer scratchBuffer = m_alloc->createBuffer(maxScratch, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV);


    // Query size of compact BLAS
    VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpci.queryCount = (uint32_t)m_blas.size();
    qpci.queryType  = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_NV;
    VkQueryPool queryPool;
    vkCreateQueryPool(m_device, &qpci, nullptr, &queryPool);


    // Create a command buffer containing all the BLAS builds
    nvvk::CommandPool            genCmdBuf(m_device, m_queueIndex);
    int                          ctr{0};
    std::vector<VkCommandBuffer> allCmdBufs;
    allCmdBufs.reserve(m_blas.size());
    for(auto& blas : m_blas)
    {
      VkCommandBuffer cmdBuf = genCmdBuf.createCommandBuffer();
      allCmdBufs.push_back(cmdBuf);

      vkCmdBuildAccelerationStructureNV(cmdBuf, &blas.asInfo, nullptr, 0, VK_FALSE, blas.as.accel, nullptr,
                                        scratchBuffer.buffer, 0);

      // Since the scratch buffer is reused across builds, we need a barrier to ensure one build
      // is finished before starting the next one
      VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV;
      barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV;
      vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
                           VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV, 0, 1, &barrier, 0, nullptr, 0, nullptr);

      // Query the compact size
      if(doCompaction)
      {
        vkCmdWriteAccelerationStructuresPropertiesNV(cmdBuf, 1, &blas.as.accel,
                                                     VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_NV, queryPool, ctr++);
      }
    }
    genCmdBuf.submitAndWait(allCmdBufs);
    allCmdBufs.clear();


    // Compacting all BLAS
    if(doCompaction)
    {
      VkCommandBuffer cmdBuf = genCmdBuf.createCommandBuffer();

      // Get the size result back
      std::vector<VkDeviceSize> compactSizes(m_blas.size());
      vkGetQueryPoolResults(m_device, queryPool, 0, (uint32_t)compactSizes.size(), compactSizes.size() * sizeof(VkDeviceSize),
                            compactSizes.data(), sizeof(VkDeviceSize), VK_QUERY_RESULT_WAIT_BIT);


      // Compacting
      std::vector<nvvk::AccelNV> cleanupAS(m_blas.size());
      uint32_t                   totOriginalSize{0}, totCompactSize{0};
      for(int i = 0; i < m_blas.size(); i++)
      {
        LOGI("Reducing %i, from %d to %d \n", i, originalSizes[i], compactSizes[i]);
        totOriginalSize += (uint32_t)originalSizes[i];
        totCompactSize += (uint32_t)compactSizes[i];

        // Creating a compact version of the AS
        VkAccelerationStructureInfoNV asInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV};
        asInfo.type  = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV;
        asInfo.flags = flags;
        VkAccelerationStructureCreateInfoNV asCreateInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV};
        asCreateInfo.compactedSize = compactSizes[i];
        asCreateInfo.info          = asInfo;
        auto as                    = m_alloc->createAcceleration(asCreateInfo);

        // Copy the original BLAS to a compact version
        vkCmdCopyAccelerationStructureNV(cmdBuf, as.accel, m_blas[i].as.accel, VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_NV);

        cleanupAS[i] = m_blas[i].as;
        m_blas[i].as = as;
      }
      genCmdBuf.submitAndWait(cmdBuf);

      // Destroying the previous version
      for(auto as : cleanupAS)
        m_alloc->destroy(as);

      LOGI("------------------\n");
      LOGI("Total: %d -> %d = %d (%2.2f%s smaller) \n", totOriginalSize, totCompactSize,
           totOriginalSize - totCompactSize, (totOriginalSize - totCompactSize) / float(totOriginalSize) * 100.f, "%%");
    }

    vkDestroyQueryPool(m_device, queryPool, nullptr);
    m_alloc->destroy(scratchBuffer);
    m_alloc->finalizeAndReleaseStaging();
  }

  //--------------------------------------------------------------------------------------------------
  // Convert an Instance object into a VkGeometryInstanceNV

  VkGeometryInstanceNV instanceToVkGeometryInstanceNV(const Instance& instance)
  {
    Blas& blas{m_blas[instance.blasId]};
    // For each BLAS, fetch the acceleration structure handle that will allow the builder to
    // directly access it from the device
    uint64_t asHandle = 0;
    vkGetAccelerationStructureHandleNV(m_device, blas.as.accel, sizeof(uint64_t), &asHandle);

    VkGeometryInstanceNV gInst{};
    // The matrices for the instance transforms are row-major, instead of column-major in the
    // rest of the application
    nvmath::mat4f transp = nvmath::transpose(instance.transform);
    // The gInst.transform value only contains 12 values, corresponding to a 4x3 matrix, hence
    // saving the last row that is anyway always (0,0,0,1). Since the matrix is row-major,
    // we simply copy the first 12 values of the original 4x4 matrix
    memcpy(gInst.transform, &transp, sizeof(gInst.transform));
    gInst.instanceId                  = instance.instanceId;
    gInst.mask                        = instance.mask;
    gInst.hitGroupId                  = instance.hitGroupId;
    gInst.flags                       = static_cast<uint32_t>(instance.flags);
    gInst.accelerationStructureHandle = asHandle;

    return gInst;
  }

  //--------------------------------------------------------------------------------------------------
  // Creating the top-level acceleration structure from the vector of Instance
  // - See struct of Instance
  // - The resulting TLAS will be stored in m_tlas
  //
  void buildTlas(const std::vector<Instance>&        instances,
                 VkBuildAccelerationStructureFlagsNV flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_NV)
  {
    // Set the instance count required to determine how much memory the TLAS will use
    m_tlas.asInfo.instanceCount = static_cast<uint32_t>(instances.size());
    m_tlas.asInfo.flags         = flags;
    VkAccelerationStructureCreateInfoNV accelerationStructureInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV};
    accelerationStructureInfo.info = m_tlas.asInfo;
    // Create the acceleration structure object and allocate the memory required to hold the TLAS data
    m_tlas.as = m_alloc->createAcceleration(accelerationStructureInfo);
    m_debug.setObjectName(m_tlas.as.accel, "Tlas");

    // Compute the amount of scratch memory required by the acceleration structure builder
    VkAccelerationStructureMemoryRequirementsInfoNV memoryRequirementsInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV};
    memoryRequirementsInfo.type                  = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_BUILD_SCRATCH_NV;
    memoryRequirementsInfo.accelerationStructure = m_tlas.as.accel;

    VkMemoryRequirements2 reqMem{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    vkGetAccelerationStructureMemoryRequirementsNV(m_device, &memoryRequirementsInfo, &reqMem);
    VkDeviceSize scratchSize = reqMem.memoryRequirements.size;


    // Allocate the scratch memory
    nvvk::Buffer scratchBuffer = m_alloc->createBuffer(scratchSize, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV);

    // For each instance, build the corresponding instance descriptor
    std::vector<VkGeometryInstanceNV> geometryInstances;
    geometryInstances.reserve(instances.size());
    for(const auto& inst : instances)
    {
      geometryInstances.push_back(instanceToVkGeometryInstanceNV(inst));
    }

    // Building the TLAS
    nvvk::CommandPool genCmdBuf(m_device, m_queueIndex);
    VkCommandBuffer   cmdBuf = genCmdBuf.createCommandBuffer();

    // Allocate the instance buffer and copy its contents from host to device memory
    m_instBuffer = m_alloc->createBuffer(cmdBuf, geometryInstances, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV);
    m_debug.setObjectName(m_instBuffer.buffer, "TLASInstances");

    // Make sure the copy of the instance buffer are copied before triggering the
    // acceleration structure build
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);


    // Build the TLAS
    vkCmdBuildAccelerationStructureNV(cmdBuf, &m_tlas.asInfo, m_instBuffer.buffer, 0, VK_FALSE, m_tlas.as.accel,
                                      nullptr, scratchBuffer.buffer, 0);


    genCmdBuf.submitAndWait(cmdBuf);

    m_alloc->finalizeAndReleaseStaging();
    m_alloc->destroy(scratchBuffer);
  }

  //--------------------------------------------------------------------------------------------------
  // Refit the TLAS using new instance matrices
  //
  void updateTlasMatrices(const std::vector<Instance>& instances)
  {
    VkDeviceSize bufferSize = instances.size() * sizeof(VkGeometryInstanceNV);
    // Create a staging buffer on the host to upload the new instance data
    nvvk::Buffer stagingBuffer = m_alloc->createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
#if defined(NVVK_ALLOC_VMA)
                                                       VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU
#else
                                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
#endif
    );

    // Copy the instance data into the staging buffer
    auto* gInst = reinterpret_cast<VkGeometryInstanceNV*>(m_alloc->map(stagingBuffer));
    for(int i = 0; i < instances.size(); i++)
    {
      gInst[i] = instanceToVkGeometryInstanceNV(instances[i]);
    }
    m_alloc->unmap(stagingBuffer);

    // Compute the amount of scratch memory required by the AS builder to update the TLAS
    VkAccelerationStructureMemoryRequirementsInfoNV memoryRequirementsInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV};
    memoryRequirementsInfo.type                  = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_UPDATE_SCRATCH_NV;
    memoryRequirementsInfo.accelerationStructure = m_tlas.as.accel;

    VkMemoryRequirements2 reqMem{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    vkGetAccelerationStructureMemoryRequirementsNV(m_device, &memoryRequirementsInfo, &reqMem);
    VkDeviceSize scratchSize = reqMem.memoryRequirements.size;


    // Allocate the scratch buffer
    nvvk::Buffer scratchBuffer = m_alloc->createBuffer(scratchSize, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV);

    // Update the instance buffer on the device side and build the TLAS
    nvvk::CommandPool genCmdBuf(m_device, m_queueIndex);
    VkCommandBuffer   cmdBuf = genCmdBuf.createCommandBuffer();

    VkBufferCopy region{0, 0, bufferSize};
    vkCmdCopyBuffer(cmdBuf, stagingBuffer.buffer, m_instBuffer.buffer, 1, &region);

    // Make sure the copy of the instance buffer are copied before triggering the
    // acceleration structure build
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);


    // Update the acceleration structure. Note the VK_TRUE parameter to trigger the update,
    // and the existing TLAS being passed and updated in place
    vkCmdBuildAccelerationStructureNV(cmdBuf, &m_tlas.asInfo, m_instBuffer.buffer, 0, VK_TRUE, m_tlas.as.accel,
                                      m_tlas.as.accel, scratchBuffer.buffer, 0);


    genCmdBuf.submitAndWait(cmdBuf);

    m_alloc->destroy(scratchBuffer);
    m_alloc->destroy(stagingBuffer);
  }

  //--------------------------------------------------------------------------------------------------
  // Refit the BLAS from updated buffers
  //
  void updateBlas(uint32_t blasIdx)
  {
    Blas& blas = m_blas[blasIdx];

    // Compute the amount of scratch memory required by the AS builder to update the TLAS
    VkAccelerationStructureMemoryRequirementsInfoNV memoryRequirementsInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV};
    memoryRequirementsInfo.type                  = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_UPDATE_SCRATCH_NV;
    memoryRequirementsInfo.accelerationStructure = blas.as.accel;

    VkMemoryRequirements2 reqMem{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    vkGetAccelerationStructureMemoryRequirementsNV(m_device, &memoryRequirementsInfo, &reqMem);
    VkDeviceSize scratchSize = reqMem.memoryRequirements.size;

    // Allocate the scratch buffer
    nvvk::Buffer scratchBuffer = m_alloc->createBuffer(scratchSize, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV);

    // Update the instance buffer on the device side and build the TLAS
    nvvk::CommandPool genCmdBuf(m_device, m_queueIndex);
    VkCommandBuffer   cmdBuf = genCmdBuf.createCommandBuffer();


    // Update the acceleration structure. Note the VK_TRUE parameter to trigger the update,
    // and the existing BLAS being passed and updated in place
    vkCmdBuildAccelerationStructureNV(cmdBuf, &blas.asInfo, nullptr, 0, VK_TRUE, blas.as.accel, blas.as.accel,
                                      scratchBuffer.buffer, 0);

    genCmdBuf.submitAndWait(cmdBuf);
    m_alloc->destroy(scratchBuffer);
  }

private:
  // Bottom-level acceleration structure
  struct Blas
  {
    nvvk::AccelNV                 as;
    VkAccelerationStructureInfoNV asInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV, nullptr,
                                         VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV};
    VkGeometryNV                  geometry;
  };

  // Top-level acceleration structure
  struct Tlas
  {
    nvvk::AccelNV                 as;
    VkAccelerationStructureInfoNV asInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV, nullptr,
                                         VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV};
  };

  //--------------------------------------------------------------------------------------------------
  // Vector containing all the BLASes built and referenced by the TLAS
  std::vector<Blas> m_blas;
  // Top-level acceleration structure
  Tlas m_tlas;
  // Instance buffer containing the matrices and BLAS ids
  nvvk::Buffer m_instBuffer;

  VkDevice m_device;
  uint32_t m_queueIndex{0};

  nvvk::Allocator* m_alloc = nullptr;
  nvvk::DebugUtil  m_debug;

#ifdef VULKAN_HPP
public:
  void buildBlas(const std::vector<std::vector<VkGeometryNV>>& geoms, vk::BuildAccelerationStructureFlagsNV flags)
  {
    buildBlas(geoms, static_cast<VkBuildAccelerationStructureFlagsNV>(flags));
  }

  void buildTlas(const std::vector<Instance>& instances, vk::BuildAccelerationStructureFlagsNV flags)

  {
    buildTlas(instances, static_cast<VkBuildAccelerationStructureFlagsNV>(flags));
  }
#endif
};

}  // namespace nvvk

#else
  #error This include requires VK_NV_ray_tracing support in the Vulkan SDK.
#endif
