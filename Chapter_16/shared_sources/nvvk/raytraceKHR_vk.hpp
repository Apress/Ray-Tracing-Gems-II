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

# class nvvk::RaytracingBuilderKHR

Base functionality of raytracing

This class acts as an owning container for a single top-level acceleration
structure referencing any number of bottom-level acceleration structures.
We provide functions for building (on the device) an array of BLASs and a
single TLAS from vectors of BlasInput and Instance, respectively, and
a destroy function for cleaning up the created acceleration structures.

Generally, we reference BLASs by their index in the stored BLAS array,
rather than using raw device pointers as the pure Vulkan acceleration
structure API uses.

This class does not support replacing acceleration structures once
built, but you can update the acceleration structures. For educational
purposes, this class prioritizes (relative) understandability over
performance, so vkQueueWaitIdle is implicitly used everywhere.

# Setup and Usage
~~~~ C++
// Borrow a VkDevice and memory allocator pointer (must remain
// valid throughout our use of the ray trace builder), and
// instantiate an unspecified queue of the given family for use.
m_rtBuilder.setup(device, memoryAllocator, queueIndex);

// You create a vector of RayTracingBuilderKHR::BlasInput then
// pass it to buildBlas.
std::vector<RayTracingBuilderKHR::BlasInput> inputs = // ...
m_rtBuilder.buildBlas(inputs);

// You create a vector of RaytracingBuilder::Instance and pass to
// buildTlas. The blasId member of each instance must be below
// inputs.size() (above).
std::vector<RayTracingBuilderKHR::Instance> instances = // ...
m_rtBuilder.buildTlas(instances);

// Retrieve the handle to the acceleration structure.
const VkAccelerationStructureKHR tlas = m.rtBuilder.getAccelerationStructure()
~~~~
*/

#include <mutex>
#include <vulkan/vulkan.h>

#include "allocator_vk.hpp"
#include "commands_vk.hpp"
#include "debug_util_vk.hpp"
#include "nvh/nvprint.hpp"
#include "nvmath/nvmath.h"

#include "nvh/nvprint.hpp"

#if VK_KHR_acceleration_structure

namespace nvvk {
struct RaytracingBuilderKHR
{
  RaytracingBuilderKHR(RaytracingBuilderKHR const&) = delete;
  RaytracingBuilderKHR& operator=(RaytracingBuilderKHR const&) = delete;

  RaytracingBuilderKHR() = default;

  // Inputs used to build Bottom-level acceleration structure.
  // You manage the lifetime of the buffer(s) referenced by the
  // VkAccelerationStructureGeometryKHRs within. In particular, you must
  // make sure they are still valid and not being modified when the BLAS
  // is built or updated.
  struct BlasInput
  {
    // Data used to build acceleration structure geometry
    std::vector<VkAccelerationStructureGeometryKHR>       asGeometry;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> asBuildOffsetInfo;
  };

private:
  // Bottom-level acceleration structure, along with the information needed to re-build it.
  struct BlasEntry
  {
    // User-provided input.
    BlasInput input;

    // VkAccelerationStructureKHR plus extra info needed for our memory allocator.
    // The RaytracingBuilderKHR that created this DOES destroy it when destroyed.
    nvvk::AccelKHR as;

    // Additional parameters for acceleration structure builds
    VkBuildAccelerationStructureFlagsKHR flags = 0;

    BlasEntry() = default;
    BlasEntry(BlasInput input_)
        : input(std::move(input_))
        , as()
    {
    }
  };

public:
  //--------------------------------------------------------------------------------------------------
  // Initializing the allocator and querying the raytracing properties
  //

  void setup(const VkDevice& device, nvvk::Allocator* allocator, uint32_t queueIndex)
  {
    m_device     = device;
    m_queueIndex = queueIndex;
    m_debug.setup(device);
    m_alloc = allocator;
  }

  // This is an instance of a BLAS
  struct Instance
  {
    uint32_t                   blasId{0};            // Index of the BLAS in m_blas
    uint32_t                   instanceCustomId{0};  // Instance Index (gl_InstanceCustomIndexEXT)
    uint32_t                   hitGroupId{0};        // Hit group index in the SBT
    uint32_t                   mask{0xFF};           // Visibility mask, will be AND-ed with ray mask
    VkGeometryInstanceFlagsKHR flags{VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR};
    nvmath::mat4f              transform{nvmath::mat4f(1)};  // Identity
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
    m_blas.clear();
    m_tlas = {};
  }

  // Returning the constructed top-level acceleration structure
  VkAccelerationStructureKHR getAccelerationStructure() const { return m_tlas.as.accel; }

  //--------------------------------------------------------------------------------------------------
  // Create all the BLAS from the vector of BlasInput
  // - There will be one BLAS per input-vector entry
  // - There will be as many BLAS as input.size()
  // - The resulting BLAS (along with the inputs used to build) are stored in m_blas,
  //   and can be referenced by index.

  void buildBlas(const std::vector<RaytracingBuilderKHR::BlasInput>& input,
                 VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR)
  {
    // Cannot call buildBlas twice.
    assert(m_blas.empty());

    // Make our own copy of the user-provided inputs.
    m_blas          = std::vector<BlasEntry>(input.begin(), input.end());
    uint32_t nbBlas = static_cast<uint32_t>(m_blas.size());

    // Preparing the build information array for the acceleration build command.
    // This is mostly just a fancy pointer to the user-passed arrays of VkAccelerationStructureGeometryKHR.
    // dstAccelerationStructure will be filled later once we allocated the acceleration structures.
    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> buildInfos(nbBlas);
    for(uint32_t idx = 0; idx < nbBlas; idx++)
    {
      buildInfos[idx].sType                    = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
      buildInfos[idx].flags                    = flags;
      buildInfos[idx].geometryCount            = (uint32_t)m_blas[idx].input.asGeometry.size();
      buildInfos[idx].pGeometries              = m_blas[idx].input.asGeometry.data();
      buildInfos[idx].mode                     = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
      buildInfos[idx].type                     = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
      buildInfos[idx].srcAccelerationStructure = VK_NULL_HANDLE;
    }

    // Finding sizes to create acceleration structures and scratch
    // Keep the largest scratch buffer size, to use only one scratch for all build
    VkDeviceSize              maxScratch{0};          // Largest scratch buffer for our BLAS
    std::vector<VkDeviceSize> originalSizes(nbBlas);  // use for stats

    for(size_t idx = 0; idx < nbBlas; idx++)
    {
      // Query both the size of the finished acceleration structure and the  amount of scratch memory
      // needed (both written to sizeInfo). The `vkGetAccelerationStructureBuildSizesKHR` function
      // computes the worst case memory requirements based on the user-reported max number of
      // primitives. Later, compaction can fix this potential inefficiency.
      std::vector<uint32_t> maxPrimCount(m_blas[idx].input.asBuildOffsetInfo.size());
      for(auto tt = 0; tt < m_blas[idx].input.asBuildOffsetInfo.size(); tt++)
        maxPrimCount[tt] = m_blas[idx].input.asBuildOffsetInfo[tt].primitiveCount;  // Number of primitives/triangles
      VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
      vkGetAccelerationStructureBuildSizesKHR(m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                              &buildInfos[idx], maxPrimCount.data(), &sizeInfo);

      // Create acceleration structure object. Not yet bound to memory.
      VkAccelerationStructureCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
      createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
      createInfo.size = sizeInfo.accelerationStructureSize;  // Will be used to allocate memory.

      // Actual allocation of buffer and acceleration structure. Note: This relies on createInfo.offset == 0
      // and fills in createInfo.buffer with the buffer allocated to store the BLAS. The underlying
      // vkCreateAccelerationStructureKHR call then consumes the buffer value.
      m_blas[idx].as = m_alloc->createAcceleration(createInfo);
      m_debug.setObjectName(m_blas[idx].as.accel, (std::string("Blas" + std::to_string(idx)).c_str()));
      buildInfos[idx].dstAccelerationStructure = m_blas[idx].as.accel;  // Setting the where the build lands

      // Keeping info
      m_blas[idx].flags = flags;
      maxScratch        = std::max(maxScratch, sizeInfo.buildScratchSize);

      // Stats - Original size
      originalSizes[idx] = sizeInfo.accelerationStructureSize;
    }

    // Allocate the scratch buffers holding the temporary data of the
    // acceleration structure builder
    nvvk::Buffer scratchBuffer =
        m_alloc->createBuffer(maxScratch, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VkBufferDeviceAddressInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    bufferInfo.buffer              = scratchBuffer.buffer;
    VkDeviceAddress scratchAddress = vkGetBufferDeviceAddress(m_device, &bufferInfo);


    // Is compaction requested?
    bool doCompaction = (flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR)
                        == VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;

    // Allocate a query pool for storing the needed size for every BLAS compaction.
    VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpci.queryCount = nbBlas;
    qpci.queryType  = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
    VkQueryPool queryPool;
    vkCreateQueryPool(m_device, &qpci, nullptr, &queryPool);


    // Allocate a command pool for queue of given queue index.
    // To avoid timeout, record and submit one command buffer per AS build.
    nvvk::CommandPool            genCmdBuf(m_device, m_queueIndex);
    std::vector<VkCommandBuffer> allCmdBufs(nbBlas);

    // Building the acceleration structures
    for(uint32_t idx = 0; idx < nbBlas; idx++)
    {
      auto&           blas   = m_blas[idx];
      VkCommandBuffer cmdBuf = genCmdBuf.createCommandBuffer();
      allCmdBufs[idx]        = cmdBuf;

      // All build are using the same scratch buffer
      buildInfos[idx].scratchData.deviceAddress = scratchAddress;

      // Convert user vector of offsets to vector of pointer-to-offset (required by vk).
      // Recall that this defines which (sub)section of the vertex/index arrays
      // will be built into the BLAS.
      std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> pBuildOffset(blas.input.asBuildOffsetInfo.size());
      for(size_t infoIdx = 0; infoIdx < blas.input.asBuildOffsetInfo.size(); infoIdx++)
        pBuildOffset[infoIdx] = &blas.input.asBuildOffsetInfo[infoIdx];

      // Building the AS
      vkCmdBuildAccelerationStructuresKHR(cmdBuf, 1, &buildInfos[idx], pBuildOffset.data());

      // Since the scratch buffer is reused across builds, we need a barrier to ensure one build
      // is finished before starting the next one
      VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
      barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
      vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                           VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier, 0, nullptr, 0, nullptr);

      // Write compacted size to query number idx.
      if(doCompaction)
      {
        vkCmdWriteAccelerationStructuresPropertiesKHR(cmdBuf, 1, &blas.as.accel,
                                                      VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, queryPool, idx);
      }
    }
    genCmdBuf.submitAndWait(allCmdBufs);  // vkQueueWaitIdle behind this call.
    allCmdBufs.clear();

    // Compacting all BLAS
    if(doCompaction)
    {
      VkCommandBuffer cmdBuf = genCmdBuf.createCommandBuffer();

      // Get the size result back
      std::vector<VkDeviceSize> compactSizes(nbBlas);
      vkGetQueryPoolResults(m_device, queryPool, 0, (uint32_t)compactSizes.size(), compactSizes.size() * sizeof(VkDeviceSize),
                            compactSizes.data(), sizeof(VkDeviceSize), VK_QUERY_RESULT_WAIT_BIT);


      // Compacting
      std::vector<nvvk::AccelKHR> cleanupAS(nbBlas);  // previous AS to destroy
      uint32_t                    statTotalOriSize{0}, statTotalCompactSize{0};
      for(uint32_t idx = 0; idx < nbBlas; idx++)
      {
        // LOGI("Reducing %i, from %d to %d \n", i, originalSizes[i], compactSizes[i]);
        statTotalOriSize += (uint32_t)originalSizes[idx];
        statTotalCompactSize += (uint32_t)compactSizes[idx];

        // Creating a compact version of the AS
        VkAccelerationStructureCreateInfoKHR asCreateInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        asCreateInfo.size = compactSizes[idx];
        asCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        auto as           = m_alloc->createAcceleration(asCreateInfo);

        // Copy the original BLAS to a compact version
        VkCopyAccelerationStructureInfoKHR copyInfo{VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR};
        copyInfo.src  = m_blas[idx].as.accel;
        copyInfo.dst  = as.accel;
        copyInfo.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
        vkCmdCopyAccelerationStructureKHR(cmdBuf, &copyInfo);
        cleanupAS[idx] = m_blas[idx].as;
        m_blas[idx].as = as;
      }
      genCmdBuf.submitAndWait(cmdBuf);  // vkQueueWaitIdle within.

      // Destroying the previous version
      for(auto as : cleanupAS)
        m_alloc->destroy(as);

      LOGI(" RT BLAS: reducing from: %u to: %u = %u (%2.2f%s smaller) \n", statTotalOriSize, statTotalCompactSize,
           statTotalOriSize - statTotalCompactSize,
           (statTotalOriSize - statTotalCompactSize) / float(statTotalOriSize) * 100.f, "%%");
    }

    vkDestroyQueryPool(m_device, queryPool, nullptr);
    m_alloc->destroy(scratchBuffer);
    m_alloc->finalizeAndReleaseStaging();
  }


  //--------------------------------------------------------------------------------------------------
  // Convert an Instance object into a VkAccelerationStructureInstanceKHR

  VkAccelerationStructureInstanceKHR instanceToVkGeometryInstanceKHR(const Instance& instance)
  {
    assert(size_t(instance.blasId) < m_blas.size());
    BlasEntry& blas{m_blas[instance.blasId]};

    VkAccelerationStructureDeviceAddressInfoKHR addressInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    addressInfo.accelerationStructure = blas.as.accel;
    VkDeviceAddress blasAddress       = vkGetAccelerationStructureDeviceAddressKHR(m_device, &addressInfo);

    VkAccelerationStructureInstanceKHR gInst{};
    // The matrices for the instance transforms are row-major, instead of
    // column-major in the rest of the application
    nvmath::mat4f transp = nvmath::transpose(instance.transform);
    // The gInst.transform value only contains 12 values, corresponding to a 4x3
    // matrix, hence saving the last row that is anyway always (0,0,0,1). Since
    // the matrix is row-major, we simply copy the first 12 values of the
    // original 4x4 matrix
    memcpy(&gInst.transform, &transp, sizeof(gInst.transform));
    gInst.instanceCustomIndex                    = instance.instanceCustomId;
    gInst.mask                                   = instance.mask;
    gInst.instanceShaderBindingTableRecordOffset = instance.hitGroupId;
    gInst.flags                                  = instance.flags;
    gInst.accelerationStructureReference         = blasAddress;
    return gInst;
  }

  //--------------------------------------------------------------------------------------------------
  // Creating the top-level acceleration structure from the vector of Instance
  // - See struct of Instance
  // - The resulting TLAS will be stored in m_tlas
  // - update is to rebuild the Tlas with updated matrices
  void buildTlas(const std::vector<Instance>&         instances,
                 VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
                 bool                                 update = false)
  {
    // Cannot call buildTlas twice except to update.
    assert(m_tlas.as.accel == VK_NULL_HANDLE || update);

    nvvk::CommandPool genCmdBuf(m_device, m_queueIndex);
    VkCommandBuffer   cmdBuf = genCmdBuf.createCommandBuffer();

    m_tlas.flags = flags;

    // Convert array of our Instances to an array native Vulkan instances.
    std::vector<VkAccelerationStructureInstanceKHR> geometryInstances;
    geometryInstances.reserve(instances.size());
    for(const auto& inst : instances)
    {
      geometryInstances.push_back(instanceToVkGeometryInstanceKHR(inst));
    }

    // Create a buffer holding the actual instance data (matrices++) for use by the AS builder
    VkDeviceSize instanceDescsSizeInBytes = instances.size() * sizeof(VkAccelerationStructureInstanceKHR);

    // Allocate the instance buffer and copy its contents from host to device memory
    if(update)
      m_alloc->destroy(m_instBuffer);
    m_instBuffer = m_alloc->createBuffer(cmdBuf, geometryInstances, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    m_debug.setObjectName(m_instBuffer.buffer, "TLASInstances");
    VkBufferDeviceAddressInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    bufferInfo.buffer               = m_instBuffer.buffer;
    VkDeviceAddress instanceAddress = vkGetBufferDeviceAddress(m_device, &bufferInfo);

    // Make sure the copy of the instance buffer are copied before triggering the
    // acceleration structure build
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);


    //--------------------------------------------------------------------------------------------------

    // Create VkAccelerationStructureGeometryInstancesDataKHR
    // This wraps a device pointer to the above uploaded instances.
    VkAccelerationStructureGeometryInstancesDataKHR instancesVk{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};
    instancesVk.arrayOfPointers    = VK_FALSE;
    instancesVk.data.deviceAddress = instanceAddress;

    // Put the above into a VkAccelerationStructureGeometryKHR. We need to put the
    // instances struct in a union and label it as instance data.
    VkAccelerationStructureGeometryKHR topASGeometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    topASGeometry.geometryType       = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    topASGeometry.geometry.instances = instancesVk;

    // Find sizes
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    buildInfo.flags         = flags;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries   = &topASGeometry;
    buildInfo.mode = update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.type                     = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;

    uint32_t                                 count = (uint32_t)instances.size();
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR(m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &count, &sizeInfo);


    // Create TLAS
    if(update == false)
    {
      VkAccelerationStructureCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
      createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
      createInfo.size = sizeInfo.accelerationStructureSize;

      m_tlas.as = m_alloc->createAcceleration(createInfo);
      m_debug.setObjectName(m_tlas.as.accel, "Tlas");
    }

    // Allocate the scratch memory
    nvvk::Buffer scratchBuffer =
        m_alloc->createBuffer(sizeInfo.buildScratchSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                                                             | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    bufferInfo.buffer              = scratchBuffer.buffer;
    VkDeviceAddress scratchAddress = vkGetBufferDeviceAddress(m_device, &bufferInfo);


    // Update build information
    buildInfo.srcAccelerationStructure  = update ? m_tlas.as.accel : VK_NULL_HANDLE;
    buildInfo.dstAccelerationStructure  = m_tlas.as.accel;
    buildInfo.scratchData.deviceAddress = scratchAddress;


    // Build Offsets info: n instances
    VkAccelerationStructureBuildRangeInfoKHR        buildOffsetInfo{static_cast<uint32_t>(instances.size()), 0, 0, 0};
    const VkAccelerationStructureBuildRangeInfoKHR* pBuildOffsetInfo = &buildOffsetInfo;

    // Build the TLAS
    vkCmdBuildAccelerationStructuresKHR(cmdBuf, 1, &buildInfo, &pBuildOffsetInfo);

    genCmdBuf.submitAndWait(cmdBuf);  // queueWaitIdle inside.
    m_alloc->finalizeAndReleaseStaging();
    m_alloc->destroy(scratchBuffer);
  }

  //--------------------------------------------------------------------------------------------------
  // Refit BLAS number blasIdx from updated buffer contents.
  //
  void updateBlas(uint32_t blasIdx)
  {
    assert(size_t(blasIdx) < m_blas.size());
    BlasEntry& blas = m_blas[blasIdx];  // The blas to update

    // Preparing all build information, acceleration is filled later
    VkAccelerationStructureBuildGeometryInfoKHR buildInfos{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    buildInfos.flags                    = blas.flags;
    buildInfos.geometryCount            = (uint32_t)blas.input.asGeometry.size();
    buildInfos.pGeometries              = blas.input.asGeometry.data();
    buildInfos.mode                     = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;  // UPDATE
    buildInfos.type                     = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfos.srcAccelerationStructure = blas.as.accel;  // UPDATE
    buildInfos.dstAccelerationStructure = blas.as.accel;

    // Find size to build on the device
    std::vector<uint32_t> maxPrimCount(blas.input.asBuildOffsetInfo.size());
    for(auto tt = 0; tt < blas.input.asBuildOffsetInfo.size(); tt++)
      maxPrimCount[tt] = blas.input.asBuildOffsetInfo[tt].primitiveCount;  // Number of primitives/triangles
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR(m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfos,
                                            maxPrimCount.data(), &sizeInfo);

    // Allocate the scratch buffer and setting the scratch info
    nvvk::Buffer scratchBuffer =
        m_alloc->createBuffer(sizeInfo.buildScratchSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                                                             | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    VkBufferDeviceAddressInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    bufferInfo.buffer                    = scratchBuffer.buffer;
    buildInfos.scratchData.deviceAddress = vkGetBufferDeviceAddress(m_device, &bufferInfo);


    std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> pBuildOffset(blas.input.asBuildOffsetInfo.size());
    for(size_t i = 0; i < blas.input.asBuildOffsetInfo.size(); i++)
      pBuildOffset[i] = &blas.input.asBuildOffsetInfo[i];

    // Update the instance buffer on the device side and build the TLAS
    nvvk::CommandPool genCmdBuf(m_device, m_queueIndex);
    VkCommandBuffer   cmdBuf = genCmdBuf.createCommandBuffer();


    // Update the acceleration structure. Note the VK_TRUE parameter to trigger the update,
    // and the existing BLAS being passed and updated in place
    vkCmdBuildAccelerationStructuresKHR(cmdBuf, 1, &buildInfos, pBuildOffset.data());

    genCmdBuf.submitAndWait(cmdBuf);
    m_alloc->destroy(scratchBuffer);
  }

private:
  // Top-level acceleration structure
  struct Tlas
  {
    nvvk::AccelKHR                       as;
    VkBuildAccelerationStructureFlagsKHR flags = 0;
  };

  //--------------------------------------------------------------------------------------------------
  // Vector containing all the BLASes built in buildBlas (and referenced by the TLAS)
  std::vector<BlasEntry> m_blas;
  // Top-level acceleration structure
  Tlas m_tlas;
  // Instance buffer containing the matrices and BLAS ids
  nvvk::Buffer m_instBuffer;

  VkDevice m_device{VK_NULL_HANDLE};
  uint32_t m_queueIndex{0};

  nvvk::Allocator* m_alloc = nullptr;
  nvvk::DebugUtil  m_debug;

#ifdef VULKAN_HPP
public:
  void buildBlas(const std::vector<RaytracingBuilderKHR::BlasInput>& blas_,
                 vk::BuildAccelerationStructureFlagsKHR              flags)
  {
    buildBlas(blas_, static_cast<VkBuildAccelerationStructureFlagsKHR>(flags));
  }

  void buildTlas(const std::vector<Instance>&           instances,
                 vk::BuildAccelerationStructureFlagsKHR flags,
                 bool                                   update = false)
  {
    buildTlas(instances, static_cast<VkBuildAccelerationStructureFlagsKHR>(flags), update);
  }

#endif
};

}  // namespace nvvk

#else
#error This include requires VK_KHR_ray_tracing support in the Vulkan SDK.
#endif
