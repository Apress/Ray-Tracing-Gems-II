// ======================================================================== //
// Copyright 2019-2021 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "owl/owl.h"
#include "owl/common.h"
#include "owl/DeviceMemory.h"
#include "owl/helper/optix.h"

namespace owl {

  /*! tracks which ID regions in the SBT have already been used -
    newly created groups allocate ranges of IDs in the SBT (to allow
    its geometries to be in successive SBT regions), and this struct
    keeps track of whats already used, and what is available */
  struct RangeAllocator {
    int alloc(size_t size);
    void release(size_t begin, size_t size);
    size_t maxAllocedID = 0;
  private:
    struct FreedRange {
      size_t begin;
      size_t size;
    };
    std::vector<FreedRange> freedRanges;
  };

  /*! helper clas to handle device-side shader binding table
      creation */
  struct SBT {
    size_t rayGenRecordCount   = 0;
    size_t rayGenRecordSize    = 0;
    DeviceMemory rayGenRecordsBuffer;

    size_t hitGroupRecordSize  = 0;
    size_t hitGroupRecordCount = 0;
    DeviceMemory hitGroupRecordsBuffer;

    size_t missProgRecordSize  = 0;
    size_t missProgRecordCount = 0;
    DeviceMemory missProgRecordsBuffer;
    
    DeviceMemory launchParamsBuffer;
  };

  /*! what will eventually containt the whole owl context across all gpus */
  struct Context;

  /*! optix and cuda context for a single, specific GPU */
  struct DeviceContext : public std::enable_shared_from_this<DeviceContext>  {
    typedef std::shared_ptr<DeviceContext> SP;

    /*! create a new device context with given context object, using
        given GPU "cudaID", and serving the rols at the "owlID"th GPU
        in that context */
    DeviceContext(Context *parent,
                  int owlID,
                  int cudaID);
    ~DeviceContext();
    
    /*! helper function - return cuda name of this device */
    std::string getDeviceName() const;
      
    /*! helper function - return cuda device ID of this device */
    int getCudaDeviceID() const;

    /*! return the optix default stream for this device. launch params
        may use their own stream */
    CUstream getStream() const { return stream; }

    /*! configures the optixPipeline link options and compile options,
        based on what values (motion blur on/off, multi-level
        instnacing, etc) are set in the context */
    void configurePipelineOptions(bool debug = false);
      
    void buildPrograms();
    void buildMissPrograms();
    void buildRayGenPrograms();
    void buildHitGroupPrograms();

    void destroyPrograms();
    void destroyMissPrograms();
    void destroyRayGenPrograms();
    void destroyHitGroupPrograms();

    void destroyPipeline();
    void buildPipeline();

    /*! collects all compiled programs during 'buildPrograms', such
        that all active progs can then be passed to optix durign
        pipeline creation */
    std::vector<OptixProgramGroup> allActivePrograms;

    OptixDeviceContext optixContext = nullptr;
    CUcontext          cudaContext  = nullptr;
    CUstream           stream       = nullptr;

    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixPipelineLinkOptions    pipelineLinkOptions    = {};
    OptixModuleCompileOptions   moduleCompileOptions   = {};
    OptixPipeline               pipeline               = nullptr;
    SBT                         sbt                    = {};

    /*! the owl context that this device is in */
    Context *const parent;

    /*! linear ID (0,1,2,...) of how *we* number devices (i.e.,
      'first' device is always device 0, no matter if it runs on
      another physical/cuda device) */
    const int      ID;

    /* the cuda device ID that this logical device runs on */
    const int      cudaDeviceID;
  };

  /*! creates the N device contexts with the given device IDs. If list
      of device is nullptr, and number requested devices is > 1, then
      the first N devices will get used; invalid device IDs in the
      list will automatically get dropped */
  std::vector<DeviceContext::SP> createDeviceContexts(Context *parent,
                                                      int32_t *requestedDeviceIDs,
                                                      int      numRequestedDevices);

  /*! helper class that will set the active cuda device (to the device
      associated with a given Context::DeviceData) for the duration fo
      the lifetime of this class, and resets it to whatever it was
      after class dies */
  struct SetActiveGPU {
    inline SetActiveGPU(const DeviceContext::SP &device)
    {
      OWL_CUDA_CHECK(cudaGetDevice(&savedActiveDeviceID));
      OWL_CUDA_CHECK(cudaSetDevice(device->cudaDeviceID));
    }
    inline SetActiveGPU(const DeviceContext *device)
    {
      OWL_CUDA_CHECK(cudaGetDevice(&savedActiveDeviceID));
      OWL_CUDA_CHECK(cudaSetDevice(device->cudaDeviceID));
    }
    inline ~SetActiveGPU()
    {
      OWL_CUDA_CHECK_NOTHROW(cudaSetDevice(savedActiveDeviceID));
    }
  private:
    int savedActiveDeviceID = -1;
  };
  
} // ::owl

