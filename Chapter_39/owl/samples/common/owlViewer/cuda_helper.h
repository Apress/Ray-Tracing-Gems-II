// ======================================================================== //
// Copyright 2018 Ingo Wald                                                 //
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

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

namespace ooawe {


  static const char *_cudaGetErrorEnum(cudaError_t error) {
    switch (error) {
    case cudaSuccess:
      return "cudaSuccess";

    case cudaErrorMissingConfiguration:
      return "cudaErrorMissingConfiguration";

    case cudaErrorMemoryAllocation:
      return "cudaErrorMemoryAllocation";

    case cudaErrorInitializationError:
      return "cudaErrorInitializationError";

    case cudaErrorLaunchFailure:
      return "cudaErrorLaunchFailure";

    case cudaErrorPriorLaunchFailure:
      return "cudaErrorPriorLaunchFailure";

    case cudaErrorLaunchTimeout:
      return "cudaErrorLaunchTimeout";

    case cudaErrorLaunchOutOfResources:
      return "cudaErrorLaunchOutOfResources";

    case cudaErrorInvalidDeviceFunction:
      return "cudaErrorInvalidDeviceFunction";

    case cudaErrorInvalidConfiguration:
      return "cudaErrorInvalidConfiguration";

    case cudaErrorInvalidDevice:
      return "cudaErrorInvalidDevice";

    case cudaErrorInvalidValue:
      return "cudaErrorInvalidValue";

    case cudaErrorInvalidPitchValue:
      return "cudaErrorInvalidPitchValue";

    case cudaErrorInvalidSymbol:
      return "cudaErrorInvalidSymbol";

    case cudaErrorMapBufferObjectFailed:
      return "cudaErrorMapBufferObjectFailed";

    case cudaErrorUnmapBufferObjectFailed:
      return "cudaErrorUnmapBufferObjectFailed";

    case cudaErrorInvalidHostPointer:
      return "cudaErrorInvalidHostPointer";

    case cudaErrorInvalidDevicePointer:
      return "cudaErrorInvalidDevicePointer";

    case cudaErrorInvalidTexture:
      return "cudaErrorInvalidTexture";

    case cudaErrorInvalidTextureBinding:
      return "cudaErrorInvalidTextureBinding";

    case cudaErrorInvalidChannelDescriptor:
      return "cudaErrorInvalidChannelDescriptor";

    case cudaErrorInvalidMemcpyDirection:
      return "cudaErrorInvalidMemcpyDirection";

    case cudaErrorAddressOfConstant:
      return "cudaErrorAddressOfConstant";

    case cudaErrorTextureFetchFailed:
      return "cudaErrorTextureFetchFailed";

    case cudaErrorTextureNotBound:
      return "cudaErrorTextureNotBound";

    case cudaErrorSynchronizationError:
      return "cudaErrorSynchronizationError";

    case cudaErrorInvalidFilterSetting:
      return "cudaErrorInvalidFilterSetting";

    case cudaErrorInvalidNormSetting:
      return "cudaErrorInvalidNormSetting";

    case cudaErrorMixedDeviceExecution:
      return "cudaErrorMixedDeviceExecution";

    case cudaErrorCudartUnloading:
      return "cudaErrorCudartUnloading";

    case cudaErrorUnknown:
      return "cudaErrorUnknown";

    case cudaErrorNotYetImplemented:
      return "cudaErrorNotYetImplemented";

    case cudaErrorMemoryValueTooLarge:
      return "cudaErrorMemoryValueTooLarge";

    case cudaErrorInvalidResourceHandle:
      return "cudaErrorInvalidResourceHandle";

    case cudaErrorNotReady:
      return "cudaErrorNotReady";

    case cudaErrorInsufficientDriver:
      return "cudaErrorInsufficientDriver";

    case cudaErrorSetOnActiveProcess:
      return "cudaErrorSetOnActiveProcess";

    case cudaErrorInvalidSurface:
      return "cudaErrorInvalidSurface";

    case cudaErrorNoDevice:
      return "cudaErrorNoDevice";

    case cudaErrorECCUncorrectable:
      return "cudaErrorECCUncorrectable";

    case cudaErrorSharedObjectSymbolNotFound:
      return "cudaErrorSharedObjectSymbolNotFound";

    case cudaErrorSharedObjectInitFailed:
      return "cudaErrorSharedObjectInitFailed";

    case cudaErrorUnsupportedLimit:
      return "cudaErrorUnsupportedLimit";

    case cudaErrorDuplicateVariableName:
      return "cudaErrorDuplicateVariableName";

    case cudaErrorDuplicateTextureName:
      return "cudaErrorDuplicateTextureName";

    case cudaErrorDuplicateSurfaceName:
      return "cudaErrorDuplicateSurfaceName";

    case cudaErrorDevicesUnavailable:
      return "cudaErrorDevicesUnavailable";

    case cudaErrorInvalidKernelImage:
      return "cudaErrorInvalidKernelImage";

    case cudaErrorNoKernelImageForDevice:
      return "cudaErrorNoKernelImageForDevice";

    case cudaErrorIncompatibleDriverContext:
      return "cudaErrorIncompatibleDriverContext";

    case cudaErrorPeerAccessAlreadyEnabled:
      return "cudaErrorPeerAccessAlreadyEnabled";

    case cudaErrorPeerAccessNotEnabled:
      return "cudaErrorPeerAccessNotEnabled";

    case cudaErrorDeviceAlreadyInUse:
      return "cudaErrorDeviceAlreadyInUse";

    case cudaErrorProfilerDisabled:
      return "cudaErrorProfilerDisabled";

    case cudaErrorProfilerNotInitialized:
      return "cudaErrorProfilerNotInitialized";

    case cudaErrorProfilerAlreadyStarted:
      return "cudaErrorProfilerAlreadyStarted";

    case cudaErrorProfilerAlreadyStopped:
      return "cudaErrorProfilerAlreadyStopped";

      /* Since CUDA 4.0*/
    case cudaErrorAssert:
      return "cudaErrorAssert";

    case cudaErrorTooManyPeers:
      return "cudaErrorTooManyPeers";

    case cudaErrorHostMemoryAlreadyRegistered:
      return "cudaErrorHostMemoryAlreadyRegistered";

    case cudaErrorHostMemoryNotRegistered:
      return "cudaErrorHostMemoryNotRegistered";

      /* Since CUDA 5.0 */
    case cudaErrorOperatingSystem:
      return "cudaErrorOperatingSystem";

    case cudaErrorPeerAccessUnsupported:
      return "cudaErrorPeerAccessUnsupported";

    case cudaErrorLaunchMaxDepthExceeded:
      return "cudaErrorLaunchMaxDepthExceeded";

    case cudaErrorLaunchFileScopedTex:
      return "cudaErrorLaunchFileScopedTex";

    case cudaErrorLaunchFileScopedSurf:
      return "cudaErrorLaunchFileScopedSurf";

    case cudaErrorSyncDepthExceeded:
      return "cudaErrorSyncDepthExceeded";

    case cudaErrorLaunchPendingCountExceeded:
      return "cudaErrorLaunchPendingCountExceeded";

    case cudaErrorNotPermitted:
      return "cudaErrorNotPermitted";

    case cudaErrorNotSupported:
      return "cudaErrorNotSupported";

      /* Since CUDA 6.0 */
    case cudaErrorHardwareStackError:
      return "cudaErrorHardwareStackError";

    case cudaErrorIllegalInstruction:
      return "cudaErrorIllegalInstruction";

    case cudaErrorMisalignedAddress:
      return "cudaErrorMisalignedAddress";

    case cudaErrorInvalidAddressSpace:
      return "cudaErrorInvalidAddressSpace";

    case cudaErrorInvalidPc:
      return "cudaErrorInvalidPc";

    case cudaErrorIllegalAddress:
      return "cudaErrorIllegalAddress";

      /* Since CUDA 6.5*/
    case cudaErrorInvalidPtx:
      return "cudaErrorInvalidPtx";

    case cudaErrorInvalidGraphicsContext:
      return "cudaErrorInvalidGraphicsContext";

    case cudaErrorStartupFailure:
      return "cudaErrorStartupFailure";

    case cudaErrorApiFailureBase:
      return "cudaErrorApiFailureBase";

      /* Since CUDA 8.0*/
    // case cudaErrorNvlinkUncorrectable:
    //   return "cudaErrorNvlinkUncorrectable";

    //   /* Since CUDA 8.5*/
    // case cudaErrorJitCompilerNotFound:
    //   return "cudaErrorJitCompilerNotFound";

    //   /* Since CUDA 9.0*/
    // case cudaErrorCooperativeLaunchTooLarge:
    //   return "cudaErrorCooperativeLaunchTooLarge";
    }

    return "<unknown>";
  }


  template <typename T>
  void checkCudaCall(T result,
                     char const *const func,
                     const char *const file,
                     int const line)
  {
    if (result) {
      fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
              static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
      cudaDeviceReset();
      // Make sure we call CUDA Device Reset before exiting
      exit(EXIT_FAILURE);
    }
  }
  
#define checkCudaErrors(val) checkCudaCall((val), #val, __FILE__, __LINE__)





  inline int gpuDeviceInit(int devID) {
    int device_count;
    checkCudaErrors(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
      fprintf(stderr,
              "gpuDeviceInit() CUDA error: "
              "no devices supporting CUDA.\n");
      exit(EXIT_FAILURE);
    }

    if (devID < 0) {
      devID = 0;
    }

    if (devID > device_count - 1) {
      fprintf(stderr, "\n");
      fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n",
              device_count);
      fprintf(stderr,
              ">> gpuDeviceInit (-device=%d) is not a valid"
              " GPU device. <<\n",
              devID);
      fprintf(stderr, "\n");
      return -devID;
    }

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    if (deviceProp.computeMode == cudaComputeModeProhibited) {
      fprintf(stderr,
              "Error: device is running in <Compute Mode "
              "Prohibited>, no threads can use cudaSetDevice().\n");
      return -1;
    }

    if (deviceProp.major < 1) {
      fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
      exit(EXIT_FAILURE);
    }

    checkCudaErrors(cudaSetDevice(devID));
    printf("gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID, deviceProp.name);

    return devID;
  }




  // Beginning of GPU Architecture definitions
  inline int _ConvertSMVer2Cores(int major, int minor) {
    // Defines for GPU Architecture types (using the SM version to determine
    // the # of cores per SM
    typedef struct {
      int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
      // and m = SM minor version
      int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192},  // Kepler Generation (SM 3.0) GK10x class
      {0x32, 192},  // Kepler Generation (SM 3.2) GK10x class
      {0x35, 192},  // Kepler Generation (SM 3.5) GK11x class
      {0x37, 192},  // Kepler Generation (SM 3.7) GK21x class
      {0x50, 128},  // Maxwell Generation (SM 5.0) GM10x class
      {0x52, 128},  // Maxwell Generation (SM 5.2) GM20x class
      {0x53, 128},  // Maxwell Generation (SM 5.3) GM20x class
      {0x60, 64},   // Pascal Generation (SM 6.0) GP100 class
      {0x61, 128},  // Pascal Generation (SM 6.1) GP10x class
      {0x62, 128},  // Pascal Generation (SM 6.2) GP10x class
      {0x70, 64},   // Volta Generation (SM 7.0) GV100 class
      {0x72, 64},   // Volta Generation (SM 7.2) GV11b class
      {-1, -1}};

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1) {
      if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
        return nGpuArchCoresPerSM[index].Cores;
      }

      index++;
    }

    // If we don't find the values, we default use the previous one
    // to run properly
    printf(
           "MapSMtoCores for SM %d.%d is undefined."
           "  Default to use %d Cores/SM\n",
           major, minor, nGpuArchCoresPerSM[index - 1].Cores);
    return nGpuArchCoresPerSM[index - 1].Cores;
  }


  // This function returns the best GPU (with maximum GFLOPS)
  inline int gpuGetMaxGflopsDeviceId() {
    int current_device = 0, sm_per_multiproc = 0;
    int max_perf_device = 0;
    int device_count = 0, best_SM_arch = 0;
    int devices_prohibited = 0;

    uint64_t max_compute_perf = 0;
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
      fprintf(stderr,
              "gpuGetMaxGflopsDeviceId() CUDA error:"
              " no devices supporting CUDA.\n");
      exit(EXIT_FAILURE);
    }

    // Find the best major SM Architecture GPU device
    while (current_device < device_count) {
      cudaGetDeviceProperties(&deviceProp, current_device);

      // If this GPU is not running on Compute Mode prohibited,
      // then we can add it to the list
      if (deviceProp.computeMode != cudaComputeModeProhibited) {
        if (deviceProp.major > 0 && deviceProp.major < 9999) {
          best_SM_arch = std::max<int>(best_SM_arch, deviceProp.major);
        }
      } else {
        devices_prohibited++;
      }

      current_device++;
    }

    if (devices_prohibited == device_count) {
      fprintf(stderr,
              "gpuGetMaxGflopsDeviceId() CUDA error:"
              " all devices have compute mode prohibited.\n");
      exit(EXIT_FAILURE);
    }

    // Find the best CUDA capable GPU device
    current_device = 0;

    while (current_device < device_count) {
      cudaGetDeviceProperties(&deviceProp, current_device);

      // If this GPU is not running on Compute Mode prohibited,
      // then we can add it to the list
      if (deviceProp.computeMode != cudaComputeModeProhibited) {
        if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
          sm_per_multiproc = 1;
        } else {
          sm_per_multiproc =
            _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
        }

        uint64_t compute_perf = (uint64_t)deviceProp.multiProcessorCount *
          sm_per_multiproc * deviceProp.clockRate;

        if (compute_perf > max_compute_perf) {
          // If we find GPU with SM major > 2, search only these
          if (best_SM_arch > 2) {
            // If our device==dest_SM_arch, choose this, or else pass
            if (deviceProp.major == best_SM_arch) {
              max_compute_perf = compute_perf;
              max_perf_device = current_device;
            }
          } else {
            max_compute_perf = compute_perf;
            max_perf_device = current_device;
          }
        }
      }

      ++current_device;
    }

    return max_perf_device;
  }


  // Initialization code to find the best CUDA Device
  inline int findCudaDevice(int argc, const char **argv) {
    cudaDeviceProp deviceProp;
    int devID = 0;

    // If the command-line has a device number specified, use it
    // if (checkCmdLineFlag(argc, argv, "device")) {
    //   devID = getCmdLineArgumentInt(argc, argv, "device=");

    //   if (devID < 0) {
    //     printf("Invalid command line parameter\n ");
    //     exit(EXIT_FAILURE);
    //   } else {
    //     devID = gpuDeviceInit(devID);

    //     if (devID < 0) {
    //       printf("exiting...\n");
    //       exit(EXIT_FAILURE);
    //     }
    //   }
    // } else {
    // Otherwise pick the device with highest Gflops/s
    devID = gpuGetMaxGflopsDeviceId();
    checkCudaErrors(cudaSetDevice(devID));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID,
           deviceProp.name, deviceProp.major, deviceProp.minor);
    // }

    return devID;
  }


  // This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

  inline void __getLastCudaError(const char *errorMessage, const char *file,
                                 const int line) {
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err) {
      fprintf(stderr,
              "%s(%i) : getLastCudaError() CUDA error :"
              " %s : (%d) %s.\n",
              file, line, errorMessage, static_cast<int>(err),
              cudaGetErrorString(err));
      // DEVICE_RESET
        exit(EXIT_FAILURE);
    }
  }


  
}
