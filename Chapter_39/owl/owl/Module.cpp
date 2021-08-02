// ======================================================================== //
// Copyright 2019-2020 Ingo Wald                                            //
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

#include "Module.h"
#include "Context.h"

#define LOG(message)                            \
  if (Context::logging())                       \
    std::cout                                   \
      << OWL_TERMINAL_LIGHT_BLUE                \
      << "#owl: "                               \
      << message                                \
      << OWL_TERMINAL_DEFAULT << std::endl

#define LOG_OK(message)                         \
  if (Context::logging())                       \
    std::cout                                   \
      << OWL_TERMINAL_BLUE                      \
      << "#owl: "                               \
      << message                                \
      << OWL_TERMINAL_DEFAULT << std::endl


namespace owl {

  /*! get next single line of PTX code */
  std::string getNextLine(const char *&s)
  {
    std::stringstream line;
    while (*s) {
      char c = *s++;
      line << c;
      if (c == '\n') break;
    }
    return line.str();
  }

  /*! given the original PTX code, create a version of this PTX code
      in which all lines that refer to an internal optix symbol (ie,
      that contains ' _optix_' get commented out. This will make this
      PTX code invalid for all optix functions, but makes it
      compilable by cude for the non-optix bounds program */
  std::string killAllInternalOptixSymbolsFromPtxString(const char *orignalPtxCode)
  {
    std::stringstream fixed;

    for (const char *s = orignalPtxCode; *s; ) {
      std::string line = getNextLine(s);
      if (line.find(" _optix_") != line.npos)
        fixed << "//dropped: " << line;
      else
        fixed << line;
    }
    return fixed.str();
  }


  // ------------------------------------------------------------------
  // Module::DeviceData
  // ------------------------------------------------------------------
  
  /*! constructor */
  Module::DeviceData::DeviceData(Module *parent, DeviceContext::SP device)
    : RegisteredObject::DeviceData(device),
      parent(parent)
  {}
  
  /*! destructor */
  Module::DeviceData::~DeviceData()
  {
    destroy();
  }

  /*! destroy the optix data for this module; the owl data for the
    module itself remains valid */
  void Module::DeviceData::destroy()
  {
    SetActiveGPU forLifeTime(device);
    
    if (module)
      optixModuleDestroy(module);
    module = 0;
  }

  /*! build the optix side of this module on this device */
  void Module::DeviceData::build()
  {
    assert(module == 0);
    SetActiveGPU forLifeTime(device);
    
    LOG("building module #" + parent->toString());
    
    char log[2048];
    size_t sizeof_log = sizeof( log );

    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(device->optixContext,
                                             &device->moduleCompileOptions,
                                             &device->pipelineCompileOptions,
                                             parent->ptxCode.c_str(),
                                             strlen(parent->ptxCode.c_str()),
                                             log,      // Log string
                                             &sizeof_log,// Log string sizse
                                             &module
                                             ));
    assert(module != nullptr);

    // ------------------------------------------------------------------
    // Now, build separate cuda-only module that does not contain
    // any optix-internal symbols. Note this does not actually
    // *remove* any potentially existing anyhit/closesthit/etc.
    // programs in this module - it just removed all optix-related
    // calls from this module, but leaves the remaining (now
    // dysfunctional) anyhit/closesthit/etc. programs still in that
    // PTX code. It would obviously be cleaner to completely
    // remove those programs, but that would require significantly
    // more advanced parsing of the PTX string, so right now we'll
    // just leave them in (and as it's in a module that never gets
    // used by optix, this should actually be OK).
    // ------------------------------------------------------------------
    LOG("generating second, 'non-optix' version of that module, too");
    CUresult rc = (CUresult)0;
    const std::string fixedPtxCode
      = killAllInternalOptixSymbolsFromPtxString(parent->ptxCode.c_str());
    strcpy(log,"(no log yet)");
    CUjit_option options[] = {
                              CU_JIT_TARGET_FROM_CUCONTEXT,
                              CU_JIT_ERROR_LOG_BUFFER,
                              CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
    };
    void *optionValues[] = {
                            (void*)0,
                            (void*)log,
                            (void*)sizeof(log)
    };
    
    rc = cuModuleLoadDataEx(&boundsModule, (void *)fixedPtxCode.c_str(),
                            3, options, optionValues);
    if (rc != CUDA_SUCCESS) {
      const char *errName = 0;
      cuGetErrorName(rc,&errName);
      throw std::runtime_error("unknown CUDA error when building module "
                               "for bounds program kernel"
                               +std::string(errName));
    }
    LOG_OK("created module #" << parent->ID << " (both optix and cuda)");
  }
  


  // ------------------------------------------------------------------
  // Module
  // ------------------------------------------------------------------
  
  /*! constructor - ptxCode contains the prec-ompiled ptx code with
    the compiled functions */
  Module::Module(Context *const context,
                 const std::string &ptxCode)
    : RegisteredObject(context,context->modules),
      ptxCode(ptxCode)
  {}

  /*! destructor, to release data if required */
  Module::~Module()
  {
    for (auto device : context->getDevices())
      getDD(device).destroy();
  }
  
  /*! create this object's device-specific data for the device */
  RegisteredObject::DeviceData::SP Module::createOn(const DeviceContext::SP &device) 
  {
    return std::make_shared<DeviceData>(this,device);
  }
  
  /*! pretty-printer, for printf-debugging */
  std::string Module::toString() const
  {
    return "Module";
  }
    
} // ::owl
