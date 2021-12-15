// ======================================================================== //
// Copyright 2019 Ingo Wald                                                 //
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

#include "MissProg.h"
#include "Context.h"

namespace owl {

  // ------------------------------------------------------------------
  // MissProgType::DeviceData
  // ------------------------------------------------------------------

  /*! constructor, only pass-through to parent class */
  MissProgType::DeviceData::DeviceData(const DeviceContext::SP &device)
    : RegisteredObject::DeviceData(device)
  {}

  // ------------------------------------------------------------------
  // MissProgType
  // ------------------------------------------------------------------

  MissProgType::MissProgType(Context *const context,
                             Module::SP module,
                             const std::string &progName,
                             size_t varStructSize,
                             const std::vector<OWLVarDecl> &varDecls)
    : SBTObjectType(context,context->missProgTypes,varStructSize,varDecls),
      module(module),
      progName(progName),
      annotatedProgName("__miss__"+progName)
  {}
  
  /*! pretty-printer, for printf-debugging */
  std::string MissProgType::toString() const
  {
    return "MissProgType";
  }

  /*! creates the device-specific data for this group */
  RegisteredObject::DeviceData::SP
  MissProgType::createOn(const DeviceContext::SP &device) 
  {
    return std::make_shared<DeviceData>(device);
  }
  
  // ------------------------------------------------------------------
  // MissProg
  // ------------------------------------------------------------------
  
  MissProg::MissProg(Context *const context,
                     MissProgType::SP type) 
    : SBTObject(context,context->missProgs,type)
  {}
  
  /*! pretty-printer, for printf-debugging */
  std::string MissProg::toString() const
  {
    return "MissProg";
  }

  /*! write the given SBT record, using the given device's
  corresponding device-side data represenataion */
  void MissProg::writeSBTRecord(uint8_t *const sbtRecord,
                                const DeviceContext::SP &device)
  {
    auto &dd = type->getDD(device);
    
    // first, compute pointer to record:
    uint8_t *const sbtRecordHeader = sbtRecord;
    uint8_t *const sbtRecordData   = sbtRecord+OPTIX_SBT_RECORD_HEADER_SIZE;

    // ------------------------------------------------------------------
    // pack record header with the corresponding hit group:
    // ------------------------------------------------------------------
    OPTIX_CALL(SbtRecordPackHeader(dd.pg,sbtRecordHeader));
    
    // ------------------------------------------------------------------
    // then, write the data for that record
    // ------------------------------------------------------------------
    writeVariables(sbtRecordData,device);
  }  
  
} // ::owl

