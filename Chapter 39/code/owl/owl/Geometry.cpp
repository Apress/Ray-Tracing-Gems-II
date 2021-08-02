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

#include "Geometry.h"
#include "Context.h"

namespace owl {

  // ------------------------------------------------------------------
  // GeomType::DeviceData
  // ------------------------------------------------------------------
  
  /*! construtor - passthrough to parent class */
  GeomType::DeviceData::DeviceData(const DeviceContext::SP &device)
    : RegisteredObject::DeviceData(device)
  {}

  /*! fill in an OptixProgramGroup descriptor with the module and
    program names for this type */
  void GeomType::DeviceData::fillPGDesc(OptixProgramGroupDesc &pgDesc,
                                        GeomType *parent, int rt)
  {
    pgDesc.hitgroup = {};

    // ----------- closest hit -----------
    if (rt < (int)parent->closestHit.size()) {
      const ProgramDesc &pd = parent->closestHit[rt];
      if (pd.module && pd.progName != "") {
        pgDesc.hitgroup.moduleCH = pd.module->getDD(device).module;
        pgDesc.hitgroup.entryFunctionNameCH = pd.progName.c_str();
      }
    }
    // ----------- any hit -----------
    if (rt < (int)parent->anyHit.size()) {
      const ProgramDesc &pd = parent->anyHit[rt];
      if (pd.module && pd.progName != "") {
        pgDesc.hitgroup.moduleAH = pd.module->getDD(device).module;
        pgDesc.hitgroup.entryFunctionNameAH = pd.progName.c_str();
      }
    }
  }

  // ------------------------------------------------------------------
  // GeomType
  // ------------------------------------------------------------------
  
  /*! constructor - mostly pass through to parent class */
  GeomType::GeomType(Context *const context,
                     size_t varStructSize,
                     const std::vector<OWLVarDecl> &varDecls)
    : SBTObjectType(context,context->geomTypes,
                    varStructSize,varDecls),
      closestHit(context->numRayTypes),
      anyHit(context->numRayTypes)
  {}
  
  /*! creates the device-specific data for this group */
  RegisteredObject::DeviceData::SP
  GeomType::createOn(const DeviceContext::SP &device) 
  {
    return std::make_shared<DeviceData>(device);
  }
  
  /*! pretty-printer, for printf-debugging */
  std::string GeomType::toString() const
  {
    return "GeomType";
  }

    /*! sets the closest program to run for given ray type */
  void GeomType::setClosestHitProgram(int rayType,
                                      Module::SP module,
                                      const std::string &progName)
  {
    assert(rayType >=0 && rayType < (int)closestHit.size());
      
    closestHit[rayType].progName = "__closesthit__"+progName;
    closestHit[rayType].module   = module;
  }

    /*! sets the anyhit program to run for given ray type */
  void GeomType::setAnyHitProgram(int rayType,
                                  Module::SP module,
                                  const std::string &progName)
  {
    assert(rayType >=0 && rayType < (int)anyHit.size());
      
    anyHit[rayType].progName = "__anyhit__"+progName;
    anyHit[rayType].module   = module;
  }

  // ------------------------------------------------------------------
  // Geom
  // ------------------------------------------------------------------

  /*! constructor - mostly pass through to parent class */
  Geom::Geom(Context *const context,
             GeomType::SP geomType)
    : SBTObject(context,context->geoms,geomType), geomType(geomType)
  {
    assert(geomType);
  }

  /*! pretty-printer, for printf-debugging */
  std::string Geom::toString() const
  {
    return "Geom";
  }
  

  /*! write the SBT record for this object and ray type; this
    requires finding the proper programs (from the type and ray
    type), and writign the variables */
  void Geom::writeSBTRecord(uint8_t *const sbtRecord,
                            const DeviceContext::SP &device,
                            int rayTypeID)
  {
    // first, compute pointer to record:
    uint8_t *const sbtRecordHeader = sbtRecord;
    uint8_t *const sbtRecordData   = sbtRecord+OPTIX_SBT_RECORD_HEADER_SIZE;

    // ------------------------------------------------------------------
    // pack record header with the corresponding hit group:
    // ------------------------------------------------------------------
    auto &dd = geomType->getDD(device);
    assert(rayTypeID < (int)dd.hgPGs.size());
    OPTIX_CALL(SbtRecordPackHeader(dd.hgPGs[rayTypeID],sbtRecordHeader));
    
    // ------------------------------------------------------------------
    // then, write the data for that record
    // ------------------------------------------------------------------
    writeVariables(sbtRecordData,device);
  }  

} //::owl
