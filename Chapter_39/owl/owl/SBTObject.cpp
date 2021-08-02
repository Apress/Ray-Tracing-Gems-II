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

#include "SBTObject.h"

namespace owl {

  // ------------------------------------------------------------------
  // SBTObjectType
  // ------------------------------------------------------------------

  /*! creates a full "deep" copy of the vardecls, just in case the app
      created variable names on the heap and releases them afterwards
      (we'll then have a clean copy) */
  std::vector<OWLVarDecl> copyVarDecls(const std::vector<OWLVarDecl> &varDecls)
  {
    std::vector<OWLVarDecl> result;
    for (auto vd : varDecls) {
      OWLVarDecl copy_vd = vd;
      copy_vd.name = strdup(vd.name);
      result.push_back(copy_vd);
    }
    return result;
  }
  
  SBTObjectType::SBTObjectType(Context *const context,
                               ObjectRegistry &registry,
                               size_t varStructSize,
                               const std::vector<OWLVarDecl> &varDecls)
    : RegisteredObject(context,registry),
      varStructSize(varStructSize),
      varDecls(copyVarDecls(varDecls))
  {
    for (auto &var : varDecls)
      assert(var.name != nullptr);
    /* TODO: at least in debug mode, do some 'duplicate variable
       name' and 'overlap of variables' checks etc */
  }

    /*! clean up; in particular, frees the vardecls */
  SBTObjectType::~SBTObjectType()
  {
    for (auto &vd : varDecls) {
      if (vd.name) free((void *)vd.name);
    }
  }
    
  int SBTObjectType::getVariableIdx(const std::string &varName)
  {
    for (int i=0;i<(int)varDecls.size();i++) {
      assert(varDecls[i].name);
      if (!strcmp(varName.c_str(),varDecls[i].name))
        return i;
    }
    return -1;
  }

  bool SBTObjectType::hasVariable(const std::string &varName)
  {
    return getVariableIdx(varName) >= 0;
  }

  /*! create one instance each of a given type's variables */
  std::vector<Variable::SP> SBTObjectType::instantiateVariables()
  {
    std::vector<Variable::SP> variables(varDecls.size());
    for (size_t i=0;i<varDecls.size();i++) {
      variables[i] = Variable::createInstanceOf(&varDecls[i]);
      assert(variables[i]);
    }
    return variables;
  }

  /*! pretty-typecast into derived classes */
  std::string SBTObjectType::toString() const
  {
    return "SBTObjectType";
  }
  
  // ------------------------------------------------------------------
  // SBTObjectBase
  // ------------------------------------------------------------------
  
  /*! create a new SBTOBject with this type descriptor, and register
    it in that registry */
  SBTObjectBase::SBTObjectBase(Context *const context,
                               ObjectRegistry &registry,
                               std::shared_ptr<SBTObjectType> type)
    : RegisteredObject(context,registry),
      type(type),
      variables(type->instantiateVariables())
  {}

  /*! this function is arguably the heart of the owl variable layer:
    given an SBT Object's set of variables, create the SBT entry
    that writes the given variables' values into the specified
    format, prorperly translating per-device data (buffers,
    traversable) while doing so (also works for launch params, even
    though those, strictly speaking, are not part of the SBT)*/
  void SBTObjectBase::writeVariables(uint8_t *sbtEntryBase,
                                     const DeviceContext::SP &device) const
  {
    for (auto var : variables) {
      auto decl = var->varDecl;
      var->writeToSBT(sbtEntryBase + decl->offset,device);
    }
  }
  
} // ::owl
