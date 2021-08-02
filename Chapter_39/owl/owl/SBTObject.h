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

#pragma once

#include "RegisteredObject.h"
#include "Variable.h"

namespace owl {

  /*! base class for describing the 'type' (eg, set of named
      variabels, progrma name, etc) of anything that can store
      variables, and that will eithe be written into the SBT, or other
      device representations. Object *types* store the *description*
      of objects (eg, which variables they have, and how those are
      layouted in device memory, while actual *objects* (basically,
      the equivalent of C++ instances of tihs type) then store the
      *values* for the given type */
  struct SBTObjectType : public RegisteredObject
  {
    typedef std::shared_ptr<SBTObjectType> SP;

    SBTObjectType(Context *const context,
                  ObjectRegistry &registry,
                  size_t varStructSize,
                  const std::vector<OWLVarDecl> &varDecls);

    /*! clean up; in particular, frees the vardecls */
    virtual ~SBTObjectType();
    
    /*! find index of variable with given name, or -1 if not exists */
    int getVariableIdx(const std::string &varName);

    /*! check if we have this variable (to error out if app tries to
        set variable that we do not own */
    bool hasVariable(const std::string &varName);

    /*! pretty-printer, for printf-debugging */
    std::string toString() const override;

    /*! add a new variable of given type and name, and with given byte
        offset within the device representation of this class */
    void declareVariable(const std::string &varName,
                         OWLDataType type,
                         size_t offset);

    /*! create a set of actual instances of the variables of this
        type, to be attached to an actual object of this type, so this
        object can then store variable values */
    std::vector<Variable::SP> instantiateVariables();

    /*! the total size of the variables struct */
    const size_t         varStructSize;

    /*! the high-level semantic description of variables in the
        variables struct */
    const std::vector<OWLVarDecl> varDecls;
  };



  /*! abstract base classs for any object that can store variables and
      write itself into a device-side shader binding table (ie, raygen
      programs, closest hit programs, etc. Each SBTObjectBase has a
      type that describes its variables, but the actual work for
      subclasses of this type will be done in the subclass. */
  struct SBTObjectBase : public RegisteredObject
  {
    /*! create a new SBTOBject with this type descriptor, and register
        it in that registry */
    SBTObjectBase(Context *const context,
                  ObjectRegistry &registry,
                  std::shared_ptr<SBTObjectType> type);

    /*! returns whether this object has a variable of this name */
    inline bool hasVariable(const std::string &name);
    
    /*! return shared-ptr to this variable - should only be called for
        variables that we actually own */
    inline Variable::SP getVariable(const std::string &name);

    /*! this function is arguably the heart of the owl variable layer:
      given an SBT Object's set of variables, create the SBT entry
      that writes the given variables' values into the specified
      format, prorperly translating per-device data (buffers,
      traversable) while doing so (also works for launch params, even
      though those, strictly speaking, are not part of the SBT)*/
    void writeVariables(uint8_t *sbtEntry,
                        const DeviceContext::SP &device) const;
    
    /*! our own type description, that tells us which variables (of
      which type, etc) we have */
    std::shared_ptr<SBTObjectType> const type;
    
    /*! the actual variable *values* */
    const std::vector<Variable::SP> variables;
  };


  /*! same as a SBTObjectBase (ie, still virtual abstract), but adds
      some type information to help make it easier to differentiate
      between RayGens, MissPorgs, etc */
  template<typename ObjectType>
  struct SBTObject : public SBTObjectBase
  {
    typedef std::shared_ptr<SBTObject> SP;

    /*! create a new SBTOBject with this type descriptor, and register
        it in that registry */
    SBTObject(Context *const context,
              ObjectRegistry &registry,
              std::shared_ptr<ObjectType> type)
      : SBTObjectBase(context,registry,type),
        type(type)
    {}
    
    virtual std::string toString() const { return "SBTObject<"+type->toString()+">"; }
    
    /*! our own type description, that tells us which variables (of
      which type, etc) we have */
    std::shared_ptr<ObjectType> const type;
  };



  // ------------------------------------------------------------------
  // implementation section
  // ------------------------------------------------------------------
  
  /*! returns whether this object has a variable of this name */
  inline bool SBTObjectBase::hasVariable(const std::string &name)
  {
    return type->hasVariable(name);
  }
  
    /*! return shared-ptr to this variable - should only be called for
      variables that we actually own */
  inline Variable::SP SBTObjectBase::getVariable(const std::string &name)
  {
    int varID = type->getVariableIdx(name);
    assert(varID >= 0);
    assert(varID < (int)variables.size());
    Variable::SP var = variables[varID];
    assert(var);
    return var;
  }

} // ::owl

