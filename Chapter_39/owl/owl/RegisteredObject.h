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

#include "Object.h"

namespace owl {

  struct ObjectRegistry;

  /*! a object that is managed/kept track of in a registry that
      assigns linear IDs (so that, for example, the SBT builder can
      easily iterate over all geometries, all geometry types, etc. The
      sole job of this class is to properly register and unregister
      itself in the given registry when it gets created/destroyed */
  struct RegisteredObject : public ContextObject {

    RegisteredObject(Context *const context,
                     ObjectRegistry &registry);
    ~RegisteredObject();

    /*! the ID we're registered by - should only ever get set to any
        useful value in the constructor, and get set to -1 when the
        object is removed from this registry */
    int ID;
    
    /*! the registry (int he context) that we're registered in */
    ObjectRegistry &registry;
  };

} // ::owl

