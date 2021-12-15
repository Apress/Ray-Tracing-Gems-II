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

#include "owl/Context.h"
#include <mutex>

namespace owl {

  struct APIHandle;
  
  struct APIContext : public Context {
    typedef std::shared_ptr<APIContext> SP;

    APIContext(int32_t *requestedDeviceIDs,
               int      numRequestedDevices)
      : Context(requestedDeviceIDs,
                numRequestedDevices)
    {}
    
    /*! pretty-printer, for printf-debugging */
    virtual std::string toString() const override { return "owl::APIContext"; }

    APIHandle *createHandle(Object::SP object);

    void track(APIHandle *object);
    
    void forget(APIHandle *object);

    /*! delete - and thereby, release - all handles that we still
      own. */
    void releaseAll();

    std::set<APIHandle *> activeHandles;    
    std::mutex monitor;
  };
  
} // ::owl  
