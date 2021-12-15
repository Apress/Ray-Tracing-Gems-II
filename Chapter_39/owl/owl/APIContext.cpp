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

#include "APIContext.h"
#include "APIHandle.h"

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
  
  void APIContext::forget(APIHandle *object)
  {
    std::lock_guard<std::mutex> lock(monitor);
    assert(object);
    auto it = activeHandles.find(object);
    assert(it != activeHandles.end());
    activeHandles.erase(it);
  }

  void APIContext::releaseAll()
  {
    LOG("#owl: context is dying; number of API handles (other than context itself) "
        << "that have not yet been released (incl this context): "
        << (activeHandles.size()));
    for (auto handle : activeHandles)
      LOG(" - " + handle->toString());

    // create one reference that won't get removed when removing all API handles (caller should actually have one, but just in case)
    std::shared_ptr<APIContext> self = shared_from_this()->as<APIContext>();
    std::vector<APIHandle*> handlesToFree;
    for (auto &it : activeHandles) {
        if (it && it->object //&& it->object.get() != this
            ) {
            it->object = {};
            it->context = {};
            handlesToFree.push_back(it);
        }
    }
    activeHandles.clear();
    for (auto handle : handlesToFree)
      delete handle;
  }
  
  void APIContext::track(APIHandle *object)
  {
    assert(object);

    std::lock_guard<std::mutex> lock(monitor);
    
    auto it = activeHandles.find(object);
    assert(it == activeHandles.end());
    activeHandles.insert(object);
  }

  APIHandle *APIContext::createHandle(Object::SP object)
  {
    assert(object);
    APIHandle *handle = new APIHandle(object,this);
    track(handle);
    return handle;
  }

} // ::owl  
