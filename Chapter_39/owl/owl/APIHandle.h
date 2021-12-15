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

namespace owl {
  
  struct APIContext;

  /*! class that wraps an internal object-sharedptr to a 64-bit
      'handle' that is accessible through the C-API, and does the
      app-side create/release. Handle are owned by an API context,
      which creates handles and will, upon 'ContextDestroy' release
      all of them (and of course, release them individually if the app
      releases them).

      Note that the app releasing a handle does _not_ mean that the
      object itself will be freed at this time; as the objects are
      themselves internally refcounted by owl.
  */
  struct APIHandle {
    APIHandle(Object::SP object, APIContext *context);
    virtual ~APIHandle();
    template<typename T> inline std::shared_ptr<T> get();
    inline std::shared_ptr<APIContext> getContext() const { return context; }
    inline bool isContext() const
    {
      return ((void*)object.get() == (void*)context.get());
    }
    std::string toString() const
    {
      assert(object);
      return object->toString();
    }
    void clear() { object = nullptr; }//context = nullptr; }
    std::shared_ptr<Object>     object;
    std::shared_ptr<APIContext> context;
  };

  /*! helper functoin that, for a given handle, retrieves a shared-ptr
      to the obejct referenced by this handle, with automatic
      type-cast to the expected type, and error handling if this
      handle does not match the expected type */
  template<typename T> inline std::shared_ptr<T> APIHandle::get()
  {
    assert(object);
    std::shared_ptr<T> asT = std::dynamic_pointer_cast<T>(object);
    if (object && !asT) {
      const std::string objectTypeID = typeid(*object.get()).name();
	
      const std::string tTypeID = typeid(T).name();
      OWL_RAISE("could not convert APIHandle of type "
                + objectTypeID
                + " to object of type "
                + tTypeID);
    }
    assert(asT);
    return asT;
  }
  
} // ::owl  
