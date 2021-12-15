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

#include "APIContext.h"
#include "APIHandle.h"

namespace owl {

  APIHandle::APIHandle(Object::SP object, APIContext *_context)
  {
    assert(object);
    assert(_context);
    this->object  = object;
    this->context = std::dynamic_pointer_cast<APIContext>
      (_context->shared_from_this());
    assert(this->object);
    assert(this->context);
  }

  APIHandle::~APIHandle()
  {
    // iw: every active handle _should_ have a context, but if context
    // itself is killing off remaining handles this pointer may
    // already be null.
    if (context) context->forget(this);
    object  = nullptr;
    context = nullptr;
  }

} // ::owl  
