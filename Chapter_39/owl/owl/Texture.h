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

#pragma once

#include "RegisteredObject.h"

namespace owl {

  struct Texture : public RegisteredObject
  {
    typedef std::shared_ptr<Texture> SP;
    
    Texture(Context *const context,
            vec2i                size,
            uint32_t             linePitchInBytes,
            OWLTexelFormat       texelFormat,
            OWLTextureFilterMode filterMode,
            OWLTextureAddressMode addressMode,
            OWLTextureColorSpace colorSpace,
            const void          *texels
            );
    
    /*! destructor - free device data, de-regsiter, and destruct */
    virtual ~Texture();
    
    std::string toString() const override { return "Texture"; }

    /* return the cuda texture object corresponding to the specified 
       device ID*/
    cudaTextureObject_t getObject(int deviceID);

    
    /*! destroy whatever resources this texture's ll-layer handle this
        may refer to; this will not destruct the current object
        itself, but should already release all its references */
    void destroy();

    /*! one entry per device */
    std::vector<cudaTextureObject_t> textureObjects;
    std::vector<cudaArray_t>         textureArrays;
    
    vec2i                size;
    uint32_t             linePitchInBytes;
    OWLTexelFormat       texelFormat;
    OWLTextureFilterMode filterMode;
  };

} // ::owl
