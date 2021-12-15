// ======================================================================== //
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

#include <cstdint>
#include <string>
#include <vector>

#include "ogt_vox.h"

// Load from byte buffer
const ogt_vox_scene* loadVoxScene(const uint8_t *voxBuffer, size_t voxBufferLen);

// Load from file
const ogt_vox_scene* loadVoxScene(const char *filename);

// Load multiple files and merge into one scene
const ogt_vox_scene* loadVoxScenes(const std::vector<std::string> &filenames);



