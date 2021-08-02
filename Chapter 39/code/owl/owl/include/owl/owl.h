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

/* since CUDA can actually understand full C++ we'll have the
   host-side API included on both host and device side; this allows the
  user's device programs to include shared haeders that use host-side
  OWL types like OWLContext etc */
#include "owl_host.h"

/* since the device API contains CUDA types such as float4 etc we'll
   include this only for files compiles with nvcc */
#ifdef __CUDACC__
# include "owl_device.h"
#endif

