// ======================================================================== //
// Copyright 2019 Ingo Wald                                                 //
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

// The Ray Tracing in One Weekend scene, but with cubes substituted for some
// spheres. This program shows how different geometric types in a single scene
// are handled.

// public owl API
#include <owl/owl.h>
// our device-side data structures
#include "GeomTypes.h"
// external helper stuff for image output
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include <random>

#define LOG(message)                                            \
  std::cout << OWL_TERMINAL_BLUE;                               \
  std::cout << "#owl.sample(main): " << message << std::endl;    \
  std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
  std::cout << OWL_TERMINAL_LIGHT_BLUE;                         \
  std::cout << "#owl.sample(main): " << message << std::endl;    \
  std::cout << OWL_TERMINAL_DEFAULT;

template<typename Lambda>
void runTest(const Lambda &alloc)
{
  int size = 1 + random() % 10000000;
  OWLBuffer buffer = alloc(size);
  owlBufferDestroy(buffer);
  owlBufferRelease(buffer);
}

int main(int ac, char **av)
{
  size_t numRepeats = 1000;
  OWLContext context = owlContextCreate(nullptr,1);

  if (ac == 2)
    numRepeats = std::atol(av[1]);
  
  for (int i=0;i<numRepeats;i++) {
    switch(i%3) {
    case 0:
      runTest([context](size_t n)
              {return owlDeviceBufferCreate(context,OWL_FLOAT,n,nullptr);});
      break;
    case 1:
      runTest([context](size_t n)
              {return owlManagedMemoryBufferCreate(context,OWL_FLOAT,n,nullptr);});
      break;
    case 3:
      runTest([context](size_t n)
              {return owlHostPinnedBufferCreate(context,OWL_FLOAT,n);});
      break;
    };
  }
  
  LOG_OK("seems all went OK; app is done, this should be the last output ...");
}
