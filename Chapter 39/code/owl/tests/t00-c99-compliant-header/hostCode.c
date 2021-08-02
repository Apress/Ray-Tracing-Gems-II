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

/*! \file c99-compliant-header-test.c - unit test for compiling an owl
    program with a C99-only (ie, non-C++, and non-C++-11) compiler */

// public owl node-graph api
#include "owl/owl.h"
#include <stdio.h>

#ifdef __cplusplus
#  error "This is not a C99 compiler"
#endif

#define UNUSED(x) (void)(x)

int main(int ac, char **av)
{
  UNUSED(ac);
  UNUSED(av);
  printf("hello world from C99\n");

  OWLContext owl = owlContextCreate(NULL,0);
  if (owl == NULL) {
    printf("error - could not create owl context\n");
    exit(1);
  }
  
  printf("owl context successfully created\n");
  owlContextDestroy(owl);
  
  exit(0);
}
