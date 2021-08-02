#ifndef ROOT_ARGUMENTS_H
#define ROOT_ARGUMENTS_H

#include "RootComponents.h"

// A root argument is a structure that constains all substructures that compose a root signature.

struct TriangleRootArguments {
    PrimitiveConstantBuffer materialCb;
};

struct ProceduralRootArguments {
    PrimitiveConstantBuffer materialCb;
    PrimitiveInstanceConstantBuffer aabbCB;
};

#endif