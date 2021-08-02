#ifndef SHADER_COMPAT_H
#define SHADER_COMPAT_H

#include "Payloads.h"
#include "AttribStructs.h"
#include "RootArguments.h"

// PERFORMANCE TIP: Set max recursion depth as low as needed
// as drivers may apply optimization strategies for low recursion depths.
#define MAX_RAY_RECURSION_DEPTH 3    // ~ primary rays + reflections + shadow rays from reflected geometry.

// Ray types traced in this sample.
namespace RayType {
    enum Enum {
        Radiance = 0,   // ~ Primary, reflected camera/view rays calculating color for each hit.
        Shadow,         // ~ Shadow/visibility rays, only testing for occlusion
        Count
    };
}

namespace TraceRayParameters
{
    static const uint InstanceMask = ~0;   // Everything is visible.
    namespace HitGroup {
        static const uint Offset[RayType::Count] =
        {
            0, // Radiance ray
            1  // Shadow ray
        };
        static const uint GeometryStride = RayType::Count;
    }
    namespace MissShader {
        static const uint Offset[RayType::Count] =
        {
            0, // Radiance ray
            1  // Shadow ray
        };
    }
}

static const float4 BackgroundColor = float4(0.6f, 0.7f, 0.9f, 1.0f);//float4(0.8f, 0.9f, 1.0f, 1.0f);
static const float InShadowRadiance = 0.35f;

#endif