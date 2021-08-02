#define HLSL
#include "RaytracingHlslCompat.h"

Texture2D<float4> Albedo[NUM_TEXTURES] : register(t3);
sampler g_trilinear : register(s0);

float4 main(float2 uv : TEX0) : SV_Target0
{
    float4 tex = Albedo[0].Sample(g_trilinear,uv);
	return tex;//float4(uv,0,1);
}