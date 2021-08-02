/* Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// -------------------------------------------------------------------------
//    C++ compatibility
// -------------------------------------------------------------------------

#if __cplusplus
#pragma once
#define matrix DirectX::XMMATRIX
#define float4 DirectX::XMFLOAT4
#define float3 DirectX::XMFLOAT3
#define float2 DirectX::XMFLOAT2
#define float16_t uint16_t
#define float16_t2 uint32_t
#define uint uint32_t
#define OUT_PARAMETER(X) X&
#else
#define OUT_PARAMETER(X) out X
#endif

// -------------------------------------------------------------------------
//    Stuff shared between C++ and HLSL code
// -------------------------------------------------------------------------

#define MAX_INSTANCES_COUNT 0x4000
#define MAX_MATERIALS_COUNT 0x400
#define MAX_TEXTURES_COUNT 0xFFFF

#define STANDARD_RAY_INDEX 0
#define SHADOW_RAY_INDEX 1

#define POINT_LIGHT 1
#define DIRECTIONAL_LIGHT 2

#define ALPHA_MODE_OPAQUE 0
#define ALPHA_MODE_BLEND 1
#define ALPHA_MODE_MASK 2

#define INVALID_ID -1

// Functions for encoding/decoding material and geometry ID into single integer
inline uint packInstanceID(uint materialID, uint geometryID) {
	return ((geometryID & 0x3FFF) << 10) | (materialID & 0x3FF);
}

inline void unpackInstanceID(uint instanceID, OUT_PARAMETER(uint) materialID, OUT_PARAMETER(uint) geometryID) {
	materialID = instanceID & 0x3FF;
	geometryID = (instanceID >> 10) & 0x3FFF;
}

// Conversion between linear and sRGB color spaces
inline float linearToSrgb(float linearColor)
{
	if (linearColor < 0.0031308f) return linearColor * 12.92f;
	else return 1.055f * float(pow(linearColor, 1.0f / 2.4f)) - 0.055f;
}

inline float srgbToLinear(float srgbColor)
{
	if (srgbColor < 0.04045f) return srgbColor / 12.92f;
	else return float(pow((srgbColor + 0.055f) / 1.055f, 2.4f));
}

struct Light {
	float3 position;
	uint type;
	float3 intensity;
	uint pad;
};

struct HitInfo
{
	float4 encodedNormals;

	float3 hitPosition;
	uint materialID;

	float16_t2 uvs;

	bool hasHit() {
		return materialID != INVALID_ID;
	}
};

struct ShadowHitInfo
{
	bool hasHit;
};

struct MaterialData
{
	float3 baseColor;
	int baseColorTexIdx;

	float3 emissive;
	int emissiveTexIdx;

	float metalness;
	float roughness;
	float opacity;
	int roughnessMetalnessTexIdx;

	int     alphaMode;                  //< 0: Opaque, 1: Blend, 2: Masked
	float   alphaCutoff;
	int     doubleSided;                //< 0: false, 1: true
	int     normalTexIdx;               //< Tangent space XYZ
};

struct RaytracingData
{
	matrix view;
	matrix proj;

	float skyIntensity;
	uint lightCount;
	uint frameNumber;
	uint maxBounces;

	float exposureAdjustment;
	uint accumulatedFrames;
	bool enableAntiAliasing;
	float focusDistance;

	float apertureSize;
	bool enableAccumulation;
	float pad;
	float pad2;

	Light lights[4];
};


#if __cplusplus
#undef matrix
#undef float4
#undef float3
#undef float2
#undef uint
#undef float16_t
#undef float16_t2
#undef OUT_PARAMETER
#endif
