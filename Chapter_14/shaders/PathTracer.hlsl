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

// This is a code sample accompanying "The Reference Path Tracer" chapter in Ray Tracing Gems 2
// v1.0, April 2021

#include "shared.h"
#include "brdf.h"

// -------------------------------------------------------------------------
//    Structures
// -------------------------------------------------------------------------

struct Attributes
{
	float2 uv;
};

struct VertexAttributes
{
	float3 position;
	float3 shadingNormal;
	float3 geometryNormal;
	float2 uv;
};

// -------------------------------------------------------------------------
//    Resources
// -------------------------------------------------------------------------

// Constant buffer with data needed for path tracing
cbuffer RaytracingDataCB : register(b0)
{
	RaytracingData gData;
}

// Output buffer with accumulated and tonemapped image
RWTexture2D<float4> RTOutput						: register(u0);
RWTexture2D<float4> accumulationBuffer				: register(u1);

// TLAS of our scene
RaytracingAccelerationStructure sceneBVH			: register(t0);

// Bindless materials, geometry, and texture buffers (for all the scene geometry)
StructuredBuffer<MaterialData> materials			: register(t1);
ByteAddressBuffer indices[MAX_INSTANCES_COUNT]		: register(t0, space1);
ByteAddressBuffer vertices[MAX_INSTANCES_COUNT]		: register(t0, space2);
Texture2D<float4> textures[MAX_TEXTURES_COUNT]		: register(t0, space3);

// Texture Sampler
SamplerState linearSampler							: register(s0);

// -------------------------------------------------------------------------
//    Defines
// -------------------------------------------------------------------------

#define FLT_MAX 3.402823466e+38F

// Defines after how many bounces will be the Russian Roulette applied
#define MIN_BOUNCES 3

// Switches between two RNGs
#define USE_PCG 1

// Number of candidates used for resampling of analytical lights
#define RIS_CANDIDATES_LIGHTS 8

// Enable this to cast shadow rays for each candidate during resampling. This is expensive but increases quality
#define SHADOW_RAY_IN_RIS 0

 // -------------------------------------------------------------------------
 //    RNG
 // -------------------------------------------------------------------------

#if USE_PCG
	#define RngStateType uint4
#else
	#define RngStateType uint
#endif

// PCG random numbers generator
// Source: "Hash Functions for GPU Rendering" by Jarzynski & Olano
uint4 pcg4d(uint4 v)
{
	v = v * 1664525u + 1013904223u;

	v.x += v.y * v.w; 
	v.y += v.z * v.x; 
	v.z += v.x * v.y; 
	v.w += v.y * v.z;

	v = v ^ (v >> 16u);

	v.x += v.y * v.w; 
	v.y += v.z * v.x; 
	v.z += v.x * v.y; 
	v.w += v.y * v.z;

	return v;
}

// 32-bit Xorshift random number generator
uint xorshift(inout uint rngState)
{
	rngState ^= rngState << 13;
	rngState ^= rngState >> 17;
	rngState ^= rngState << 5;
	return rngState;
}

// Jenkins's "one at a time" hash function
uint jenkinsHash(uint x) {
	x += x << 10;
	x ^= x >> 6;
	x += x << 3;
	x ^= x >> 11;
	x += x << 15;
	return x;
}

// Converts unsigned integer into float int range <0; 1) by using 23 most significant bits for mantissa
float uintToFloat(uint x) {
	return asfloat(0x3f800000 | (x >> 9)) - 1.0f;
}

#if USE_PCG

// Initialize RNG for given pixel, and frame number (PCG version)
RngStateType initRNG(uint2 pixelCoords, uint2 resolution, uint frameNumber) {
	return RngStateType(pixelCoords.xy, frameNumber, 0); //< Seed for PCG uses a sequential sample number in 4th channel, which increments on every RNG call and starts from 0
}

// Return random float in <0; 1) range  (PCG version)
float rand(inout RngStateType rngState) {
	rngState.w++; //< Increment sample index
	return uintToFloat(pcg4d(rngState).x);
}

#else

// Initialize RNG for given pixel, and frame number (Xorshift-based version)
RngStateType initRNG(uint2 pixelCoords, uint2 resolution, uint frameNumber) {
	RngStateType seed = dot(pixelCoords, uint2(1, resolution.x)) ^ jenkinsHash(frameNumber);
	return jenkinsHash(seed);
}

// Return random float in <0; 1) range (Xorshift-based version)
float rand(inout RngStateType rngState) {
	return uintToFloat(xorshift(rngState));
}

#endif

// Maps integers to colors using the hash function (generates pseudo-random colors)
float3 hashAndColor(int i) {
	uint hash = jenkinsHash(i);
	float r = ((hash >> 0) & 0xFF) / 255.0f;
	float g = ((hash >> 8) & 0xFF) / 255.0f;
	float b = ((hash >> 16) & 0xFF) / 255.0f;
	return float3(r, g, b);
}

// -------------------------------------------------------------------------
//    Utilities
// -------------------------------------------------------------------------

// Clever offset_ray function from Ray Tracing Gems chapter 6
// Offsets the ray origin from current position p, along normal n (which must be geometric normal)
// so that no self-intersection can occur.
float3 offsetRay(const float3 p, const float3 n)
{
	static const float origin = 1.0f / 32.0f;
	static const float float_scale = 1.0f / 65536.0f;
	static const float int_scale = 256.0f;

	int3 of_i = int3(int_scale * n.x, int_scale * n.y, int_scale * n.z);

	float3 p_i = float3(
		asfloat(asint(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
		asfloat(asint(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
		asfloat(asint(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

	return float3(abs(p.x) < origin ? p.x + float_scale * n.x : p_i.x,
		abs(p.y) < origin ? p.y + float_scale * n.y : p_i.y,
		abs(p.z) < origin ? p.z + float_scale * n.z : p_i.z);
}

// Calculates probability of selecting BRDF (specular or diffuse) using the approximate Fresnel term
float getBrdfProbability(MaterialProperties material, float3 V, float3 shadingNormal) {
	
	// Evaluate Fresnel term using the shading normal
	// Note: we use the shading normal instead of the microfacet normal (half-vector) for Fresnel term here. That's suboptimal for rough surfaces at grazing angles, but half-vector is yet unknown at this point
	float specularF0 = luminance(baseColorToSpecularF0(material.baseColor, material.metalness));
	float diffuseReflectance = luminance(baseColorToDiffuseReflectance(material.baseColor, material.metalness));
	float Fresnel = saturate(luminance(evalFresnel(specularF0, shadowedF90(specularF0), max(0.0f, dot(V, shadingNormal)))));

	// Approximate relative contribution of BRDFs using the Fresnel term
	float specular = Fresnel;
	float diffuse = diffuseReflectance * (1.0f - Fresnel); //< If diffuse term is weighted by Fresnel, apply it here as well

	// Return probability of selecting specular BRDF over diffuse BRDF
	float p = (specular / max(0.0001f, (specular + diffuse)));

	// Clamp probability to avoid undersampling of less prominent BRDF
	return clamp(p, 0.1f, 0.9f);
}

// Helpers to convert between linear and sRGB color spaces
float3 linearToSrgb(float3 linearColor)
{
	return float3(linearToSrgb(linearColor.x), linearToSrgb(linearColor.y), linearToSrgb(linearColor.z));
}

float3 srgbToLinear(float3 srgbColor)
{
	return float3(srgbToLinear(srgbColor.x), srgbToLinear(srgbColor.y), srgbToLinear(srgbColor.z));
}

// Helpers for octahedron encoding of normals
float2 octWrap(float2 v)
{
	return float2((1.0f - abs(v.y)) * (v.x >= 0.0f ? 1.0f : -1.0f), (1.0f - abs(v.x)) * (v.y >= 0.0f ? 1.0f : -1.0f));
}

float2 encodeNormalOctahedron(float3 n)
{
	float2 p = float2(n.x, n.y) * (1.0f / (abs(n.x) + abs(n.y) + abs(n.z)));
	p = (n.z < 0.0f) ? octWrap(p) : p;
	return p;
}

float3 decodeNormalOctahedron(float2 p)
{
	float3 n = float3(p.x, p.y, 1.0f - abs(p.x) - abs(p.y));
	float2 tmp = (n.z < 0.0f) ? octWrap(float2(n.x, n.y)) : float2(n.x, n.y);
	n.x = tmp.x;
	n.y = tmp.y;
	return normalize(n);
}

float4 encodeNormals(float3 geometryNormal, float3 shadingNormal) {
	return float4(encodeNormalOctahedron(geometryNormal), encodeNormalOctahedron(shadingNormal));
}

void decodeNormals(float4 encodedNormals, out float3 geometryNormal, out float3 shadingNormal) {
	geometryNormal = decodeNormalOctahedron(encodedNormals.xy);
	shadingNormal = decodeNormalOctahedron(encodedNormals.zw);
}

// -------------------------------------------------------------------------
//    Lights & Shadows
// -------------------------------------------------------------------------

// Returns intensity of given light at specified distance
float3 getLightIntensityAtPoint(Light light, float distance) {
	if (light.type == POINT_LIGHT) {
		
#if 0
		// This is version with simple attenuation by inverse square root of distance
		return light.intensity / (distance * distance); 
#else
		// Cem Yuksel's improved attenuation avoiding singularity at distance=0
		// Source: http://www.cemyuksel.com/research/pointlightattenuation/
		const float radius = 0.5f; //< We hardcode radius at 0.5, but this should be a light parameter
		const float radiusSquared = radius * radius;
		const float distanceSquared = distance * distance;
		const float attenuation = 2.0f / (distanceSquared + radiusSquared + distance * sqrt(distanceSquared + radiusSquared));

		return light.intensity * attenuation;
#endif

	} else if (light.type == DIRECTIONAL_LIGHT) {
		return light.intensity;
	} else {
		return float3(1.0f, 0.0f, 1.0f);
	}
}

// Decodes light vector and distance from Light structure based on the light type
void getLightData(Light light, float3 hitPosition, out float3 lightVector, out float lightDistance) {
	if (light.type == POINT_LIGHT) {
		lightVector = light.position - hitPosition;
		lightDistance = length(lightVector);
	} else if (light.type == DIRECTIONAL_LIGHT) {
		lightVector = light.position; //< We use position field to store direction for directional light
		lightDistance = FLT_MAX;
	} else {
		lightDistance = FLT_MAX;
		lightVector = float3(0.0f, 1.0f, 0.0f);
	}
}

// Casts a shadow ray and returns true if light is unoccluded
// Note that we use dedicated hit group with simpler shaders for shadow rays
bool castShadowRay(float3 hitPosition, float3 surfaceNormal, float3 directionToLight, float TMax)
{
	RayDesc ray;
	ray.Origin = offsetRay(hitPosition, surfaceNormal);
	ray.Direction = directionToLight;
	ray.TMin = 0.0f;
	ray.TMax = TMax;

	ShadowHitInfo payload;
	payload.hasHit = true; //< Initialize hit flag to true, it will be set to false on a miss

	// Trace the ray
	TraceRay(
		sceneBVH,
		RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
		0xFF,
		SHADOW_RAY_INDEX,
		0,
		SHADOW_RAY_INDEX,
		ray,
		payload);

	return !payload.hasHit;
}

// Samples a random light from the pool of all lights using simplest uniform distirbution
bool sampleLightUniform(inout RngStateType rngState, float3 hitPosition, float3 surfaceNormal, out Light light, out float lightSampleWeight) {

	if (gData.lightCount == 0) return false;

	uint randomLightIndex = min(gData.lightCount - 1, uint(rand(rngState) * gData.lightCount));
	light = gData.lights[randomLightIndex];

	// PDF of uniform distribution is (1/light count). Reciprocal of that PDF (simply a light count) is a weight of this sample
	lightSampleWeight = float(gData.lightCount);

	return true;
}

// Samples a random light from the pool of all lights using RIS (Resampled Importance Sampling)
bool sampleLightRIS(inout RngStateType rngState, float3 hitPosition, float3 surfaceNormal, out Light selectedSample, out float lightSampleWeight) {
	
	if (gData.lightCount == 0) return false;

	selectedSample = (Light) 0;
	float totalWeights = 0.0f;
	float samplePdfG = 0.0f;

	for (int i = 0; i < RIS_CANDIDATES_LIGHTS; i++) {

		float candidateWeight;
		Light candidate;
		if (sampleLightUniform(rngState, hitPosition, surfaceNormal, candidate, candidateWeight)) {

			float3	lightVector;
			float lightDistance;
			getLightData(candidate, hitPosition, lightVector, lightDistance);

			// Ignore backfacing light
			float3 L = normalize(lightVector);
			if (dot(surfaceNormal, L) < 0.00001f) continue;

#if SHADOW_RAY_IN_RIS
			// Casting a shadow ray for all candidates here is expensive, but can significantly decrease noise
			if (!castShadowRay(hitPosition, surfaceNormal, L, lightDistance)) continue;
#endif

			float candidatePdfG = luminance(getLightIntensityAtPoint(candidate, length(lightVector)));
			const float candidateRISWeight = candidatePdfG * candidateWeight;

			totalWeights += candidateRISWeight;
			if (rand(rngState) < (candidateRISWeight / totalWeights)) {
				selectedSample = candidate;
				samplePdfG = candidatePdfG;
			}
		}
	}

	if (totalWeights == 0.0f) {
		return false;
	} else {
		lightSampleWeight = (totalWeights / float(RIS_CANDIDATES_LIGHTS)) / samplePdfG;
		return true;
	}
}

// -------------------------------------------------------------------------
//    Materials
// -------------------------------------------------------------------------

// Helper to read vertex indices of the triangle from index buffer
uint3 GetIndices(uint geometryID, uint triangleIndex)
{
	uint baseIndex = (triangleIndex * 3);
	int address = (baseIndex * 4);
	return indices[geometryID].Load3(address);
}

// Helper to interpolate vertex attributes at hit point from triangle vertices
VertexAttributes GetVertexAttributes(uint geometryID, uint triangleIndex, float3 barycentrics)
{
	// Get the triangle indices
	uint3 indices = GetIndices(geometryID, triangleIndex);
	VertexAttributes v = (VertexAttributes)0;
	float3 triangleVertices[3];

	// Interpolate the vertex attributes
	for (uint i = 0; i < 3; i++)
	{
		int address = (indices[i] * 12) * 4;

		// Load and interpolate position and transform it to world space
		triangleVertices[i] = mul(ObjectToWorld3x4(), float4(asfloat(vertices[geometryID].Load3(address)), 1.0f)).xyz;
		v.position += triangleVertices[i] * barycentrics[i];
		address += 12;

		// Load and interpolate normal
		v.shadingNormal += asfloat(vertices[geometryID].Load3(address)) * barycentrics[i];
		address += 12;

		// Load and interpolate tangent
		address += 12;

		// Load bitangent direction
		address += 4;

		// Load and interpolate texture coordinates
		v.uv += asfloat(vertices[geometryID].Load2(address)) * barycentrics[i];
	}

	// Transform normal from local to world space
	v.shadingNormal = normalize(mul(ObjectToWorld3x4(), float4(v.shadingNormal, 0.0f)).xyz);

	// Calculate geometry normal from triangle vertices positions
	float3 edge20 = triangleVertices[2] - triangleVertices[0];
	float3 edge21 = triangleVertices[2] - triangleVertices[1];
	float3 edge10 = triangleVertices[1] - triangleVertices[0];
	v.geometryNormal = normalize(cross(edge20, edge10));

	return v;
}

// Loads material properties (including textures) for selected material
MaterialProperties loadMaterialProperties(uint materialID, float2 uvs) {
	MaterialProperties result = (MaterialProperties) 0;

	// Read base data
	MaterialData mData = materials[materialID];

	result.baseColor = mData.baseColor;
	result.emissive = mData.emissive;
	result.metalness = mData.metalness;
	result.roughness = mData.roughness;
	result.opacity = mData.opacity;
	
	// Load textures (using mip level 0)
	if (mData.baseColorTexIdx != INVALID_ID) {
		result.baseColor *= textures[mData.baseColorTexIdx].SampleLevel(linearSampler, uvs, 0.0f).rgb;
	}

	if (mData.emissiveTexIdx != INVALID_ID) {
		result.emissive *= textures[mData.emissiveTexIdx].SampleLevel(linearSampler, uvs, 0.0f).rgb;
	}

	if (mData.roughnessMetalnessTexIdx != INVALID_ID) {
		float3 occlusionRoughnessMetalness = textures[mData.roughnessMetalnessTexIdx].SampleLevel(linearSampler, uvs, 0.0f).rgb;
		result.metalness *= occlusionRoughnessMetalness.b;
		result.roughness *= occlusionRoughnessMetalness.g;
	}

	return result;
}

// Performs an opacity test in any hit shader for potential hit. Returns true if hit point is transparent and can be ignored
bool testOpacityAnyHit(Attributes attrib) {

	// Load material at hit point
	uint materialID;
	uint geometryID;
	unpackInstanceID(InstanceID(), materialID, geometryID);

	MaterialData mData = materials[materialID];
	float opacity = mData.opacity;
	
	// Also load the opacity texture if available
	if (mData.baseColorTexIdx != INVALID_ID) {
		float3 barycentrics = float3((1.0f - attrib.uv.x - attrib.uv.y), attrib.uv.x, attrib.uv.y);
		VertexAttributes vertex = GetVertexAttributes(geometryID, PrimitiveIndex(), barycentrics);
		opacity *= textures[mData.baseColorTexIdx].SampleLevel(linearSampler, vertex.uv, 0.0f).a;
	}

	// Decide whether this hit is opaque or not according to chosen alpha testing mode
	if (mData.alphaMode == ALPHA_MODE_MASK) {
		return (opacity < mData.alphaCutoff);
	} else {
		// Alpha blending mode
		float u = 0.5f; // If you want alpha blending, there should be a random u. Semi-transparent things are, however, better rendered using refracted rays with real IoR
		return (opacity < u);
	}
}

// -------------------------------------------------------------------------
//    Camera
// -------------------------------------------------------------------------

// Generates a primary ray for pixel given in NDC space using pinhole camera
RayDesc generatePinholeCameraRay(float2 pixel)
{
	// Setup the ray
	RayDesc ray;
	ray.Origin = gData.view[3].xyz;
	ray.TMin = 0.f;
	ray.TMax = FLT_MAX;

	// Extract the aspect ratio and field of view from the projection matrix
	float aspect = gData.proj[1][1] / gData.proj[0][0];
	float tanHalfFovY = 1.0f / gData.proj[1][1];

	// Compute the ray direction for this pixel
	ray.Direction = normalize(
		(pixel.x * gData.view[0].xyz * tanHalfFovY * aspect) -
		(pixel.y * gData.view[1].xyz * tanHalfFovY) +
			gData.view[2].xyz);

	return ray;
}

// Helper to generate aperture samples of the thin lens model
float2 getApertureSample(inout RngStateType rngState)
{
	// Generate a sample within a circular aperture. Other shapes can be implemented here
	// Using just. xy coordinates of hemisphere sample gives samples within a disk
	return sampleHemisphere(float2(rand(rngState), rand(rngState))).xy;
}

// Generates a primary ray for pixel given in NDC space using thin lens model (with depth of field)
RayDesc generateThinLensCameraRay(float2 pixel, inout RngStateType rngState)
{
	// First find the point in distance at which we want perfect focus 
	RayDesc ray = generatePinholeCameraRay(pixel);
	float3 focalPoint = ray.Origin + ray.Direction * gData.focusDistance;

	// Sample the aperture shape
	float2 apertureSample = getApertureSample(rngState) * gData.apertureSize;

	// Jitter the ray origin within camera plane using aperture sample
	float3 rightVector = gData.view[0].xyz;
	float3 upVector = gData.view[1].xyz;
	ray.Origin = ray.Origin + rightVector * apertureSample.x + upVector * apertureSample.y;

	// Set ray direction from jittered origin towards the focal point
	ray.Direction = normalize(focalPoint - ray.Origin);

	return ray;
}

// Generates primary ray either using pinhole camera (for zero-sized apertures) or thin lens model
RayDesc generatePrimaryRay(float2 posNdcXy, inout RngStateType rngState)
{
	if (gData.apertureSize == 0.0f)
		return generatePinholeCameraRay(posNdcXy);
	else
		return generateThinLensCameraRay(posNdcXy, rngState);
}

// -------------------------------------------------------------------------
//    Sky
// -------------------------------------------------------------------------

float3 loadSkyValue(float3 rayDirection) {

	// Load the sky value for given direction here, e.g. from environment map, procedural sky, etc.
	// Make sure to only account for sun once - either on the skybox or as an analytical light (if sun is included as explicit directional light, it shouldn't be on the skybox)
	return gData.skyIntensity;
}

// -------------------------------------------------------------------------
//    Raytracing shaders
// -------------------------------------------------------------------------

[shader("closesthit")]
void ClosestHit(inout HitInfo payload, Attributes attrib)
{
	// At closest hit, we first load material and geometry ID packed into InstanceID 
	uint materialID;
	uint geometryID;
	unpackInstanceID(InstanceID(), materialID, geometryID);

	// Read hit point properties (position, normal, UVs, ...) from vertex buffer
	float3 barycentrics = float3((1.0f - attrib.uv.x - attrib.uv.y), attrib.uv.x, attrib.uv.y);
	VertexAttributes vertex = GetVertexAttributes(geometryID, PrimitiveIndex(), barycentrics);

	// Encode hit point properties and material ID into payload
	payload.encodedNormals = encodeNormals(vertex.geometryNormal, vertex.shadingNormal);
	payload.hitPosition = vertex.position;
	payload.materialID = materialID;
	payload.uvs = float16_t2(vertex.uv);
}

[shader("anyhit")]
void AnyHit(inout HitInfo payload : SV_RayPayload, Attributes attrib : SV_IntersectionAttributes)
{
	// At any hit, we test opacity and discard the hit if it's transparent
	if (testOpacityAnyHit(attrib)) IgnoreHit();
}

[shader("anyhit")]
void AnyHitShadow(inout ShadowHitInfo payload : SV_RayPayload, Attributes attrib : SV_IntersectionAttributes)
{
	// At any hit for shadow rays, we test opacity and discard the hit if it's transparent
	// But also end the search if we encounter any opaque surface
	if (testOpacityAnyHit(attrib))
		IgnoreHit();
	else
		AcceptHitAndEndSearch();
}

[shader("miss")]
void Miss(inout HitInfo payload)
{
	// We indicate miss by storing invalid material ID in the payload
    payload.materialID = INVALID_ID;
}

[shader("miss")]
void MissShadow(inout ShadowHitInfo payload)
{
	// For shadow rays, miss means that light is unoccluded. Note that there's no closest hit shader for shadows. That's because we don't need to know details of the closest hit, just whether any opaque hit occured or not.
	payload.hasHit = false;
}

[shader("raygeneration")]
void RayGen()
{
	uint2 LaunchIndex = DispatchRaysIndex().xy;
	uint2 LaunchDimensions = DispatchRaysDimensions().xy;

	// Initialize random numbers generator
	RngStateType rngState = initRNG(LaunchIndex, LaunchDimensions, gData.frameNumber);

	// Figure out pixel coordinates being raytraced
	float2 pixel = float2(DispatchRaysIndex().xy);
	const float2 resolution = float2(DispatchRaysDimensions().xy);

	// Antialiasing (optional)
	if (gData.enableAntiAliasing) {
		// Add a random offset to the pixel 's screen coordinates .
		float2 offset = float2(rand(rngState), rand(rngState));
		pixel += lerp(-0.5.xx, 0.5.xx, offset);
	}

	pixel = (((pixel + 0.5f) / resolution) * 2.0f - 1.0f);

	// Initialize ray to the primary ray
	RayDesc ray = generatePrimaryRay(pixel, rngState);
	HitInfo payload = (HitInfo) 0;

	// Initialize path tracing data
	float3 radiance = float3(0.0f, 0.0f, 0.0f);
	float3 throughput = float3(1.0f, 1.0f, 1.0f);
	
	// Start the ray tracing loop
	for (int bounce = 0; bounce < gData.maxBounces; bounce++) {

		// Trace the ray
		TraceRay(
			sceneBVH,
			RAY_FLAG_NONE,
			0xFF,
			STANDARD_RAY_INDEX,
			0,
			STANDARD_RAY_INDEX,
			ray,
			payload);

		// On a miss, load the sky value and break out of the ray tracing loop
		if (!payload.hasHit()) {
			radiance += throughput * loadSkyValue(ray.Direction);
			break;
		}

		// Decode normals and flip them towards the incident ray direction (needed for backfacing triangles)
		float3 geometryNormal;
		float3 shadingNormal;
		decodeNormals(payload.encodedNormals, geometryNormal, shadingNormal);

		float3 V = -ray.Direction;
		if (dot(geometryNormal, V) < 0.0f) geometryNormal = -geometryNormal;
		if (dot(geometryNormal, shadingNormal) < 0.0f) shadingNormal = -shadingNormal;

		// Load material properties at the hit point
		MaterialProperties material = loadMaterialProperties(payload.materialID, payload.uvs);

		// Account for emissive surfaces
		radiance += throughput * material.emissive;

		// Evaluate direct light (next event estimation), start by sampling one light 
		Light light;
		float lightWeight;
		if (sampleLightRIS(rngState, payload.hitPosition, geometryNormal, light, lightWeight)) {

			// Prepare data needed to evaluate the light
			float3 lightVector;
			float lightDistance;
			getLightData(light, payload.hitPosition, lightVector, lightDistance);
			float3 L = normalize(lightVector);

			// Cast shadow ray towards the selected light
			if (SHADOW_RAY_IN_RIS || castShadowRay(payload.hitPosition, geometryNormal, L, lightDistance))
			{
				// If light is not in shadow, evaluate BRDF and accumulate its contribution into radiance
				radiance += throughput * evalCombinedBRDF(shadingNormal, L, V, material) * (getLightIntensityAtPoint(light, lightDistance) * lightWeight);
			}
		}

		// Terminate loop early on last bounce (we don't need to sample BRDF)
		if (bounce == gData.maxBounces - 1) break;

		// Russian Roulette
		if (bounce > MIN_BOUNCES) {
			float rrProbability = min(0.95f, luminance(throughput));
			if (rrProbability < rand(rngState)) break;
			else throughput /= rrProbability;
		}

		// Sample BRDF to generate the next ray
		// First, figure out whether to sample diffuse or specular BRDF
		int brdfType;
		if (material.metalness == 1.0f && material.roughness == 0.0f) {
			// Fast path for mirrors
			brdfType = SPECULAR_TYPE;
		} else {

			// Decide whether to sample diffuse or specular BRDF (based on Fresnel term)
			float brdfProbability = getBrdfProbability(material, V, shadingNormal);

			if (rand(rngState) < brdfProbability) {
				brdfType = SPECULAR_TYPE;
				throughput /= brdfProbability;
			} else {
				brdfType = DIFFUSE_TYPE;
				throughput /= (1.0f - brdfProbability);
			}
		}

		// Run importance sampling of selected BRDF to generate reflecting ray direction
		float3 brdfWeight;
		float2 u = float2(rand(rngState), rand(rngState));
		if (!evalIndirectCombinedBRDF(u, shadingNormal, geometryNormal, V, material, brdfType, ray.Direction, brdfWeight)) {
			break; // Ray was eaten by the surface :(
		}

		// Account for surface properties using the BRDF "weight"
		throughput *= brdfWeight;

		// Offset a new ray origin from the hitpoint to prevent self-intersections
		ray.Origin = offsetRay(payload.hitPosition, geometryNormal);
	}

	// Temporal accumulation
	float3 previousColor = accumulationBuffer[LaunchIndex].rgb;
	float3 accumulatedColor = radiance;
	if (gData.enableAccumulation) accumulatedColor = previousColor + radiance;
	accumulationBuffer[LaunchIndex] = float4(accumulatedColor, 1.0f);

	// Copy accumulated result into output buffer (this one is only RGB8, so precision is not good enough for accumulation)
	// Note: Conversion from linear to sRGB here is not be necessary if conversion is applied later in the pipeline
	RTOutput[LaunchIndex] = float4(linearToSrgb(accumulatedColor / gData.accumulatedFrames * gData.exposureAdjustment), 1.0f);
}
