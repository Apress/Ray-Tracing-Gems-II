#ifndef PAYLOADS_H
#define PAYLOADS_H

// The type of the ray payloads that are passed through the shaders.

struct RayPayload
{
	float4 color;
	uint   recursionDepth;
	float dist;
	int count;
	bool hit;
};

struct ShadowRayPayload
{
	float dist;
	bool hit;
};

#endif