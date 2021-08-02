#ifndef ROOT_COMPONENTS_H
#define ROOT_COMPONENTS_H

// Root components are the components of Root arguments.

// Attributes per primitive type.
struct PrimitiveConstantBuffer
{
	float4 albedo;
	float reflectanceCoef;
	float diffuseCoef;
	float specularCoef;
	float specularPower;
	float stepScale;                      // Step scale for ray marching of signed distance primitives. 
										  // - Some object transformations don't preserve the distances and 
										  //   thus require shorter steps.
	float3 padding;
};

// Attributes per primitive instance.
struct PrimitiveInstanceConstantBuffer
{
	uint instanceIndex;
	uint primitiveType; // Procedural primitive type
};

struct SceneConstantBuffer
{
	float4x4 projectionToWorld;
	float4 cameraPosition;
	float4 lightPosition;
	float4 lightAmbientColor;
	float4 lightDiffuseColor;
	float    reflectance;
	float    elapsedTime;                 // Elapsed application time.
	int		 debugFlag;
};

// Dynamic attributes per primitive instance.
struct InstanceBuffer
{
	float4x4 localSpaceToBottomLevelAS;   // Matrix from local primitive space to bottom-level object space.
	float4x4 bottomLevelASToLocalSpace;   // Matrix from bottom-level object space to local primitive space.
};

#endif