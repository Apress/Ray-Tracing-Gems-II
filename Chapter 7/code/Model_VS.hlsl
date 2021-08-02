#define HLSL
#include "RaytracingHlslCompat.h"

ConstantBuffer<SceneConstantBuffer> g_sceneCB : register(b0);
StructuredBuffer<Vertex> Vertices : register(t2);

struct VS_OUT
{
	float2 uv : TEX0;
	float4 pos : SV_Position;
};

VS_OUT main(uint vertexIdx : SV_VertexID)
{
	Vertex vtx = Vertices[vertexIdx];

	VS_OUT Out;
	Out.uv = vtx.uv;
	Out.pos = mul(float4(vtx.position,1),g_sceneCB.worldToProjection);
	return Out;
}