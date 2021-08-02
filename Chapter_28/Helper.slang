/***************************************************************************
 # Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/

// structs -----------

struct BillboardShadingData
{
    float3 posW; // position in world space
    float3 V;   // direction to the eye at shading hit
    float3 N;   // shading normal
    float3 diffuse; // diffuse albedo
    float  opacity; 
};

struct BillboardVertex
{
    float3 posW; // position in world space
    float3 normalW; // shading normal
    float2 texC; // texture coordinates
    float  billboardDepth; // depth ds of the billboard vertex
    float  billboardStrength; // For particles: soft particle opacity factor; For sphericals: Normalized distance to the billboard center (1-(d/r))*(ds / 2r)
#if BILLBOARD_TYPE != BILLBOARD_TYPE_SPHERICAL
    float3 tangentW; // tangent for ray differentials
    float3 bitangentW; // bitangent for ray differentials
#endif
};

// helper function to fetch material data from texture
BillboardShadingData prepareBillboardShadingData<L:ITextureSampler>(BillboardVertex v, uint materialID, MaterialData md, MaterialResources mr, float3 viewDir, L lod)
{
    BillboardShadingData sd;
    float4 baseColor = sampleTexture(mr.baseColor, mr.samplerState, v.texC, md.baseColor, EXTRACT_DIFFUSE_TYPE(md.flags), lod);
    sd.diffuse = baseColor.rgb;
    sd.opacity = baseColor.a;
    sd.posW = v.posW;
    sd.V = viewDir;
    sd.N = v.normalW;
    return sd;
}

// GetTangentSpace() function from the article with option for y-axis alignment
float3x3 GetBillboardTangentSpace(float3 normalDir, bool yAlign) {
	float3x3 planeBase;

	// y axis alignment
	if(yAlign) planeBase[2] = normalize(float3(normalDir.x, 0.0, normalDir.z));
    else planeBase[2] = normalize(normalDir);  

	float3 up = float3(0.0, 1.0, 0.0);
	planeBase[0] = normalize(cross(up, planeBase[2]));
	planeBase[1] = cross(planeBase[2], planeBase[0]);
	return planeBase;
}

// contrast function from soft particles (smooth gradient between 0 and 1)
float getSoftParticleContrast(float x)
{
    const float c = 2.0; // contrast in [1, 5]
    x = saturate(x);
    if(x <= 0.5) return 0.5 * pow(2.0 * x, c);
    return 1.0 - 0.5 * pow(2.0 * (1.0 - x), c);
}

// Returns vertex data for impostors and particle billboards
BillboardVertex getBillboardVertexData(uint aabbOffset, float3 rayOrigin, float3 rayDirection, float rayTCurrent, float rayTMin, float rayTMax)
{
    BillboardVertex v;
    const AABB bbox = gScene.proceduralPrimitiveAABBs[aabbOffset];
    const float2 dim = bbox.extent().xy;
    const float3 C = bbox.center();

    // billboard plane base
    const bool yAlign = BILLBOARD_TYPE == BILLBOARD_TYPE_IMPOSTOR;
    const float3x3 planeBase = GetBillboardTangentSpace(rayOrigin - C, yAlign);

    // fill out important vertex data
    v.posW = rayOrigin + rayDirection * rayTCurrent;
#if BILLBOARD_TYPE != BILLBOARD_TYPE_SPHERICAL
    v.tangentW = planeBase[0];
    v.bitangentW = planeBase[1];
#endif
    v.normalW = planeBase[2];
    v.texC.x = dot(C - v.posW, planeBase[0]) / dim.x + 0.5;
    v.texC.y = dot(C - v.posW, planeBase[1]) / dim.y + 0.5;
    v.billboardStrength = 1.0;
    if(BILLBOARD_TYPE == BILLBOARD_TYPE_PARTICLE && USE_SOFT_PARTICLES)
    {
        float blendScale = 8.0 / dim.x;
        float outFade = getSoftParticleContrast((rayTMax - rayTCurrent) * blendScale);
        float inFade = getSoftParticleContrast((rayTCurrent - rayTMin) * blendScale);
        v.billboardStrength = min(outFade, inFade);
    }

    v.billboardDepth = dim.x - length(C - v.posW) * 2.0; // for soft shadows
    //v.billboardDepth = sqrt(dim.x * dim.x * 0.25 - dot(C - v.posW, C - v.posW)) * 2.0;
    return v;
}

#define BILLBOARD_UV_OFFSET(id) (id * 0.912)

float2 CalcUVFromNormal(float3 n, uint aabbOffset)
{
    float2 uv;
    //uv.x = (n.x == 0.0 ? 0.25 * sign(n.z) : atan2(n.z, n.x) / (2.0 * M_PI)) + 0.5;
    uv.x = atan2(n.z, n.x) / (2.0 * M_PI);
    uv.y = acos(n.y) / M_PI;
    uv.x += BILLBOARD_UV_OFFSET(aabbOffset); // give each billboard a random texture rotation
    return uv;
}

// Returns vertex data for spherical billboards
BillboardVertex getSphericalBillboardVertexData(uint aabbOffset, float3 rayOrigin, float3 rayDirection, float rayTCurrent, float rayTMin, float rayTMax)
{
    BillboardVertex v;

    // reconstruct billboard plane
    const AABB bbox = gScene.proceduralPrimitiveAABBs[aabbOffset];
    const float2 dim = bbox.extent().xy;
    const float3 C = bbox.center();

    // determine depth
    const float r = 0.5 * dim.x;
    const float t = dot(rayDirection, C - rayOrigin); // rayT to closest point inside billboard
    const float l = length(rayOrigin + t * rayDirection - C); // smallest distance
    const float w = sqrt(r * r - l * l);
    const float ds = min(rayTMax, t + w) - rayTCurrent;

    // fill out vertex data
    v.billboardDepth = ds;
    v.billboardStrength = (1.0 - (l / r)) * (ds / dim.x);
    v.posW = rayOrigin + rayDirection * rayTCurrent; // closest entry point
    v.normalW = normalize(rayOrigin + rayDirection * (t - w) - C); // direction to unclipped entry point (for consistent texture coordinates when traversing)
    v.texC = CalcUVFromNormal(v.normalW, aabbOffset);

    return v;
}

// shading data: pos, N, diffuse
float3 evalBillboardMaterial(BillboardShadingData sd, LightData light)
{
    LightSample ls;
    if (light.type == uint(LightType::Directional)) ls = evalDirectionalLight(light, sd.posW);
    else if (light.type == uint(LightType::Point)) ls = evalPointLight(light, sd.posW);
    else return 0.0;

    // volumetric shading?
    if(BILLBOARD_TYPE != BILLBOARD_TYPE_IMPOSTOR) return max(ls.diffuse * sd.diffuse, 0.0);

    // If the light doesn't hit the surface or we are viewing the surface from the back, return
    ls.NdotL = dot(sd.N, ls.L);
    if (ls.NdotL <= 0) return 0.0;

    // Calculate the diffuse term
    float3 color = ls.diffuse * sd.diffuse.rgb * (1 / M_PI) * ls.NdotL;
    return color;
}

// shading function for billboards that iterates though all lights and traces shadow rays for the primary light source
// returns rgb color and transmittance (in alpha channel)
float4 shadeBillboard(BillboardShadingData sd, float billboardDepth, float billboardOffset = 0.0)
{
    float3 curColor = 0.0;
    // forward render all light sources
    const uint lightCount = gScene.getLightCount();
    for(uint i = 0; i < lightCount; ++i)
    {
        LightData l = gScene.getLight(i);
        [branch] if(BILLBOARD_SHADOW_SAMPLES > 1 &&  (i == 0))
        {
            float3 color = evalBillboardMaterial(sd, l);
            // obtain #BILLBOARD_SHADOW_SAMPLES shadow ray samples
            // Note: only for visualization purposes, please use fourier opacity maps or something similar in a real-time context
            float shadowSum = 0.0;
            const uint nSamples = BILLBOARD_SHADOW_SAMPLES;
            float scale = 1.0 / (nSamples - 1);
            for(uint j = 0; j < nSamples; ++j)
            {
                shadowSum += TraceShadowRay(sd.posW - sd.V * (billboardDepth * j * scale - billboardOffset), l);
            }
            curColor += color * (shadowSum / nSamples);
        }
        else
        {
            float shadowFac = 1.0;
            if(i == 0) shadowFac = TraceShadowRay(sd.posW, l); // single shadow ray sample for the primary light
            curColor += evalBillboardMaterial(sd, l) * shadowFac;
        }
    }
    curColor += sd.diffuse * AMBIENT_LIGHT;
    return float4(curColor * sd.opacity, 1.0 - sd.opacity);
}

// simple shading function that iterates though all lights and traces shadow rays for the primary light source
// returns rgb color and transmittance (in alpha channel)
float4 shade(ShadingData sd)
{
    float3 curColor = sd.emissive;

    // forward render all light sources
    const uint lightCount = gScene.getLightCount();
    for(uint i = 0; i < lightCount; ++i)
    {
        LightData l = gScene.getLight(i);
        float shadowFac = 1.0;
        if(i == 0) shadowFac = TraceShadowRay(sd.posW, l); // single shadow ray sample for the primary light
        curColor += evalMaterial(sd, l, shadowFac).color;
    }
    curColor += sd.diffuse * AMBIENT_LIGHT;
    return float4(curColor * sd.opacity, 1.0 - sd.opacity);
}

// returns depth weight for Weighted-Blended OIT
float getDepthWeight(float alpha, float z)
{
    // transform z to non-linear
    // Phenomenological:
    const float near = gScene.camera.data.nearZ;
    const float far = gScene.camera.data.farZ;
    float f = ((near * far) / clamp(z, near, far) - far) / (near - far);
    
    float w = 10 * (1 - 0.99 * f); // modification: did not include alpha here (looks better for smoke)
    return clamp(w * w * w, 0.01, 3000.0) * alpha;
    //return 1.0;
}
