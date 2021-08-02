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

// imported and modified ray differentials
#include "RenderPasses/Shared/PathTracer/RayFootprintModes.slangh"

#if (RAY_FOOTPRINT_MODE == RAY_FOOTPRINT_MODE_RAYDIFFS_ANISO)

bool SolveLinearSystem2x2(float2x2 A, float2 B, out float2 x) {
    float det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    if (abs(det) < 1e-10f)
        return false;
    x[0] = (A[1][1] * B[0] - A[0][1] * B[1]) / det;
    x[1] = (A[0][0] * B[1] - A[1][0] * B[0]) / det;
    if (isnan(x[0]) || isnan(x[1]))
        return false;
    return true;
}

extension RayFootprint
{
    static RayFootprint create(uint2 pixel, uint2 frameDim)
    {
        RayFootprint res;
        const float3 nonNormalizedRayDir = gScene.camera.computeNonNormalizedRayDirPinhole(pixel, frameDim);
        float3 dDdx, dDdy;
        computeRayDirectionDifferentials(nonNormalizedRayDir, gScene.camera.data.cameraU, gScene.camera.data.cameraV, frameDim, dDdx, dDdy);
        res.rayDiff = RayDiff.create(float3(0.0), float3(0.0), dDdx, dDdy);
        return res;
    }

    [mutating]
    void hitSurface(float3 faceNormal, float rayT, float3 rayDir)
    {
        this.rayDiff = this.rayDiff.propagate(rayDir, rayT, faceNormal);   // Propagate the ray differential to the current hit point.
    }

    [mutating]
    void hitSphere(float3 faceNormal, float3 rayOrg, float3 rayDir, float rayT, float3 sphereCenter, float sphereRadius)
    {
        // Propagate the ray differential to the current hit point.
        this.rayDiff = this.rayDiff.propagate(rayDir, rayT, faceNormal);
        // this version is more precise but does not make a significant difference in practice
        //this.rayDiff = this.rayDiff.propagateSphere(rayOrg, rayDir, sphereCenter, sphereRadius);   
    }

    ShadingData computeShadingData(VertexData v, StaticVertexData triangleVertices[3], HitInfo hit, float rayT, float3 rayDir)
    {
        // Get hit point adapted parameters.
        float2 dUVdx, dUVdy;    // Ray differential variables for the texture lookup.
        prepareRayDiffAtHitPoint(v, triangleVertices, hit.getBarycentricWeights(), rayDir, rayT, gScene.getWorldMatrix(hit.instanceID), gScene.getInverseTransposeWorldMatrix(hit.instanceID),
            this.rayDiff, dUVdx, dUVdy);

        // Compute shading data.
        float3 viewDir = -rayDir;
        ShadingData sd;
        const uint materialID = gScene.getMaterialID(hit.instanceID);
        // Use the dUVdx and dUVdy directly --> gives you anisotropic filtering.
        sd = prepareShadingData(v, materialID, gScene.materials[materialID], gScene.materialResources[materialID], viewDir, dUVdx, dUVdy);

        return sd;
    }

    BillboardShadingData computeBillboardShadingData(BillboardVertex v, uint aabbIndex, float3 rayOrg, float3 rayDir)
    {
        // reconstruct billboard plane
        const AABB bbox = gScene.proceduralPrimitiveAABBs[aabbIndex];
        const float2 dim = bbox.extent().xy;
        const float r = dim.x * 0.5;

        float2 dUVdx, dUVdy;    // Ray differential variables for the texture lookup.
        #if BILLBOARD_TYPE == BILLBOARD_TYPE_SPHERICAL
        {
            // from pbrt http://www.pbr-book.org/3ed-2018/Texture/Sampling_and_Antialiasing.html#FindingtheTextureSamplingRate
            const float3 n = v.normalW;
            const float2 uv = v.texC;

            int dim[2];
            if(abs(n.x) > abs(n.y) && abs(n.x) > abs(n.z)) {
                dim[0] = 1; dim[1] = 2;
            } else if (abs(n.y) > abs(n.z)) {
                dim[0] = 0; dim[1] = 2;
            } else {
                dim[0] = 0; dim[1] = 1;
            }
            
            // change of position per texture coordinate direction (UV-sphere)
            //float theta = (uv.x - BILLBOARD_UV_OFFSET(aabbIndex)) * 2.0 * M_PI;
            //float phi = M_PI * uv.y;
            //float sinphi = sin(phi);
            //float cosphi = cos(phi);
            //float sintheta = sin(theta);
            //float costheta = cos(theta);
            //const float3 dpdu = float3(-sinphi*sintheta,0.0,sinphi*costheta) * 2.0 * M_PI * r;
            //const float3 dpdv = float3(cosphi*costheta,-sinphi,cosphi*sintheta) * M_PI * r;
            // simplified computation when replacing theta and phi with the normal (theta = arctan(n.z, n.x), phi = arccos(n.y))
            float denom = 1.0 / sqrt(n.x * n.x + n.z * n.z);
            float invysq = sqrt(1.0 - n.y * n.y);
            const float3 dpdu = float3(-n.z * invysq * denom, 0.0, n.x * invysq * denom) * 2.0 * M_PI * r;
            const float3 dpdv = float3(n.y * n.x * denom, -invysq, n.z * n.y * denom) * M_PI * r;
            
            float2x2 A;
            A[0][0] = dpdu[dim[0]]; A[0][1] = dpdv[dim[0]];
            A[1][0] = dpdu[dim[1]]; A[1][1] = dpdv[dim[1]];
            
            float2 Bx = float2(this.rayDiff.dOdx[dim[0]], this.rayDiff.dOdx[dim[1]]);
            float2 By = float2(this.rayDiff.dOdy[dim[0]], this.rayDiff.dOdy[dim[1]]);
            
            if(!SolveLinearSystem2x2(A, Bx, dUVdx)) dUVdx = 0.0;
            if(!SolveLinearSystem2x2(A, By, dUVdy)) dUVdy = 0.0;
        }
        #else
        {
            dUVdx.x = dot(this.rayDiff.dOdx, v.tangentW) / dim[0];
            dUVdx.y = dot(this.rayDiff.dOdx, v.bitangentW) / dim[1];
            dUVdy.x = dot(this.rayDiff.dOdy, v.tangentW) / dim[0];
            dUVdy.y = dot(this.rayDiff.dOdy, v.bitangentW) / dim[1];
        }
        #endif

        // Compute shading data.
        float3 viewDir = -rayDir;
        const uint materialID = BILLBOARD_MATERIAL_ID;
        // Use the dUVdx and dUVdy directly --> gives you anisotropic filtering.
        ExplicitRayDiffsLodTextureSampler explicitRayDiffsLOD = { dUVdx, dUVdy };
        BillboardShadingData sd = prepareBillboardShadingData(v, materialID, gScene.materials[materialID], gScene.materialResources[materialID], viewDir, explicitRayDiffsLOD);

        return sd;
    }

    [mutating]
    void bounceOnSurface(VertexData v, StaticVertexData triangleVertices[3], HitInfo hit, float3 normal, float3 rayDirIn, float3 rayDirOut, bool isTransmission)
    {
        // Check if scattered ray is refracted.
        if (isTransmission || dot(rayDirOut, normal) <= 0.0) return; // Leaves the footprint unchanged until refraction if properly handled.

        // TODO: Avoid recomputing everything (dUVdx, dUVdy and  dBarydx, dBarydy) by keeping intermediate variables live in registers.
        float2 dUVdx, dUVdy;    // Ray differential variables for the texture lookup.
        reflectRayDiffUsingVertexData(v, triangleVertices, hit.getBarycentricWeights(), rayDirIn, gScene.getWorldMatrix(hit.instanceID), gScene.getInverseTransposeWorldMatrix(hit.instanceID),
                                        this.rayDiff, dUVdx, dUVdy);
    }
};

/** Ray footprint for tex LOD.
    Force LOD0 version.
    Makes the interface consistant even when ray footprint is disabled.
*/
#else
extension RayFootprint
{
    static RayFootprint create(uint2 pixel, uint2 frameDim)
    {
        RayFootprint res;
        return res;
    }

    [mutating]
    void hitSurface(float3 faceNormal, float rayT, float3 rayDir)
    {
    }

    [mutating]
    void hitSphere(float3 faceNormal, float3 rayOrg, float3 rayDir, float rayT, float3 sphereCenter, float sphereRadius)
    {        
    }

    [mutating]
    void bounceOnSurface(VertexData v, StaticVertexData triangleVertices[3], HitInfo hit, float3 normal, float3 rayDirIn, float3 rayDirOut, bool isTransmission)
    {
    }

    ShadingData computeShadingData(VertexData v, StaticVertexData triangleVertices[3], HitInfo hit, float rayT, float3 rayDir)
    {
        const uint materialID = gScene.getMaterialID(hit.instanceID);
        return prepareShadingData(v, materialID, gScene.materials[materialID], gScene.materialResources[materialID], -rayDir, 0.f);
    }

    BillboardShadingData computeBillboardShadingData(BillboardVertex v, uint aabbIndex, float3 rayOrg, float3 rayDir)
    {
        const uint materialID = BILLBOARD_MATERIAL_ID;
        ExplicitLodTextureSampler explicitLOD = { 0.0 };
        return prepareBillboardShadingData(v, materialID, gScene.materials[materialID], gScene.materialResources[materialID], -rayDir, explicitLOD);
    }
};
#endif


/** Prepare shading data using ray footprint and TexLOD.
   \param[in] v The mesh vertex data interpolated at hit point.
   \param[in] rayFootprint The ray footprint to use for TexLOD.
   \param[in] triangleVertices The vertex data of the 3 vertices of the triangle.
   \param[in] hit Ray hit info at hit point.
   \param[in] rayOrg Ray origin.
   \param[in] rayDir Ray direction.
   \return Shading data computed at hit point.
*/
ShadingData prepareShadingData(VertexData v, RayFootprint rayFootprint, StaticVertexData triangleVertices[3], HitInfo hit, float rayT, float3 rayDir)
{
    return rayFootprint.computeShadingData(v, triangleVertices, hit, rayT, rayDir);
}

static const float3 gRandomColors[] = {float3(1.0, 0, 0), float3(1.0, 0.1, 0.9), float3(0.1, 1, 0.1), float3(0, 1.0, 0.95), float3(1.0, 0.95, 0.0), float3(0.1, 0.1, 1)};

BillboardShadingData prepareBillboardShadingData(BillboardVertex v, RayFootprint rayFootprint, uint aabbIndex, /*long version*/ float3 rayOrg,  float3 rayDir)
{
    BillboardShadingData sd = rayFootprint.computeBillboardShadingData(v, aabbIndex, rayOrg, rayDir);
#ifdef BILLBOARD_ALPHA_MULTIPLIER
    sd.opacity *= BILLBOARD_ALPHA_MULTIPLIER;
#endif
    if(BILLBOARD_TYPE == BILLBOARD_TYPE_PARTICLE) sd.opacity *= v.billboardStrength;
    if(BILLBOARD_TYPE == BILLBOARD_TYPE_SPHERICAL && sd.opacity < 1.0) sd.opacity = clamp(1.0 - pow(1.0 - sd.opacity, v.billboardStrength), 0.0, 1.0);
    if(USE_RANDOM_BILLBOARD_COLORS)
    {
        sd.diffuse *= gRandomColors[aabbIndex % 6];
    }
    return sd;
}
