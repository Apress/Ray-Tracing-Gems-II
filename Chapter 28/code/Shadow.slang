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

// use ray query to trace approximate shadow ray
float TraceShadowRay(float3 origin, LightData l)
{
    if(!USE_SHADOWS) return 1.0;

    // set up shadow ray
    RayDesc ray;
    ray.Origin = origin;
    ray.TMin = 0.01;
    ray.TMax = FLT_MAX;
    switch(l.type)
    {
    case uint(LightType::Point): {
        float3 dir = l.posW - origin;
        ray.TMax = length(dir);
        ray.Direction = dir / ray.TMax;
        } break;
    case uint(LightType::Directional):
        ray.Direction = -l.dirW;
        break;
    default: return 1.0; // not supported
    }

    // prepare ray query (do all computations in the "any-hit shader")
#if BILLBOARD_TYPE == BILLBOARD_TYPE_IMPOSTOR
    RayQuery<RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_FORCE_NON_OPAQUE | /* no shadows for impostors */ RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
#else
    RayQuery<RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_FORCE_NON_OPAQUE> q;
#endif
    q.TraceRayInline(gRtScene, RAY_FLAG_NONE, 0xff /* instanceInclusionMask */, ray);
    float transmit = 1.0;

    while(q.Proceed())
    {
        if(q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) // triangle
        {
           float2 rayBary = q.CandidateTriangleBarycentrics();
           float3 barycentrics = float3(1.0 - rayBary.x - rayBary.y, rayBary.x, rayBary.y);
           VertexData v = gScene.getVertexData(q.CandidateInstanceID() + q.CandidateGeometryIndex(), q.CandidatePrimitiveIndex(), barycentrics);
           const uint materialID = gScene.getMaterialID(q.CandidateInstanceID() + q.CandidateGeometryIndex());
           ShadingData sd = prepareShadingData(v, materialID, gScene.materials[materialID], gScene.materialResources[materialID], -ray.Direction, 0.f);
           transmit *= 1.0 - sd.opacity;
        }
        else // billboard
        {
            // do constant colored shadows in spherical billboard style
            const uint aabbOffset = gScene.proceduralPrimitives[q.CandidateGeometryIndex()].AABBOffset;
            const AABB bbox = gScene.proceduralPrimitiveAABBs[aabbOffset];
            const float2 dim = bbox.extent().xy;
            const float3 C = bbox.center();

            const float r = 0.5 * dim.x;
            const float t = dot(ray.Direction, C - ray.Origin);
            const float l = length(ray.Origin + t * ray.Direction - C);
            if(l >= r) continue; // no hit
            const float w = sqrt(r * r - l * l);
            const float ds = min(ray.TMax, t + w) - max(ray.TMin, t - w);
            if(ds <= 0.0) continue;
            // hard coded opacity
            float alpha = 0.3;
            #ifdef BILLBOARD_ALPHA_MULTIPLIER
                alpha *= BILLBOARD_ALPHA_MULTIPLIER;
            #endif

            transmit *= pow(alpha, (1 - (l / r)) * (0.5 * ds / r));
        }

        if(transmit <= SHADOW_MIN)
        {
            q.Abort();
            break;
        }
    }

    return max(transmit, SHADOW_MIN);
}
