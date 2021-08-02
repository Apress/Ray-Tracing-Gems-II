#ifndef JULIASETS_H
#define JULIASETS_H

#define kPrecis 0.00025f
#define kc float4(-2.f, 6.f, 15.f, -6.f) / 22.0f

float3 float3_ctor(float x0)
{
    return float3(x0, x0, x0);
}

float4 QuatSquare(in float4 q)
{
    return float4(((((q.x * q.x) - (q.y * q.y)) - (q.z * q.z)) - (q.w * q.w)), ((2.0 * q.x) * q.yzw));
}

float4 QuatCube(in float4 q)
{
    float4 q2 = (q * q);
    return float4((q.x * (((q2.x - (3.0 * q2.y)) - (3.0 * q2.z)) - (3.0 * q2.w))), (q.yzw * ((((3.0 * q2.x) - q2.y) - q2.z) - q2.w)));
}

float QuatLength(in float4 q)
{
    return dot(q, q);
}

float2 iSphere(in float3 ro, in float3 rd, in float rad)
{
    float b = dot(ro, rd);
    float c = dot(ro, ro) - (rad * rad);
    float h = (b * b) - c;
    if (h < 0.0)
    {
        return float2(-1.0, -1.0);
    }
    h = sqrt(h);
    return float2(-b - h, -b + h);
}

float2 JDist(in float3 p, in float4 c, in int iMax)
{
    float4 z = float4(p, 0.0);
    float dz2 = 1.0;
    float m2 = 0.0;
    float n = 0.0;
    
    for (int i = 0; i < iMax; i++)
    {
        dz2 *= 9.0 * QuatLength(QuatSquare(z));
        z = QuatCube(z) + c;
        m2 = QuatLength(z);
        if (m2 > 256.0)
        {
            break;
        }
        n += 1.0;
    }
    
    float d = (0.25 * log(m2)) * sqrt((m2 / dz2));
    return float2(d, n);
}
float3 CalcNormal(in float3 pos, in float4 c = kc, in int iMax = 200)
{
    const float2 e = float2(1.0f, -1.0f) * 0.5773f * kPrecis;
    
    return normalize(
        e.xyy * JDist(pos + e.xyy, c, iMax).x +
        e.yyx * JDist(pos + e.yyx, c, iMax).x +
        e.yxy * JDist(pos + e.yxy, c, iMax).x +
        e.xxx * JDist(pos + e.xxx, c, iMax).x
    );
}
float2 SphereTracing(in float3 ro, in float3 rd, in float4 c = kc, in float deltaT = 0.0, in int iMax = 200)
{
    float tmax = 7000.f;
    float tmin = kPrecis;
 
    float upperPlane = deltaT;
    
    float tpS = ( upperPlane - ro.y) / rd.y;
    
    bool isCamAboveUpper = (ro.y > upperPlane);
    
    if (tpS > 0.0)
    {
        if (isCamAboveUpper)
        {
            tmin = max(tmin, tpS);
        }
        else
        {
            tmax = min(tmax, tpS);
        }
    }
    else
    {
        if (isCamAboveUpper)
        {
            return float2(-2.0, 0.0);
        }
    }
    
    float lowerPlane = -1.1;
    float tpF = (lowerPlane - ro.y) / rd.y;
    
    bool isCamBellowLower = (ro.y < lowerPlane);
    
    if (tpF > 0.0)
    {
        if (isCamBellowLower)
        {
            tmin = max(tmin, tpF);
        }
        else
        {
            tmax = min(tmax, tpF);
        }
    }
    else
    {
        if (isCamBellowLower)
        {
            return float2(-2.0, 0.0);
        }
    }
    
    float2 bv = iSphere(ro, rd, 1.2);
    if (bv.y < 0.0)
    {
        return float2(-2.0, 0.0);
    }
    tmin = max(tmin, bv.x);
    tmax = min(tmax, bv.y);
    float2 res = { -1.0, -1.0 };
    float t = tmin;
    float lt = { 0.0 };
    float lh = { 0.0 };
    {
        for (int i = 0; i < 1024; i++)
        {
            res = JDist(ro + (rd * t), c, iMax);
            if (res.x < kPrecis)
            {
                break;
            }
            lt = t;
            lh = res.x;
            t += min(res.x, 0.2);
            if (t > tmax)
            {
                break;
            }
        }
    }
    if ((lt > 9.9999997e-05) && (res.x < 0.0))
    {
        t = lt - ((lh * (t - lt)) / (res.x - lh));
    }
    float s = 0 ;
    if (t < tmax)
    {
        s = t;
    }
    else
    {
        s = -1.0;
    }
    res.x = s;
    return res;
}
float3 ColorSurface(in float3 pos, in float2 tn)
{
    float3 col = 0.5 + 0.5 * cos(log2(tn.y) * 0.9 + 3.5 + float3(0.0, 0.6, 1.0));
    
    if (pos.y > 0.0)
    {
        col = lerp(col, float3(1.0, 1.0, 1.0), 0.2);
    }
    float inside = smoothstep(14., 15., tn.y);
    
    col *= float3(0.45, 0.42, 0.40) + float3(0.55, 0.58, 0.60) * inside;
    col = lerp(col * col * (3.0 - 2.0 * col), col, inside);
    col = lerp(lerp(col, float3_ctor(dot(col, float3(0.3333, 0.333, 0.333))), -0.4), col, inside);
    
    float3 surfaceColor = clamp(col * 0.65, 0.0, 1.0);
    
    return surfaceColor;
}

bool IntersectionJuliaTest(in float3 ro, in float3 rd, inout float3 normal, inout float2 resT, in float time=0.3f)
{
    resT = 1e20;
    
    ro *= 0.4;
    
    ro.y += -1.f; //for the scene with multiples objects

    float2 tn = SphereTracing(ro, rd);
    
    bool cond = (tn.x >= 0.0);
    if (cond)
    {
        float3 pos = (ro + (tn.x * rd));
        normal = CalcNormal(pos);
        resT = tn;
    }
    return cond;
}

[shader("intersection")]
void IntersectionJulia()
{
    Ray localRay = GetRayInAABBPrimitiveLocalSpace();

    float2 thit;
    ProceduralPrimitiveAttributes attr = { {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 1.f} };
    
    float3 pos;
    bool primitiveTest = IntersectionJuliaTest(localRay.origin, localRay.direction, attr.normal, thit, g_sceneCB.elapsedTime);
   
    if (primitiveTest && thit.x < RayTCurrent())
    {
        InstanceBuffer aabbAttribute = g_instanceBuffer[l_aabbCB.instanceIndex];
        attr.normal = normalize(mul(attr.normal, (float3x3) WorldToObject3x4()));
        attr.color = float4(thit, 0.f, 0.f); // Using the color parameter to send intersection min and max t.
        
        ReportHit(thit.x, /*hitKind*/ 0, attr);
    }
}

[shader("closesthit")]
void ClosestHitJulia(inout RayPayload rayPayload, in ProceduralPrimitiveAttributes attr)
{
    // Shadow component.
    // Trace a shadow ray.
    float3 hitPosition = HitWorldPosition();
    Ray shadowRay = { hitPosition, normalize(g_sceneCB.lightPosition.xyz - hitPosition) };
    bool shadowRayHit = false;
    
    float3 pos = ObjectRayOrigin() + RayTCurrent() * ObjectRayDirection();
    float3 dir = WorldRayDirection();
    
    float4 albedo = float4(3.5*ColorSurface(pos, attr.color.xy), 1.f);
    
    if (rayPayload.recursionDepth == MAX_RAY_RECURSION_DEPTH - 1)
    {
       albedo += 1.65 * step(0.0, abs(pos.y));
    }
    
    // Reflected component.
    float4 reflectedColor = float4(0, 0, 0, 0);
    
    float reflecCoef = 0.1;
    
    if (reflecCoef > 0.001)
    {
        // Trace a reflection ray.
        Ray reflectionRay = { hitPosition, reflect(WorldRayDirection(), attr.normal) };
        float4 reflectionColor = TraceRadianceRay(reflectionRay, rayPayload.recursionDepth);
        
        float3 fresnelR = FresnelReflectanceSchlick(WorldRayDirection(), attr.normal, albedo.xyz);
        reflectedColor = reflecCoef * float4(fresnelR, 1) * reflectionColor;
    }

    float diffuseCoef = 0.6;
    float specularCoef = 0.08;
    float specularPower = 0.2;
    // Calculate final color.
    float4 phongColor = CalculatePhongLighting(albedo, attr.normal, shadowRayHit, diffuseCoef, specularCoef, specularPower);
    float4 color = phongColor + reflectedColor;

    color += rayPayload.color;
    rayPayload.color = color;
}

#endif