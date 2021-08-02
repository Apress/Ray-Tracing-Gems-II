#ifndef MANDELBULB_H
#define MANDELBULB_H

float mdist( in float3 p, out float4 resColor )
{
    float3 w = p;
    float m = dot(w,w);

    float4 trap = float4(abs(w),m);
	float dz = 1.;
    
    
	for( int i=0; i<4; i++ )
    {
//#define PERFORMANCE
#ifdef PERFORMANCE
        float m2 = m*m;
        float m4 = m2*m2;
		dz = 8.0*sqrt(m4*m2*m)*dz + 1.0;

        float x = w.x; float x2 = x*x; float x4 = x2*x2;
        float y = w.y; float y2 = y*y; float y4 = y2*y2;
        float z = w.z; float z2 = z*z; float z4 = z2*z2;

        float k3 = x2 + z2;
        float k2 = rsqrt( k3*k3*k3*k3*k3*k3*k3 );
        float k1 = x4 + y4 + z4 - 6.0*y2*z2 - 6.0*x2*y2 + 2.0*z2*x2;
        float k4 = x2 - y2 + z2;

        w.x = p.x +  64.0*x*y*z*(x2-z2)*k4*(x4-6.0*x2*z2+z4)*k1*k2;
        w.y = p.y + -16.0*y2*k3*k4*k4 + k1*k1;
        w.z = p.z +  -8.0*y*k4*(x4*x4 - 28.0*x4*x2*z2 + 70.0*x4*z4 - 28.0*x2*z2*z4 + z4*z4)*k1*k2;
#else
        dz = 8.0*pow(sqrt(m),7.0)*dz + 1.0;
        float r = length(w);
        float b = 8.0*acos( w.y/r);
        float a = 8.0*atan2( w.x, w.z );
        w = p + pow(r,8.0) * float3( sin(b)*sin(a), cos(b), sin(b)*cos(a) );
#endif        
        
        trap = min( trap, float4(abs(w),m) );

        m = dot(w,w);
		if( m > 256.0 )
            break;
    }

    resColor = float4(m,trap.yzw);

    return 0.25*log(m)*sqrt(m)/dz;
}

float3 CalcNormal( in float3 pos, in float t )
{
    float4 tmp;
    float2 e = float2(1.0,-1.0)*0.5773 * 0.0001;
    return normalize( e.xyy*mdist( pos + e.xyy,tmp ) + 
					  e.yyx*mdist( pos + e.yyx,tmp ) + 
					  e.yxy*mdist( pos + e.yxy,tmp ) + 
					  e.xxx*mdist( pos + e.xxx,tmp ) );
}

bool MandelbulbDistance(in Ray ray, in float time, int instanceId, out float thit, out ProceduralPrimitiveAttributes attr,
                        in float stepScale = 1.0f)
{
    float res = -1.0;

    float3 ro = ray.origin;
    float3 rd = ray.direction;
    
    // bounding sphere
    float2 dis = isphere( float4(0.0,0.0,0.0, 1.25), ro, rd );
    if( dis.y<0.0 )
        return false;
    dis.x = max( dis.x, 0.0 );
    dis.y = min( dis.y, 1000.0 );

    // raymarch fractal distance field
	float4 trap = {0.f, 0.f, 0.f, 0.f};

	float t = dis.x;
    float3 pos = {0.f, 0.f, 0.f};
    
    // Number of iterations is used to animate the Mandelbulbs.
    int iAnimMin = 1;
    int iAnimMax = 128;
    int iMax = (0.5*pow(time,2)+instanceId) % (iAnimMax*2);
    if(iMax>iAnimMax) 
    {
        iMax = 2*iAnimMax - iMax; 
    }
   
    iMax += iAnimMin;
    
	for( int i=0; i<iMax; i++  )
    { 
        pos = ro + rd*t;
        float th =  0.0001*t;
		float h = mdist( pos, trap );
		if( t>dis.y || h<th ) break;
        t += h;
    }
    
    if( t<dis.y )
    {
        float3 hitSurfaceNormal = CalcNormal(pos, t);
        thit = t;
        attr.normal = hitSurfaceNormal;
        attr.color = trap;
        return true;
    }

    return false;
}

[shader("intersection")]
void Intersection_Mandelbulb()
{
    Ray localRay = GetRayInAABBPrimitiveLocalSpace();

    float thit;
    ProceduralPrimitiveAttributes attr = { {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 1.f} };
    
    bool primitiveTest = MandelbulbDistance(localRay, g_sceneCB.elapsedTime, l_aabbCB.instanceIndex, thit, attr, l_materialCB.stepScale);
    
    if (primitiveTest && thit < RayTCurrent())
    {
        InstanceBuffer aabbAttribute = g_instanceBuffer[l_aabbCB.instanceIndex];
        attr.normal = normalize(mul(attr.normal, (float3x3) WorldToObject3x4()));
        
        ReportHit(thit, /*hitKind*/ 0, attr);
    }
}

[shader("closesthit")]
void ClosestHit_Mandelbulb(inout RayPayload rayPayload, in ProceduralPrimitiveAttributes attr)
{
    // color
    float3 col = float3(0.01, 0.01, 0.01);
	col = lerp( col, float3(0.10,0.20,0.30), clamp(attr.color.y,0.0,1.0) );
	col = lerp( col, float3(0.02,0.10,0.30), clamp(attr.color.z*attr.color.z,0.0,1.0) );
    col = lerp( col, float3(0.30,0.10,0.02), clamp(pow(attr.color.w,6.0),0.0,1.0) );
    col *= 0.5;
        
    // lighting terms
    float3 light1 = float3(0.577, 0.577, -0.577);
    float3 light2 = float3(-0.707, 0.000,  0.707);
            
    float3 pos = HitWorldPosition();
    float3 nor = attr.normal;
    float3 hal = normalize( light1-WorldRayDirection());
    float3 ref = reflect( WorldRayDirection(), nor );
    float occ = clamp(0.05*log(attr.color.x),0.0,1.0);
    float fac = clamp(1.0+dot(WorldRayDirection(),nor),0.0,1.0);
            
    // sun
    float sha1 = 1.f;
    float dif1 = clamp( dot( light1, nor ), 0.0, 1.0 )*sha1;
    float spe1 = pow( clamp(dot(nor,hal),0.0,1.0), 32.0 )*dif1*(0.04+0.96*pow(clamp(1.0-dot(hal,light1),0.0,1.0),5.0));
    // bounce
    float dif2 = clamp( 0.5 + 0.5*dot( light2, nor ), 0.0, 1.0 )*occ;
    // sky
    float dif3 = (0.7+0.3*nor.y)*(0.2+0.8*occ);
        
	float3 lin = float3(0.0, 0.0, 0.0); 
		    lin += 7.0*float3(1.50,1.10,0.70)*dif1;
		    lin += 4.0*float3(0.25,0.20,0.15)*dif2;
        	lin += 1.5*float3(0.10,0.20,0.30)*dif3;
            lin += 2.5*float3(0.35,0.30,0.25)*(0.05+0.95*occ);  // ambient
        	lin += 4.0*fac*occ;                                 // fake SSS
	col *= lin;
	col = pow( col, float3(0.7,0.9,1.0) );                      // fake SSS
    col += spe1*15.0;
        
    // gamma
	col = sqrt( col );
            
    float4 color = float4(col, 1.f);
    
    // Reflected component.
    float4 reflectedColor = float4(0, 0, 0, 0);
    if (l_materialCB.reflectanceCoef > 0.001)
    {
        // Trace a reflection ray.
        Ray reflectionRay = {pos, ref};
        float4 reflectionColor = TraceRadianceRay(reflectionRay, rayPayload.recursionDepth);

        float3 fresnelR = FresnelReflectanceSchlick(WorldRayDirection(), attr.normal, l_materialCB.albedo.xyz);
        reflectedColor = l_materialCB.reflectanceCoef * float4(fresnelR, 1) * reflectionColor;
    }

    // Calculate final color.
    color = color + reflectedColor;
    color += rayPayload.color;
    
    // Apply visibility falloff.
    rayPayload.dist+=RayTCurrent();
   
    rayPayload.color = color;
}

#endif