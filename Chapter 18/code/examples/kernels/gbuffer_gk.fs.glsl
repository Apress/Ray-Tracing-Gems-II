#version 300 es

precision highp float;
precision highp sampler2DArray;

uniform sampler2D u_materialBuffer;
uniform int u_material_index;
uniform sampler2DArray u_texturesBuffer;
uniform vec3 u_camera_position;
uniform sampler2D env_map;
uniform sampler2D u_blue_noise_map;

#define RG_PI 3.14159
#define WR_INV_PI 0.31830988618
#define WR_EPS_COS 0.001
#define WR_EPSILON 0.0001

#define PERFECT_SPECULAR 0.0
#define FRESNEL_SPECULAR 1.0
#define CT_GGX 2.0
#define LAMBERT 3.0
#define DISNEY 4.0
#define THICK_GLASS 5.0

// we need to declare an output for the fragment shader
layout(location = 0) out vec4 outColor; //R11G11B10
layout(location = 1) out vec4 outNormal; // RGB10_A2 (Normal.xyz, unused)
layout(location = 2) out vec4 outParams; // RGB10_A2 (UV.xy, materialID, unused)

in vec2 v_UV;
in vec3 v_Normal;
in vec3 v_PosWCS;

// Create ONB from the normalized vector (N). This is an optimized, branchless version without the need to use sqrt to normalize the vectors.
void createONBOpt(const in vec3 n, out vec3 b1, out vec3 b2)
{
  //float sign = copysignf(1.0f, n.z);
  float sign = (n.z >= 0.0) ? 1.0 : -1.0;
  float a = -1.0 / (sign + n.z);
  float b = n.x * n.y * a;
  b1 = vec3(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
  b2 = vec3(b, sign + n.y * n.y * a, -n.y);
}

// cosine weighted hemisphere sampling. pdf = cosTheta / PI
// Point in Tangent Coordinates need to be converted to World Coordinates
vec3 cosine_sample_hemisphere(float u1, float u2)
{
  // Uniformly sample disk.
  float r = sqrt(u1);
  float phi = 2.0 * RG_PI * u2;
  vec3 p;
  p.x = r * cos(phi);
  p.y = r * sin(phi);

  // Project up to hemisphere.
  p.z = sqrt(max(0.0, 1.0 - p.x*p.x - p.y*p.y));
  return p;
}

// Sample the Half Vector from the GGX NDF. pdf = D(m) * cosTheta_m: D(m) * (NdotH)
vec3 ImportanceSampleGGX(vec2 random, float roughness)
{
	float a = roughness * roughness;
	float phi = 2.0 * RG_PI * random.x;
	float cosTheta = sqrt((1.0 - random.y) / (1.0 + (a*a - 1.0)*random.y));
	float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
	
	vec3 H;
	H.x = sinTheta * cos(phi);
	H.y = sinTheta * sin(phi);
	H.z = cosTheta;
	
  return H;	
}

// Sample the Half Vector from the GGX VNDF. pdf = D(m) * cosTheta_m: D(m) * (NdotH)
// From the paper: Sampling the GGX Distribution of Visible Normals
vec3 ImportanceSampleGGX_VNDF(vec2 random, float roughness, vec3 Ve)
{
	vec3 Vh = normalize(vec3(roughness * Ve.x, roughness * Ve.y, Ve.z));
	float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
	vec3 T1 = lensq > 0.0 ? vec3(-Vh.y, Vh.x, 0) * inversesqrt(lensq) : vec3(1,0,0);
	vec3 T2 = cross(Vh, T1);
	float r = sqrt(random.x);
	float phi = 2.0 * RG_PI * random.y;
	float t1 = r * cos(phi);
	float t2 = r * sin(phi);
	float s = 0.5 * (1.0 + Vh.z);
	t2 = (1.0 - s) * sqrt(1.0 - t1*t1) + s*t2;
	vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1*t1 - t2*t2)) * Vh;
	vec3 Ne = normalize(vec3(roughness * Nh.x, roughness * Nh.y, max(0.0, Nh.z)));
	return Ne;	
}

float convertFromLinearReflectance(float refl)
{
    return 0.16 * refl * refl;
}

vec3 convertFromLinearReflectance3(vec3 refl)
{
    return 0.16 * refl * refl;
}

float geometricTermSmithGGX2(float NdotV, float NdotL, float a)
{
  float a2 = a * a;
  float nom = 2.0 * NdotL * NdotV;
  float denom = NdotV * sqrt(a2 + (1.0 - a2) * NdotL * NdotL) + NdotL * sqrt(a2 + (1.0 - a2) * NdotV * NdotV);
  return nom / denom;
}

// GLTF also use this G1 and G2 is aproximated as G1(wo) * G1(wi).
// cosTheta is the angle with the normal. Either NdotV or NdoL
float geometricTermSmithGGX1(float cosTheta, float a)
{
  float a2 = a * a;
  float nom = 2.0 * cosTheta;
  float denom = cosTheta  + sqrt(a2 + (1.0 - a2) * cosTheta * cosTheta);
  return nom / denom;
}

// Trowbridge-Reitz (GGX) NDF isotropic distribution.
float ggxDistribution(float NdotH, float roughness)
{
    float a2 = roughness * roughness;
    float denominator = NdotH * NdotH * (a2 - 1.0) + 1.0;

    return a2 / (RG_PI * denominator * denominator);
}

vec3 schlickFresnel(vec3 F0, float LdotH)
{
    float power = 1.0 - LdotH;
    float power2 = power * power;
    float power5 = power2 * power2 * power;
    return F0 + (1.0 - F0) * power5;       
}

mat3 cotangent_frame( vec3 N, vec3 p, vec2 uv )
{
    // get edge vectors of the pixel triangle
    vec3 dp1 = dFdx( p );
    vec3 dp2 = dFdy( p );
    vec2 duv1 = dFdx( uv );
    vec2 duv2 = dFdy( uv );

    // solve the linear system
    vec3 dp2perp = cross( dp2, N );
    vec3 dp1perp = cross( N, dp1 );
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;

    // construct a scale-invariant frame 
    float invmax = inversesqrt( max( dot(T,T), dot(B,B) ) );
    return mat3( T * invmax, B * invmax, N );
}

vec3 computeNormal(vec3 viewDirection, int normalsTextureIndex)
{
  normalsTextureIndex = -1;
  if(normalsTextureIndex >= 0)
  {
    vec3 normal = texture(u_texturesBuffer, vec3(v_UV, normalsTextureIndex)).rgb;
    normal = 2.0 * normal - 1.0; // Convert to [-1 1]

    mat3 TBN = cotangent_frame(v_Normal, -viewDirection, v_UV);
    
    return normalize(TBN * normalize(normal));
  }
  else
  {
    return normalize(v_Normal);
  }
}

vec3 evaluateCookTorrance(in vec3 viewDirection, in vec3 lightDirection,
    in vec3 shadingNormal, in vec3 baseColor, float metallic, float roughness, float reflectance);
vec3 evaluateLambert(in vec3 baseColor);
vec3 sampleEnvPerfectSpecular(in vec3 view_direction, in vec3 shading_normal, in vec3 baseColor);
vec3 sampleEnvCookTorrance(in vec3 view_direction, in vec3 shading_normal, 
    uint sample_index, uint sample_count,
    in vec3 baseColor, float metallic, float roughness, float reflectance);

void main() {
    vec3 lightDirection = normalize(vec3(1,0.2,1));
    const vec3 lightColor = 8.0 * vec3(1);
    vec3 viewDirection = normalize(u_camera_position - v_PosWCS);
    
    // Type, BaseColor
    vec4 TypeBaseColor = texelFetch(u_materialBuffer, ivec2(0, u_material_index), 0);
    // Index, Metallic, Roughness, Reflectance
    vec4 IMRR = texelFetch(u_materialBuffer, ivec2(1, u_material_index), 0).rgba;
    // r: baseColor, g: metallicRoughness, b: normalmap, a: unused
    vec3 textureProperties = texelFetch(u_materialBuffer, ivec2(2, u_material_index), 0).rgb; // a == -1, for now
    
    float basecolorTextureIndex = textureProperties.r;
    float metallicTextureIndex = textureProperties.g;
    vec3 shadingNormal = computeNormal(viewDirection, int(textureProperties.b));

    vec3 baseColor = basecolorTextureIndex == -1.0? TypeBaseColor.gba : texture(u_texturesBuffer, vec3(v_UV.xy, basecolorTextureIndex)).rgb;
    float ior = IMRR.x;
    float reflectance = convertFromLinearReflectance(IMRR.w); // [0 16%] reflectance (4% == 0.5 value, most common)
    float metallic = metallicTextureIndex == -1.0? IMRR.y : texture(u_texturesBuffer, vec3(v_UV.xy, metallicTextureIndex)).r;
    float roughness = metallicTextureIndex == -1.0? IMRR.z : texture(u_texturesBuffer, vec3(v_UV.xy, metallicTextureIndex)).g;
    roughness = roughness * roughness; // TODO: CHECK. Added to much rendering with blender

    if(TypeBaseColor.x == CT_GGX)
    {
        float NdotL = max(dot(lightDirection, shadingNormal), 0.0);
        vec3 BRDF = evaluateCookTorrance(viewDirection, lightDirection, shadingNormal, baseColor, metallic, roughness, reflectance);
        outColor = vec4(lightColor * BRDF * NdotL, 1.0);
        vec3 illumination = vec3(0);
        uint sample_count = 4u;
        for(uint sample_index = 0u; sample_index < sample_count; sample_index++)
            illumination += sampleEnvCookTorrance(viewDirection, shadingNormal, sample_index, sample_count, baseColor, metallic, roughness, reflectance);
        outColor = vec4(illumination / float(sample_count), 1.0);
        /*path_throughput = sampleCookTorrance(ray_origin, ray_direction, 
            ray_origin, firstBounce_normal.xyz, view_direction, baseColor, metallic, roughness, reflectance);*/
    }
    else if(TypeBaseColor.x == PERFECT_SPECULAR)
    {
      /*path_throughput = samplePerfectSpecular(ray_origin, ray_direction, 
        ray_origin, firstBounce_normal.xyz, view_direction, baseColor);*/
        outColor = vec4(sampleEnvPerfectSpecular(viewDirection, shadingNormal, baseColor), 1.0);
    }
    else if(TypeBaseColor.x == FRESNEL_SPECULAR)
    {
      /*path_throughput = sampleFresnelSpecular(ray_origin, ray_direction, 
        ray_origin, firstBounce_normal.xyz, view_direction, baseColor, IOR_AIR, ior);*/
        outColor = vec4(sampleEnvPerfectSpecular(viewDirection, shadingNormal, baseColor), 1.0);
    }
    else // LAMBERT and Fallback
    {
        float NdotL = max(dot(lightDirection, shadingNormal), 0.0);
        vec3 BRDF = evaluateLambert(baseColor);
        outColor = vec4(lightColor * BRDF * NdotL, 1.0);
      /*path_throughput = sampleLambert(ray_origin, ray_direction, 
        ray_origin, firstBounce_normal.xyz, view_direction, baseColor);*/
    }
    
    outNormal = vec4(0.5 * shadingNormal + 0.5, 0.0);
    outParams = vec4(abs(fract(v_UV)), 1, 0);
    //outParams = vec4(v_UV, 1, 0);
    vec2 uv = fract(v_UV);
    ivec2 iff = ivec2(uv * 4.0);
    vec2 ff = vec2(iff.x < 1? 0.0 : iff.x < 2? 0.3 : iff.x < 3? 0.7 : 1.0, iff.y < 1? 0.0 : iff.y < 2? 0.3 : iff.y < 3? 0.7 : 1.0);
    uv = fract(uv * 4.0);
    outParams = vec4(uv, (float(u_material_index) + 1.5) / 255.0, ff.y);
    outNormal.a = ff.x;
}

vec3 evaluateCookTorrance(in vec3 viewDirection, in vec3 lightDirection,
    in vec3 shadingNormal, in vec3 baseColor, float metallic, float roughness, float reflectance) { 

  vec3  BRDF = vec3(1);
  
  vec3 cdiff = mix(baseColor.rgb * (1.0 - reflectance), vec3(0), metallic);
  vec3 F0 = mix(vec3(reflectance), baseColor.rgb, metallic);

  vec3 halfVector = normalize(lightDirection + viewDirection);
      
  float NdotL = max(dot(lightDirection, shadingNormal), 0.0);
  float NdotH = max(dot(halfVector, shadingNormal), 0.0);
  float VdotH = max(dot(viewDirection, halfVector), 0.0); // same as LdotH
  float NdotV = max(dot(viewDirection, shadingNormal), 0.0);
  
  float GeometryTerm = geometricTermSmithGGX2(NdotV, NdotL, roughness);
  vec3  Fresnel = schlickFresnel(F0, VdotH);
  float Distribution = ggxDistribution(NdotH, roughness);
	
  // metals do not have diffuse
  vec3 specularReflection = (NdotL > 0.0 && NdotV > 0.0)? Fresnel * GeometryTerm * Distribution / (4.0 * NdotL * NdotV) : vec3(0);

  vec3 diffuseReflection = (vec3(1.0) - Fresnel) * cdiff;
  BRDF = diffuseReflection + specularReflection;

  return BRDF;
}

vec3 evaluateLambert(in vec3 baseColor)
{
    return baseColor.rgb / RG_PI;
}

vec3 sampleEnvMap(vec3 dir)
{
    float u = 0.5 + atan(dir.z, dir.x) / (2.0 * RG_PI);
	float v = 0.5 - sin(dir.y) / RG_PI;
	return 2.0*texture(env_map, vec2(u, 1.0 - v)).rgb;
}

vec3 sampleEnvMapLod(vec3 dir, float lod)
{
    float u = 0.5 + atan(dir.z, dir.x) / (2.0 * RG_PI);
	float v = 0.5 - sin(dir.y) / RG_PI;
	return 2.0*textureLod(env_map, vec2(u, 1.0 - v), lod).rgb;
}

vec4 rg_Random(uint index, uint seed0, uint seed1);
vec4 rg_RandomHalton(inout uint index);

// Returns the illumination from the environment map
vec3 sampleEnvPerfectSpecular(in vec3 view_direction, in vec3 shading_normal, in vec3 baseColor)
{
    vec3 light_direction = reflect(-view_direction, shading_normal);		  
    /* BTDF */
    vec3  BRDF = baseColor; // No loss of energy
    return BRDF * sampleEnvMap(light_direction); // compute the path throughput  
}


uint wang(uint v) {
    v = (v ^ 61u) ^ (v >> 16u);
    v *= 9u;
    v ^= v >> 4u;
    v *= 0x27d4eb2du;
    v ^= v >> 15u;
    return v;
}
vec3 sampleEnvCookTorrance(in vec3 view_direction, in vec3 shading_normal, 
    uint sample_index, uint sample_count,
    in vec3 baseColor, float metallic, float roughness, float reflectance)
{
    vec3 tangent, bitangent;
	createONBOpt(shading_normal, tangent, bitangent);

	// Generate random numbers
    uvec2 seed = sample_index == 0u ? uvec2(123231u, 6734762u) : uvec2(31731u,9373u);
    seed = uvec2(wang(sample_index), wang(sample_index + 397u));
    vec2 frame = vec2(512);
	uint counter = sample_count * (uint(gl_FragCoord.x) + uint(gl_FragCoord.y) * uint(frame.x) + 1u) + sample_index;
    //counter = 512u * uint(gl_FragCoord.x) + uint(gl_FragCoord.y);
	//vec4 randoms = rg_Random(counter, seed.x, seed.y);	  
    //vec4 randoms = rg_RandomHalton(counter);
    //vec4 randoms = texture(u_blue_noise_map, vec2(int(gl_FragCoord.x) * int(gl_FragCoord.y)) / frame);
    //vec4 randoms = texture(u_blue_noise_map, rg_Random(counter, seed.x, seed.y).xy);
    //vec4 randoms = texture(u_blue_noise_map, rg_RandomHalton(counter).xy);
    vec4 randoms = texture(u_blue_noise_map, vec2(
        (uint(gl_FragCoord.x) + seed.x) & 127u, 
        (uint(gl_FragCoord.y) + seed.y) & 127u
    ) / 128.0);

	vec3 light_direction;
    /* BTDF */
	vec3  BRDF = vec3(0);

    vec3 cdiff = mix(baseColor.rgb * (1.0 - reflectance), vec3(0), metallic);
	vec3 F0 = mix(vec3(reflectance), baseColor.rgb, metallic);

	//if(randoms.z < 0.5) // evaluate Diffuse
	{      
        // cosine weighted importance sampling
  	    vec3 dir = cosine_sample_hemisphere(randoms.x, randoms.y); // importance sample using the pdf = cosTheta / PI
  	    dir = dir.x * tangent + dir.y * bitangent + dir.z * shading_normal;
  	    dir = normalize(dir);
  
  	    light_direction = dir;  
  	    float NdotL = max(dot(light_direction, shading_normal), 0.0);
    
		vec3 halfVector = normalize(view_direction + light_direction);  
		float NdotH = max(dot(halfVector, shading_normal), 0.0);
  	    float VdotH = max(dot(view_direction, halfVector), 0.0); // same as LdotH
    
		vec3 Fresnel = schlickFresnel(F0, VdotH);
    
		vec3 diffuseReflection = (vec3(1.0) - Fresnel) * cdiff;
      
  	    // BRDF * cosTheta / pdf = (rho / PI) * cosTheta / (cosTheta / PI) = rho
  	    BRDF += diffuseReflection * sampleEnvMapLod(light_direction, 3.0); // weight with the 50% probability
	}
	//else
    {
        // Convert the viewDirection to tangent space
        vec3 vd = vec3(dot(tangent, view_direction), dot(bitangent, view_direction), dot(shading_normal, view_direction));
        vd = normalize(vd);

        // importance sample the GGX VNDF
  	    vec3 halfVector = ImportanceSampleGGX_VNDF(randoms.zw, roughness, vd);
  	    halfVector = halfVector.x * tangent + halfVector.y * bitangent + halfVector.z * shading_normal;
  	    halfVector = normalize(halfVector);
      
  	    light_direction = reflect(-view_direction, halfVector);
    
        // maybe check if NdotL and VdotH are > 0.0
        float NdotL = max(dot(light_direction, shading_normal), 0.0);
  	    float NdotH = max(dot(halfVector, shading_normal), 0.0);
  	    float VdotH = max(dot(view_direction, halfVector), 0.0); // same as LdotH
  	    float NdotV = max(dot(view_direction, shading_normal), 0.0);
  
  	    float GeometryTerm 		 = geometricTermSmithGGX2(NdotV, NdotL, roughness);
        float GeometryTerm1		 = geometricTermSmithGGX1(NdotV, roughness);
  	    vec3 Fresnel 			 = schlickFresnel(F0, VdotH);
	
  	    // metals do not have diffuse
  	    vec3 specularReflection = (NdotL > 0.0 && NdotV > 0.0)? Fresnel * GeometryTerm / (GeometryTerm1) : vec3(0);
    
        // BRDF * cosTheta / pdf = (F * G2 * D / (4 * NdotL * NdotV)) * cosTheta / (pdf) = F * G2 / G1(wo)
  	    BRDF += specularReflection * sampleEnvMap(light_direction); // weight with the 50% probability
    }  

   	return BRDF;
}




/* compute the upper 32 bits of the product of two unsigned 32-bit integers */
void umulExtended_(uint a, uint b, out uint hi, out uint lo) {
    const uint WHALF = 16u;
    const uint LOMASK = (1u<<WHALF)-1u;
    lo = a*b;               /* full low multiply */
    uint ahi = a>>WHALF;
    uint alo = a& LOMASK;
    uint bhi = b>>WHALF;
    uint blo = b& LOMASK;

    uint ahbl = ahi*blo;
    uint albh = alo*bhi;

    uint ahbl_albh = ((ahbl&LOMASK) + (albh&LOMASK));
    hi = ahi*bhi + (ahbl>>WHALF) +  (albh>>WHALF);
    hi += ahbl_albh >> WHALF; /* carry from the sum of lo(ahbl) + lo(albh) ) */
    /* carry from the sum with alo*blo */
    hi += ((lo >> WHALF) < (ahbl_albh&LOMASK)) ? 1u : 0u;
}

uvec2 philox4x32Bumpkey(uvec2 key) {
    uvec2 ret = key;
    ret.x += 0x9E3779B9u;
    ret.y += 0xBB67AE85u;
    return ret;
}

uvec4 philox4x32Round(uvec4 state, uvec2 key) {
    const uint M0 = 0xD2511F53u, M1 = 0xCD9E8D57u;
    uint hi0, lo0, hi1, lo1;
//    umulExtended(M0, state.x, hi0, lo0);
//    umulExtended(M1, state.z, hi1, lo1);
    umulExtended_(M0, state.x, hi0, lo0);
    umulExtended_(M1, state.z, hi1, lo1);

    return uvec4(
        hi1^state.y^key.x, lo1,
        hi0^state.w^key.y, lo0);
}

float uintToFloat(uint src) {
    return uintBitsToFloat(0x3f800000u | (src & 0x7fffffu))-1.0;
}

vec4 uintToFloat(uvec4 src) {
    return vec4(uintToFloat(src.x), uintToFloat(src.y), uintToFloat(src.z), uintToFloat(src.w));
}

vec4
rg_Random(uint index, uint seed0, uint seed1) {
  
  uvec2 key = uvec2( index , seed0 );
  uvec4 ctr = uvec4( 0u , 0xf00dcafeu, 0xdeadbeefu, seed1 ); 

  uvec4 state = ctr;
  uvec2 round_key = key;

  for(int i=0; i<7; ++i) {
    state = philox4x32Round(state, round_key);
    round_key = philox4x32Bumpkey(round_key);
  }

  return uintToFloat(state);
  /*return vec4(
    float(ctr[0]) / float(0xFFFFFFFFu), float(ctr[1]) / float(0xFFFFFFFFu),
    float(ctr[2]) / float(0xFFFFFFFFu), float(ctr[3]) / float(0xFFFFFFFFu)
  );*/
}

float computeHalton(int i, int b)
{
	float f = 1.0;
	float r = 0.0;
			
	int j = i;
	
	while( j > 0)
	{
		f = f / float(b);
		r = r + f * float(j % b);
		//j = int(floor(float(j) / float(b)));
		j = j / b;
	}
	return r;
}

vec4
rg_RandomHalton(inout uint index) {
  int index2 = int(index);
  vec2 s1 = vec2(computeHalton(index2 + 0, 2), computeHalton(index2 + 1, 3));
  vec2 s2 = vec2(computeHalton(index2 + 2, 2), computeHalton(index2 + 3, 3));
  index += 4u;

  return vec4(s1, s2);
}