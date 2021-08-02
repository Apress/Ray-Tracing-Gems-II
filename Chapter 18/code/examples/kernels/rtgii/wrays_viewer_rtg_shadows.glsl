#line 0
//precision highp float;
precision highp sampler2D;
precision highp isampler2D;

layout(location = 0) out vec4 ray_accumulation_OUT;

uniform int ads; /* Always zero for now */
uniform ivec4 tile;
uniform ivec2 frame;
uniform uvec2 seed;
#define MAX_DEPTH 0
uint sample_count = uint(MAX_DEPTH + 1);
uniform uint frame_count;

uniform sampler2D env_map;
uniform sampler2D u_blue_noise_map;

uniform sampler2D u_materialBuffer;
uniform sampler2DArray u_texturesBuffer;

struct LightData {
  vec4 a;
  vec4 b;
};

layout(std140) uniform u_lightsArray {
  LightData lights[8];
};

int GetLightType(LightData d)
{
  return int(floatBitsToUint(d.a.x)) & 0xFF;
}

vec3 GetLightColor(LightData d)
{
  uint colorI = (floatBitsToUint(d.a.x) >> 8u) & 0xFFFFFFu;
  uint red = (colorI ) & 0xFFu;
  uint green = (colorI >> 8) & 0xFFu;
  uint blue = (colorI >> 16) & 0xFFu;
    
  return vec3(vec3(red, green, blue) / 255.0);
}

float GetLightIntensity(LightData d)
{
  return d.b.w;
}

vec3 GetLightDirection(LightData d)
{
  return d.a.yzw;
}

vec3 GetLightPosition(LightData d)
{
  return d.a.yzw;
}

#define RG_PI 3.14159

#define PERFECT_SPECULAR 0.0
#define FRESNEL_SPECULAR 1.0
#define CT_GGX 2.0
#define LAMBERT 3.0
#define DISNEY 4.0
#define THICK_GLASS 5.0

#define IOR_AIR 1.000293
#define IOR_GLASS 1.52

#define POSITION_OFFSET 0.01

const float epsilon	= 1e-7;

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

vec4 rg_Random(uint index, uint seed0, uint seed1);
vec4 rg_RandomHalton(inout uint index);

// for microfacet BRDFs. Else NdotL is required
vec3 schlickFresnel(vec3 F0, float LdotH)
{
    float power = 1.0 - LdotH;
    float power2 = power * power;
    float power5 = power2 * power2 * power;
    return F0 + (1.0 - F0) * power5;       
}

float schlickFresnel(float F0, float LdotH)
{
	float power = 1.0 - LdotH;
	float power2 = power * power;
	float power5 = power2 * power2 * power;
	return F0 + (1.0 - F0) * power5;
}

float schlickFresnel(float F0, float F90, float LdotH)
{
	float power = 1.0 - LdotH;
	float power2 = power * power;
	float power5 = power2 * power2 * power;
	return F0 + (F90 - F0) * power5;
}

float dielectricFresnel(float etaI, float etaT, float cosThetaI, float cosThetaT)
{
    float sinThetaI = sqrt(max(0.0, 1.0 - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;
	  //if(sinThetaT >  0.0) // not used since we compute this equation when we have a transmission
	  //	return 1.0;
    cosThetaT = sqrt(max(0.0, 1.0 - sinThetaT * sinThetaT));
    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
                  ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
                  ((etaI * cosThetaI) + (etaT * cosThetaT));
    return (Rparl * Rparl + Rperp * Rperp) / 2.0;
}

/// Distributions

// Trowbridge-Reitz (GGX) NDF isotropic distribution.
float ggxDistribution(float NdotH, float roughness)
{
    float a2 = roughness * roughness;
    float denominator = NdotH * NdotH * (a2 - 1.0) + 1.0;

    return a2 / (RG_PI * denominator * denominator);
}

float blinnPhongDistribution(float NdotH, float roughness)
{
    return (roughness + 2.0) * pow(NdotH, roughness) / (2.0 * RG_PI);
}

// Geometric Terms

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

// G / (4 * NdotV * NdotL)
float smithJointGeometricOcclusionGGX(float NdotL, float NdotV, float roughness)
{
    float a2 = roughness * roughness;
    float denominator1 = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
    float denominator2 = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
    return 0.5 / (denominator1 + denominator2);
}

/// Diffuse
vec3 lambertDiffuse(vec3 diffuseColor)
{
    return diffuseColor / RG_PI;
}

vec3 burleyDiffuse(vec3 diffuseColor, float roughness, float NdotV, float NdotL, float LdotH)
{
  float f90 = 0.5 + 2.0 * roughness * LdotH * LdotH;
	float lightScatter = schlickFresnel(1.0, f90, NdotL);
	float viewScatter = schlickFresnel(1.0, f90, NdotV);
	
	return diffuseColor * lightScatter * viewScatter / RG_PI;
}

// Others
float convertFromLinearReflectance(float refl)
{
  return 0.16 * refl * refl;
}

vec3 computeCookTorrance(vec3 diffuseColor, vec3 F0, float roughness, float NdotL, float NdotV, float NdotH)
{
    // reflection is only defined for light and view directions above the surface
    if(NdotL == 0.0)
        return vec3(0.0);
    if(NdotV == 0.0)
        return vec3(0.0);

    vec3 Fresnel = schlickFresnel(F0, NdotL);
    float GeometricTerm = geometricTermSmithGGX2(NdotV, NdotL, roughness);
    float Distribution = ggxDistribution(NdotH, roughness);
    
    vec3 specular =  (Fresnel * GeometricTerm * Distribution) / (4.0 * NdotL * NdotV);    
    //vec3 specular = Fresnel * smithJointGeometricOcclusionGGX(NdotL, NdotV, roughness) * Distribution;
    vec3 diffuse = lambertDiffuse(diffuseColor);
    return (1.0 - Fresnel) * diffuse + specular;
}

/// Evaluate BSDFs
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


// Sample BSDFs

// returns pathThroughput
vec3 sampleCookTorrance(inout vec3 ray_origin, out vec3 ray_direction, vec4 randoms,
vec3 position, vec3 shading_normal, vec3 view_direction, 
vec3 baseColor, float metallic, float roughness, float reflectance)
{
  vec3 geom_normal = shading_normal;
  vec3 tangent, bitangent;
	createONBOpt(shading_normal, tangent, bitangent);

	vec3 light_direction;
  /* BTDF */
	vec3  BRDF = vec3(1);

  vec3 cdiff = mix(baseColor.rgb * (1.0 - reflectance), vec3(0), metallic);
	vec3 F0 = mix(vec3(reflectance), baseColor.rgb, metallic);

  float diffuse_prob = clamp(1.0-metallic, 0.1, 0.9);
  diffuse_prob = 0.5;

	if(randoms.z < diffuse_prob) // evaluate Diffuse
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
  	BRDF = diffuseReflection / diffuse_prob; // weight with the 50% probability
	}
	else
  {
    // Convert the viewDirection to tangent space
    vec3 vd = vec3(dot(tangent, view_direction), dot(bitangent, view_direction), dot(shading_normal, view_direction));
    vd = normalize(vd);

    // importance sample the GGX VNDF
  	vec3 halfVector = ImportanceSampleGGX_VNDF(randoms.xy, roughness, vd);
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
  	vec3 Fresnel 			     = schlickFresnel(F0, VdotH);
	
  	// metals do not have diffuse
  	vec3 specularReflection = (NdotL > 0.0 && NdotV > 0.0)? Fresnel * GeometryTerm / (GeometryTerm1) : vec3(0);
    
    // BRDF * cosTheta / pdf = (F * G2 * D / (4 * NdotL * NdotV)) * cosTheta / (pdf) = F * G2 / G1(wo)
  	BRDF = specularReflection / (1.0 - diffuse_prob); // weight with the 50% probability
  }  

  ray_origin = position + geom_normal * POSITION_OFFSET;		
  ray_direction = light_direction;  
	return BRDF; // compute the path throughput  
}

// returns pathThroughput
vec3 samplePerfectSpecular(inout vec3 ray_origin, out vec3 ray_direction, 
vec3 position, vec3 shading_normal, vec3 view_direction, 
vec3 baseColor)
{
  vec3 geom_normal = shading_normal;
  vec3 light_direction = reflect(-view_direction, shading_normal);
		  
	/* BTDF */
	vec3  BRDF = baseColor; // No loss of energy

  ray_origin = position + geom_normal * 0.001;		
  ray_direction = light_direction;  
	return BRDF; // compute the path throughput  
}

// returns pathThroughput
vec3 sampleLambert(inout vec3 ray_origin, out vec3 ray_direction, vec2 randoms,
vec3 position, vec3 shading_normal, vec3 view_direction, 
vec3 baseColor)
{
  vec3 geom_normal = shading_normal;
  vec3 tangent, bitangent;
	createONBOpt(shading_normal, tangent, bitangent);

	// cosine weighted importance sampling
  vec3 dir = cosine_sample_hemisphere(randoms.x, randoms.y); // importance sample using the pdf = cosTheta / PI
  dir = dir.x * tangent + dir.y * bitangent + dir.z * shading_normal;
  vec3 light_direction = normalize(dir);

  /* BTDF */    
  // BRDF * cosTheta / pdf = (rho / PI) * cosTheta / (cosTheta / PI) = rho  
	vec3  BRDF = baseColor.rgb;
	
  ray_origin = position + geom_normal * POSITION_OFFSET;		
  ray_direction = light_direction;  
	return BRDF; // compute the path throughput  
}

// returns pathThroughput (Both Reflection and Refraction) 
vec3 sampleFresnelSpecular(inout vec3 ray_origin, out vec3 ray_direction, 
vec3 position, vec3 shading_normal, vec3 view_direction, 
vec3 baseColor, float etaI, float etaT)
{
  vec3 geom_normal = shading_normal;

  bool backfacing = dot(geom_normal, ray_direction) > 0.0;
  float eta_ratio = etaI / etaT;
  vec3 ff_normal = (backfacing) ? -shading_normal : shading_normal;
  eta_ratio = (backfacing) ? 1.0 / eta_ratio : eta_ratio; // (glass to air) else (air to glass)
  vec3 refractionVector = refract(ray_direction, ff_normal, eta_ratio);
  float Fr = 0.0;
  if(refractionVector.x == 0.0 && refractionVector.y == 0.0 && refractionVector.z == 0.0)
  {
    refractionVector = reflect(ray_direction, ff_normal); // internal reflection
  }
  else
  { 
    Fr = dielectricFresnel(etaI, etaT, -dot(ff_normal, ray_direction), -dot(ff_normal, refractionVector));
  }
  vec3 light_direction = normalize(refractionVector);

  /* BTDF */
  vec3  BRDF = baseColor * (1.0 - Fr);
	if(dot(ff_normal, light_direction) > 0.0) // reflection
		ray_origin = position + POSITION_OFFSET * ff_normal;
	else // refraction
		ray_origin = position - POSITION_OFFSET * ff_normal;
  ray_direction = light_direction;

  return BRDF; // compute the path throughput  
}

vec3 sampleEnvMap(vec3 dir)
{
  float u = 0.5 + atan(dir.z, dir.x) / (2.0 * RG_PI);
	float v = 0.5 - sin(dir.y) / RG_PI;
  return 0.0 * texture(env_map, vec2(u, 1.0 - v)).rgb;
}

vec3 sampleEnvMapNoScale(vec3 dir)
{
  float u = 0.5 + atan(dir.z, dir.x) / (2.0 * RG_PI);
	float v = 0.5 - sin(dir.y) / RG_PI;
	return 0.1*texture(env_map, vec2(u, 1.0 - v)).rgb;
}

// theta [0 pi], phi[0 2pi]
vec3 getDirectionFromPhiTheta(float _phi, float _theta)
{
  float theta = radians(_theta);
  float phi = radians(_phi);
  return vec3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
}

vec3 computeDirectionalLightContribution(vec3 pos, vec3 geom_normal, out vec3 light_direction)
{
  // for fireplace (90, 70);
  //const vec3 light_direction = normalize(vec3(1,1.2,-1));
  //light_direction = getDirectionFromPhiTheta(-110.0, 40.0); // areashadows3
  light_direction = getDirectionFromPhiTheta(90.0, 80.0); // areashadows3
  const float light_intensity = 10.5;
  const vec3 light_color = vec3(1,1,1);
  vec3 directLight = vec3(0);
  {
    // generate random direction on the cone
    vec2 randoms = texture(u_blue_noise_map, vec2(
        (uint(gl_FragCoord.x) + seed.x) & 127u, 
        (uint(gl_FragCoord.y) + seed.x) & 127u
    ) / 128.0).xy;

    vec3 tangent, bitangent;
	  createONBOpt(light_direction, tangent, bitangent);
    float coneCosineX = cos(10.0 * 0.0174532925);
    const float coneCosine10 = 0.9848;
    const float coneCosine6 = 0.995;
    const float coneCosine3 = 0.99862953;    
    float coneCosine = coneCosine3;
    float square_root = sqrt(1.0 - pow(1.0 - randoms.y * (1.0 - coneCosine), 2.0));
    vec3 dir = vec3(
      cos(2.0 * RG_PI * randoms.x) * square_root,
      sin(2.0 * RG_PI * randoms.x) * square_root,
      1.0 - randoms.y * (1.0 - coneCosine)
    );
    dir = dir.x * tangent + dir.y * bitangent + dir.z * light_direction;
    //dir = light_direction;

    float occluded = 1.0 - float(wr_query_occlusion(0, pos + POSITION_OFFSET * geom_normal, dir, 10000.0));
    float NdotL = max(0.0, dot(light_direction, geom_normal));
    directLight = occluded * NdotL * light_intensity * light_color;
  }
  return 0.2*directLight;
}

vec3 computeAreaLightQuadContribution(vec3 pos, vec3 geom_normal, out vec3 light_direction)
{
  vec3 directLight = vec3(0);
  {
    const float light_intensity = 2.5;
    // generate random direction on the cone
    vec2 randoms = texture(u_blue_noise_map, vec2(
        (uint(gl_FragCoord.x) + seed.x) & 127u, 
        (uint(gl_FragCoord.y) + seed.x) & 127u
    ) / 128.0).xy;
    uint counter = sample_count * (uint(gl_FragCoord.x) + uint(gl_FragCoord.y) * uint(frame.x) + 1u) + 0u;
    randoms = rg_Random(counter, seed.x, seed.y).xy;

    //randoms.xy = vec2(0.5);
    randoms.x = randoms.x * 0.5 + 0.5;
    //randoms.x = 0.6;
    //randoms.y = 0.5;

    vec3 light_color = 8.0*texture(u_texturesBuffer, vec3(randoms.xy, 2)).rgb;
    //light_color = vec3(1,1,1);

    vec3 light_pos = vec3(-5.1, 1, 0);
    light_pos.yz += (2.0 * randoms.xy - 1.0) * vec2(1, 1.25);

    vec3 dir = normalize(light_pos - pos);
    light_direction = dir;    

    float occluded = 1.0 - float(wr_query_occlusion(0, pos + POSITION_OFFSET * geom_normal.rgb, dir, 10000.0));
    float NdotL = max(0.0, dot(light_direction, geom_normal));
    directLight = occluded * NdotL * light_intensity * light_color;
  }
  return directLight;
}

vec3 computeAreaLightQuadContribution2(vec3 pos, vec3 geom_normal, out vec3 light_direction)
{
  /*if(GetLightType(lights[1]) != 5)
    return vec3(0.0);*/
  
  vec3 directLight = vec3(0);
  float light_intensity = 2.5 * GetLightIntensity(lights[1]);
  // generate random direction on the cone
  vec2 randoms = texture(u_blue_noise_map, vec2(
      (uint(gl_FragCoord.x) + seed.x) & 127u, 
      (uint(gl_FragCoord.y) + seed.x) & 127u
  ) / 128.0).xy;
  uint counter = sample_count * (uint(gl_FragCoord.x) + uint(gl_FragCoord.y) * uint(frame.x) + 1u) + 0u;
  randoms = rg_Random(counter, seed.x, seed.y).xy;

  vec3 light_color = 8.0 * texture(u_texturesBuffer, vec3(randoms.xy, 2)).rgb;
  //light_color = vec3(1,1,1);

  vec3 light_pos = vec3(-5.1, 1, 0);
  light_pos.yz += (2.0 * randoms.xy - 1.0) * vec2(1, 1.25);

  vec3 dir = normalize(light_pos - pos);
  light_direction = dir;    

  float occluded = 1.0 - float(wr_query_occlusion(0, pos + POSITION_OFFSET * geom_normal.rgb, dir, 10000.0));
  float NdotL = max(0.0, dot(light_direction, geom_normal));
  directLight = occluded * NdotL * light_intensity * light_color;

  return directLight;
}

vec3 computeDirectLightContribution(vec3 pos, vec3 geom_normal, out vec3 light_direction)
{
  vec3 directLight = computeDirectionalLightContribution(pos, geom_normal, light_direction);  
  //vec3 directLight = computeAreaLightQuadContribution2(pos, geom_normal, light_direction);
  //directLight *= 0.0;
  
  return directLight;
}


/* Camera */
uniform vec3 camera_pos;

uniform mat4 u_ProjectionInvMatrix;
uniform mat4 u_ViewInvMatrix;
uniform sampler2D u_gBuffer_depth;
uniform sampler2D u_gBuffer_material;
uniform sampler2D u_gBuffer_normal;

vec3 reconstruct_position_from_depth(vec2 uv)
{
	vec4 pndc = vec4(2.0 * vec3(uv.xy, texture(u_gBuffer_depth, uv.xy).r) - 1.0, 1.0);
  vec4 pecs = (u_ProjectionInvMatrix) * pndc;
	pecs = u_ViewInvMatrix * pecs;
	pecs.xyz = pecs.xyz/pecs.w;
	return pecs.xyz;
}

void main() {

  // Gather data from the First Pass (G Buffer)
  vec4 firstBounce_mat = texelFetch(u_gBuffer_material, ivec2(gl_FragCoord.xy), 0);
  int material_index = int(firstBounce_mat.b * 255.0) - 1;
  vec4 firstBounce_normal = texelFetch(u_gBuffer_normal, ivec2(gl_FragCoord.xy), 0);
  firstBounce_normal.xyz = 2.0 * firstBounce_normal.xyz - 1.0;
  firstBounce_mat.x = firstBounce_mat.x / 4.0 + firstBounce_normal.a * (3.0 / 4.0);
  firstBounce_mat.y = firstBounce_mat.y / 4.0 + firstBounce_mat.a * (3.0 / 4.0);
    
  vec3 ray_origin = reconstruct_position_from_depth(vec2(gl_FragCoord.xy) / vec2(frame));
  vec3 ray_direction = normalize(ray_origin - camera_pos); 

  // first hit is miss
  if(firstBounce_mat.b == 0.0)
  {
	  ray_accumulation_OUT.rgb = 0.5*sampleEnvMapNoScale(ray_direction.xyz);
    ray_accumulation_OUT.a = 1.0 / float(frame_count);
    return;
  }

  // Print only the normals
  /*{
    //vec3 shading_normal = wr_GetInterpolatedNormal(ads, intersection);
    vec3 shading_normal = firstBounce_normal.rgb;
    ray_accumulation_OUT = shading_normal.rgbr;
    ray_accumulation_OUT.a = 1.0 / float(frame_count);
    return;
  }*/
  
  vec3 light_direction = getDirectionFromPhiTheta(290.0, 40.0);
  vec3 directLight = computeDirectLightContribution(ray_origin, firstBounce_normal.xyz, light_direction);

  // Generate random numbers
	uint pixel_seed = uint(MAX_DEPTH + 1) * (uint(gl_FragCoord.x) + uint(gl_FragCoord.y) * uint(frame.x) + 1u) + 0u;
  vec3 path_throughput = vec3(1);
  {
    // Type, BaseColor
    vec4 TypeBaseColor = texelFetch(u_materialBuffer, ivec2(0,material_index), 0);
    // Index, Metallic, Roughness, Reflectance
    vec4 IMRR = texelFetch(u_materialBuffer, ivec2(1,material_index), 0).rgba;
    // r: baseColor, g: metallicRoughness, b: normalmap, a: unused
    vec3 textureProperties = texelFetch(u_materialBuffer, ivec2(2,material_index), 0).rgb; // a == -1, for now
    
    float basecolorTextureIndex = textureProperties.r;
    float metallicTextureIndex = textureProperties.g;

    vec3 baseColor = basecolorTextureIndex == -1.0? TypeBaseColor.gba : pow(texture(u_texturesBuffer, vec3(firstBounce_mat.xy, basecolorTextureIndex)).rgb, vec3(2.2));
    float ior = IMRR.x;
    float reflectance = convertFromLinearReflectance(IMRR.w); // [0 16%] reflectance (4% == 0.5 value, most common)
    float metallic = metallicTextureIndex == -1.0? IMRR.y : texture(u_texturesBuffer, vec3(firstBounce_mat.xy, metallicTextureIndex)).r;
    float roughness = metallicTextureIndex == -1.0? IMRR.z : texture(u_texturesBuffer, vec3(firstBounce_mat.xy, metallicTextureIndex)).g;
    roughness = roughness * roughness; // TODO: CHECK. Added to match rendering with blender

    // Generate random numbers
    vec4 randoms = rg_Random(pixel_seed, seed.x, seed.y);
    pixel_seed++;
    //pixel_seed = pixel_seed % 125u; 
    ///vec4 randoms = rg_RandomHalton(pixel_seed);
    /*vec4 randoms = texture(u_blue_noise_map, vec2(
        (uint(gl_FragCoord.x) + seed.x) & 127u, 
        (uint(gl_FragCoord.y) + seed.x) & 127u
    ) / 128.0);*/
    if(false)
    {
      randoms = texture(u_blue_noise_map, vec2(
        (uint(gl_FragCoord.x) + seed.x) & 127u, 
        (uint(gl_FragCoord.y) + seed.y) & 127u
      ) / 128.0);
      //randoms = 0.5 * randoms + 0.25;
      //randoms.z = 0.6;
    }
            
    vec3 view_direction = -ray_direction;
    if(TypeBaseColor.x == CT_GGX)
    {      
      // evaluate Direct Light
      directLight = directLight * evaluateCookTorrance(view_direction, light_direction,
        firstBounce_normal.xyz, baseColor, metallic, roughness, reflectance); 

      // sample Indirect Path 
      path_throughput = sampleCookTorrance(ray_origin, ray_direction, randoms,
        ray_origin, firstBounce_normal.xyz, view_direction, baseColor, metallic, roughness, reflectance);      
    }
    else if(TypeBaseColor.x == PERFECT_SPECULAR)
    {
      path_throughput = samplePerfectSpecular(ray_origin, ray_direction, 
        ray_origin, firstBounce_normal.xyz, view_direction, baseColor);
    }
    else if(TypeBaseColor.x == FRESNEL_SPECULAR)
    {
      path_throughput = sampleFresnelSpecular(ray_origin, ray_direction, 
        ray_origin, firstBounce_normal.xyz, view_direction, baseColor, IOR_AIR, ior);
    }
    else // LAMBERT and Fallback
    {
      // evaluate Direct Light
      directLight = directLight * lambertDiffuse(baseColor);

      // sample Indirect Path 
      path_throughput = sampleLambert(ray_origin, ray_direction, randoms.xy,
        ray_origin, firstBounce_normal.xyz, view_direction, baseColor);
    }
  }

  // Cast a Ray
  ivec4 intersection = wr_query_intersection(0, ray_origin, ray_direction.xyz, 10000.0); // or 1.e27
  // If we miss sample the Environment map
  if(intersection.x < 0)
  {
	  ray_accumulation_OUT.rgb = directLight + path_throughput * sampleEnvMap(ray_direction.xyz);
    ray_accumulation_OUT.a = 1.0 / float(frame_count);
    return;
  }

  vec3 color = directLight;//vec3(0,0,0);  
  for(int depth = 0; depth < MAX_DEPTH; depth++)
  {
	  int materialID = wr_GetIndices(ads, intersection.x).w;
	  vec3 geom_normal = wr_GetGeomNormal(ads, intersection);
	  vec3 position = wr_GetInterpolatedPosition(ads, intersection);
	  //vec3 position = ray_origin.xyz + ray_direction.xyz * hit_distance;
	  vec3 shading_normal = wr_GetInterpolatedNormal(ads, intersection);
    vec2 uv = wr_GetInterpolatedTexCoords(ads, intersection);

    // Type, BaseColor
    vec4 TypeBaseColor = texelFetch(u_materialBuffer, ivec2(0,materialID), 0);
    // Index, Metallic, Roughness, Reflectance
    vec4 IMRR = texelFetch(u_materialBuffer, ivec2(1,materialID), 0).rgba;
    // r: baseColor, g: metallicRoughness, b: normalmap, a: unused
    vec3 textureProperties = texelFetch(u_materialBuffer, ivec2(2,materialID), 0).rgb; // a == -1, for now
    
    float basecolorTextureIndex = textureProperties.r;
    float metallicTextureIndex = textureProperties.g;

    vec3 baseColor = basecolorTextureIndex == -1.0? TypeBaseColor.gba : pow(texture(u_texturesBuffer, vec3(uv.xy, basecolorTextureIndex)).rgb, vec3(2.2));
    float ior = IMRR.x;
    float reflectance = convertFromLinearReflectance(IMRR.w); // [0 16%] reflectance (4% == 0.5 value, most common)
    float metallic = metallicTextureIndex == -1.0? IMRR.y : texture(u_texturesBuffer, vec3(uv.xy, metallicTextureIndex)).r;
    float roughness = metallicTextureIndex == -1.0? IMRR.z : texture(u_texturesBuffer, vec3(uv.xy, metallicTextureIndex)).g;

    // Generate random numbers
	  vec4 randoms = rg_Random(pixel_seed, seed.x, seed.y);
    pixel_seed++;
    //vec4 randoms = rg_RandomHalton(pixel_seed);
        
    vec3 view_direction = -ray_direction;
    if(TypeBaseColor.x == CT_GGX)
    {
      // evaluate Direct Light
      directLight = computeDirectLightContribution(position, shading_normal.xyz, light_direction);  
      color += path_throughput * directLight * evaluateCookTorrance(view_direction, light_direction,
        shading_normal.xyz, baseColor, metallic, roughness, reflectance); 
      
      // sample Indirect path
      path_throughput *= sampleCookTorrance(ray_origin, ray_direction, randoms,
        position, shading_normal.xyz, view_direction, baseColor, metallic, roughness, reflectance);      
    }
    else if(TypeBaseColor.x == PERFECT_SPECULAR)
      // No direct Light

      // Sample Indirect path
      path_throughput *= samplePerfectSpecular(ray_origin, ray_direction, 
        position, shading_normal.xyz, view_direction, baseColor);
    else if(TypeBaseColor.x == FRESNEL_SPECULAR)
      // No direct Light

      // Sample Indirect path
      path_throughput *= sampleFresnelSpecular(ray_origin, ray_direction, 
        position, shading_normal.xyz, view_direction, baseColor, IOR_AIR, ior);
    else // LAMBERT and Fallback
    {
      // evaluate Direct Light
      directLight = computeDirectLightContribution(position, shading_normal.xyz, light_direction);  
      color += path_throughput * directLight * lambertDiffuse(baseColor);

      // sample Indirect Path 
      path_throughput *= sampleLambert(ray_origin, ray_direction, randoms.xy,
        position, shading_normal.xyz, view_direction, baseColor);      
    }
    
	  intersection = wr_query_intersection(0, ray_origin, ray_direction.xyz, 10000.0);

    // On miss add contribution from the Sky Map
    if (intersection.x < 0) {
		  color += path_throughput * sampleEnvMap(ray_direction.xyz).rgb;
      break;
	  }
  } 
  ray_accumulation_OUT = vec4(color, 1.0 / float(frame_count));
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

