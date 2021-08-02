#line 0
//#define USE_LAMBERT
//#define USE_GLOSSY
#define USE_FULL

precision highp sampler2D;
precision highp sampler2DArray;
precision highp isampler2D;

layout(location = 0) out vec4 accumulation_OUT;
layout(location = 1) out vec4 origin_OUT;
layout(location = 2) out vec4 direction_OUT;
layout(location = 3) out vec4 payload_OUT;

uniform int ads; 
uniform int sample_index;
uniform int depth;
uniform ivec4 tile;
uniform ivec2 frame;
uniform uvec2 seed;

uniform sampler2D env_map;
uniform sampler2D ray_origins;
uniform sampler2D ray_accumulations;
uniform sampler2D ray_directions;
uniform sampler2D ray_payloads;
uniform isampler2D intersections;
uniform isampler2D occlusions;

uniform sampler2D u_materialBuffer;
uniform sampler2D u_lightBuffer;
uniform int light_count; 
uniform sampler2DArray u_texturesBuffer;

#define WR_PI 3.14159265359
#define WR_2PI 6.28318530718
#define WR_INV_PI 0.31830988618
#define WR_INV_2PI 0.15915494309
#define WR_4PI 12.5663706144

/* Utilities */
vec3 hemisphere_cosine_sample(vec2 u);
bool hemisphere_same(vec3 a, vec3 b, vec3 n);
bool is_zero(vec3 u);
bool is_zerof(float u);
vec4 rg_Random(uint index, uint seed0, uint seed1);

struct onb
{
  vec3 tangent;
  vec3 bitangent;
  vec3 normal;
};

struct event
{
  onb   basis;
  vec3  reflectance;
  vec3  base_color;
  vec3  shading_normal;
  vec3  position;
  float type;
  float alpha;
};

/* Lighting */
struct wr_light
{
  int type;
  vec3 position;
  vec3 power;
};

#define WR_LIGHT_TYPE_POINT 1
#define WR_LIGHT_TYPE_QUAD  2
wr_light Lights_sample(float u, out float pdf);
vec3 Light_sample(const in event evt, const in wr_light light, const in vec2 u, out vec3 EtL, out float pdf, out float dist);

onb create_onb(const in vec3 n)
{
  onb  basis;
  //float sign = copysignf(1.0f, n.z);
  float sign = (n.z >= 0.0) ? 1.0 : -1.0;
  float a = -1.0 / (sign + n.z);
  float b = n.x * n.y * a;
  basis.tangent = vec3(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
  basis.bitangent = vec3(b, sign + n.y * n.y * a, -n.y);
  basis.normal = n;
  return basis;
}

vec3
onb_from(const in onb basis, vec3 v)
{
  return normalize(v.x * basis.tangent + v.y * basis.bitangent + v.z * basis.normal);
}

vec3
onb_to(const in onb basis, vec3 v)
{
  return normalize(vec3(dot(basis.tangent, v), dot(basis.bitangent, v), dot(basis.normal, v)));
}

float hemisphere_cosine_pdf(float costheta);

float ggx_isotropic_ndf(const in event evt, const vec3 h);
vec3  ggx_isotropic_ndf_sample(const in event evt, const vec3 wo, const vec2 u);
float ggx_isotropic_ndf_pdf(const in event evt, const vec3 wo, const vec3 h);

vec3  fresnel_schlick(const in event evt, const float costheta);

float ggx_isotropic_lambda( const in event evt, const vec3 dir);
float ggx_isotropic_geometric(const in event evt, const vec3 wi, const vec3 wo);

float BxDF_pdfD(const in event evt, const in vec3 wo, const in vec3 wi);
float BxDF_pdfG(const in event evt, const in vec3 wo, const in vec3 wi);
vec3  BxDF_evalD(const in event evt, const in vec3 wo, const in vec3 wi);
vec3  BxDF_evalG(const in event evt, const in vec3 wo, const in vec3 wi);
vec3  BxDF_sample(const in event evt, const in vec3 wo, const in vec2 u, out vec3 wi, out float pdf);

void main() {
  vec4 tilef = vec4(tile);
  vec2 framef = vec2(frame);
  vec2 pixel = tilef.xy * tilef.zw + gl_FragCoord.xy;
  
  vec4  direction_IN = texelFetch(ray_directions, ivec2(gl_FragCoord.xy), 0);
  vec4  origin_IN = texelFetch(ray_origins, ivec2(gl_FragCoord.xy), 0);
  vec4  payload_IN = texelFetch(ray_payloads, ivec2(gl_FragCoord.xy), 0);
  vec4  accumulation_IN = texelFetch(ray_accumulations, ivec2(gl_FragCoord.xy), 0);
  ivec4 intersection = texelFetch(intersections, ivec2(gl_FragCoord.xy), 0);
  
  vec3 wo = -direction_IN.xyz;
  vec3 throughput = payload_IN.xyz;
  
  if (direction_IN.w == 0.0)  {
    accumulation_OUT = accumulation_IN;
    direction_OUT    = vec4(0.0);
    origin_OUT       = vec4(0.0);
    payload_OUT      = vec4(0.0);
    return;
  }

  if (intersection.x < 0 && depth == 0)  {
    //vec2 env_coords = vec2( 0.5 + atan(wo.z, wo.x) / (2.0 * WR_PI), 0.5 - sin(wo.y) / WR_PI);
    //accumulation_OUT = texture(env_map, env_coords);
    accumulation_OUT = vec4(0);
    direction_OUT    = vec4(0);
    origin_OUT       = vec4(0);
    payload_OUT      = vec4(0);
    return;
  }

  
  // Generate random numbers   
  uint counter = uint(pixel.x) + uint(pixel.y) * uint(frame.x) + 1u;
  vec4 random_lights = rg_Random((uint(sample_index) * 2u + 0u) * counter, seed.x, seed.y);
  vec4 random_bxdfs  = rg_Random((uint(sample_index) * 2u + 1u) * counter, seed.x, seed.y);

  ivec4 face = wr_GetFace(ads, intersection);
  vec3 geom_normal = wr_GetGeomNormal(ads, intersection);
  vec3 shading_normal = normalize(wr_GetInterpolatedNormal(ads, intersection));
  shading_normal = faceforward(shading_normal, direction_IN.xyz, shading_normal);
  vec2 uv              = wr_GetInterpolatedTexCoords(ads, intersection);
  vec3 origin          = wr_GetInterpolatedPosition(ads, intersection);
  origin = origin_IN.xyz + direction_IN.xyz * wr_GetHitDistance(intersection);
  
  vec4 type_base_color   = texelFetch(u_materialBuffer, ivec2(0,face.w), 0);
  vec3 textureProperties = texelFetch(u_materialBuffer, ivec2(2,face.w), 0).rgb; // a == -1, for now

  int basecolorTextureIndex = int(textureProperties.r);
  int metallicTextureIndex = int(textureProperties.g);
  int normalTextureIndex = int(textureProperties.b);

  type_base_color.rgb = (basecolorTextureIndex == -1) ? type_base_color.rgb : texture(u_texturesBuffer, vec3(uv.xy, basecolorTextureIndex)).rgb;
  vec4 normal_emission = (normalTextureIndex == -1) ? vec4(1,0,0,0) : texture(u_texturesBuffer, vec3(uv.xy, normalTextureIndex));
  vec4 metal_refl_gloss = (metallicTextureIndex == -1) ? vec4(0,0,0.06,0.8) : texture(u_texturesBuffer, vec3(uv.xy, metallicTextureIndex));

  if ( normal_emission.a > 0.0) 
  {
	float flux = 3.0;
	if (face.w==5)
	{
		flux = 30.0;
	}
	else if (face.w==9)
	{
		flux = 2.0;
	}
	vec3 Ld = (throughput * type_base_color.rgb * normal_emission.a * flux);
		
    accumulation_OUT = vec4(accumulation_IN.xyz + Ld, 1.0);
    direction_OUT = vec4(0.0);
    origin_OUT = vec4(0.0);
    payload_OUT = vec4(0.0); 
    return;
  }

  vec3 v0  = wr_GetPosition(ads, face.x);
  vec3 v1  = wr_GetPosition(ads, face.y);
  vec3 v2  = wr_GetPosition(ads, face.z);
  vec2 uv0 = wr_GetTexCoords(ads, face.x);
  vec2 uv1 = wr_GetTexCoords(ads, face.y);
  vec2 uv2 = wr_GetTexCoords(ads, face.z);
  
  vec3 deltaPos1 = v1 - v0;
  vec3 deltaPos2 = v2 - v0;

    // UV delta
  vec2 deltaUV1 = uv1 - uv0;
  vec2 deltaUV2 = uv2 - uv0;

  float r = 1.0 / (deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x);
  r = 0.3;
  vec3 tangent = normalize(deltaPos1 * deltaUV2.y - deltaPos2 * deltaUV1.y)*r;
  vec3 bitangent = normalize(deltaPos2 * deltaUV1.x - deltaPos1 * deltaUV2.x)*r;

  vec3 NN = normal_emission.rgb;
  NN = NN * 2.0 - 1.0;
  NN.y = -NN.y;
  vec3 N = normalize(tangent * NN.x + bitangent * NN.y + shading_normal * NN.z);
  shading_normal = N;

  event surface;
  surface.basis = create_onb(shading_normal);
  surface.base_color = (basecolorTextureIndex == -1) ? type_base_color.gba : texture(u_texturesBuffer, vec3(uv.xy, basecolorTextureIndex)).rgb;
  surface.type = type_base_color.x;
  surface.shading_normal = shading_normal;
  surface.position = origin;
  surface.alpha = 1.0 - metal_refl_gloss.a;//pow(metal_refl_gloss.a,1.2);//*metal_refl_gloss.a;
  //surface.base_color = vec3(1.0);
  surface.reflectance = mix(vec3(0.1),surface.base_color, metal_refl_gloss.r) * vec3(metal_refl_gloss.a);
  surface.base_color *= (1.0-metal_refl_gloss.r);
  //vec3 shading_normal_lcs = onb_to(surface.basis, shading_normal);
  //shading_normal = onb_from(surface.basis, shading_normal_lcs);

  /* Estimate Direct Lighting */
  float light_selection_pdf;
  wr_light light = Lights_sample(random_lights.x, light_selection_pdf);

  float light_pdf;
  float light_distance;
  vec3  wi;
  vec3  Li = Light_sample(surface, light, random_lights.yz, wi, light_pdf, light_distance);
  vec3  Ld = vec3(0);
  vec3 offset_oriign = origin + geom_normal * 0.005;
  /*
  if ( light_pdf > 0.0 && !is_zero(Li)) {
    float scattering_pdf = BxDF_pdf(surface, wo, wi);
    vec3 BxDF = BxDF_eval(surface, wo, wi);

    if(scattering_pdf > 0.0 && !is_zero(BxDF)) {
      float NdL = max(0.001, abs(dot(shading_normal, wi)));
      bool occluded = wr_query_occlusion(ads, offset_oriign, wi, light_distance - 0.05);

      if(!occluded)
        Ld += (throughput * BxDF * NdL * Li) / (light_pdf * light_selection_pdf);
    }
  }
  */
  /* Next Event */
  float scattering_pdf;
  vec3 BxDF = BxDF_sample(surface, wo, random_bxdfs.xy, wi, scattering_pdf);
  if(scattering_pdf == 0.0 || is_zero(BxDF)) {
    accumulation_OUT = accumulation_IN;
    direction_OUT = vec4(0.0);
    origin_OUT = vec4(0.0);
    payload_OUT = vec4(0.0); 
    return;
  }

  float NdL = max(0.001, abs(dot(shading_normal, wi)));

  accumulation_OUT = vec4(accumulation_IN.xyz + Ld, 1.0);
  direction_OUT = vec4(wi, 10000.0);
  origin_OUT = vec4(offset_oriign, 0.0);
  payload_OUT = vec4((throughput * BxDF * NdL) / max(0.001, scattering_pdf), 0); // throughput
}

float BxDF_pdfD(const in event evt, const in vec3 wo, const in vec3 wi)
{
  return hemisphere_same(wo, wi, evt.shading_normal) ? hemisphere_cosine_pdf(abs(dot(evt.shading_normal, wi))) : 0.0;
}

float BxDF_pdfG(const in event evt, const in vec3 wo, const in vec3 wi)
{
  if ( !hemisphere_same(wo, wi, evt.shading_normal) ) return 0.0;

  vec3 wh = normalize(wo + wi);

  float hdo = abs(dot(evt.shading_normal, wh));
  float pdf_wh = ggx_isotropic_ndf_pdf(evt, wo, wh);

	return pdf_wh / (4.0 * hdo);
}


vec3 fresnel_schlick(const in event evt, const float costheta) {
  float v = 1.0 - costheta;
  float v5 = (v * v) * (v * v) * v;
  return clamp(evt.reflectance + (vec3(1) - evt.reflectance) * v5, vec3(0), vec3(1));
}

vec3 BxDF_evalD(const in event evt, const in vec3 wo, const in vec3 wi)
{
	return evt.base_color * WR_INV_PI;
}

vec3 BxDF_evalG(const in event evt, const in vec3 wo, const in vec3 wi)
{
  float ndo = abs(dot(evt.shading_normal, wo));
  float ndi = abs(dot(evt.shading_normal, wi));
  vec3  h = wo + wi;

  if (is_zerof(ndi) || is_zerof(ndo) || is_zero(h)) return vec3(0.0);

  h = normalize(h);

  float hdo = max(dot(wi, h), 0.0); // same as hdi
	float D = ggx_isotropic_ndf(evt, h);
  //vec3  F = fresnel_schlick(evt, hdo);
	float G = ggx_isotropic_geometric(evt, wi, wo);

  return vec3(D * G) / (4.0 * ndo * ndi);
}

vec3 BxDF_sample(const in event evt, const in vec3 wo, const in vec2 u, out vec3 wi, out float pdf) 
{
	if ( is_zerof( dot( wo, evt.shading_normal ) ) ) 
		return vec3(0.0);

	vec2 u_remap = u;
	vec3 wi_wcs;
	
	const float Pg = 0.5;
	
	if (u.x < Pg) 
	{
		vec3 wh_lcs = ggx_isotropic_ndf_sample(evt, wo, u);
		vec3 wh = onb_from(evt.basis, wh_lcs);

		if (dot(wo, wh) < 0.0) return vec3(0.0);

		vec3 wi_wcs = reflect(-wo, wh);
		wi = wi_wcs;
		pdf = BxDF_pdfG(evt, wo, wi_wcs);
		vec3  h = wo + wi;
		h = normalize(h);
		float hdo = max(dot(wi, h), 0.0); // same as hdi
		vec3  F = fresnel_schlick(evt, hdo);
				
		return F*BxDF_evalG(evt, wo, wi_wcs)/Pg;
	} 
	else 
	{
		vec3 wi_lcs = hemisphere_cosine_sample(u);
		vec3 wi_wcs = onb_from(evt.basis, wi_lcs);
		pdf = BxDF_pdfD(evt, wo, wi_wcs);
		wi = wi_wcs;
		vec3  h = wo + wi;
		h = normalize(h);
		float hdo = max(dot(wi, h), 0.0); // same as hdi
		vec3  F = fresnel_schlick(evt, hdo);
				
		return (vec3(1)-F)*BxDF_evalD(evt, wo, wi_wcs)/(1.0-Pg);
	}
}

vec3 ggx_isotropic_ndf_sample(const in event evt, const vec3 wo, const vec2 u) {
  float alphaUV = evt.alpha;
  float roughness = alphaUV;
  float phi = WR_2PI * u.y;
  float tantheta2 = roughness * roughness * u.x / (1.0 - u.x);
  float costheta = 1.0 / sqrt(1.0 + tantheta2);
  float sintheta = sqrt(max(0.0, 1.0 - costheta * costheta));
  vec3  h = vec3(sintheta * cos(phi), sintheta * sin(phi), costheta);

	return h;
}

float ggx_isotropic_ndf(const in event evt, const vec3 h) {
  float costheta = dot(evt.shading_normal, h);
  float costheta2 = costheta * costheta;
  float sintheta2 = 1.0 - costheta2;
  float sintheta  = sqrt(sintheta2);
  float tantheta2 = sintheta2 / costheta2;

  if (isinf(tantheta2)) return 0.f;

  float costheta4 = costheta2 * costheta2;
  float alphaUV = evt.alpha;
  float a2 = alphaUV * alphaUV;
 
  float sinphi = (sintheta == 0.0) ? 1.0 : clamp(h.y / sintheta, -1.0, 1.0);
  float sinphi2 = sinphi * sinphi;
  float cosphi2 = 1.0 - sinphi2;

  float e = (cosphi2 / a2) 
          + (sinphi2 / a2) * tantheta2;

  return 1.0 / (WR_PI * a2 * costheta4 * (1.0 + e) * (1.0 + e));
}

float ggx_isotropic_ndf_pdf(const in event evt, const vec3 wo, const vec3 h) {
  float D = ggx_isotropic_ndf(evt, h);
	return D * abs(dot(h, evt.shading_normal));
}

float ggx_isotropic_lambda( const in event evt, const vec3 dir) {
  float costheta = dot(evt.shading_normal, dir);
  float sintheta = sqrt(1.0 - costheta * costheta);
  float tantheta = abs(sintheta / costheta);
  float alpha = evt.alpha;

  if (isinf(tantheta)) return 0.0;

  float alphaUV = alpha;
  float a2 = alphaUV * alphaUV;

  float sinphi = (sintheta == 0.0) ? 1.0 : clamp(dir.y / sintheta, -1.0, 1.0);
  float sinphi2 = sinphi * sinphi;
  float cosphi2 = 1.0 - sinphi2;

  float roughness = sqrt(cosphi2 * a2 + sinphi2 * a2);

  float alpha2Tan2Theta = (alpha * tantheta) * (alpha * tantheta);

  return (-1.0 + sqrt(1.0 + alpha2Tan2Theta)) / 2.0;
}

float ggx_isotropic_geometric(const in event evt, const vec3 wi, const vec3 wo) {
  return 1.0 / (1.0 +
         ggx_isotropic_lambda(evt, wi) + 
         ggx_isotropic_lambda(evt, wo));
}

bool is_zerof(float u) {
  return (u == 0.0);
}

bool is_zero(vec3 u) {
  return (dot(u,u) == 0.0);
}

bool hemisphere_same(vec3 a, vec3 b, vec3 n) {
  return (dot(a, n) * dot(b, n) > 0.0);
}

float hemisphere_cosine_pdf(float costheta) {
  return costheta * WR_INV_PI;
}

vec3 hemisphere_cosine_sample(vec2 u) {
  // Uniformly sample disk.
  float r = sqrt(u.x);
  float phi = 2.0 * WR_PI * u.y;
  vec3 p;
  p.x = r * cos(phi);
  p.y = r * sin(phi);

  // Project up to hemisphere.
  p.z = sqrt(max(0.0, 1.0 - p.x*p.x - p.y*p.y));
  return p;
}

wr_light Lights_sample(float u, out float pdf) {
  int light_index = clamp(int(u * float(light_count)), 0, light_count - 1);
  //int light_index = int(u * float(light_count));
  //int light_index = sample_index % light_count;
  vec4 position_type = texelFetch(u_lightBuffer, ivec2(0,light_index), 0);
  vec4 power_count   = texelFetch(u_lightBuffer, ivec2(1,light_index), 0);
  wr_light light;
  light.type = int(position_type.w);

  if (light.type == WR_LIGHT_TYPE_POINT) {
    light.position = position_type.xyz;
    light.power = power_count.xyz; 
    pdf = 1.0 / float(light_count);
  } else {
    pdf = 0.000001;
  }

  return light;
}

vec3 Light_sample(const in event evt, const in wr_light light, const in vec2 u, out vec3 EtL, out float pdf, out float dist) {
  dist = 0.0;
  pdf = 0.0;
  EtL = vec3(0);
  if (light.type == WR_LIGHT_TYPE_POINT) {
    vec3 light_direction = light.position - evt.position; 
    float light_dist = length(light_direction);
    float light_dist2 = light_dist * light_dist;
    pdf = 1.0;
    dist = light_dist;
    EtL = normalize(light_direction);
    return (light.power / WR_4PI) / max(0.001, light_dist2);
  }
  return vec3(100,100,100);
}

#define m4x32_0 0xD2511F53u
#define m4x32_1 0xCD9E8D57u
#define w32_0 0x9E3779B9u
#define w32_1 0xBB67AE85u

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

uvec4 philox4x32_7(uvec4 plain, uvec2 key) {
    uvec4 state = plain;
    uvec2 round_key = key;

    for(int i=0; i<7; ++i) {
        state = philox4x32Round(state, round_key);
        round_key = philox4x32Bumpkey(round_key);
    }

    return state;
}

float uintToFloat(uint src) {
    return uintBitsToFloat(0x3f800000u | (src & 0x7fffffu))-1.0;
}

vec4 uintToFloat(uvec4 src) {
    return vec4(uintToFloat(src.x), uintToFloat(src.y), uintToFloat(src.z), uintToFloat(src.w));
}

vec4
rg_Random(uint index, uint seed0, uint seed1) {
  //philox4x32_key_t k = {{tid, 0xdecafbad}};
  //philox4x32_ctr_t c = {{0, 0xf00dcafe, 0xdeadbeef, 0xbeeff00d}};

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
