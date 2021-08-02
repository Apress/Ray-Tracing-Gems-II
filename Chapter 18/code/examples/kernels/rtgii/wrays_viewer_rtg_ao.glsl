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

#define RG_PI 3.14159

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

/* Camera */
uniform vec3 camera_pos;

uniform mat4 u_ProjectionInvMatrix;
uniform mat4 u_ViewInvMatrix;
uniform sampler2D u_gBuffer_depth;
uniform sampler2D u_gBuffer_material;
uniform sampler2D u_gBuffer_normal;


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
  // first hit is miss
  if(firstBounce_mat.b == 0.0)
  {
    ray_accumulation_OUT.rgb = vec3(0);
    ray_accumulation_OUT.a = 1.0 / float(frame_count);
    return;
  }

  vec4 firstBounce_normal = texelFetch(u_gBuffer_normal, ivec2(gl_FragCoord.xy), 0);
  vec3 shading_normal = 2.0 * firstBounce_normal.xyz - 1.0;
    
  vec3 ray_origin = reconstruct_position_from_depth(vec2(gl_FragCoord.xy) / vec2(frame));
  
  // Generate random numbers
	uint pixel_seed = uint(MAX_DEPTH + 1) * (uint(gl_FragCoord.x) + uint(gl_FragCoord.y) * uint(frame.x) + 1u) + 0u;
              
  vec3 tangent, bitangent;
  createONBOpt(shading_normal, tangent, bitangent);

  const uint sample_count = 1u;
  float occlusion = 0.0f;
  for (uint sample_index = 0u; sample_index < sample_count; sample_index++) {
    // Generate random numbers
    uint counter = ( uint(gl_FragCoord.x) + uint(gl_FragCoord.y) * uint(frame.x) + 1u ) + sample_index;
    vec4 randoms = rg_Random(counter, seed.x, seed.y);

    // cosine weighted importance sampling
    vec3 dir = cosine_sample_hemisphere(randoms.x, randoms.y);
    dir = dir.x * tangent + dir.y * bitangent + dir.z * shading_normal;
    dir = normalize(dir);

    float occluded = wr_query_occlusion(ads, ray_origin + shading_normal * POSITION_OFFSET, dir, 1.5) ? 1.0 : 0.0;

    occlusion += occluded;
  }
  occlusion /= float(sample_count);

  vec3 color = vec3(1.0 - occlusion);
  ray_accumulation_OUT = vec4(color, 1.0 / float(frame_count));
}