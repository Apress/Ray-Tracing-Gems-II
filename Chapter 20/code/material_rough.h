#include "functions.h"

// Parts of the code in this file was adapter from code by Thomas Mueller.
float Epsilon = 1E-6f;

float material_weight_diffuse(const Material &m) {
  float w_diffuse = max(m.diffuse.x, max(m.diffuse.y, m.diffuse.z));
  float w_specular = m.specular;
  return w_diffuse / (w_diffuse + w_specular);
}

float D(float alpha, float cosTheta) {
  float cosThetaSq = cosTheta * cosTheta;
  float alphaSq = alpha * alpha;
  return exp((cosThetaSq - 1) / (cosThetaSq * alphaSq)) / (PI * alphaSq * cosThetaSq * cosThetaSq);
}

float roughness_to_alpha(float roughness) { return roughness * roughness; }

vec3 random_beckman(vec3 normal, float alpha) {
  float u0 = uniform(), u1 = uniform();
  
  // In spherical coordinates
  float log_sample = logf(1.0f-u0);
  if (isinf(log_sample)) log_sample = 0.0f;
  float tan2_theta = -alpha*alpha * log_sample;
  float phi = u1*(float)(2.0*PI);

  float cos_theta = 1.0f/sqrtf(1.0f + tan2_theta);
  float sin_theta = sqrtf(max(0.0f, 1.0f-cos_theta*cos_theta));

  // Construct the half-vector (or micro-normal)
  vec3 wh = rotate_frame(sin_theta*sinf(phi), sin_theta*cosf(phi), cos_theta, normal);

  return wh;
}

float random_beckman_pdf(vec3 normal, float alpha, vec3 wh) {
	// I assume m always has unit length.
  float cosTheta = dot(normal, wh);
	if(cosTheta <= 0) {
		return 0.0f;
	}
	float cosThetaSq = cosTheta * cosTheta;
	float alphaSq = alpha * alpha;
	return exp((cosThetaSq - 1.0f) / (cosThetaSq * alphaSq)) / (PI * alphaSq * cosThetaSq * cosTheta);
}

float Frame_tanTheta(vec3 normal, vec3 v) {
  // tan=sin/cos with sin from 1=sin^2+cos^2.
  float cos_theta = dot(normal, v);
  return sqrtf(1.0-cos_theta*cos_theta)/cos_theta;
}

float G1(float alpha, vec3 normal, const vec3& w, const vec3& wh) {
  if(dot(w, wh) / dot(normal, w) <= 0) {
    return 0;
  }
  float b = 1.0f / (alpha * Frame_tanTheta(normal, w));
  if(b >= 1.6f) return 1;
  float bSq = b * b;
  return (3.535f * b + 2.181f * bSq) / (1.0f + 2.276f * b + 2.577f * bSq);
}

float G(float alpha, vec3 normal, const vec3& wi, const vec3& wo, const vec3& wh) {
  return G1(alpha, normal, wi, wh) * G1(alpha, normal, wo, wh);
}

float fresnel(float cosThetaI, float extIOR, float intIOR) {
  float etaI = extIOR, etaT = intIOR;
  if (extIOR == intIOR) return 0.0f;
  // Swap the indices of refraction if the interaction starts at the inside of the object
  if (cosThetaI < 0.0f) {
    float temp = etaI;
    etaI = etaT;
    etaT = temp;
    cosThetaI = -cosThetaI;
  }
  // Using Snell's law, calculate the squared sine of the angle between the normal and the transmitted ray
  float eta = etaI / etaT, sinThetaTSqr = eta*eta * (1-cosThetaI*cosThetaI);
  if (sinThetaTSqr > 1.0f) return 1.0f; // Total internal reflection!
  float cosThetaT = sqrtf(1.0f - sinThetaTSqr);
  float Rs = (etaI * cosThetaI - etaT * cosThetaT) / (etaI * cosThetaI + etaT * cosThetaT);
  float Rp = (etaT * cosThetaI - etaI * cosThetaT) / (etaT * cosThetaI + etaI * cosThetaT);
  return (Rs * Rs + Rp * Rp) / 2.0f;
}

vec3 evaluate_material(Material m, vec3 normal, vec3 wo, vec3 wi) {
  vec3 wh = wi + wo;
  float alpha = roughness_to_alpha(m.roughness);

  float ndotwi = dot(normal, wi);
  float ndotwo = dot(normal, wo);
  // This is a smooth BRDF -- return zero when queried for illumination on the backside
  if(ndotwi <= 0.0f || ndotwo <= 0.0f) {
    return 0.0f;
  }

  vec3 result = vec3(0.0f);

  // D can't handle alpha being 0.0
  if (alpha > Epsilon) {
    float whLength = length(wh);
    if(whLength < Epsilon) return vec3(0.0);
    wh /= whLength;
    float int_ior = 1.5046f; // Glass
    float ext_ior = 1.000277f; // Air

    float f = fresnel(dot(wh, wi), ext_ior, int_ior);
    float g = G(alpha, normal, wi, wo, wh);
    float d = D(alpha, dot(normal, wh));

    result += m.specular * (d * f * g / (4.0f * ndotwi * ndotwo));
  }

  result += m.diffuse * INV_PI;

  return result;
}

/// Evaluate the sampling density of \ref sample() wrt. solid angles
float sample_material_pdf(Material m, vec3 normal, vec3 wo, vec3 wi) {
  // This is a smooth BRDF -- return zero when queried for illumination on the backside
  float ndotwi = dot(normal, wi);
  float ndotwo = dot(normal, wo);
  if(ndotwi <= 0.0f || ndotwo <= 0.0f) {
    return 0.0f;
  }

  float p_diffuse = material_weight_diffuse(m);
  float p_specular = 1.0f-p_diffuse;

  float alpha = roughness_to_alpha(m.roughness);

  float total_pdf = 0.0f;

  if (alpha > Epsilon) {
    vec3 wh = wi + wo;
    float whLength = length(wh);
    if(whLength < Epsilon) return 0.0;
    wh /= whLength;

    total_pdf += p_specular * random_beckman_pdf(normal, alpha, wh) / (4.0f * clamped_dot(wo, wh));
  }

  total_pdf += p_diffuse * ndotwo/PI;

  return total_pdf;
}

bool sample_material(Material m, vec3 normal, vec3 wo, vec3 *wi, float *out_pdf) {
  if (dot(wo, normal) <= 0.0f) {
    return false;
  }

  float alpha = roughness_to_alpha(m.roughness);

  float p_diffuse = material_weight_diffuse(m);
  float p_specular = 1.0f-p_diffuse;
  float p = uniform();

  if (uniform() <= p_specular) {
    vec3 wh = random_beckman(normal, alpha);
    *wi = reflect(-wo, wh);
  } else {
    *wi = random_cosine_hemisphere(normal);
  }

  // NOTE: Here we give out the pdf of BOTH the method even if we just used one of them
  // Same with evaluate_material, it always does both
  float pdf = sample_material_pdf(m, normal, wo, *wi);
  if (pdf == 0.0f) return false;
  *out_pdf = pdf;
  return true;
}
