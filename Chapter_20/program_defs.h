#pragma once

#define _USE_MATH_DEFINES
#include <math.h>

float min(float a, float b) {
	if (a<b) return a;
	return b;
}
float max(float a, float b) {
	if (a>b) return a;
	return b;
}

struct vec3 {
	float x, y, z;
	vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
	vec3(float _v) : x(_v), y(_v), z(_v) {}
	vec3() : x(0), y(0), z(0) {}

	vec3 operator*(vec3 v) const { return vec3(x * v.x, y * v.y, z * v.z); }
	vec3 operator+(vec3 v) const { return vec3(x + v.x, y + v.y, z + v.z); }
	vec3 operator-(vec3 v) const { return vec3(x - v.x, y - v.y, z - v.z); }

	vec3 operator*(float v) const { return vec3(x * v, y * v, z * v); }
	vec3 operator/(float v) const { return vec3(x / v, y / v, z / v); }

	vec3 operator-() const { return vec3(-x, -y, -z); }

	void operator/=(float v) { x/=v; y/=v; z/=v; }
	void operator+=(vec3 v) { x+=v.x; y+=v.y; z+=v.z; }
	void operator*=(vec3 v) { x*=v.x; y*=v.y; z*=v.z; }
	float sum() const { return x + y + z; }
};

struct float2 {
	float x, y;
	float2(float _x, float _y) : x(_x), y(_y) {}
	float2(float _v) : x(_v), y(_v) {}
	float2() : x(0), y(0) {}
};

float dot(vec3 a, vec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }

float clamped_dot(vec3 a, vec3 b) {
	return max(dot(a, b), 0.0);
}
float length (vec3 a) {
	return sqrtf(a.x*a.x+a.y*a.y+a.z*a.z);
}

float distance(vec3 a, vec3 b) {
	return length(a - b);
}

float distance_squared(vec3 a, vec3 b) {
	vec3 d = a - b;
	return d.x * d.x + d.y * d.y + d.z * d.z;
}

vec3 normalize(vec3 a) {
	float len = length(a);
	return vec3(a.x/len, a.y/len, a.z/len);
}

vec3 cross(vec3 a, vec3 b) {
	return vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x * b.z, a.x*b.y - a.y*b.x);
}

// incident should point into the surface
vec3 reflect(vec3 incident, vec3 normal) {
	return incident - normal * (2.0*dot(incident, normal));
}

static const float PI = (float)M_PI;
static const float INV_PI = (float)(1.0/M_PI);

vec3 mix(vec3 x, vec3 y, float a) {
	return x*(a-1.0)+y*a;
}

float square(float v) { return v*v; }

float uniform();
