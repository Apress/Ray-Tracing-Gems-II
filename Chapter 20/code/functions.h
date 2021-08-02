#pragma once

#include "program_defs.h"

float infinity = 100000.0;

struct Material {
	vec3 emissive;
	vec3 diffuse;
	float specular;
	float roughness;
	bool is_emissive;
};

enum class GeometryType { Sphere, Quad };

struct Intersect {
	vec3 pos;
	vec3 nor;
	vec3 nor_emissive; // Normal facing the emissive side of an emissive object
	int object_index;
	Material mat;
	float distance;
	bool hit;
};

struct Object {
	GeometryType type;
	vec3 pos;
	vec3 normal;
	vec3 size;
	Material material;
};

#define NUM_LIGHTS 6

// Lights must come first
// Emissive objects are assumed to not have a brdf, ie they don't reflect light. Also only one side emit.
// Since there is no acceleration structure each new objects cost a lot!
static const Object objects[]={
	// Emissive spheres
	{GeometryType::Sphere, vec3(-2.0, 2.0, 0.0), vec3(0.0), vec3(0.05), {vec3(1.0, 0.0, 0.0)*250, vec3(0.0), 0.0, 0.0, true}},
	{GeometryType::Sphere, vec3(-1.0, 2.0, 0.0), vec3(0.0), vec3(0.1), {vec3(0.0, 1.0, 0.0)*60, vec3(0.0), 0.0, 0.0, true}},
	{GeometryType::Sphere, vec3(-0.0, 2.0, 0.0), vec3(0.0), vec3(0.2), {vec3(0.0, 0.0, 1.0)*40, vec3(0.0), 0.0, 0.0, true}},
	{GeometryType::Sphere, vec3( 1.0, 2.0, 0.0), vec3(0.0), vec3(0.3), {vec3(1.0, 1.0, 0.0)*30, vec3(0.0), 0.0, 0.0, true}},
	{GeometryType::Sphere, vec3( 2.0, 1.9, 0.0), vec3(0.0), vec3(0.4), {vec3(0.0, 1.0, 1.0)*20, vec3(0.0), 0.0, 0.0, true}},
	
	// Emissive quad
	{GeometryType::Quad, vec3(-3.7f, 0.25f, 2.0f), vec3(1.0f, 0.0f, 0.0f), vec3(0.5f, 5.0f, 0.0f), {vec3(0.2f, 0.2f, 1.0f)*30.0f, vec3(0.0f), 0.0f, 0.0f, true}},

	// Ground
	{GeometryType::Quad, vec3(0.0f), vec3(0.0f, 1.0f, 0.0f), vec3(300.0f), {vec3(0.0f), vec3(0.6f), 0.3f, 0.5f, false}},

	// Side quads. They are diffuse since making them reflect our actual scene is hard
	{GeometryType::Quad, vec3(-4.0f, 2.0f, 1.0f), vec3( 1.0f, 0.0f, 0.0f), vec3(4.0f, 5.0f, 0.0f), {vec3(0.0), vec3(0.6), 0.2f, 1.0f, false}},
	{GeometryType::Quad, vec3( 4.0f, 2.0f, 1.0f), vec3(-1.0f, 0.0f, 0.0f), vec3(4.0f, 5.0f, 0.0f), {vec3(0.0), vec3(0.6), 0.2f, 1.0f, false}},

	// Reflective half-balls
	{GeometryType::Sphere, vec3(-2.0f, 0.0f, 0.0f), vec3(0.0f), vec3(0.5f), {vec3(0.0f), vec3(0.1f), 0.9f, 0.8f, false}},
	{GeometryType::Sphere, vec3(-1.0f, 0.0f, 0.0f), vec3(0.0f), vec3(0.5f), {vec3(0.0f), vec3(0.1f), 0.9f, 0.6f, false}},
	{GeometryType::Sphere, vec3( 0.0f, 0.0f, 0.0f), vec3(0.0f), vec3(0.5f), {vec3(0.0f), vec3(0.1f), 0.9f, 0.4f, false}},
	{GeometryType::Sphere, vec3( 1.0f, 0.0f, 0.0f), vec3(0.0f), vec3(0.5f), {vec3(0.0f), vec3(0.1f), 0.9f, 0.2f, false}},
	{GeometryType::Sphere, vec3( 2.0f, 0.0f, 0.0f), vec3(0.0f), vec3(0.5f), {vec3(0.0f), vec3(0.1f), 0.9f, 0.01f, false}}, // NOTE: Roughness 0.0 doesn't work
};

vec3 minor_axis(vec3 v) {
	if (abs(v.x) < abs(v.y)) {
		if (abs(v.x) < abs(v.z)) {
			return vec3(1,0,0);
		} else {
			return vec3(0,0,1);
		}
	} else {
		if (abs(v.y) < abs(v.z)) {
			return vec3(0,1,0);
		} else {
			return vec3(0,0,1);
		}
	}
}

void construct_frame(vec3 normal, vec3 *x_axis, vec3 *y_axis) {
	vec3 m = minor_axis(normal);
	*x_axis = normalize(cross(normal, m));
	*y_axis = cross(*x_axis, normal);
}

// note: x,y,z are scalars, forward is a vec3
vec3 rotate_frame(float x, float y, float z, vec3 forward) {
	vec3 x_axis, y_axis;
	construct_frame(forward, &x_axis, &y_axis);
	return x_axis * x + y_axis * y + forward * z;
}

vec3 random_quad(vec3 normal, vec3 size) {
	float u = uniform(), v = uniform();
	return rotate_frame((u-0.5f)*size.x, (v-0.5f)*size.y, 0.0f, normal);
}

vec3 random_on_sphere() {
	float u0 = uniform(), u1 = uniform();
	float phi = (2.0*PI)*u0;
	float cos_theta = u1*2.0f-1.0f;
	float sin_theta = sqrtf(max(0.0f, 1.0f-cos_theta*cos_theta));
	float x = sin_theta * cos(phi);
	float y = sin_theta * sin(phi);
	float z = cos_theta;
	return vec3(x,y,z);
}

vec3 random_hemisphere_cosine_pow(vec3 normal,float n) {
	float u0 = uniform(), u1 = uniform();
	float phi = 2*PI*u0;
	float cos_theta = pow(u1, 1/(1+n));
	float sin_theta = sqrt(1.0-cos_theta*cos_theta);
	float x = sin_theta * cos(phi);
	float y = sin_theta * sin(phi);
	float z = cos_theta;
	return rotate_frame(x,y,z,normal);
}

vec3 random_hemisphere(vec3 normal) {
	vec3 s = random_on_sphere();
	if (dot(s, normal)<0.0) {
		return -s;
	}
	return s;
}

vec3 random_cosine_hemisphere(vec3 normal) {
	return normalize(random_on_sphere() + normal);
}

// https://www.iquilezles.org/www/articles/intersectors/intersectors.htm
float sphIntersect( vec3 ro, vec3 rd, vec3 ce, float ra, float tmax ) {
  vec3 oc = ro - ce;
  float b = dot( oc, rd );
  float c = dot( oc, oc ) - ra*ra;
  float h = b*b - c;
  if( h<0.0 ) return -1.0; // no intersection
  h = sqrt( h );
	float r0 = -b-h, r1 = -b+h;
	if (r0 > 0.0 && r0 < tmax) return r0;
	if (r1 > 0.0 && r1 < tmax) return r1;
  return -1.0;
}

float intersect_quad(vec3 pos, vec3 dir, vec3 quad_center, vec3 quad_normal, vec3 quad_size, float tmax) {
		float t = dot(pos - quad_center, quad_normal) / -dot(dir, quad_normal);
		if (t<0.0 || t > tmax) return -1.0;
		vec3 u_axis, v_axis;
		construct_frame(quad_normal, &u_axis, &v_axis);
		vec3 point_on_plane = pos + dir * t - quad_center;
		if (abs(dot(point_on_plane, u_axis)) > quad_size.x*0.5f) return -1.0;
		if (abs(dot(point_on_plane, v_axis)) > quad_size.y*0.5f) return -1.0;
		return t;
}

// Intersector iterates intersection in random order.
// Can be used to find ANY intersection, closest intersection or all intersections (in some unspecified order).
// NOTE: We do not care about precision. See RTG Chapter 7 for a discussion on precision.
struct Intersector {
	vec3 pos, dir;
	float best_t;
	int current_object = -1;
	int num_objects; // How many objects do we iterate? Can be used to restrict to the range of lights since they come first
	Intersect result;

	Intersector(vec3 _pos, vec3 _dir, float _tmax, int _num_objects = sizeof(objects)/sizeof(Object)) : pos(_pos), dir(_dir), best_t(_tmax), num_objects(_num_objects) {
		result.hit = false;
	}

	void update_pos_and_emissive_normal() {
		assert(result.hit);
		Object best = objects[result.object_index];
		result.pos = pos + dir * result.distance;
		if (best.type == GeometryType::Sphere) {
			result.nor_emissive = normalize(result.pos - best.pos);
		} else if (best.type == GeometryType::Quad) {
			// NOTE: While the quads are only emissive on one side they are two-sided
			// This will be reflected in result.nor
			result.nor_emissive = best.normal;
		}
	}

	Intersect finalize_intersect() {
		if (!result.hit) {
			return result;
		}
		Object best = objects[result.object_index];
		result.mat = best.material;
		update_pos_and_emissive_normal();
		if (dot(dir, result.nor_emissive)>0.0) {
			result.nor = -result.nor_emissive;
		} else {
			result.nor = result.nor_emissive;
		}
		return result;
	}

	// Iterate intersections
	// Note that this.result is not fully filled in after a call to next.
	// If restrict_t we will always get closer intersections for each call
	// If do_update_pos_and_emissive_normal is set to true the result.pos and result.nor_emissive are valid after each step.
	// Return true when it finds an intersection, false when it is done
	bool next(bool restrict_t, bool do_update_pos_and_emissive_normal = false) {
		bool got_hit = false;
		while (++current_object < num_objects) {
			const Object &o = objects[current_object];
			if (o.type == GeometryType::Sphere) {
				float t = sphIntersect(pos, dir, o.pos, o.size.x, best_t);
				if (t>=0.0) {
					result.distance = t;
					got_hit = true;
					break;
				}
			} else if (o.type == GeometryType::Quad) {
				float t = intersect_quad(pos, dir, o.pos, o.normal, o.size, best_t);
				if (t>= 0.0) {
					result.distance = t;
					got_hit = true;
					break;
				}
			}
		}
		if (got_hit) {
			result.object_index = current_object;
			if (restrict_t) best_t = result.distance;
			result.hit = true;
			if (do_update_pos_and_emissive_normal) {
				update_pos_and_emissive_normal();
			}
			return true;
		}
		return false;
	}
};

// Offset an intersection point out of self-intersection
// See RTG1 chapter 6 for a more proper way to do this
vec3 safe(vec3 pos, vec3 normal) {
	return pos + normal * 0.00001;
}

// Find closest intersection in direction dir from pos, but not further than tmax
// Dir assumed to be normalized
// Pos assumed to be non-self-intersecting (use safe(pos, normal) to offset it).
Intersect intersect(vec3 pos, vec3 dir, float tmax = infinity) {
	Intersector intersector(pos, dir, tmax);
	while (intersector.next(true));
	return intersector.finalize_intersect();
}

// return true if there is nothing between a and b, otherwise false
// a and b are assumed to be non-self intersecting (use safe(a, normal))
bool intersect_visibility(vec3 a, vec3 b) {
  vec3 dir = b - a; 
	float tmax = length(dir);
	dir = normalize(dir);	
	Intersector intersector(a, dir, tmax);
	while (intersector.next(false)) {
		return false; // We hit something so no visibility, no need to find closest
	}
	return true;
}

vec3 evaluate_emissive(Intersect i, vec3 dir) {
	if (!i.mat.is_emissive) return vec3(0.0); // Not an emissive surface
	if (dot(dir, i.nor_emissive)>=0.0) return vec3(0.0); // Backside test
	return i.mat.emissive;
}

bool russian_roulette(vec3 *tp) {
  float p = uniform();
	float p_continue = 0.95f;
	if (p<p_continue) {
		*tp /= p_continue;
		return false;
	}
	return true;
}
