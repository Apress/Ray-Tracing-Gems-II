#include "program_defs.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <stdint.h>
#include <vector>
#include <random>

#ifdef WIN32
	#define PCG_LITTLE_ENDIAN 1
#endif
#include "pcg_random.hpp"

// NOTE: Include order is a bit brittle due to some of these appearing exactly as they are int he RTG2 chapter.
#include "functions.h"
#include "balance_heuristic.h" // In article
#include "sample_lights.h" // In article
#include "sample_lights_pdf.h" // In article
//#include "material_lambert.h"
#include "material_rough.h"
#include "direct_light.h" // In article
#include "direct_cos.h" // In article
#include "direct_mat.h" // In article
#include "direct_mis.h" // In article
#include "pathtrace_mis.h" // In article
#include "pathtrace.h"
#include "trace_wrappers.h" // Add camera-trace to functions so they can be used

struct Random {
	pcg32 rng;
	std::uniform_real_distribution<float> uniform;
	Random() {
		pcg_extras::seed_seq_from<std::random_device> seed_source;
		rng.seed(seed_source);
	};
};

thread_local Random random;

// NOTE: uniform is thread-safe since random generator is in a thread_local variable
float uniform() {
	return random.uniform(random.rng);
}

// NOTE: This function only applies gamma.
uint32_t convert_sRGB(vec3 color) {
	float r = color.x, g = color.y, b = color.z;
	r = min(max(r, 0.0), 1.0);
	g = min(max(g, 0.0), 1.0);
	b = min(max(b, 0.0), 1.0);
	// TODO: Is this correct? Validate
	r = powf(r, 1.0f/2.2f);
	g = powf(g, 1.0f/2.2f);
	b = powf(b, 1.0f/2.2f);
	uint8_t rb = floor(r * 255.0f);
	uint8_t gb = floor(g * 255.0f);
	uint8_t bb = floor(b * 255.0f);
	return 0xFF000000|(bb<<16)|(gb<<8)|rb;
}

inline vec3 camera_direction(float2 uv, int w, int h, float fov_horizontal_degrees) {
	float aspect = float(h) / float(w);
	float fov_factor = tanf(fov_horizontal_degrees * (float)(2.0 * PI / 360.0 * 0.5));
	vec3 camera_forward = vec3(0.0f, 0.0f, 1.0f);
	vec3 camera_right = vec3(1.0f, 0.0f, 0.0f);
	vec3 camera_up = vec3(0.0f, 1.0f, 0.0f);
	return normalize(camera_forward + camera_right * ((uv.x * 2.0 - 1.0) * fov_factor) + camera_up * ((uv.y * 2.0 - 1.0) * fov_factor * aspect));
}

typedef vec3(*renderFunction)(vec3 pos, vec3 normal);

void render(const std::string &dir, const std::string &filename, const renderFunction per_pixel, int subpixels, int w = 800, int h = 600, const std::string &filename_variance = "") {
	printf("Rendering %s%s\n", dir.c_str(), filename.c_str());
	int ws = subpixels, hs = subpixels;

	// Camera definition
	vec3 camera_pos = vec3(0.0f, 1.2f, -7.0f);

	std::vector<uint32_t> result(w*h);
	std::vector<uint32_t> result_variance;
	
	if (!filename_variance.empty()) {
		result_variance.resize(w * h);
	}

	#pragma omp parallel for // NOTE: Comment away this line to get single-threaded execution
	for (int y=0; y<h; y++) {
		for (int x=0; x<w; x++) {
			vec3 mean = vec3(0);
			vec3 M2 = vec3(0);
			int N = 0;
			for (int sy=0; sy<hs; sy++) {
				for (int sx=0; sx<ws; sx++) {
					float xc = (x + (sx + uniform()) / ws) * (1.0f / w);
					float yc = (y + (sy + uniform()) / hs) * (1.0f / h);
					yc = 1.0 - yc; // Turn y-coordinate upside down since image (0,0) is upper-left but we want (0,0) to be lower-left

					// NOTE: A "real" renderer would have some sort of tone mapper here and use some filter kernel.
					vec3 dir = camera_direction(float2(xc, yc), w, h, 70.0f);
					vec3 color = per_pixel(camera_pos, dir);

					// Welford's online algorithm so we get variance as well
					// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
					N++;
					vec3 delta = color - mean;
					mean += delta/N;
					vec3 delta2 = color - mean;
					M2 += delta * delta2;
				}
			}
			vec3 variance = M2/(N-1)/N; // Sample variance from Welford's online algorithm

			result[y*w+x]= convert_sRGB(mean);
			if (!result_variance.empty()) {
				if (y<8) {
					// Fill top-most 8 lines with pink so we don't mix the images
					result_variance[y * w + x] = convert_sRGB(vec3(1.0, 0.0, 1.0));
				} else {
					result_variance[y * w + x] = convert_sRGB(vec3(sqrtf(variance.x), sqrtf(variance.y), sqrtf(variance.z)));
				}
			}
		}
	}

	stbi_write_png((dir+filename).c_str(), w, h, 4, result.data(), w*4);
	if (!filename_variance.empty()) {
		stbi_write_png((dir + filename_variance).c_str(), w, h, 4, result_variance.data(), w * 4);
	}
}

void render_mis_weights(const std::string& dir, const std::string &filename, int subpixels, int w = 800, int h = 600, const char* const filename_variance = nullptr) {
	printf("Rendering %s%s\n", dir.c_str(), filename.c_str());
	int ws = subpixels, hs = subpixels;

	// Camera definition
	vec3 camera_pos = vec3(0.0f, 1.2f, -7.0f);

	std::vector<uint32_t> result(w*h);

	#pragma omp parallel for // NOTE: Comment away this line to get single-threaded execution
	for (int y=0; y<h; y++) {
		for (int x=0; x<w; x++) {
			vec3 mean_light = vec3(0);
			vec3 mean_material = vec3(0);
			int N = 0;
			for (int sy=0; sy<hs; sy++) {
				for (int sx=0; sx<ws; sx++) {
					float xc = (x + (sx + uniform()) / ws) * (1.0f / w);
					float yc = (y + (sy + uniform()) / hs) * (1.0f / h);
					yc = 1.0 - yc; // Turn y-coordinate upside down since image (0,0) is upper-left but we want (0,0) to be lower-left

					// NOTE: A "real" renderer would have some sort of tone mapper here and use some filter kernel.
					vec3 dir = camera_direction(float2(xc, yc), w, h, 70.0f);
					vec3 color_light = trace_direct_mis_light(camera_pos, dir);
					vec3 color_material = trace_direct_mis_material(camera_pos, dir);

					mean_light += color_light / (ws*hs);
					mean_material += color_material / (ws*hs);
				}
			}

			vec3 mean_full = mean_light + mean_material;
			float a = dot(mean_light, mean_full);
			float b = dot(mean_material, mean_full);
	
			vec3 full = mean_light + mean_material;
			mean_light.x /= full.x;
			mean_light.y /= full.y;
			mean_light.z /= full.z;
			mean_material.x /= full.x;
			mean_material.y /= full.y;
			mean_material.z /= full.z;

			float l = a/(a+b);
			float m = b/(a+b);

			vec3 color = vec3(m,0,0) + vec3(0,l,0);

			result[y*w+x]= convert_sRGB(color);
		}
	}

	stbi_write_png((dir + filename).c_str(), w, h, 4, result.data(), w*4);
}

void rtg2_figures(const char * const dir) {
	// This is the code that was used to generate the images in the RTG2 chapter

	int w = 512, h = 300;
	// Direct light sampling
	int direct_N = 10;
	
	render(dir, "direct_light_sampling.png", { trace_direct_light_sampling}, direct_N, w, h);
	render(dir, "direct_material_sampling.png", {trace_direct_material_sampling }, direct_N, w, h);
	render(dir, "direct_cos_sampling.png", { trace_direct_cos_sampling }, direct_N, w, h);
	render(dir, "direct_mis.png", { trace_direct_mis }, direct_N, w, h);
	render(dir, "direct_mis_light.png", { trace_direct_mis_light<2> }, direct_N, w, h);
	render(dir, "direct_mis_material.png", { trace_direct_mis_material<2> }, direct_N, w, h);

	render(dir, "scene_description.png", { show_scene }, direct_N, w, h);
	
	// Path tracing
	int pathtrace_N = 15;
	render(dir, "pathtrace_mis.png", { pathtrace_mis_helper }, pathtrace_N, w, h);
	render(dir, "pathtrace_light_sampling.png", { pathtrace<direct_light> }, pathtrace_N, w, h);
	//render(dir, "pathtrace_material_sampling.png", { pathtrace<direct_mat> }, pathtrace_N, w, h);
	//render(dir, "pathtrace_hemisphere_sampling.png", { pathtrace<direct_hemi> }, pathtrace_N, w, h);

	// Weight image for direct MIS	
	render_mis_weights(dir, "direct_mis_weights.png", 35, w, h);
}

int main(void) {
	rtg2_figures("../figures/");
}