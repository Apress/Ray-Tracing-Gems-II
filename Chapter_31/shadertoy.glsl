// ------------------------- Image Tab -------------------------

// Self-contained live demonstration for
// "Tilt-Shift Rendering Using a Thin Lens Model"
// by Andrew Kensler, from Ray Tracing Gems II, Chapter 31.
//
// View this at https://www.shadertoy.com/view/tlcBzN.
//
// See Buffer A for the main camera model shown in the chapter
// and the path tracing code built around it for demonstration.
// This is a live demonstration: try changing the constants at
// the top of that buffer to experiment with moving the camera or
// focus, switching the scene, selecting between the thin lens or
// tilt shift model, adjusting the camera parameters, etc.  Then
// recompile and restart the shader to see the effect.
//
// Refer to the chapter for an explanation of the camera model.

void mainImage(
    out vec4 fragColor,
    in vec2 fragCoord)
{
	vec2 ndc = fragCoord.xy / vec2(iResolution);
    vec3 pixel = textureLod(iChannel0, ndc, 0.0).rgb;
    fragColor = vec4(pixel, 1.0);
}


// ------------------------ Buffer A Tab ------------------------

// Scene parameters.  Change these to select whether to render a
// scene more like Figure 11a or Figure 12 in the chapter or
// adjust the position of the spheres (Y-up coordinates).

#define GRID 0 // 0 = diffuse grey grid, 1 = emissive spots
const vec3 blue_sphere = vec3(0.75, 0.5, 0.25);
const vec3 green_sphere = vec3(-0.5, -0.25, 0.75);
const vec3 red_sphere = vec3(0.25, -0.75, -0.5);

// Camera parameters.  Change these to reposition the camera,
// move the points of focus, or change the lens and camera
// parameters.  Not all combinations are physically possible.

#define CAMERA tilt_shift // thin_lens or tilt_shift
const vec3 eye = vec3(0.0, 0.0, -2.75);
const vec3 look_at = vec3(0.0, 0.0, 0.0);
const vec3 world_middle = look_at;
const vec3 world_focus_a = blue_sphere;
const vec3 world_focus_b = green_sphere;
const vec3 world_focus_c = red_sphere;
const float sensor_size = 1.8;
const float f_stop = 5.0;
const float focal_length = 1.0;

// Derived constants from above settings.

const vec3 gaze = normalize(look_at - eye);
const vec3 right = normalize(cross(gaze, vec3(0.0, 1.0, 0.0)));
const vec3 up = cross(gaze, right);

const vec3 middle = vec3(dot(right, world_middle - eye),
                         dot(up, world_middle - eye),
                         dot(gaze, world_middle - eye));
const vec3 focus_a = vec3(dot(right, world_focus_a - eye),
                          dot(up, world_focus_a - eye),
                          dot(gaze, world_focus_a - eye));
const vec3 focus_b = vec3(dot(right, world_focus_b - eye),
                          dot(up, world_focus_b - eye),
                          dot(gaze, world_focus_b - eye));
const vec3 focus_c = vec3(dot(right, world_focus_c - eye),
                          dot(up, world_focus_c - eye),
                          dot(gaze, world_focus_c - eye));

const float focal_distance = focus_a.z;

const float epsilon = 0.001;

// First sample code listing from the chapter.  This function
// models a simple thin lens projection with a focal plane
// parallel to the lens and sensor.  It takes in a coordinate in
// screen space and a pair of random numbers in the unit square
// to generate a ray origin and direction.

void thin_lens(vec2 screen, vec2 random,
               out vec3 ray_origin, out vec3 ray_direction)
{
    // f  : focal_length      p : focal_distance
    // n  : f_stop            P : focused
    // s  : image_plane       O : ray_origin
    // P' : sensor            d : ray_direction

    // Lens values (precomputable)
    float aperture = focal_length / f_stop;
    // Image plane values (precomputable)
    float image_plane = focal_distance * focal_length /
        (focal_distance - focal_length);

    // Image plane values (render-time)
    vec3 sensor = vec3(screen * 0.5 * sensor_size, -image_plane);
    // Lens values (render-time)
    float theta = 6.28318531 * random.x;
    float r = aperture * sqrt(random.y);
    vec3 lens = vec3(cos(theta) * r, sin(theta) * r, 0.0);
    // Focal plane values (render-time)
    vec3 focused = sensor * focal_length /
        (focal_length - image_plane);

    ray_origin = lens;
    ray_direction = normalize(focused - lens);
}

// Second sample code listing from the chapter.  This function
// extends the thin lens model with shift for perspective control
// and tilt to allow an oblique plane of focus.  Refer to the
// chapter for details on how it works.

void tilt_shift(vec2 screen, vec2 random,
                out vec3 ray_origin, out vec3 ray_direction)
{
    // n  : normal      A : focus_a
    // t  : tilt        B : focus_b
    // M  : middle      C : focus_c
    // M' : shift

    // Focal plane values (precomputable)
    vec3 normal = normalize(cross(focus_b - focus_a,
                                  focus_c - focus_a));
    // Lens values (precomputable)
    vec3 tilt = vec3(0.0);
    if (abs(normal.x) > abs(normal.y))
    {
        tilt.x = (focus_a.z - focus_b.z) * focal_length /
            (focus_a.z * focus_b.x - focus_b.z * focus_a.x +
             (focus_a.z * focus_b.y - focus_b.z * focus_a.y) *
             normal.y / normal.x);
        tilt.y = tilt.x * normal.y / normal.x;
    }
    else if (abs(normal.y) > 0.0)
    {
        tilt.y = (focus_a.z - focus_b.z) * focal_length /
            (focus_a.z * focus_b.y - focus_b.z * focus_a.y +
             (focus_a.z * focus_b.x - focus_b.z * focus_a.x) *
             normal.x / normal.y);
        tilt.x = tilt.y * normal.x / normal.y;
    }
    tilt.z = sqrt(1.0 - tilt.x * tilt.x - tilt.y * tilt.y);
    vec3 basis_u = normalize(cross(tilt,
        abs(tilt.x) > abs(tilt.y) ? vec3(0.0, 1.0, 0.0)
                                  : vec3(1.0, 0.0, 0.0)));
    vec3 basis_v = cross(tilt, basis_u);
    float aperture = focal_length / f_stop;
    // Image plane values (precomputable)
    float image_plane = focus_a.z * focal_length /
        (dot(focus_a, tilt) - focal_length);
    vec2 shift = middle.xy / middle.z * -image_plane;

    // Image plane values (render-time)
    vec3 sensor = vec3(screen * 0.5 * sensor_size + shift,
                       -image_plane);
    // Lens values (render-time)
    float theta = 6.28318531 * random.x;
    float r = 0.5 * aperture * sqrt(random.y);
    vec3 lens = (cos(theta) * basis_u +
                 sin(theta) * basis_v) * r;
    // Focal plane values (render-time)
    vec3 focused = sensor * focal_length /
        (focal_length + dot(sensor, tilt));
    float flip = sign(dot(tilt, focused));

    ray_origin = lens;
    ray_direction = flip * normalize(focused - lens);
}

// Ray/primitive intersection routines.  Take the ray origin and
// direction, and the primitive position as input.  If the
// t-value for the distance to the intersection is closer than
// the existing t-value passed in, then updates the t-value, the
// hit coordinates, and surface normal at the intersection.

int plane(
    vec3 origin,
    vec3 direction,
    vec3 orient,
    float offset,
    inout float t,
    inout vec3 hit,
    inout vec3 normal)
{
    float t_plane = (offset - dot(origin, orient)) / dot(direction, orient);
    if (t_plane < epsilon || t < t_plane)
        return 0;
    t = t_plane;
    hit = origin + direction * t;
    normal = orient;
    return 1;
}

int sphere(
    vec3 origin,
    vec3 direction,
    vec3 center,
    float radius,
    inout float t,
    inout vec3 hit,
    inout vec3 normal)
{
    vec3 offset = origin - center;
    float b = dot(offset, direction);
    float c = dot(offset, offset) - radius * radius;
    float discriminant = b * b - c;
    if (discriminant <= 0.0)
        return 0;
    float t_sphere = -b - sqrt(discriminant);
    if (t_sphere < epsilon || t < t_sphere)
        return 0;
    t = t_sphere;
    hit = origin + direction * t;
    normal = normalize(hit - center);
    return 1;
}

int cylinder(
    vec3 origin,
    vec3 direction,
    vec3 center,
    vec3 orient,
    float radius,
    inout float t,
    inout vec3 hit,
    inout vec3 normal)
{
    vec3 approach = cross(direction, orient);
    float distance = abs(dot(origin - center, normalize(approach)));
    if (distance > radius)
        return 0;
    float t_center = dot(cross(orient, origin - center), approach) /
        dot(approach, approach);
    float t_half = sqrt(radius * radius - distance * distance) /
        dot(direction, normalize(cross(orient, approach)));
    float t_cylinder = t_center - t_half;
    if (t_cylinder < epsilon || t < t_cylinder)
        return 0;
    t = t_cylinder;
    hit = origin + direction * t;
    normal = ((hit - center) - dot(hit - center, orient) * orient) / radius;
    return 1;
}

// Shade a 2D position to compute either a grey grid pattern with
// lines to use for diffuse shading or else a grey pattern with a
// grid of small dots to use for emissive shading.

float grid_diffuse(
    vec2 position)
{
#if GRID == 0
    vec2 cell = fract(position);
    vec2 grid_1 = step(0.015, cell) - step(0.985, cell);
    vec2 grid_2 = 1.0 - step(0.475, cell) + step(0.525, cell);
    float blend = mod(floor(position.x) + floor(position.y), 2.0);
    return mix(0.4, grid_1.x * grid_1.y * mix(grid_2.x * grid_2.y, 0.9, blend), 0.6);
#else
    return 0.0;
#endif
}

float grid_emissive(
    vec2 position)
{
#if GRID == 1
    vec2 cell = fract(position);
    vec2 centered = cell - vec2(0.5);
    return 20.0 * (1.0 - step(0.001, dot(centered, centered)));
#else
    return 0.0;
#endif
}

// Random number generation.  Map a 3D seed value to a group of
// three pseudorandom each in [0,1].  (I.e., a uniformly
// distributed sample in the unit 3D cube.)

vec3 rng(
    vec3 seed)
{
    uvec3 v = uvec3(abs(seed) * 1048576.0);
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y*v.z;
    v.y += v.z*v.x;
    v.z += v.x*v.y;
    return vec3(v) * (1.0 / float(0xffffffffu));
}

// Given a ray and a maximum distance t-value, intersect the ray
// against the scene and get information out about what was hit
// if anything.  The position and shading values at the hit are
// output.  Returns 1 if the ray hit anything, or 0 if it missed.

int trace(
    vec3 origin,
    vec3 direction,
    float t,
    out vec3 hit,
    out vec3 normal,
    out vec3 diffuse,
    out vec3 specular,
    out vec3 emissive)
{
    const vec3 x_axis = vec3(1.0, 0.0, 0.0);
    const vec3 y_axis = vec3(0.0, 1.0, 0.0);
    const vec3 z_axis = vec3(0.0, 0.0, 1.0);
    float original_t = t;
    if (plane(origin, direction, -z_axis, -1.0, t, hit, normal) > 0)
    {
        diffuse = vec3(grid_diffuse(hit.xy * 4.0));
        specular = vec3(0.0);
        emissive = vec3(grid_emissive(hit.xy * 4.0));
    }
    if (plane(origin, direction,  x_axis, -1.0, t, hit, normal) +
        plane(origin, direction, -x_axis, -1.0, t, hit, normal) > 0)
    {
        diffuse = vec3(grid_diffuse(hit.yz * 4.0));
        specular = vec3(0.0);
        emissive = vec3(grid_emissive(hit.yz * 4.0));
    }
    if (plane(origin, direction,  y_axis, -1.0, t, hit, normal) +
        plane(origin, direction, -y_axis, -1.0, t, hit, normal) > 0)
    {
        diffuse = vec3(grid_diffuse(hit.xz * vec2(-4.0, 4.0)));
        specular = vec3(0.0);
        emissive = vec3(grid_emissive(hit.xz * vec2(-4.0, 4.0)));
    }
    if (sphere(origin, direction, red_sphere, 0.1, t, hit, normal) > 0)
    {
        diffuse = vec3(1.0, 0.1, 0.1);
        specular = vec3(0.9, 0.4, 0.4);
        emissive = vec3(0.0);
    }
    if (cylinder(origin, direction, red_sphere, x_axis, 0.01, t, hit, normal) +
        cylinder(origin, direction, red_sphere, y_axis, 0.01, t, hit, normal) +
        cylinder(origin, direction, red_sphere, z_axis, 0.01, t, hit, normal) > 0)
    {
        diffuse = vec3(1.0, 0.1, 0.1);
        specular = vec3(0.0);
        emissive = vec3(0.0);
    }
    if (sphere(origin, direction, green_sphere, 0.1, t, hit, normal) > 0)
    {
        diffuse = vec3(0.1, 1.0, 0.1);
        specular = vec3(0.4, 0.9, 0.4);
        emissive = vec3(0.0);
    }
    if (cylinder(origin, direction, green_sphere, x_axis, 0.01, t, hit, normal) +
        cylinder(origin, direction, green_sphere, y_axis, 0.01, t, hit, normal) +
        cylinder(origin, direction, green_sphere, z_axis, 0.01, t, hit, normal) > 0)
    {
        diffuse = vec3(0.1, 1.0, 0.1);
        specular = vec3(0.0);
        emissive = vec3(0.0);
    }
    if (sphere(origin, direction, blue_sphere, 0.1, t, hit, normal) > 0)
    {
        diffuse = vec3(0.1, 0.1, 1.0);
        specular = vec3(0.4, 0.4, 0.9);
        emissive = vec3(0.0);
    }
    if (cylinder(origin, direction, blue_sphere, x_axis, 0.01, t, hit, normal) +
        cylinder(origin, direction, blue_sphere, y_axis, 0.01, t, hit, normal) +
        cylinder(origin, direction, blue_sphere, z_axis, 0.01, t, hit, normal) > 0)
    {
        diffuse = vec3(0.1, 0.1, 1.0);
        specular = vec3(0.0);
        emissive = vec3(0.0);
    }
    return original_t == t ? 0 : 1;
}

// Path trace a ray against the scene and return an single sample
// estimating the radiance arriving at the sensor back along the
// ray.  This orchestrates intersecting the ray against the
// scene, shading the ray at the hit point for the surface
// properties, next-event estimation for direct lighting from an
// invisible area light on the ceiling, and following several
// bounces of indirect rays.

vec3 shade(
    inout vec3 origin,
    inout vec3 direction)
{
    vec3 shaded = vec3(0.0);
    vec3 throughput = vec3(1.0);
    for (int bounce = 0; bounce < 3; ++bounce)
    {
        float t = 1.0e30;
        vec3 hit = vec3(0.0);
        vec3 normal = vec3(0.0);
        vec3 diffuse = vec3(0.0);
        vec3 specular = vec3(0.0);
        vec3 emissive = vec3(0.0);
        trace(origin, direction, t, hit, normal, diffuse, specular, emissive);

        // Add in emissive contribution.

        shaded += throughput * emissive;

        // Add in direct lighting via next event estimation.

        vec3 xi_1 = rng(hit);
        vec3 light = vec3((xi_1.x - 0.5) * 0.1, 0.99, (xi_1.z - 0.5) * 0.1);
        vec3 light_direction = light - hit;
        if (dot(normal, light_direction) > 0.0)
        {
            float light_distance = length(light_direction);
            light_direction /= light_distance;
            vec3 shadow_hit = vec3(0.0);
            vec3 shadow_normal = vec3(0.0);
            vec3 shadow_diffuse = vec3(0.0);
            vec3 shadow_specular = vec3(0.0);
            vec3 shadow_emissive = vec3(0.0);
            int intersected = trace(hit, light_direction, light_distance,
                                    shadow_hit, shadow_normal,
                                    shadow_diffuse, shadow_specular, shadow_emissive);
            if (intersected == 0)
            {
                vec3 halfway = normalize(light_direction + normalize(origin - hit));
                float lambert = max(0.0, dot(normal, light_direction));
                float blinn_phong = 3.0 * pow(max(0.0, dot(halfway, normal)), 64.0);
                shaded += throughput * (diffuse * lambert + specular * blinn_phong);
            }
        }

        // Update for indirect lighting: adjust path throughput
        // and choose new ray for next path segment.

        float diffuse_weight = dot(diffuse, vec3(1.0));
        float specular_weight = dot(specular, vec3(1.0));
        if (xi_1.y * (diffuse_weight + specular_weight) <= diffuse_weight)
        {
            vec3 xi_2 = rng(hit + vec3(239.0, 491.0, 128.0));
            float phi = 6.28318531 * xi_2.x;
            float cos_theta_sq = xi_2.y;
            float sin_theta = sqrt(1.0 - cos_theta_sq);
            float sgn = normal.z < 0.0 ? -1.0 : 1.0;
            float a = -1.0 / (sgn + normal.z);
            float b = normal.x * normal.y * a;
            direction =
                (vec3(b, sgn + normal.y * normal.y * a, -normal.y) * (cos(phi) * sin_theta) +
                 vec3(1.0 + sgn * normal.x * normal.x * a, sgn * b, -sgn * normal.x) * (sin(phi) * sin_theta) +
                 normal * sqrt(cos_theta_sq));
            throughput *= diffuse;
        }
        else
        {
            direction = reflect(direction, normal);
            throughput *= specular;
        }

        origin = hit + direction * 0.001;
    }
    return shaded;
}

// Main driver routine.  Trace a new batch of paths for each
// pixel and mix it with the average of the previous batches
// (stored in the render target texture) to get the new average
// and return it.  One pixel in the corner tracks the current
// state to reset the accumulation on resolution changes or stop
// rendering at 32K spp.

void mainImage(
    out vec4 fragColor,
    in vec2 fragCoord)
{
    const float batch = 32.0;

    vec2 ndc = fragCoord.xy / vec2(iResolution);
    vec3 pixel = textureLod(iChannel0, ndc, 0.0).rgb;
    vec4 data = textureLod(iChannel0, vec2(0.0), 0.0);
    vec2 old_resolution = data.xy;
    float samples = data.z;
    if (old_resolution != vec2(iResolution))
        samples = 0.0;
    if (uvec2(fragCoord) == uvec2(0))
    {
        fragColor = vec4(iResolution.x, iResolution.y, samples + batch, 0.0);
        return;
    }
    if (samples >= 32768.0)
    {
        fragColor = vec4(pixel, 1.0);
        return;
    }

    vec3 shaded = vec3(0.0);
    for (float ray = 1.0; ray <= batch; ++ray)
    {
        vec2 jittered = fragCoord + rng(vec3(fragCoord, iTime * ray)).xy;
        vec2 screen = (2.0 * jittered - iResolution.xy) / min(iResolution.x, iResolution.y);
        vec2 random = rng(vec3(screen, iTime)).xy;
        vec3 origin, direction;
        CAMERA(screen, random, origin, direction);
        origin = eye + origin.x * right + origin.y * up + origin.z * gaze;
        direction = normalize(direction.x * right + direction.y * up + direction.z * gaze);
        shaded += shade(origin, direction);
    }

    fragColor = vec4((pixel * samples + shaded) / (samples + batch), 1.0);
}
