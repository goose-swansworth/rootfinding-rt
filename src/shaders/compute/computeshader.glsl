#version 430 core

#include "lighting/lighting.glsl"
#include "lygia/geometry/aabb.glsl"


struct Camera {
    vec3 position;
    vec3 fowards;
    vec3 up;
    vec3 right;
};

struct Box {
    vec3 dims;
    vec3 position;
    vec4 color;
};

struct Material {
    vec3 diffuse;
    float bump;
    vec3 normal;
};

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(rgba32f, binding = 0) uniform image2D imgOutput;
layout(std430, binding = 1) readonly buffer sceneData {
    Box boxes[];
};
layout(rgba32f, binding = 2) readonly uniform image2D lumpedTexture;

uniform Camera viewer;
uniform bool AA;
uniform vec3 light;
uniform float FLOOR;
uniform vec4 FLOOR1;
uniform vec4 FLOOR2;
uniform vec4 sky_color;
uniform int num_boxes;

#define EPSILON  1e-3
#define MAX_ITERS 128
#define T_MIN 1
#define T_MAX 40
#define TEX_SIZE 1024
#define PI 3.1415926538

#define DIFFUSE_OFFSET 0
#define NORMAL_OFFSET 1

#define SHADOW_HARDNESS 8

const float inf = uintBitsToFloat(0x7F800000);
const vec3 zeros = vec3(0);
const vec3 up = vec3(0, 1, 0);
const vec4 white = vec4(1.0);

Material sampleMaterial(int texture_index, float s, float t) {
    Material material;
    material.diffuse = imageLoad(lumpedTexture, ivec2(floor(TEX_SIZE * (s + DIFFUSE_OFFSET)), floor(TEX_SIZE * (t + texture_index)))).rgb;
    material.normal = imageLoad(lumpedTexture, ivec2(floor(TEX_SIZE * (s + NORMAL_OFFSET)), floor(TEX_SIZE * (t + texture_index)))).rgb;
    material.normal = 2 * material.normal - white.xyz;
    return material;
}

// vec2 stMapSphere(vec3 center, vec3 p) {
//     vec3 d = normalize(center - p);
//     vec2 txCoords;
//     txCoords.s = 0.5 + atan(d.z, d.x) / (2 * PI);
//     txCoords.t = 0.5 + asin(d.y) / PI;
//     return txCoords;
// }

// vec2 txCoordsBox(int index, vec3 p) {
//     Box box = boxes[index];
//     vec3 t = p - box.position;
//     vec2 txCoords;
//     if (abs(t.x) >= box.dims.x) {
//         txCoords.s = (box.dims.z - t.z) / (2 * box.dims.z);
//         txCoords.t = (box.dims.y - t.y) / (2 * box.dims.y);
//     }
//     if (abs(t.y) >= box.dims.y) {
//         txCoords.s = (box.dims.z - t.z) / (2 * box.dims.z);
//         txCoords.t = (box.dims.x - t.x) / (2 * box.dims.x);
//     }
//     if (abs(t.z) >= box.dims.z) {
//         txCoords.s = (box.dims.x - t.x) / (2 * box.dims.x);
//         txCoords.t = (box.dims.y - t.y) / (2 * box.dims.y);
//     }
//     return txCoords;

// }

// AABB boxToAABB(Box box) {
//     AABB aabb;
//     aabb.min = box.position - box.dims;
//     aabb.max = box.position + box.dims;
//     return aabb;
// }


// float metaball(vec3 x, vec3 p0, float t) {
//     float r = length(x - p0);
//     return 1 / r - t;
// }




vec4 check(vec3 point) {
    vec2 q = floor(point.xz);
    if (mod(q.x + q.y, 2.0) == 0) {
        return FLOOR1;
    }
    return FLOOR2;
}


// float map(float value, float inMin, float inMax, float outMin, float outMax) {
//     return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
// }


// vec4 shadeSphere(vec3 L, vec3 p, vec3 I) {
//     vec2 coords = stMapSphere(vec3(0), p);
//     Material material = sampleMaterial(0, coords.s, coords.t);
//     vec4 diff_col = vec4(material.diffuse.rgb * 0.75, 1.0);
//     // Normal mapping
//     vec3 N = normalize(p - vec3(0, 1, 0));
//     float theta = atan(p.y / p.x);
//     vec3 T = vec3(-sin(theta), 0, cos(theta));
//     vec3 B = normalize(cross(N, T));
//     mat3 TBN = mat3(T, B, N);
//     N = TBN * material.normal;
//     // --------------
//     vec4 c1 = diffuse(L, N, 0.2, 0.8, diff_col);
//     vec4 c2;
//     // Refection
//     vec3 d = reflect(I, N);
//     vec2 res = trace(p + EPSILON * d, d);
//     float t = res.x;
//     if (t > 0) {
//         int i = int(res.y);
//         if (i == -2) {
//             c2 = shadeFloor(L, p + t * d);
//         } else {
//             c2 = shadeBox(i, L, p + t * d);
//         }
//     } else {
//         c2 = mix(sky_color, white, I.y);
//     }
//     return c1 + 0.2 * c2;
// }



// vec4 shade(int obj_index, vec3 p, vec3 I) {
//     vec3 L = normalize(light - p);
//     if (obj_index == -1) {
//         return shadeSphere(L, p, I);
//     }
//     if (obj_index == -2) {
//         return shadeFloor(L, p);
//     }
//     if (obj_index > -1) {
//         return shadeBox(obj_index, L, p);
//     }
// }



float intersectSphere(vec3 r0, vec3 d, vec3 center, float radius) {
    float l = center - r0;
    float b = dot(l, d);
    if (b )
}

float intersectFloor(vec3 r0, vec3 d) {
    return - r0.y / d.y;
}

vec4 shadePoint(vec3 p, int index) {
    vec3 L = normalize(light - p);
    if (index == 0) {
        return diffuse(L, vec3(0, 1, 0), 0.3, 0.7, check(p));
    }
}

vec4 render(vec3 r0, vec3 d) {
    vec4 color;
    float t = intersectFloor(r0, d);
    int index = 0;
    if (t > 0) {
        vec3 p = r0 + t * d;
        return shadePoint(p, index);
    }
    return mix(sky_color, white, d.y);

}

const vec2 s4[] = {{0.375f, 0.125f}, {0.875f, 0.375f}, {0.625f, 0.875f}, {0.125f, 0.625f}};

vec3 get_ray_dir(vec2 c) {
    return normalize(viewer.fowards + c.x * viewer.right + c.y * viewer.up);
}

void main() {
    vec4 color = vec4(0.0, 0.0, 0.0, 1.0);
    ivec2 texel_coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 dims = imageSize(imgOutput);
    float aspect = float(dims.x) / dims.y;
    float x = ((float(texel_coords.x) * 2 - dims.x) / dims.x); // transform to [-h/w , h/w]
    float y = ((float(texel_coords.y) * 2 - dims.y) / dims.x); // transform to [-1, 1]
    vec2 pixel_center = vec2(x, y);
    vec3 r0 = viewer.position;
    vec3 d, d1, d2;
    vec2 offset;
    float pixel_width = 2.0 / dims.x;
    if (AA) {
        for (int i = 0; i < 4; i++) {
            offset = pixel_center + s4[i] * pixel_width - vec2(0.5, 0.5) * pixel_width;
            d = normalize(viewer.fowards + offset.x * viewer.right + offset.y * viewer.up);
            color += 0.25 * render(r0, d);
        }
    } else {
        d = get_ray_dir(pixel_center);
        color = render(r0, d);
    }
    ivec2 mid = dims / 2;
    if (mid.x - 1 < texel_coords.x && texel_coords.x < mid.x + 1 && mid.y - 1 < texel_coords.y && texel_coords.y < mid.y + 1) {
        color = mix(white, color, 0.75);
    }

    imageStore(imgOutput, texel_coords, color);
}