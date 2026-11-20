#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec2 outUV;

// ---------- Set 0: сцена ----------
layout(set = 0, binding = 0) uniform SceneData {
    mat4  viewProj;
    vec4  cameraPos;
    vec4  ambient;
    vec4  dirLightDirection;
    vec4  dirLightColor;
    uvec4 lightCounts;
} scene;

// ---------- Set 1: материал (динамический uniform буфер) ----------
layout(set = 1, binding = 0) uniform ModelData {
    mat4 model;
    mat4 normalMatrix;
    vec4 specular;
} object;

void main() {
    vec4 worldPos = object.model * vec4(inPosition, 1.0);

    outWorldPos = worldPos.xyz;
    outNormal   = mat3(object.normalMatrix) * inNormal;
    outUV       = inUV;

    gl_Position = scene.viewProj * worldPos;
}
