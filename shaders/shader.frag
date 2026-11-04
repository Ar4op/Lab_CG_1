#version 450

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(location = 0) out vec4 outColor;

struct PointLight {
    vec4 position;   // xyz
    vec4 color;      // rgb + intensity
};

struct SpotLight {
    vec4 position;   // xyz + cos(inner)
    vec4 direction;  // xyz + cos(outer)
    vec4 color;      // rgb + intensity
};

// ---------- Set 0: сцена ----------
layout(set = 0, binding = 0) uniform SceneData {
    mat4  viewProj;
    vec4  cameraPos;
    vec4  ambient;
    vec4  dirLightDirection;
    vec4  dirLightColor;
    uvec4 lightCounts;       // x - point, y - spot
} scene;

layout(std430, set = 0, binding = 1) readonly buffer PointLights {
    PointLight pointLights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight spotLights[];
};

// ---------- Set 1: модель ----------
layout(set = 1, binding = 0) uniform ModelData {
    mat4 model;
    mat4 normalMatrix;
    vec4 albedo;             // rgb + vertex color mix
    vec4 specular;           // rgb + shininess
} object;

// ---------- Блинн-Фонг ----------
vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo) {
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0) {
        return vec3(0.0);
    }

    vec3  H    = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), object.specular.w);

    return radiance * (albedo * NdotL + object.specular.rgb * spec);
}

void main() {
    vec3 N = normalize(inNormal);
    vec3 V = normalize(scene.cameraPos.xyz - inWorldPos);
    vec3 albedo = object.albedo.rgb * mix(vec3(1.0), inColor, object.albedo.w);

    // ---------- Ambient ----------
    vec3 color = albedo * scene.ambient.rgb * scene.ambient.a;

    // ---------- Направленный свет ----------
    vec3 Ld = normalize(-scene.dirLightDirection.xyz);
    color += blinnPhong(N, V, Ld, scene.dirLightColor.rgb * scene.dirLightColor.a, albedo);

    // ---------- Точечные источники ----------
    for (uint i = 0; i < scene.lightCounts.x; ++i) {
        vec3  toLight  = pointLights[i].position.xyz - inWorldPos;
        float dist2    = max(dot(toLight, toLight), 1e-4);
        vec3  radiance = pointLights[i].color.rgb * pointLights[i].color.a / dist2;

        color += blinnPhong(N, V, toLight * inversesqrt(dist2), radiance, albedo);
    }

    // ---------- Прожекторы ----------
    for (uint i = 0; i < scene.lightCounts.y; ++i) {
        SpotLight light = spotLights[i];

        vec3  toLight = light.position.xyz - inWorldPos;
        float dist2   = max(dot(toLight, toLight), 1e-4);
        vec3  L       = toLight * inversesqrt(dist2);

        float cosTheta = dot(-L, normalize(light.direction.xyz));
        float cosInner = light.position.w;
        float cosOuter = light.direction.w;

        float cone = cosInner > cosOuter
            ? clamp((cosTheta - cosOuter) / (cosInner - cosOuter), 0.0, 1.0)
            : step(cosOuter, cosTheta);

        vec3 radiance = light.color.rgb * light.color.a * cone / dist2;

        color += blinnPhong(N, V, L, radiance, albedo);
    }

    outColor = vec4(color, 1.0);
}
