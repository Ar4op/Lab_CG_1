#version 450

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

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

// ---------- Set 1: материал ----------
layout(set = 1, binding = 0) uniform ModelData {
    mat4 model;
    mat4 normalMatrix;
    vec4 specular;           // rgb + shininess
} object;

layout(set = 1, binding = 1) uniform sampler2D albedoTexture;

// ---------- Push-константы: нетривиальное сэмплирование ----------
const uint FLAG_ANIMATED_SAMPLING = 1u;
const uint FLAG_COLOR_MODULATION  = 2u;

layout(push_constant) uniform SamplingParams {
    vec4  tintA;
    vec4  tintB;
    float time;
    float distortion;
    float layerScale;
    uint  flags;
} params;

// ---------- Сэмплирование альбедо ----------
vec3 sampleAlbedo(vec2 uv) {
    if ((params.flags & FLAG_ANIMATED_SAMPLING) == 0u) {
        return texture(albedoTexture, uv).rgb;
    }

    float t = params.time;

    vec2 wave = vec2(sin(uv.y * 12.0 + t * 2.0), cos(uv.x * 12.0 + t * 1.7)) * params.distortion;
    vec2 uvA  = uv + wave + vec2(t * 0.05, 0.0);

    float angle = t * 0.3;
    mat2  rot   = mat2(cos(angle), sin(angle), -sin(angle), cos(angle));
    vec2  uvB   = rot * (uv - 0.5) * params.layerScale + 0.5 + vec2(0.0, t * 0.03);

    vec3 layerA = texture(albedoTexture, uvA).rgb;
    vec3 layerB = texture(albedoTexture, uvB).rgb;
    vec3 color  = max(layerA, layerB) * 0.7 + min(layerA, layerB) * 0.3;

    if ((params.flags & FLAG_COLOR_MODULATION) != 0u) {
        float m = 0.5 + 0.5 * sin(t * 1.5 + (uv.x + uv.y) * 6.2831853);
        color *= mix(params.tintA.rgb, params.tintB.rgb, m);
    }

    return color;
}

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
    vec3 albedo = sampleAlbedo(inUV);

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
