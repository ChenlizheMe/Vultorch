#version 450

// ── Inputs ────────────────────────────────────────────────────────────
layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragUV;

// ── Output ────────────────────────────────────────────────────────────
layout(location = 0) out vec4 outColor;

// ── Uniforms ──────────────────────────────────────────────────────────
layout(set = 0, binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 cameraPos;
    float _pad0;
} ubo;

layout(set = 0, binding = 1) uniform LightUBO {
    vec3  direction;
    float intensity;
    vec3  color;
    float ambient;
    float specular;
    float shininess;
    float _pad0;
    float _pad1;
} light;

layout(set = 0, binding = 2) uniform sampler2D tensorTex;

void main() {
    // Sample tensor texture
    vec4 texColor = texture(tensorTex, fragUV);

    // Blinn-Phong lighting
    vec3 N = normalize(fragNormal);
    vec3 L = normalize(-light.direction);
    vec3 V = normalize(ubo.cameraPos - fragWorldPos);
    vec3 H = normalize(L + V);

    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), light.shininess);

    vec3 result = texColor.rgb * (light.ambient + diff * light.intensity) * light.color
                + light.specular * spec * light.color;

    outColor = vec4(result, texColor.a);
}
