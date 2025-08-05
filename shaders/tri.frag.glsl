#version 450

#define MAX_POINT_LIGHTS 8

struct PointLight {
    vec3 position;
    float _pad1;
    vec3 color;
    float intensity;
    float constant;
    float linear;
    float quadratic;
    float _pad2;
};

layout(binding = 0) uniform UniformBufferObject {
    mat4 proj;
    mat4 view;
    mat4 model;
    vec3 cameraPos;
    uint numLights;
    PointLight lights[MAX_POINT_LIGHTS];
} ubo;

layout(binding = 1) uniform sampler2D texSampler;

// ✅ New: BaseColor UBO at binding = 3
layout(binding = 3) uniform BaseColorUBO_t {
    vec4 baseColor;
} baseColorUbo;

// ✅ New: HasTexture UBO at binding = 4
layout(binding = 4) uniform HasTextureUBO_t {
    int hasTexture;
} hasTextureUbo;

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

vec3 calculatePointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir, vec3 albedo) {
    vec3 lightDir = normalize(light.position - fragPos);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), 64.0);
    float distance = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * distance * distance);

    vec3 ambient = 0.1 * light.color * light.intensity;
    vec3 diffuse = diff * light.color * light.intensity;
    vec3 specular = spec * light.color * light.intensity * 0.5;

    ambient *= attenuation;
    diffuse *= attenuation;
    specular *= attenuation;

    return (ambient + diffuse) * albedo + specular;
}

void main() {
    vec3 albedo;
    if (hasTextureUbo.hasTexture == 1) {
        albedo = texture(texSampler, fragTexCoord).rgb * baseColorUbo.baseColor.rgb;
    } else {
        albedo = baseColorUbo.baseColor.rgb;
    }

    vec3 normal = normalize(fragNormal);
    vec3 viewDir = normalize(ubo.cameraPos - fragWorldPos);
    
    vec3 result = vec3(0.0);
    for (int i = 0; i < int(ubo.numLights); ++i) {
        if (i >= MAX_POINT_LIGHTS) break;
        result += calculatePointLight(ubo.lights[i], normal, fragWorldPos, viewDir, albedo);
    }

    // Add a bit of ambient
    result += 0.15 * albedo;

    outColor = vec4(result, 1.0);
}
