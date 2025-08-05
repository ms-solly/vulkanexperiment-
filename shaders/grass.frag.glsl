#version 450

// Input from vertex shader
layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 3) in vec4 fragColor;

// Output
layout(location = 0) out vec4 outColor;

// Uniform buffer matching your existing structure
layout(binding = 0) uniform UniformBufferObject {
    mat4 proj;
    mat4 view;
    mat4 model;
    vec3 cameraPos;
    uint numLights;
} ubo;

void main() {
    vec4 col = fragColor;
    
    // Add some texture-like variation based on UV coordinates
    float texVariation = sin(fragTexCoord.x * 3.14159) * sin(fragTexCoord.y * 6.28318) * 0.1;
    col.rgb += texVariation;
    
    // Distance-based fade
    float distance = length(fragWorldPos - ubo.cameraPos);
    float fadeDistance = 60.0;
    float d = 1.0 - clamp(distance / fadeDistance, 0.0, 1.0);
    
    // Simple ambient lighting
    float ambient = 0.6;
    float diffuse = max(dot(fragNormal, normalize(vec3(0.5, 1.0, 0.3))), 0.0) * 0.4;
    float lighting = ambient + diffuse;
    
    col.rgb *= lighting;
    
    // Alpha testing for grass edges (optional)
    if (col.a < 0.1) discard;
    
    // Distance fade
    col.a *= d;
    
    outColor = col;
}