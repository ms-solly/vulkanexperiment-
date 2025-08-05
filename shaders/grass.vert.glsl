#version 450

// Instance data (grass positions)
layout(location = 0) in vec3 inPosition;
layout(location = 1) in float inScale;

// Output to fragment shader
layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragTexCoord;
layout(location = 3) out vec4 fragColor;

// Uniform buffer matching your existing structure
layout(binding = 0) uniform UniformBufferObject {
    mat4 proj;
    mat4 view;
    mat4 model;
    vec3 cameraPos;
    uint numLights;
} ubo;

// Simple noise functions
float random(float x) {
    return fract(sin(x * 12.9898) * 43758.5453);
}

float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

vec3 random3(float seed) {
    return vec3(
        random(seed),
        random(seed + 1.0),
        random(seed + 2.0)
    );
}

// Simple noise function
float noise(vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);
    
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));
    
    vec2 u = f * f * (3.0 - 2.0 * f);
    
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

void main() {
    // Get instance and vertex IDs
    uint instanceId = gl_InstanceIndex;
    uint vertexId = gl_VertexIndex;
    
    // Simple grass blade geometry - just 2 triangles forming a quad
    vec2 quadVertices[6] = vec2[6](
        vec2(-0.5, 0.0), vec2(0.5, 0.0), vec2(-0.5, 1.0),  // First triangle
        vec2(0.5, 0.0), vec2(0.5, 1.0), vec2(-0.5, 1.0)   // Second triangle
    );
    
    vec2 localPos = quadVertices[vertexId];
    
    // Get grass position and random values
    vec3 grassPos = inPosition;
    float scale = inScale;
    vec3 randomPos = random3(float(instanceId));
    
    // Extract view matrix vectors for billboarding
    vec3 right = vec3(ubo.view[0][0], ubo.view[1][0], ubo.view[2][0]);
    vec3 up = vec3(ubo.view[0][1], ubo.view[1][1], ubo.view[2][1]);
    
    // Scale grass blade
    float grassWidth = 0.05 * scale;
    float grassHeight = (0.3 + randomPos.x * 0.4) * scale;
    
    vec3 vertexPos = right * localPos.x * grassWidth + up * localPos.y * grassHeight;
    
    // Calculate gradient for wind effect and coloring
    float vgradient = localPos.y;
    
    // Wind animation (simplified - you can add time uniform later)
    float windIntensity = 0.02 + randomPos.y * 0.03;
    vec2 wind = vec2(
        noise(grassPos.xz * 0.8) * 0.5,
        noise(grassPos.xz * 0.5) * 0.3
    ) * windIntensity * vgradient * vgradient; // More wind at the top
    
    // Apply wind deformation
    vertexPos += right * wind.x + vec3(0, 0, 1) * wind.y;
    
    // Final world position
    vec3 worldPos = grassPos + vertexPos;
    
    // Transform to clip space
    gl_Position = ubo.proj * ubo.view * vec4(worldPos, 1.0);
    
    // Output to fragment shader
    fragWorldPos = worldPos;
    fragNormal = up; // Simple upward normal for grass
    fragTexCoord = vec2(localPos.x + 0.5, localPos.y); // UV mapping
    
    // Color gradient from bottom to top with some variation
    vec3 bottomColor = vec3(0.2 + randomPos.z * 0.1, 0.4 + randomPos.x * 0.2, 0.1);
    vec3 topColor = vec3(0.4 + randomPos.y * 0.2, 0.7 + randomPos.z * 0.2, 0.2 + randomPos.x * 0.1);
    fragColor = vec4(mix(bottomColor, topColor, vgradient), 1.0);
}