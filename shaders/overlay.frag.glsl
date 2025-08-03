#version 450

layout(input_attachment_index = 0, binding = 0) uniform subpassInput sceneColor;

layout(location = 0) in vec2 texCoord;
layout(location = 0) out vec4 fragColor;

void main() {
    // Read from the main scene
    vec4 scenePixel = subpassLoad(sceneColor);
    
    // Simple overlay composition
    vec4 overlayColor = vec4(1.0, 0.0, 0.0, 0.5); // Red overlay
    
    // Alpha blend
    fragColor = mix(scenePixel, overlayColor, overlayColor.a);
}