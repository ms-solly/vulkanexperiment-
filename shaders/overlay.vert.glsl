#version 450

// Rectangle vertices (4 corners)
vec2 positions[4] = vec2[](
    vec2(-1.0, -1.0),  // Bottom-left
    vec2( 1.0, -1.0),  // Bottom-right
    vec2(-1.0,  1.0),  // Top-left
    vec2( 1.0,  1.0)   // Top-right
);

vec2 texCoords[4] = vec2[](
    vec2(0.0, 0.0),
    vec2(1.0, 0.0),
    vec2(0.0, 1.0),
    vec2(1.0, 1.0)
);

layout(location = 0) out vec2 texCoord;

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    texCoord = texCoords[gl_VertexIndex];
}