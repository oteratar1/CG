#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

// Set 0: data shared by the whole scene.
layout(set = 0, binding = 0, std140) uniform SceneData {
    mat4 viewProj;
    vec4 cameraPos;
    vec4 ambient;
    vec4 dirDirection;
    vec4 dirColor;
    uvec4 counts;
} scene;

// Set 1: data belonging to the current model.
layout(set = 1, binding = 0, std140) uniform ModelData {
    mat4 model;
    vec4 albedo;
    vec4 specular; // rgb = specular color, a = shininess
} modelData;

layout(location = 0) out vec3 fragWorldPosition;
layout(location = 1) out vec3 fragWorldNormal;

void main() {
    vec4 worldPosition = modelData.model * vec4(inPosition, 1.0);
    fragWorldPosition = worldPosition.xyz;

    // Correct normal transformation, including non-uniform model scaling.
    mat3 normalMatrix = transpose(inverse(mat3(modelData.model)));
    fragWorldNormal = normalize(normalMatrix * inNormal);

    gl_Position = scene.viewProj * worldPosition;
}
