#version 450

layout(set = 0, binding = 0, std140) uniform SceneData {
    mat4 viewProj;
    vec4 cameraPos;
    vec4 ambient;
    vec4 dirDirection;
    vec4 dirColor;
    uvec4 counts;
} scene;

layout(set = 1, binding = 0, std140) uniform ModelData {
    mat4 model;
    vec4 albedo;
    vec4 specular; // rgb = specular color, a = shininess
} modelData;

struct PointLight {
    vec4 position;
    vec4 color; // rgb = light color, a = intensity
};

struct SpotLight {
    vec4 position;
    vec4 direction;
    vec4 color; // rgb = light color, a = intensity
    vec4 cone;  // x = cos(inner angle), y = cos(outer angle)
};

// Runtime-sized arrays: the shader does not declare a fixed element count.
layout(set = 0, binding = 1, std430) readonly buffer PointLightBuffer {
    PointLight pointLights[];
};

layout(set = 0, binding = 2, std430) readonly buffer SpotLightBuffer {
    SpotLight spotLights[];
};

layout(location = 0) in vec3 fragWorldPosition;
layout(location = 1) in vec3 fragWorldNormal;
layout(location = 0) out vec4 outColor;

const float MIN_DISTANCE_SQUARED = 0.0001;

// Blinn-Phong diffuse and specular contribution for one light.
vec3 blinnPhong(vec3 normal, vec3 viewDirection, vec3 lightDirection) {
    float diffuseFactor = max(dot(normal, lightDirection), 0.0);
    if (diffuseFactor <= 0.0) {
        return vec3(0.0);
    }

    vec3 halfDirection = normalize(viewDirection + lightDirection);
    float specularFactor = pow(
        max(dot(normal, halfDirection), 0.0),
        max(modelData.specular.a, 1.0)
    );

    vec3 diffuse = modelData.albedo.rgb * diffuseFactor;
    vec3 specular = modelData.specular.rgb * specularFactor;
    return diffuse + specular;
}

void main() {
    vec3 N = normalize(fragWorldNormal);
    vec3 V = normalize(scene.cameraPos.xyz - fragWorldPosition);

    // Ambient light is controlled separately in the ImGui interface.
    vec3 result = scene.ambient.rgb * modelData.albedo.rgb;

    // One directional light. dirDirection describes the direction in which
    // light rays travel, therefore the vector from the surface to the light
    // is its negation.
    if (dot(scene.dirColor.rgb, scene.dirColor.rgb) > 0.0) {
        vec3 L = normalize(-scene.dirDirection.xyz);
        result += blinnPhong(N, V, L) * scene.dirColor.rgb;
    }

    // Point lights: inverse-square attenuation, intensity / distance^2.
    for (uint i = 0; i < scene.counts.x; ++i) {
        PointLight light = pointLights[i];
        vec3 toLight = light.position.xyz - fragWorldPosition;
        float distanceSquared = max(dot(toLight, toLight), MIN_DISTANCE_SQUARED);
        vec3 L = toLight * inversesqrt(distanceSquared);
        vec3 radiance = light.color.rgb * (light.color.a / distanceSquared);
        result += blinnPhong(N, V, L) * radiance;
    }

    // Spot lights: inverse-square attenuation multiplied by a smooth cone.
    for (uint i = 0; i < scene.counts.y; ++i) {
        SpotLight light = spotLights[i];
        vec3 fromLight = fragWorldPosition - light.position.xyz;
        float distanceSquared = max(dot(fromLight, fromLight), MIN_DISTANCE_SQUARED);
        vec3 L = -fromLight * inversesqrt(distanceSquared);

        vec3 spotDirection = normalize(light.direction.xyz);
        float theta = dot(fromLight * inversesqrt(distanceSquared), spotDirection);
        float coneFactor = smoothstep(light.cone.y, light.cone.x, theta);

        vec3 radiance = light.color.rgb *
                        (light.color.a * coneFactor / distanceSquared);
        result += blinnPhong(N, V, L) * radiance;
    }

    outColor = vec4(result, 1.0);
}
