#include "application.hpp"

#include <vk_mem_alloc.h>
#include <imgui.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace application {
namespace {

constexpr uint32_t SEGMENTS = 50;
constexpr float PI = 3.14159265358979323846f;

// Capacity of the GPU storage buffers (the shader itself has no limit).
constexpr uint32_t MAX_POINT_LIGHTS = 64;
constexpr uint32_t MAX_SPOT_LIGHTS = 64;

constexpr float toRadians(float degrees) {
    return degrees * PI / 180.0f;
}

// =============================================================== math

struct Vec3 {
    float x, y, z;
};

Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }

float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

Vec3 cross(Vec3 a, Vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

Vec3 normalizeOr(Vec3 v, Vec3 fallback) {
    const float length = std::sqrt(dot(v, v));
    if (length < 1e-5f) {
        return fallback;
    }
    return v * (1.0f / length);
}

Vec3 toVec3(const float v[3]) {
    return {v[0], v[1], v[2]};
}

struct Mat4 {
    float m[16]{};
};

Mat4 identity() {
    Mat4 r{};
    r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
    return r;
}

Mat4 multiply(const Mat4& a, const Mat4& b) {
    Mat4 r{};

    for (int c = 0; c < 4; ++c) {
        for (int row = 0; row < 4; ++row) {
            for (int k = 0; k < 4; ++k) {
                r.m[c * 4 + row] +=
                    a.m[k * 4 + row] * b.m[c * 4 + k];
            }
        }
    }

    return r;
}

Mat4 scaleMatrix(float s) {
    Mat4 r = identity();
    r.m[0] = s;
    r.m[5] = s;
    r.m[10] = s;
    return r;
}

Mat4 translateMatrix(float x, float y, float z) {
    Mat4 r = identity();
    r.m[12] = x;
    r.m[13] = y;
    r.m[14] = z;
    return r;
}

Mat4 rotateXMatrix(float a) {
    Mat4 r = identity();
    const float c = std::cos(a);
    const float s = std::sin(a);
    r.m[5] = c;
    r.m[6] = s;
    r.m[9] = -s;
    r.m[10] = c;
    return r;
}

Mat4 rotateYMatrix(float a) {
    Mat4 r = identity();
    const float c = std::cos(a);
    const float s = std::sin(a);
    r.m[0] = c;
    r.m[2] = -s;
    r.m[8] = s;
    r.m[10] = c;
    return r;
}

Mat4 rotateZMatrix(float a) {
    Mat4 r = identity();
    const float c = std::cos(a);
    const float s = std::sin(a);
    r.m[0] = c;
    r.m[1] = s;
    r.m[4] = -s;
    r.m[5] = c;
    return r;
}


Mat4 makePerspective(
    float fovDegrees,
    float aspect,
    float nearValue,
    float farValue
) {
    Mat4 r{};
    const float f = 1.0f / std::tan(fovDegrees * PI / 360.0f);

    r.m[0] = f / aspect;
    r.m[5] = -f;
    r.m[10] = farValue / (nearValue - farValue);
    r.m[11] = -1.0f;
    r.m[14] = farValue * nearValue / (nearValue - farValue);

    return r;
}

Mat4 makeLookAt(Vec3 eye, Vec3 center, Vec3 up) {
    const Vec3 f = normalizeOr(center - eye, {0.0f, 0.0f, -1.0f});
    const Vec3 s = normalizeOr(cross(f, up), {1.0f, 0.0f, 0.0f});
    const Vec3 u = cross(s, f);

    Mat4 r = identity();
    r.m[0] = s.x;  r.m[4] = s.y;  r.m[8] = s.z;
    r.m[1] = u.x;  r.m[5] = u.y;  r.m[9] = u.z;
    r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z;
    r.m[12] = -dot(s, eye);
    r.m[13] = -dot(u, eye);
    r.m[14] = dot(f, eye);
    return r;
}



struct Vertex {
    float position[3];
    float normal[3];
};


struct SceneUniform {
    float viewProj[16];
    float cameraPos[4];
    float ambient[4];
    float dirDirection[4];
    float dirColor[4];
    uint32_t counts[4];
};
static_assert(sizeof(SceneUniform) == 144, "SceneUniform must match std140 layout");


struct ModelUniform {
    float model[16];
    float albedo[4];
    float specular[4]; // rgb + shininess
};
static_assert(sizeof(ModelUniform) == 96, "ModelUniform must match std140 layout");


struct PointLightGpu {
    float position[4];
    float color[4]; // rgb + intensity
};
static_assert(sizeof(PointLightGpu) == 32, "PointLightGpu must match std430 layout");


struct SpotLightGpu {
    float position[4];
    float direction[4];
    float color[4]; // rgb + intensity
    float cone[4];  // cos(inner), cos(outer)
};
static_assert(sizeof(SpotLightGpu) == 64, "SpotLightGpu must match std430 layout");

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    void* mapped = nullptr;
};

constexpr uint32_t MODEL_COUNT = 2; // 0 = cylinder, 1 = floor

struct DrawRange {
    uint32_t first;
    uint32_t count;
};

Buffer vertexBuffer;
Buffer sceneBuffer;
Buffer pointLightBuffer;
Buffer spotLightBuffer;
Buffer modelBuffers[MODEL_COUNT];

VkDescriptorSetLayout sceneSetLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout modelSetLayout = VK_NULL_HANDLE;
VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
VkDescriptorSet sceneSet = VK_NULL_HANDLE;
VkDescriptorSet modelSets[MODEL_COUNT] = {VK_NULL_HANDLE, VK_NULL_HANDLE};

VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;

std::vector<Vertex> vertices;
DrawRange drawRanges[MODEL_COUNT]{};

// =============================================================== scene state

struct Material {
    float albedo[3];
    float specular[3];
    float shininess;
};

struct PointLightState {
    float position[3];
    float color[3];
    float intensity;
};

struct SpotLightState {
    float position[3];
    float direction[3];
    float color[3];
    float intensity;
    float innerAngle; 
    float outerAngle; 
};

// Camera
float cameraPosition[3];
float cameraYaw;  
float cameraPitch; 
float cameraFov;   
float cameraSpeed;

// Ambient + directional light
float ambientColor[3];
float ambientIntensity;
bool directionalEnabled;
float directionalDirection[3];
float directionalColor[3];
float directionalIntensity;

// Lights from the storage buffers
std::vector<PointLightState> pointLights;
std::vector<SpotLightState> spotLights;

// Cylinder
float cylinderPosition[3];
float cylinderRotation[3]; // degrees
float cylinderScale;
bool autoRotate;
float rotationSpeed;
Material cylinderMaterial;

// Floor
Material floorMaterial;

double previousTime = -1.0;

PointLightState makeDefaultPointLight() {
    return PointLightState{
        {-3.0f, 2.0f, 2.0f},
        {1.0f, 0.65f, 0.35f},
        12.0f
    };
}

SpotLightState makeDefaultSpotLight() {
    return SpotLightState{
        {3.0f, 5.0f, 3.0f},
        {-0.5f, -0.8f, -0.5f},
        {1.0f, 1.0f, 1.0f},
        60.0f,
        12.0f,
        22.0f
    };
}

void resetScene() {
    cameraPosition[0] = 0.0f;
    cameraPosition[1] = 3.5f;
    cameraPosition[2] = 8.0f;
    cameraYaw = 0.0f;
    cameraPitch = -18.0f;
    cameraFov = 50.0f;
    cameraSpeed = 4.0f;

    ambientColor[0] = 0.20f;
    ambientColor[1] = 0.22f;
    ambientColor[2] = 0.30f;
    ambientIntensity = 1.0f;

    directionalEnabled = true;
    directionalDirection[0] = -0.5f;
    directionalDirection[1] = -1.0f;
    directionalDirection[2] = -0.4f;
    directionalColor[0] = 1.0f;
    directionalColor[1] = 0.97f;
    directionalColor[2] = 0.90f;
    directionalIntensity = 0.7f;

    pointLights.clear();
    pointLights.push_back(makeDefaultPointLight());

    spotLights.clear();
    spotLights.push_back(makeDefaultSpotLight());

    cylinderPosition[0] = 0.0f;
    cylinderPosition[1] = 1.0f;
    cylinderPosition[2] = 0.0f;
    cylinderRotation[0] = 0.0f;
    cylinderRotation[1] = 0.0f;
    cylinderRotation[2] = 0.0f;
    cylinderScale = 1.0f;
    autoRotate = false;
    rotationSpeed = 25.0f;

    cylinderMaterial = Material{
        {0.20f, 0.55f, 0.90f},
        {1.0f, 1.0f, 1.0f},
        64.0f
    };

    floorMaterial = Material{
        {0.60f, 0.60f, 0.60f},
        {0.20f, 0.20f, 0.20f},
        16.0f
    };
}

Mat4 makeCylinderModel() {
    const float rx = toRadians(cylinderRotation[0]);
    const float ry = toRadians(cylinderRotation[1]);
    const float rz = toRadians(cylinderRotation[2]);

    // Model = T * Rz * Ry * Rx * S
    return multiply(
        translateMatrix(cylinderPosition[0], cylinderPosition[1], cylinderPosition[2]),
        multiply(
            rotateZMatrix(rz),
            multiply(
                rotateYMatrix(ry),
                multiply(rotateXMatrix(rx), scaleMatrix(cylinderScale))
            )
        )
    );
}

Vec3 cameraForward() {
    const float yaw = toRadians(cameraYaw);
    const float pitch = toRadians(cameraPitch);
    return {
        std::sin(yaw) * std::cos(pitch),
        std::sin(pitch),
        -std::cos(yaw) * std::cos(pitch)
    };
}

Vec3 cameraRight() {
    const float yaw = toRadians(cameraYaw);
    return {std::cos(yaw), 0.0f, std::sin(yaw)};
}

// =============================================================== geometry

void addVertex(Vec3 position, Vec3 normal) {
    vertices.push_back({
        {position.x, position.y, position.z},
        {normal.x, normal.y, normal.z}
    });
}

void createGeometry() {
    vertices.clear();
    vertices.reserve(SEGMENTS * 12 + 6);

    constexpr float radius = 1.0f;
    constexpr float halfHeight = 1.0f;

    
    for (uint32_t i = 0; i < SEGMENTS; ++i) {
        const float a0 = 2.0f * PI * static_cast<float>(i) / SEGMENTS;
        const float a1 = 2.0f * PI * static_cast<float>(i + 1) / SEGMENTS;

        const float c0 = std::cos(a0);
        const float s0 = std::sin(a0);
        const float c1 = std::cos(a1);
        const float s1 = std::sin(a1);

        const Vec3 p0{radius * c0, -halfHeight, radius * s0};
        const Vec3 p1{radius * c1, -halfHeight, radius * s1};
        const Vec3 p2{radius * c1, halfHeight, radius * s1};
        const Vec3 p3{radius * c0, halfHeight, radius * s0};

       
        const Vec3 n0{c0, 0.0f, s0};
        const Vec3 n1{c1, 0.0f, s1};

        const Vec3 up{0.0f, 1.0f, 0.0f};
        const Vec3 down{0.0f, -1.0f, 0.0f};

        addVertex(p0, n0);
        addVertex(p1, n1);
        addVertex(p2, n1);

        addVertex(p0, n0);
        addVertex(p2, n1);
        addVertex(p3, n0);

        
        addVertex({0.0f, halfHeight, 0.0f}, up);
        addVertex(p3, up);
        addVertex(p2, up);

        
        addVertex({0.0f, -halfHeight, 0.0f}, down);
        addVertex(p1, down);
        addVertex(p0, down);
    }

    drawRanges[0] = {0, static_cast<uint32_t>(vertices.size())};

    // ---- floor
    constexpr float floorHalfSize = 12.0f;
    const Vec3 up{0.0f, 1.0f, 0.0f};

    const uint32_t floorStart = static_cast<uint32_t>(vertices.size());

    addVertex({-floorHalfSize, 0.0f, -floorHalfSize}, up);
    addVertex({-floorHalfSize, 0.0f, floorHalfSize}, up);
    addVertex({floorHalfSize, 0.0f, floorHalfSize}, up);

    addVertex({-floorHalfSize, 0.0f, -floorHalfSize}, up);
    addVertex({floorHalfSize, 0.0f, floorHalfSize}, up);
    addVertex({floorHalfSize, 0.0f, -floorHalfSize}, up);

    drawRanges[1] = {floorStart, 6};
}

// =============================================================== Vulkan helpers

bool createBuffer(Buffer& buffer, VkDeviceSize size, VkBufferUsageFlags usage) {
    auto& context = graphics::internal::context;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocationInfo.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
        VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo mappedInfo{};

    if (vmaCreateBuffer(
            context.allocator,
            &bufferInfo,
            &allocationInfo,
            &buffer.buffer,
            &buffer.allocation,
            &mappedInfo) != VK_SUCCESS) {
        std::cerr << "Failed to create buffer\n";
        return false;
    }

    buffer.mapped = mappedInfo.pMappedData;
    return buffer.mapped != nullptr;
}

void destroyBuffer(Buffer& buffer) {
    if (buffer.buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(
            graphics::internal::context.allocator,
            buffer.buffer,
            buffer.allocation
        );
    }
    buffer = Buffer{};
}

void uploadBuffer(Buffer& buffer, const void* data, size_t bytes) {
    if (bytes == 0 || buffer.mapped == nullptr) {
        return;
    }

    std::memcpy(buffer.mapped, data, bytes);
    vmaFlushAllocation(
        graphics::internal::context.allocator,
        buffer.allocation,
        0,
        VK_WHOLE_SIZE
    );
}

bool createBuffers() {
    if (!createBuffer(
            vertexBuffer,
            vertices.size() * sizeof(Vertex),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)) {
        return false;
    }

    uploadBuffer(vertexBuffer, vertices.data(), vertices.size() * sizeof(Vertex));

    if (!createBuffer(sceneBuffer, sizeof(SceneUniform),
                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)) {
        return false;
    }

    if (!createBuffer(pointLightBuffer, sizeof(PointLightGpu) * MAX_POINT_LIGHTS,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
        return false;
    }

    if (!createBuffer(spotLightBuffer, sizeof(SpotLightGpu) * MAX_SPOT_LIGHTS,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
        return false;
    }

    for (uint32_t i = 0; i < MODEL_COUNT; ++i) {
        if (!createBuffer(modelBuffers[i], sizeof(ModelUniform),
                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)) {
            return false;
        }
    }

    return true;
}

bool createDescriptors() {
    auto& context = graphics::internal::context;

    // ---- set 0: scene (uniform buffer + two storage buffers)
    VkDescriptorSetLayoutBinding sceneBindings[3]{};

    sceneBindings[0].binding = 0;
    sceneBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sceneBindings[0].descriptorCount = 1;
    sceneBindings[0].stageFlags =
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    sceneBindings[1].binding = 1;
    sceneBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sceneBindings[1].descriptorCount = 1;
    sceneBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    sceneBindings[2].binding = 2;
    sceneBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sceneBindings[2].descriptorCount = 1;
    sceneBindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo sceneLayoutInfo{};
    sceneLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    sceneLayoutInfo.bindingCount = 3;
    sceneLayoutInfo.pBindings = sceneBindings;

    if (vkCreateDescriptorSetLayout(
            context.device, &sceneLayoutInfo, nullptr, &sceneSetLayout) != VK_SUCCESS) {
        std::cerr << "Failed to create scene descriptor set layout\n";
        return false;
    }

    // ---- set 1: model (uniform buffer with transform + material)
    VkDescriptorSetLayoutBinding modelBinding{};
    modelBinding.binding = 0;
    modelBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    modelBinding.descriptorCount = 1;
    modelBinding.stageFlags =
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo modelLayoutInfo{};
    modelLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    modelLayoutInfo.bindingCount = 1;
    modelLayoutInfo.pBindings = &modelBinding;

    if (vkCreateDescriptorSetLayout(
            context.device, &modelLayoutInfo, nullptr, &modelSetLayout) != VK_SUCCESS) {
        std::cerr << "Failed to create model descriptor set layout\n";
        return false;
    }

    // ---- pool
    VkDescriptorPoolSize poolSizes[2]{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = 1 + MODEL_COUNT;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = 2;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1 + MODEL_COUNT;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;

    if (vkCreateDescriptorPool(
            context.device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor pool\n";
        return false;
    }

    // ---- allocate sets: [scene, model0, model1]
    const VkDescriptorSetLayout layouts[1 + MODEL_COUNT] = {
        sceneSetLayout, modelSetLayout, modelSetLayout
    };
    VkDescriptorSet sets[1 + MODEL_COUNT]{};

    VkDescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = descriptorPool;
    allocateInfo.descriptorSetCount = 1 + MODEL_COUNT;
    allocateInfo.pSetLayouts = layouts;

    if (vkAllocateDescriptorSets(context.device, &allocateInfo, sets) != VK_SUCCESS) {
        std::cerr << "Failed to allocate descriptor sets\n";
        return false;
    }

    sceneSet = sets[0];
    for (uint32_t i = 0; i < MODEL_COUNT; ++i) {
        modelSets[i] = sets[1 + i];
    }

    // ---- write descriptors
    const VkDescriptorBufferInfo sceneInfo{
        sceneBuffer.buffer, 0, sizeof(SceneUniform)};
    const VkDescriptorBufferInfo pointInfo{
        pointLightBuffer.buffer, 0, sizeof(PointLightGpu) * MAX_POINT_LIGHTS};
    const VkDescriptorBufferInfo spotInfo{
        spotLightBuffer.buffer, 0, sizeof(SpotLightGpu) * MAX_SPOT_LIGHTS};
    const VkDescriptorBufferInfo modelInfos[MODEL_COUNT] = {
        {modelBuffers[0].buffer, 0, sizeof(ModelUniform)},
        {modelBuffers[1].buffer, 0, sizeof(ModelUniform)},
    };

    std::array<VkWriteDescriptorSet, 3 + MODEL_COUNT> writes{};

    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = sceneSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[0].pBufferInfo = &sceneInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = sceneSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &pointInfo;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = sceneSet;
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].pBufferInfo = &spotInfo;

    for (uint32_t i = 0; i < MODEL_COUNT; ++i) {
        writes[3 + i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3 + i].dstSet = modelSets[i];
        writes[3 + i].dstBinding = 0;
        writes[3 + i].descriptorCount = 1;
        writes[3 + i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[3 + i].pBufferInfo = &modelInfos[i];
    }

    vkUpdateDescriptorSets(
        context.device,
        static_cast<uint32_t>(writes.size()),
        writes.data(),
        0,
        nullptr
    );

    return true;
}

std::vector<char> readShader(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);

    if (!file) {
        std::cerr << "Cannot open shader: " << path << '\n';
        return {};
    }

    const auto size = file.tellg();
    if (size <= 0) {
        return {};
    }

    std::vector<char> code(static_cast<size_t>(size));
    file.seekg(0);

    if (!file.read(code.data(), size)) {
        return {};
    }

    return code;
}

VkShaderModule createShaderModule(const std::vector<char>& code) {
    if (code.empty()) {
        return VK_NULL_HANDLE;
    }

    VkShaderModuleCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = code.size();
    info.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule module = VK_NULL_HANDLE;

    if (vkCreateShaderModule(
            graphics::internal::context.device,
            &info,
            nullptr,
            &module) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }

    return module;
}

bool createPipeline() {
    auto& context = graphics::internal::context;

    const auto vertexCode = readShader("shaders/cylinder.vert.spv");
    const auto fragmentCode = readShader("shaders/cylinder.frag.spv");

    if (vertexCode.empty() || fragmentCode.empty()) {
        std::cerr << "Shader files missing. Run from project root.\n";
        return false;
    }

    VkShaderModule vertexShader = createShaderModule(vertexCode);
    VkShaderModule fragmentShader = createShaderModule(fragmentCode);

    if (vertexShader == VK_NULL_HANDLE ||
        fragmentShader == VK_NULL_HANDLE) {
        if (vertexShader != VK_NULL_HANDLE) {
            vkDestroyShaderModule(context.device, vertexShader, nullptr);
        }
        if (fragmentShader != VK_NULL_HANDLE) {
            vkDestroyShaderModule(context.device, fragmentShader, nullptr);
        }
        return false;
    }

    // Two descriptor sets: 0 = scene, 1 = model. No push constants.
    const VkDescriptorSetLayout setLayouts[2] = {sceneSetLayout, modelSetLayout};

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 2;
    layoutInfo.pSetLayouts = setLayouts;

    if (vkCreatePipelineLayout(
            context.device,
            &layoutInfo,
            nullptr,
            &pipelineLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(context.device, vertexShader, nullptr);
        vkDestroyShaderModule(context.device, fragmentShader, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo stages[2]{};

    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertexShader;
    stages[0].pName = "main";

    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragmentShader;
    stages[1].pName = "main";

    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = sizeof(Vertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::array<VkVertexInputAttributeDescription, 2> attributes{};

    attributes[0].location = 0;
    attributes[0].binding = 0;
    attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[0].offset = offsetof(Vertex, position);

    attributes[1].location = 1;
    attributes[1].binding = 0;
    attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[1].offset = offsetof(Vertex, normal);

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(attributes.size());
    vertexInput.pVertexAttributeDescriptions = attributes.data();

    VkPipelineInputAssemblyStateCreateInfo assembly{};
    assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth{};
    depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = VK_TRUE;
    depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    depth.minDepthBounds = 0.0f;
    depth.maxDepthBounds = 1.0f;

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT |
        VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT |
        VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments = &blendAttachment;

    const VkDynamicState dynamicStates[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };

    VkPipelineDynamicStateCreateInfo dynamic{};
    dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamicStates;

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stages;
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &assembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &raster;
    pipelineInfo.pMultisampleState = &multisample;
    pipelineInfo.pDepthStencilState = &depth;
    pipelineInfo.pColorBlendState = &blend;
    pipelineInfo.pDynamicState = &dynamic;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = context.render_pass;
    pipelineInfo.subpass = 0;

    const VkResult result = vkCreateGraphicsPipelines(
        context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline
    );

    vkDestroyShaderModule(context.device, vertexShader, nullptr);
    vkDestroyShaderModule(context.device, fragmentShader, nullptr);

    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create graphics pipeline\n";
        return false;
    }

    return true;
}

// Safe to call several times / after a partial initialization.
void destroyResources() {
    auto& context = graphics::internal::context;

    if (context.device == VK_NULL_HANDLE) {
        return;
    }

    vkDeviceWaitIdle(context.device);

    if (pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(context.device, pipeline, nullptr);
        pipeline = VK_NULL_HANDLE;
    }

    if (pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);
        pipelineLayout = VK_NULL_HANDLE;
    }

    if (descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(context.device, descriptorPool, nullptr);
        descriptorPool = VK_NULL_HANDLE;
    }

    if (sceneSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(context.device, sceneSetLayout, nullptr);
        sceneSetLayout = VK_NULL_HANDLE;
    }

    if (modelSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(context.device, modelSetLayout, nullptr);
        modelSetLayout = VK_NULL_HANDLE;
    }

    sceneSet = VK_NULL_HANDLE;
    for (auto& set : modelSets) {
        set = VK_NULL_HANDLE;
    }

    destroyBuffer(vertexBuffer);
    destroyBuffer(sceneBuffer);
    destroyBuffer(pointLightBuffer);
    destroyBuffer(spotLightBuffer);
    for (auto& buffer : modelBuffers) {
        destroyBuffer(buffer);
    }
}

// =============================================================== GPU upload

void fillMaterial(ModelUniform& uniform, const Material& material) {
    for (int i = 0; i < 3; ++i) {
        uniform.albedo[i] = material.albedo[i];
        uniform.specular[i] = material.specular[i];
    }
    uniform.albedo[3] = 1.0f;
    uniform.specular[3] = std::max(material.shininess, 1.0f);
}

void uploadSceneData(const Mat4& viewProj) {
    // ---- scene uniform
    SceneUniform scene{};

    std::memcpy(scene.viewProj, viewProj.m, sizeof(scene.viewProj));

    for (int i = 0; i < 3; ++i) {
        scene.cameraPos[i] = cameraPosition[i];
        scene.ambient[i] = ambientColor[i] * ambientIntensity;
    }
    scene.cameraPos[3] = 1.0f;

    const Vec3 dir = normalizeOr(toVec3(directionalDirection), {0.0f, -1.0f, 0.0f});
    scene.dirDirection[0] = dir.x;
    scene.dirDirection[1] = dir.y;
    scene.dirDirection[2] = dir.z;

    const float dirScale = directionalEnabled ? directionalIntensity : 0.0f;
    for (int i = 0; i < 3; ++i) {
        scene.dirColor[i] = directionalColor[i] * dirScale;
    }

    // ---- point lights (storage buffer)
    std::vector<PointLightGpu> points;
    points.reserve(pointLights.size());

    for (const auto& light : pointLights) {
        if (points.size() >= MAX_POINT_LIGHTS) {
            break;
        }

        PointLightGpu gpu{};
        for (int i = 0; i < 3; ++i) {
            gpu.position[i] = light.position[i];
            gpu.color[i] = light.color[i];
        }
        gpu.color[3] = light.intensity;
        points.push_back(gpu);
    }

    // ---- spot lights (storage buffer)
    std::vector<SpotLightGpu> spots;
    spots.reserve(spotLights.size());

    for (const auto& light : spotLights) {
        if (spots.size() >= MAX_SPOT_LIGHTS) {
            break;
        }

        SpotLightGpu gpu{};
        const Vec3 direction =
            normalizeOr(toVec3(light.direction), {0.0f, -1.0f, 0.0f});

        for (int i = 0; i < 3; ++i) {
            gpu.position[i] = light.position[i];
            gpu.color[i] = light.color[i];
        }
        gpu.direction[0] = direction.x;
        gpu.direction[1] = direction.y;
        gpu.direction[2] = direction.z;
        gpu.color[3] = light.intensity;

        // The outer angle can never be smaller than the inner one.
        const float outer = std::max(light.outerAngle, light.innerAngle);
        gpu.cone[0] = std::cos(toRadians(light.innerAngle));
        gpu.cone[1] = std::cos(toRadians(outer));

        spots.push_back(gpu);
    }

    scene.counts[0] = static_cast<uint32_t>(points.size());
    scene.counts[1] = static_cast<uint32_t>(spots.size());

    uploadBuffer(sceneBuffer, &scene, sizeof(scene));
    uploadBuffer(pointLightBuffer, points.data(), points.size() * sizeof(PointLightGpu));
    uploadBuffer(spotLightBuffer, spots.data(), spots.size() * sizeof(SpotLightGpu));
}

void uploadModelData() {
    // ---- cylinder
    ModelUniform cylinder{};
    const Mat4 cylinderModel = makeCylinderModel();
    std::memcpy(cylinder.model, cylinderModel.m, sizeof(cylinder.model));
    fillMaterial(cylinder, cylinderMaterial);
    uploadBuffer(modelBuffers[0], &cylinder, sizeof(cylinder));

    // ---- floor
    ModelUniform floorUniform{};
    const Mat4 floorModel = identity();
    std::memcpy(floorUniform.model, floorModel.m, sizeof(floorUniform.model));
    fillMaterial(floorUniform, floorMaterial);
    uploadBuffer(modelBuffers[1], &floorUniform, sizeof(floorUniform));
}

// =============================================================== UI

void materialUi(const char* label, Material& material) {
    if (ImGui::TreeNodeEx(label, ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ColorEdit3("Albedo", material.albedo);
        ImGui::ColorEdit3("Specular color", material.specular);
        ImGui::SliderFloat("Shininess", &material.shininess, 1.0f, 256.0f, "%.1f");
        ImGui::TreePop();
    }
}

void cameraUi() {
    ImGui::DragFloat3("Position", cameraPosition, 0.05f);
    ImGui::SliderFloat("Yaw", &cameraYaw, -180.0f, 180.0f, "%.1f deg");
    ImGui::SliderFloat("Pitch", &cameraPitch, -89.0f, 89.0f, "%.1f deg");
    ImGui::SliderFloat("Field of view", &cameraFov, 20.0f, 120.0f, "%.1f deg");
    ImGui::SliderFloat("Move speed", &cameraSpeed, 0.5f, 20.0f, "%.1f");
    ImGui::TextDisabled("WASD + Q/E move, hold Shift to speed up,");
    ImGui::TextDisabled("hold right mouse button to look around.");

    if (ImGui::Button("Reset camera")) {
        cameraPosition[0] = 0.0f;
        cameraPosition[1] = 3.5f;
        cameraPosition[2] = 8.0f;
        cameraYaw = 0.0f;
        cameraPitch = -18.0f;
        cameraFov = 50.0f;
    }
}

void lightingUi() {
    ImGui::ColorEdit3("Ambient color", ambientColor);
    ImGui::SliderFloat("Ambient intensity", &ambientIntensity, 0.0f, 2.0f, "%.2f");
    ImGui::Separator();
    ImGui::Checkbox("Directional light enabled", &directionalEnabled);
    ImGui::DragFloat3("Light direction", directionalDirection, 0.01f, -1.0f, 1.0f);
    ImGui::ColorEdit3("Light color", directionalColor);
    ImGui::SliderFloat("Light intensity", &directionalIntensity, 0.0f, 5.0f, "%.2f");
}

void pointLightsUi() {
    ImGui::Text("Point lights: %d / %u", static_cast<int>(pointLights.size()), MAX_POINT_LIGHTS);

    int removeIndex = -1;

    // Separate the IDs of point-light widgets from spot-light widgets.
    ImGui::PushID("PointLights");

    for (size_t i = 0; i < pointLights.size(); ++i) {
        auto& light = pointLights[i];

        ImGui::PushID(static_cast<int>(i));

        if (ImGui::TreeNodeEx("light", ImGuiTreeNodeFlags_DefaultOpen,
                              "Point light %d", static_cast<int>(i) + 1)) {
            ImGui::DragFloat3("Position", light.position, 0.05f);
            ImGui::ColorEdit3("Color", light.color);
            ImGui::DragFloat("Intensity", &light.intensity, 0.1f, 0.0f, 1000.0f, "%.1f");

            if (ImGui::Button("Remove")) {
                removeIndex = static_cast<int>(i);
            }

            ImGui::TreePop();
        }

        ImGui::PopID();
    }

    ImGui::PopID();

    if (removeIndex >= 0) {
        pointLights.erase(pointLights.begin() + removeIndex);
    }

    if (pointLights.size() < MAX_POINT_LIGHTS && ImGui::Button("Add point light")) {
        pointLights.push_back(makeDefaultPointLight());
    }
}

void spotLightsUi() {
    ImGui::Text("Spot lights: %d / %u", static_cast<int>(spotLights.size()), MAX_SPOT_LIGHTS);

    int removeIndex = -1;

    // Separate the IDs of spot-light widgets from point-light widgets.
    ImGui::PushID("SpotLights");

    for (size_t i = 0; i < spotLights.size(); ++i) {
        auto& light = spotLights[i];

        ImGui::PushID(static_cast<int>(i));

        if (ImGui::TreeNodeEx("light", ImGuiTreeNodeFlags_DefaultOpen,
                              "Spot light %d", static_cast<int>(i) + 1)) {
            ImGui::DragFloat3("Position", light.position, 0.05f);
            ImGui::DragFloat3("Direction", light.direction, 0.01f, -1.0f, 1.0f);
            ImGui::ColorEdit3("Color", light.color);
            ImGui::DragFloat("Intensity", &light.intensity, 0.1f, 0.0f, 2000.0f, "%.1f");
            ImGui::SliderFloat("Inner angle", &light.innerAngle, 0.0f, 89.0f, "%.1f deg");
            ImGui::SliderFloat("Outer angle", &light.outerAngle, 1.0f, 90.0f, "%.1f deg");

            // Outer angle must not be smaller than the inner one.
            if (light.innerAngle > light.outerAngle) {
                light.innerAngle = light.outerAngle;
            }

            if (ImGui::Button("Remove")) {
                removeIndex = static_cast<int>(i);
            }

            ImGui::TreePop();
        }

        ImGui::PopID();
    }

    ImGui::PopID();

    if (removeIndex >= 0) {
        spotLights.erase(spotLights.begin() + removeIndex);
    }

    if (spotLights.size() < MAX_SPOT_LIGHTS && ImGui::Button("Add spot light")) {
        spotLights.push_back(makeDefaultSpotLight());
    }
}

void cylinderUi() {
    ImGui::DragFloat3("Position", cylinderPosition, 0.05f);
    ImGui::SliderFloat("Rotation X", &cylinderRotation[0], -180.0f, 180.0f, "%.1f deg");
    ImGui::SliderFloat("Rotation Y", &cylinderRotation[1], -180.0f, 180.0f, "%.1f deg");
    ImGui::SliderFloat("Rotation Z", &cylinderRotation[2], -180.0f, 180.0f, "%.1f deg");
    ImGui::SliderFloat("Scale", &cylinderScale, 0.3f, 2.0f, "%.2f");
    ImGui::Checkbox("Auto rotate (Y)", &autoRotate);
    ImGui::SliderFloat("Rotation speed", &rotationSpeed, 0.0f, 120.0f, "%.1f deg/s");
    materialUi("Cylinder material", cylinderMaterial);
}

void updateCamera(float delta) {
    ImGuiIO& io = ImGui::GetIO();

    // Look around: hold the right mouse button outside of ImGui windows.
    if (!io.WantCaptureMouse && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
        cameraYaw += io.MouseDelta.x * 0.15f;
        cameraPitch -= io.MouseDelta.y * 0.15f;
    }

    cameraPitch = std::clamp(cameraPitch, -89.0f, 89.0f);

    if (cameraYaw > 180.0f) cameraYaw -= 360.0f;
    if (cameraYaw < -180.0f) cameraYaw += 360.0f;

    // Move: WASD + Q/E.
    if (!io.WantCaptureKeyboard) {
        float speed = cameraSpeed * delta;
        if (ImGui::IsKeyDown(ImGuiKey_LeftShift)) {
            speed *= 3.0f;
        }

        const Vec3 forward = cameraForward();
        const Vec3 right = cameraRight();
        const Vec3 worldUp{0.0f, 1.0f, 0.0f};

        Vec3 position = toVec3(cameraPosition);

        if (ImGui::IsKeyDown(ImGuiKey_W)) position = position + forward * speed;
        if (ImGui::IsKeyDown(ImGuiKey_S)) position = position - forward * speed;
        if (ImGui::IsKeyDown(ImGuiKey_D)) position = position + right * speed;
        if (ImGui::IsKeyDown(ImGuiKey_A)) position = position - right * speed;
        if (ImGui::IsKeyDown(ImGuiKey_E)) position = position + worldUp * speed;
        if (ImGui::IsKeyDown(ImGuiKey_Q)) position = position - worldUp * speed;

        cameraPosition[0] = position.x;
        cameraPosition[1] = position.y;
        cameraPosition[2] = position.z;
    }
}

} // namespace

// =============================================================== public API

bool initialize() {
    resetScene();
    createGeometry();

    if (!createBuffers() || !createDescriptors() || !createPipeline()) {
        destroyResources();
        return false;
    }

    return true;
}

void shutdown() {
    destroyResources();
}

void update(double time) {
    float delta = 0.0f;
    if (previousTime >= 0.0) {
        delta = static_cast<float>(time - previousTime);
        delta = std::clamp(delta, 0.0f, 0.1f);
    }
    previousTime = time;

    updateCamera(delta);

    ImGui::SetNextWindowSize(ImVec2(420.0f, 640.0f), ImGuiCond_FirstUseEver);
    ImGui::Begin("Lab 2 - Variant 8: Cylinder");

    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        cameraUi();
    }

    if (ImGui::CollapsingHeader("Ambient and directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
        lightingUi();
    }

    if (ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
        pointLightsUi();
    }

    if (ImGui::CollapsingHeader("Spot lights", ImGuiTreeNodeFlags_DefaultOpen)) {
        spotLightsUi();
    }

    if (ImGui::CollapsingHeader("Cylinder", ImGuiTreeNodeFlags_DefaultOpen)) {
        cylinderUi();
    }

    if (ImGui::CollapsingHeader("Floor")) {
        materialUi("Floor material", floorMaterial);
    }

    ImGui::Separator();
    if (ImGui::Button("Reset whole scene")) {
        resetScene();
    }

    ImGui::End();

    if (autoRotate) {
        cylinderRotation[1] += delta * rotationSpeed;
        if (cylinderRotation[1] > 180.0f) cylinderRotation[1] -= 360.0f;
    }
}

void render(const graphics::internal::FrameData& fd) {
    auto& context = graphics::internal::context;

    if (pipeline == VK_NULL_HANDLE ||
        vertexBuffer.buffer == VK_NULL_HANDLE ||
        fd.command_buffer == VK_NULL_HANDLE ||
        context.swapchain_extent.width == 0 ||
        context.swapchain_extent.height == 0) {
        return;
    }

    VkCommandBuffer cmd = fd.command_buffer;

   
    const float aspect =
        static_cast<float>(context.swapchain_extent.width) /
        static_cast<float>(context.swapchain_extent.height);

    const Vec3 eye = toVec3(cameraPosition);
    const Mat4 view = makeLookAt(eye, eye + cameraForward(), {0.0f, 1.0f, 0.0f});
    const Mat4 projection = makePerspective(cameraFov, aspect, 0.1f, 100.0f);
    const Mat4 viewProj = multiply(projection, view);

    uploadSceneData(viewProj);
    uploadModelData();

    if (vkResetCommandBuffer(cmd, 0) != VK_SUCCESS) {
        return;
    }

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(cmd, &begin) != VK_SUCCESS) {
        return;
    }

    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = {{0.04f, 0.05f, 0.08f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = context.render_pass;
    renderPassInfo.framebuffer = fd.framebuffer;
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = context.swapchain_extent;
    renderPassInfo.clearValueCount =
        static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport{};
    viewport.width = static_cast<float>(context.swapchain_extent.width);
    viewport.height = static_cast<float>(context.swapchain_extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.extent = context.swapchain_extent;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    const VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer.buffer, &offset);

    // Set 0: scene data, bound once for the whole frame.
    vkCmdBindDescriptorSets(
        cmd,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        pipelineLayout,
        0,
        1,
        &sceneSet,
        0,
        nullptr
    );

    // Set 1: model data, rebound for every object.
    for (uint32_t i = 0; i < MODEL_COUNT; ++i) {
        vkCmdBindDescriptorSets(
            cmd,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipelineLayout,
            1,
            1,
            &modelSets[i],
            0,
            nullptr
        );

        vkCmdDraw(cmd, drawRanges[i].count, 1, drawRanges[i].first, 0);
    }

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}

} // namespace application