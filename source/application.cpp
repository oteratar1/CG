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

struct Vertex {
    float position[3];
    float color[3];
};

// Column-major 4x4 matrix (as expected by GLSL / Vulkan)
struct Mat4 {
    float m[16]{};
};

VkBuffer vertexBuffer = VK_NULL_HANDLE;
VmaAllocation vertexAllocation = VK_NULL_HANDLE;
VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;

std::vector<Vertex> vertices;

// Transform state
float rotationX = -20.0f;
float rotationY = 0.0f;
float rotationZ = 0.0f;
float rotationSpeed = 25.0f;
bool autoRotate = true;
double previousTime = -1.0;

float positionX = 0.0f;
float positionY = 0.0f;
float positionZ = -4.0f;

float objectScale = 1.0f;

// Projection: true = perspective, false = orthographic
bool perspectiveProjection = true;

// ---------------------------------------------------------------- matrices

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

// Right-handed perspective, depth 0..1, Y flipped for Vulkan.
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

// Right-handed orthographic, depth 0..1, Y flipped for Vulkan.
// halfHeight = half of the visible height in world units.
Mat4 makeOrthographic(
    float halfHeight,
    float aspect,
    float nearValue,
    float farValue
) {
    Mat4 r{};

    r.m[0] = 1.0f / (halfHeight * aspect);
    r.m[5] = -1.0f / halfHeight;
    r.m[10] = 1.0f / (nearValue - farValue);
    r.m[14] = nearValue / (nearValue - farValue);
    r.m[15] = 1.0f;

    return r;
}

Mat4 makeModel() {
    const float rx = rotationX * PI / 180.0f;
    const float ry = rotationY * PI / 180.0f;
    const float rz = rotationZ * PI / 180.0f;

    // Model = T * Rz * Ry * Rx * S
    return multiply(
        translateMatrix(positionX, positionY, positionZ),
        multiply(
            rotateZMatrix(rz),
            multiply(
                rotateYMatrix(ry),
                multiply(rotateXMatrix(rx), scaleMatrix(objectScale))
            )
        )
    );
}

// ---------------------------------------------------------------- geometry

void addTriangle(
    const std::array<float, 3>& a,
    const std::array<float, 3>& b,
    const std::array<float, 3>& c,
    const std::array<float, 3>& color
) {
    vertices.push_back({{a[0], a[1], a[2]},
                        {color[0], color[1], color[2]}});
    vertices.push_back({{b[0], b[1], b[2]},
                        {color[0], color[1], color[2]}});
    vertices.push_back({{c[0], c[1], c[2]},
                        {color[0], color[1], color[2]}});
}

void createCylinderGeometry() {
    vertices.clear();
    vertices.reserve(SEGMENTS * 12);

    constexpr float radius = 1.0f;
    constexpr float halfHeight = 1.0f;

    for (uint32_t i = 0; i < SEGMENTS; ++i) {
        const float a0 = 2.0f * PI * static_cast<float>(i) / SEGMENTS;
        const float a1 = 2.0f * PI * static_cast<float>(i + 1) / SEGMENTS;

        const std::array<float, 3> p0{
            radius * std::cos(a0), -halfHeight, radius * std::sin(a0)
        };
        const std::array<float, 3> p1{
            radius * std::cos(a1), -halfHeight, radius * std::sin(a1)
        };
        const std::array<float, 3> p2{
            radius * std::cos(a1), halfHeight, radius * std::sin(a1)
        };
        const std::array<float, 3> p3{
            radius * std::cos(a0), halfHeight, radius * std::sin(a0)
        };

        const float shade =
            0.65f + 0.35f * (0.5f + 0.5f * std::cos((a0 + a1) * 0.5f));

        const std::array<float, 3> sideColor{
            0.15f * shade, 0.55f * shade, 0.90f * shade
        };
        const std::array<float, 3> topColor{0.25f, 0.65f, 1.0f};
        const std::array<float, 3> bottomColor{0.12f, 0.30f, 0.65f};
        const std::array<float, 3> topCenter{0.0f, halfHeight, 0.0f};
        const std::array<float, 3> bottomCenter{0.0f, -halfHeight, 0.0f};

        // side (two triangles per segment)
        addTriangle(p0, p1, p2, sideColor);
        addTriangle(p0, p2, p3, sideColor);
        // caps
        addTriangle(topCenter, p3, p2, topColor);
        addTriangle(bottomCenter, p1, p0, bottomColor);
    }
}

// ---------------------------------------------------------------- Vulkan

bool createVertexBuffer() {
    auto& context = graphics::internal::context;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = vertices.size() * sizeof(Vertex);
    bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
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
            &vertexBuffer,
            &vertexAllocation,
            &mappedInfo) != VK_SUCCESS) {
        std::cerr << "Failed to create vertex buffer\n";
        return false;
    }

    std::memcpy(
        mappedInfo.pMappedData,
        vertices.data(),
        vertices.size() * sizeof(Vertex)
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

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(Mat4);

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;

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
    attributes[1].offset = offsetof(Vertex, color);

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
        vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);
        pipelineLayout = VK_NULL_HANDLE;
        std::cerr << "Failed to create graphics pipeline\n";
        return false;
    }

    return true;
}

} // namespace

// ---------------------------------------------------------------- public API

bool initialize() {
    createCylinderGeometry();

    if (!createVertexBuffer()) {
        return false;
    }

    if (!createPipeline()) {
        vmaDestroyBuffer(
            graphics::internal::context.allocator,
            vertexBuffer,
            vertexAllocation
        );
        vertexBuffer = VK_NULL_HANDLE;
        vertexAllocation = VK_NULL_HANDLE;
        return false;
    }

    return true;
}

void shutdown() {
    auto& context = graphics::internal::context;

    if (context.device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(context.device);

        if (pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(context.device, pipeline, nullptr);
            pipeline = VK_NULL_HANDLE;
        }

        if (pipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);
            pipelineLayout = VK_NULL_HANDLE;
        }

        if (vertexBuffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(
                context.allocator,
                vertexBuffer,
                vertexAllocation
            );
            vertexBuffer = VK_NULL_HANDLE;
            vertexAllocation = VK_NULL_HANDLE;
        }
    }
}

void update(double time) {
    float delta = 0.0f;
    if (previousTime >= 0.0) {
        delta = static_cast<float>(time - previousTime);
        delta = std::clamp(delta, 0.0f, 0.1f);
    }
    previousTime = time;

    ImGui::Begin("Lab 1 - Variant 8: Cylinder");
    ImGui::Text("Cylinder with %u segments", SEGMENTS);
    ImGui::Separator();

    ImGui::Text("Projection");
    if (ImGui::RadioButton("Perspective", perspectiveProjection)) {
        perspectiveProjection = true;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Orthographic", !perspectiveProjection)) {
        perspectiveProjection = false;
    }
    ImGui::Separator();

    ImGui::Checkbox("Auto rotate", &autoRotate);
    ImGui::SliderFloat("Rotation speed", &rotationSpeed, 0.0f, 120.0f, "%.1f deg/s");
    ImGui::SliderFloat("Rotation X", &rotationX, -180.0f, 180.0f, "%.1f deg");
    ImGui::SliderFloat("Rotation Y", &rotationY, -180.0f, 180.0f, "%.1f deg");
    ImGui::SliderFloat("Rotation Z", &rotationZ, -180.0f, 180.0f, "%.1f deg");
    ImGui::Separator();
    ImGui::SliderFloat("Position X", &positionX, -2.0f, 2.0f, "%.2f");
    ImGui::SliderFloat("Position Y", &positionY, -2.0f, 2.0f, "%.2f");
    ImGui::SliderFloat("Position Z", &positionZ, -10.0f, -2.0f, "%.2f");
    ImGui::SliderFloat("Scale", &objectScale, 0.3f, 2.0f, "%.2f");

    if (ImGui::Button("Reset")) {
        rotationX = -20.0f;
        rotationY = 0.0f;
        rotationZ = 0.0f;
        positionX = 0.0f;
        positionY = 0.0f;
        positionZ = -4.0f;
        objectScale = 1.0f;
        rotationSpeed = 25.0f;
        autoRotate = true;
        perspectiveProjection = true;
    }
    ImGui::End();

    if (autoRotate) {
        rotationY += delta * rotationSpeed;
        if (rotationY > 180.0f) rotationY -= 360.0f;
    }
}

void render(const graphics::internal::FrameData& fd) {
    auto& context = graphics::internal::context;

    if (pipeline == VK_NULL_HANDLE ||
        vertexBuffer == VK_NULL_HANDLE ||
        fd.command_buffer == VK_NULL_HANDLE ||
        context.swapchain_extent.width == 0 ||
        context.swapchain_extent.height == 0) {
        return;
    }

    VkCommandBuffer cmd = fd.command_buffer;

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
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer, &offset);

    const float aspect =
        static_cast<float>(context.swapchain_extent.width) /
        static_cast<float>(context.swapchain_extent.height);

    const Mat4 projection = perspectiveProjection
        ? makePerspective(45.0f, aspect, 0.1f, 100.0f)
        : makeOrthographic(2.0f, aspect, 0.1f, 100.0f);

    const Mat4 mvp = multiply(projection, makeModel());

    vkCmdPushConstants(
        cmd,
        pipelineLayout,
        VK_SHADER_STAGE_VERTEX_BIT,
        0,
        sizeof(Mat4),
        &mvp
    );

    vkCmdDraw(
        cmd,
        static_cast<uint32_t>(vertices.size()),
        1,
        0,
        0
    );

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}

} // namespace application // namespace application
//.\build-debug\vulkan-starter-app.exelkan-starter-app>
//cmake --build build-debug --parallel 1