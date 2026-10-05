#include "application.hpp"
#include "graphics_internal.hpp"
#include <imgui.h>
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <cstring>
#include <cstddef>

namespace application {

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
};

struct MeshPushConstants {
    glm::mat4 mvp;
    glm::vec4 baseColor;
};

VkPipeline       customPipeline         = VK_NULL_HANDLE;
VkPipelineLayout customPipelineLayout   = VK_NULL_HANDLE;
VkBuffer         vertexBuffer           = VK_NULL_HANDLE;
VmaAllocation    vertexBufferAllocation = VK_NULL_HANDLE;
VkBuffer         indexBuffer            = VK_NULL_HANDLE;
VmaAllocation    indexBufferAllocation  = VK_NULL_HANDLE;

static uint32_t g_indexCount = 0;

glm::vec3 shapePosition = glm::vec3(0.0f, 0.0f, 0.0f);
glm::vec3 shapeRotation = glm::vec3(0.0f, 0.0f, 0.0f);
glm::vec3 shapeScale    = glm::vec3(1.0f, 1.0f, 1.0f);

float uiColor[4]    = { 1.0f, 0.7f, 0.2f, 1.0f };
int   isPerspective = 1;

bool  isAnimated     = false;
float animationSpeed = 1.0f;
float pathRadius     = 2.0f;
float animationTime  = 0.0f;

// ---------------------------------------------------------------------------
// ДОДЕКАЭДР
// ---------------------------------------------------------------------------
static const float PHI = 1.61803398875f;

std::vector<Vertex> getDodecahedronVertices() {
    // 20 вершин, порядок ровно как в OBJ-файле
    std::vector<glm::vec3> positions = {
        {-1.0f,       1.0f,      -1.0f},       // 1
        {-1.61803f,   0.0f,       0.618034f},  // 2
        {-1.61803f,   0.0f,      -0.618034f},  // 3
        {-1.0f,       1.0f,       1.0f},       // 4
        {-0.618034f,  1.61803f,   0.0f},       // 5
        { 1.0f,       1.0f,       1.0f},       // 6
        { 0.618034f,  1.61803f,   0.0f},       // 7
        { 0.0f,       0.618034f,  1.61803f},   // 8
        {-1.0f,      -1.0f,       1.0f},       // 9
        { 0.0f,      -0.618034f,  1.61803f},   // 10
        {-1.0f,      -1.0f,      -1.0f},       // 11
        {-0.618034f, -1.61803f,   0.0f},       // 12
        { 0.0f,      -0.618034f, -1.61803f},   // 13
        { 0.0f,       0.618034f, -1.61803f},   // 14
        { 1.0f,       1.0f,      -1.0f},       // 15
        { 1.61803f,   0.0f,      -0.618034f},  // 16
        { 1.61803f,   0.0f,       0.618034f},  // 17
        { 1.0f,      -1.0f,       1.0f},       // 18
        { 0.618034f, -1.61803f,   0.0f},       // 19
        { 1.0f,      -1.0f,      -1.0f},       // 20
    };

    std::vector<Vertex> vertices;
    vertices.reserve(positions.size());

    for (const auto& p : positions) {
        Vertex v;
        v.pos = p;
        v.color = glm::vec3(
            0.5f + 0.5f * p.x / PHI,
            0.5f + 0.5f * p.y / PHI,
            0.5f + 0.5f * p.z / PHI
        );
        vertices.push_back(v);
    }
    return vertices;
}

std::vector<uint16_t> buildDodecahedronIndices() {
    // 36 треугольников 
    static const uint16_t tris[] = {
        1, 0, 2,    // f 2 1 3
        0, 1, 3,    // f 1 2 4
        0, 3, 4,    // f 1 4 5
        4, 5, 6,    // f 5 6 7
        5, 4, 3,    // f 6 5 4
        5, 3, 7,    // f 6 4 8
        8, 3, 1,    // f 9 4 2
        3, 8, 7,    // f 4 9 8
        7, 8, 9,    // f 8 9 10
        8, 10, 11,  // f 9 11 12
        10, 8, 2,   // f 11 9 3
        2, 8, 1,    // f 3 9 2
        0, 10, 2,   // f 1 11 3
        10, 0, 12,  // f 11 1 13
        12, 0, 13,  // f 13 1 14
        0, 14, 13,  // f 1 15 14
        14, 0, 6,   // f 15 1 7
        6, 0, 4,    // f 7 1 5
        15, 5, 16,  // f 16 6 17
        5, 15, 14,  // f 6 16 15
        5, 14, 6,   // f 6 15 7
        9, 5, 7,    // f 10 6 8
        5, 9, 17,   // f 6 10 18
        5, 17, 16,  // f 6 18 17
        18, 8, 11,  // f 19 9 12
        8, 18, 17,  // f 9 19 18
        8, 17, 9,   // f 9 18 10
        19, 10, 12, // f 20 11 13
        10, 19, 11, // f 11 20 12
        11, 19, 18, // f 12 20 19
        13, 19, 12, // f 14 20 13
        19, 13, 14, // f 20 14 15
        19, 14, 15, // f 20 15 16
        19, 17, 18, // f 20 18 19
        17, 19, 16, // f 18 20 17
        16, 19, 15, // f 17 20 16
    };

    std::vector<uint16_t> indices;
    indices.reserve(sizeof(tris) / sizeof(tris[0]));
    for (uint16_t i : tris) {
        indices.push_back(i);
    }
    return indices;
}

// ---------------------------------------------------------------------------
// Загрузка SPIR-V
// ---------------------------------------------------------------------------
std::vector<char> readShaderFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "CRITICAL: Failed to open shader file: " << filename << "\n";
        return {};
    }
    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    return buffer;
}

VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code) {
    if (code.empty() || code.size() % 4 != 0) return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size(),
        .pCode = reinterpret_cast<const uint32_t*>(code.data())
    };
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return shaderModule;
}

// ---------------------------------------------------------------------------
// Инициализация
// ---------------------------------------------------------------------------
bool initialize() {
    auto allocator = graphics::internal::context.allocator;
    auto device    = graphics::internal::context.device;

    VkPushConstantRange pushConstantRange = {
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset = 0,
        .size = sizeof(MeshPushConstants)
    };

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 0,
        .pSetLayouts = nullptr,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstantRange
    };

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
                               &customPipelineLayout) != VK_SUCCESS) {
        std::cerr << "Failed to create pipeline layout\n";
        return false;
    }

    auto vertShaderCode = readShaderFile("shaders/shader.vert.spv");
    auto fragShaderCode = readShaderFile("shaders/shader.frag.spv");

    if (vertShaderCode.empty() || fragShaderCode.empty()) {
        std::cerr << "Failed to read shader files\n";
        return false;
    }

    VkShaderModule vertModule = createShaderModule(device, vertShaderCode);
    VkShaderModule fragModule = createShaderModule(device, fragShaderCode);
    if (vertModule == VK_NULL_HANDLE || fragModule == VK_NULL_HANDLE) {
        std::cerr << "Failed to create shader modules\n";
        return false;
    }

    VkPipelineShaderStageCreateInfo shaderStages[2] = {
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertModule,
            .pName = "main",
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragModule,
            .pName = "main",
        }
    };

    VkVertexInputBindingDescription bindingDesc = {
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX
    };

    VkVertexInputAttributeDescription attribDesc[2] = {
        { .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, pos) },
        { .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, color) }
    };

    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &bindingDesc,
        .vertexAttributeDescriptionCount = 2,
        .pVertexAttributeDescriptions = attribDesc
    };

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE
    };

    VkPipelineViewportStateCreateInfo viewportState = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1
    };

    VkPipelineRasterizationStateCreateInfo rasterizer = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .lineWidth = 1.0f
    };

    VkPipelineMultisampleStateCreateInfo multisampling = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = VK_FALSE
    };

    VkPipelineDepthStencilStateCreateInfo depthStencil = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable = VK_FALSE
    };

    VkPipelineColorBlendAttachmentState colorBlendAttachment = {
        .blendEnable = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
    };

    VkPipelineColorBlendStateCreateInfo colorBlending = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = VK_FALSE,
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttachment,
        .blendConstants = {0.0f, 0.0f, 0.0f, 0.0f}
    };

    VkDynamicState dynamicStates[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicStateInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2,
        .pDynamicStates = dynamicStates
    };

    VkGraphicsPipelineCreateInfo pipelineInfo = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = shaderStages,
        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = &depthStencil,
        .pColorBlendState = &colorBlending,
        .pDynamicState = &dynamicStateInfo,
        .layout = customPipelineLayout,
        .renderPass = graphics::internal::context.render_pass,
        .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1
    };

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                  nullptr, &customPipeline) != VK_SUCCESS) {
        std::cerr << "Failed to create graphics pipeline\n";
        vkDestroyShaderModule(device, fragModule, nullptr);
        vkDestroyShaderModule(device, vertModule, nullptr);
        return false;
    }

    vkDestroyShaderModule(device, fragModule, nullptr);
    vkDestroyShaderModule(device, vertModule, nullptr);

    // --- Вершинный буфер (ДОДЕКАЭДР) ---
    std::vector<Vertex> vertices = getDodecahedronVertices();
    VkDeviceSize vertexBufferSize = sizeof(Vertex) * vertices.size();

    VkBufferCreateInfo vertexBufferInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = vertexBufferSize,
        .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };

    VmaAllocationCreateInfo vertexAllocInfo = {
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO
    };

    VmaAllocationInfo vertexAllocationInfo;
    if (vmaCreateBuffer(allocator, &vertexBufferInfo, &vertexAllocInfo,
                        &vertexBuffer, &vertexBufferAllocation,
                        &vertexAllocationInfo) != VK_SUCCESS) {
        std::cerr << "Failed to create vertex buffer\n";
        return false;
    }
    std::memcpy(vertexAllocationInfo.pMappedData, vertices.data(), vertexBufferSize);

    // --- Индексный буфер (ДОДЕКАЭДР) ---
    std::vector<uint16_t> indices = buildDodecahedronIndices();
    g_indexCount = static_cast<uint32_t>(indices.size());
    VkDeviceSize indexBufferSize = sizeof(uint16_t) * indices.size();

    VkBufferCreateInfo indexBufferInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = indexBufferSize,
        .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };

    VmaAllocationCreateInfo indexAllocInfo = {
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO
    };

    VmaAllocationInfo indexAllocationInfo;
    if (vmaCreateBuffer(allocator, &indexBufferInfo, &indexAllocInfo,
                        &indexBuffer, &indexBufferAllocation,
                        &indexAllocationInfo) != VK_SUCCESS) {
        std::cerr << "Failed to create index buffer\n";
        return false;
    }
    std::memcpy(indexAllocationInfo.pMappedData, indices.data(), indexBufferSize);

    return true;
}

// ---------------------------------------------------------------------------
// Завершение
// ---------------------------------------------------------------------------
void shutdown() {
    auto allocator = graphics::internal::context.allocator;
    auto device    = graphics::internal::context.device;

    vkDeviceWaitIdle(device);

    if (customPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, customPipeline, nullptr);
        customPipeline = VK_NULL_HANDLE;
    }
    if (customPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, customPipelineLayout, nullptr);
        customPipelineLayout = VK_NULL_HANDLE;
    }
    if (indexBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator, indexBuffer, indexBufferAllocation);
        indexBuffer = VK_NULL_HANDLE;
        indexBufferAllocation = VK_NULL_HANDLE;
    }
    if (vertexBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator, vertexBuffer, vertexBufferAllocation);
        vertexBuffer = VK_NULL_HANDLE;
        vertexBufferAllocation = VK_NULL_HANDLE;
    }
}

// ---------------------------------------------------------------------------
// Обновление (UI + анимация)
// ---------------------------------------------------------------------------
void update(double time) {
    static double lastTime = 0.0;
    float deltaTime = static_cast<float>(time - lastTime);
    lastTime = time;
    if (deltaTime < 0.0f) deltaTime = 0.0f;
    if (deltaTime > 0.1f) deltaTime = 0.1f;

    ImGui::Begin("Lab Work: Dodecahedron Controller");

    ImGui::Text("1. Projection Matrix:");
    if (ImGui::RadioButton("Perspective", isPerspective == 1)) isPerspective = 1;
    ImGui::SameLine();
    if (ImGui::RadioButton("Orthographic", isPerspective == 0)) isPerspective = 0;
    ImGui::Separator();

    ImGui::Text("2. Object Transformations:");
    ImGui::SliderFloat3("Position (X,Y,Z)", &shapePosition.x, -10.0f, 10.0f);
    ImGui::SliderFloat3("Rotation (Angles)", &shapeRotation.x, -180.0f, 180.0f);
    ImGui::SliderFloat3("Scale", &shapeScale.x, 0.1f, 5.0f);
    ImGui::Separator();

    ImGui::Text("3. Dynamic Animation (Complex Path):");
    ImGui::Checkbox("Play Animation", &isAnimated);
    ImGui::SliderFloat("Speed Multiplier", &animationSpeed, 0.0f, 5.0f);
    ImGui::SliderFloat("Orbit Radius", &pathRadius, 0.5f, 5.0f);
    ImGui::Separator();

    ImGui::Text("4. Color Settings:");
    ImGui::ColorEdit4("Base Color", uiColor);
    ImGui::End();

    if (isAnimated) {
        animationTime += deltaTime * animationSpeed;
        shapePosition.x = std::cos(animationTime) * pathRadius;
        shapePosition.z = -5.0f + std::sin(animationTime) * pathRadius;
        shapePosition.y = std::sin(animationTime * 2.0f) * 0.5f;

        shapeRotation.y += deltaTime * 40.0f * animationSpeed;
        shapeRotation.x += deltaTime * 15.0f * animationSpeed;
    }
}

// ---------------------------------------------------------------------------
// Рендер
// ---------------------------------------------------------------------------
void render(const graphics::internal::FrameData& fd) {
    VkCommandBuffer cmd = fd.command_buffer;

    VkViewport viewport = {
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(graphics::internal::context.swapchain_extent.width),
        .height = static_cast<float>(graphics::internal::context.swapchain_extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f
    };
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor = {
        .offset = { 0, 0 },
        .extent = graphics::internal::context.swapchain_extent
    };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, shapePosition);
    model = glm::rotate(model, glm::radians(shapeRotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::rotate(model, glm::radians(shapeRotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, glm::radians(shapeRotation.z), glm::vec3(0.0f, 0.0f, 1.0f));
    model = glm::scale(model, shapeScale);

    glm::mat4 view = glm::lookAt(
        glm::vec3(0.0f, 0.0f, 7.0f),
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f)
    );

    glm::mat4 proj;
    float aspect = viewport.width / viewport.height;
    if (isPerspective == 1) {
        proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
    }
    else {
        float orthoSize = 4.0f;
        proj = glm::ortho(-orthoSize * aspect, orthoSize * aspect,
            -orthoSize, orthoSize, 0.1f, 100.0f);
    }

    proj[1][1] *= -1.0f;  // Y-flip для Vulkan (нужен для обеих)

    MeshPushConstants constants;
    constants.mvp = proj * view * model;
    constants.baseColor = glm::vec4(uiColor[0], uiColor[1], uiColor[2], uiColor[3]);

    if (customPipeline != VK_NULL_HANDLE) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, customPipeline);
    }
    if (customPipelineLayout != VK_NULL_HANDLE) {
        vkCmdPushConstants(cmd, customPipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(MeshPushConstants), &constants);
    }
    if (vertexBuffer != VK_NULL_HANDLE && indexBuffer != VK_NULL_HANDLE && g_indexCount > 0) {
        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer, offsets);
        vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT16);
        vkCmdDrawIndexed(cmd, g_indexCount, 1, 0, 0, 0);
    }
}

} // namespace application