#include "application.hpp"
#include "graphics_internal.hpp"

#include <imgui.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace application {

namespace {

// ---------------------------------------------------------------------------
// Типы данных
// ---------------------------------------------------------------------------
struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec3 color;
};

struct SceneUniforms {
    glm::mat4  viewProj;
    glm::vec4  cameraPos;
    glm::vec4  ambient;
    glm::vec4  dirLightDirection;
    glm::vec4  dirLightColor;
    glm::uvec4 lightCounts;
};

struct ModelUniforms {
    glm::mat4 model;
    glm::mat4 normalMatrix;
    glm::vec4 albedo;
    glm::vec4 specular;
};

struct GpuPointLight {
    glm::vec4 position;
    glm::vec4 color;
};

struct GpuSpotLight {
    glm::vec4 position;
    glm::vec4 direction;
    glm::vec4 color;
};

struct Buffer {
    VkBuffer      buffer     = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    void*         mapped     = nullptr;
    VkDeviceSize  size       = 0;
};

struct Mesh {
    Buffer   vertices;
    Buffer   indices;
    uint32_t indexCount = 0;
};

struct Material {
    glm::vec3 albedo         = glm::vec3(1.0f);
    float     vertexColorMix = 0.0f;
    glm::vec3 specular       = glm::vec3(1.0f);
    float     shininess      = 32.0f;
};

struct SceneObject {
    std::string     name;
    Mesh*           mesh = nullptr;
    glm::vec3       position = glm::vec3(0.0f);
    glm::vec3       rotation = glm::vec3(0.0f);
    glm::vec3       scale    = glm::vec3(1.0f);
    Material        material;
    Buffer          uniforms;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
};

struct Camera {
    glm::vec3 position    = glm::vec3(0.0f, 1.5f, 9.0f);
    float     yaw         = -90.0f;
    float     pitch       = -8.0f;
    float     fov         = 60.0f;
    float     moveSpeed   = 4.0f;
    float     sensitivity = 0.15f;
    bool      perspective = true;
    float     orthoSize   = 5.0f;

    glm::vec3 forward() const {
        const float y = glm::radians(yaw);
        const float p = glm::radians(pitch);
        return glm::normalize(glm::vec3(std::cos(y) * std::cos(p), std::sin(p), std::sin(y) * std::cos(p)));
    }
};

struct DirectionalLight {
    glm::vec3 direction = glm::vec3(-0.4f, -1.0f, -0.3f);
    glm::vec3 color     = glm::vec3(1.0f, 0.95f, 0.85f);
    float     intensity = 0.25f;
};

struct PointLight {
    glm::vec3 position  = glm::vec3(0.0f, 1.0f, 2.5f);
    glm::vec3 color     = glm::vec3(1.0f);
    float     intensity = 6.0f;
};

struct SpotLight {
    glm::vec3 position     = glm::vec3(0.0f, 5.0f, 0.0f);
    glm::vec3 direction    = glm::vec3(0.0f, -1.0f, 0.0f);
    glm::vec3 color        = glm::vec3(1.0f);
    float     intensity    = 30.0f;
    float     innerAngle   = 15.0f;
    float     outerAngle   = 25.0f;
    bool      followCamera = false;
};

// ---------------------------------------------------------------------------
// Состояние
// ---------------------------------------------------------------------------
VkPipeline            pipeline              = VK_NULL_HANDLE;
VkPipelineLayout      pipelineLayout        = VK_NULL_HANDLE;
VkDescriptorSetLayout sceneSetLayout        = VK_NULL_HANDLE;
VkDescriptorSetLayout modelSetLayout        = VK_NULL_HANDLE;
VkDescriptorPool      descriptorPool        = VK_NULL_HANDLE;
VkDescriptorSet       sceneSet              = VK_NULL_HANDLE;

Buffer sceneUniformBuffer;
Buffer pointLightBuffer;
Buffer spotLightBuffer;

Mesh dodecahedronMesh;
Mesh floorMesh;

std::vector<SceneObject> objects;

Camera                  camera;
DirectionalLight        dirLight;
glm::vec3               ambientColor     = glm::vec3(0.6f, 0.7f, 1.0f);
float                   ambientIntensity = 0.08f;
std::vector<PointLight> pointLights;
std::vector<SpotLight>  spotLights;

bool  isAnimated     = false;
float animationSpeed = 1.0f;
float pathRadius     = 2.0f;
float animationTime  = 0.0f;
bool  animateLights  = false;

constexpr size_t initialLightCapacity = 4;
constexpr float  floorHeight          = -2.5f;

VkDevice device() { return graphics::internal::context.device; }
VmaAllocator allocator() { return graphics::internal::context.allocator; }

glm::vec3 safeNormalize(const glm::vec3& v, const glm::vec3& fallback) {
    const float len = glm::length(v);
    return len > 1e-4f ? v / len : fallback;
}

// ---------------------------------------------------------------------------
// Буферы
// ---------------------------------------------------------------------------
bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out) {
    const VkBufferCreateInfo bufferInfo = {
        .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size        = size,
        .usage       = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    const VmaAllocationCreateInfo allocInfo = {
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };

    VmaAllocationInfo info;
    if (vmaCreateBuffer(allocator(), &bufferInfo, &allocInfo,
                        &out.buffer, &out.allocation, &info) != VK_SUCCESS) {
        std::cerr << "Failed to create buffer\n";
        return false;
    }

    out.mapped = info.pMappedData;
    out.size   = size;
    return true;
}

void destroyBuffer(Buffer& buffer) {
    if (buffer.buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator(), buffer.buffer, buffer.allocation);
    }
    buffer = {};
}

void uploadBuffer(Buffer& buffer, const void* data, VkDeviceSize size) {
    if (size == 0) return;
    std::memcpy(buffer.mapped, data, size);
    vmaFlushAllocation(allocator(), buffer.allocation, 0, size);
}

bool ensureCapacity(Buffer& buffer, VkDeviceSize required) {
    if (required <= buffer.size) return false;

    const VkDeviceSize newSize = std::max(buffer.size * 2, required);
    destroyBuffer(buffer);
    createBuffer(newSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, buffer);
    return true;
}

bool createMesh(const std::vector<Vertex>& vertices, const std::vector<uint16_t>& indices, Mesh& mesh) {
    const VkDeviceSize vSize = sizeof(Vertex) * vertices.size();
    const VkDeviceSize iSize = sizeof(uint16_t) * indices.size();

    if (!createBuffer(vSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, mesh.vertices) ||
        !createBuffer(iSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, mesh.indices)) {
        return false;
    }

    uploadBuffer(mesh.vertices, vertices.data(), vSize);
    uploadBuffer(mesh.indices, indices.data(), iSize);
    mesh.indexCount = static_cast<uint32_t>(indices.size());
    return true;
}

void destroyMesh(Mesh& mesh) {
    destroyBuffer(mesh.vertices);
    destroyBuffer(mesh.indices);
    mesh.indexCount = 0;
}

// ---------------------------------------------------------------------------
// Геометрия: додекаэдр (плоские нормали)
// ---------------------------------------------------------------------------
constexpr float PHI = 1.61803398875f;

void buildDodecahedron(std::vector<Vertex>& vertices, std::vector<uint16_t>& indices) {
    static const glm::vec3 positions[] = {
        {-1.0f,       1.0f,      -1.0f},      {-1.61803f,   0.0f,       0.618034f},
        {-1.61803f,   0.0f,      -0.618034f}, {-1.0f,       1.0f,       1.0f},
        {-0.618034f,  1.61803f,   0.0f},      { 1.0f,       1.0f,       1.0f},
        { 0.618034f,  1.61803f,   0.0f},      { 0.0f,       0.618034f,  1.61803f},
        {-1.0f,      -1.0f,       1.0f},      { 0.0f,      -0.618034f,  1.61803f},
        {-1.0f,      -1.0f,      -1.0f},      {-0.618034f, -1.61803f,   0.0f},
        { 0.0f,      -0.618034f, -1.61803f},  { 0.0f,       0.618034f, -1.61803f},
        { 1.0f,       1.0f,      -1.0f},      { 1.61803f,   0.0f,      -0.618034f},
        { 1.61803f,   0.0f,       0.618034f}, { 1.0f,      -1.0f,       1.0f},
        { 0.618034f, -1.61803f,   0.0f},      { 1.0f,      -1.0f,      -1.0f},
    };

    static const uint16_t tris[] = {
        1, 0, 2,    0, 1, 3,    0, 3, 4,    4, 5, 6,    5, 4, 3,    5, 3, 7,
        8, 3, 1,    3, 8, 7,    7, 8, 9,    8, 10, 11,  10, 8, 2,   2, 8, 1,
        0, 10, 2,   10, 0, 12,  12, 0, 13,  0, 14, 13,  14, 0, 6,   6, 0, 4,
        15, 5, 16,  5, 15, 14,  5, 14, 6,   9, 5, 7,    5, 9, 17,   5, 17, 16,
        18, 8, 11,  8, 18, 17,  8, 17, 9,   19, 10, 12, 10, 19, 11, 11, 19, 18,
        13, 19, 12, 19, 13, 14, 19, 14, 15, 19, 17, 18, 17, 19, 16, 16, 19, 15,
    };

    vertices.clear();
    indices.clear();

    for (size_t t = 0; t < std::size(tris); t += 3) {
        glm::vec3 a = positions[tris[t + 0]];
        glm::vec3 b = positions[tris[t + 1]];
        glm::vec3 c = positions[tris[t + 2]];

        glm::vec3 n = glm::normalize(glm::cross(b - a, c - a));
        if (glm::dot(n, a + b + c) < 0.0f) {
            std::swap(b, c);
            n = -n;
        }

        for (const glm::vec3& p : { a, b, c }) {
            const glm::vec3 color = glm::vec3(0.5f) + 0.5f * p / PHI;
            indices.push_back(static_cast<uint16_t>(vertices.size()));
            vertices.push_back({ p, n, color });
        }
    }
}

// ---------------------------------------------------------------------------
// Геометрия: пол (шахматная доска)
// ---------------------------------------------------------------------------
void buildFloor(std::vector<Vertex>& vertices, std::vector<uint16_t>& indices,
                int tiles, float tileSize) {
    vertices.clear();
    indices.clear();

    const float     half   = tiles * tileSize * 0.5f;
    const glm::vec3 normal = glm::vec3(0.0f, 1.0f, 0.0f);

    for (int z = 0; z < tiles; ++z) {
        for (int x = 0; x < tiles; ++x) {
            const float     x0    = -half + x * tileSize;
            const float     z0    = -half + z * tileSize;
            const glm::vec3 color = glm::vec3((x + z) % 2 == 0 ? 0.6f : 0.3f);
            const uint16_t  base  = static_cast<uint16_t>(vertices.size());

            vertices.push_back({ { x0,            0.0f, z0            }, normal, color });
            vertices.push_back({ { x0,            0.0f, z0 + tileSize }, normal, color });
            vertices.push_back({ { x0 + tileSize, 0.0f, z0 + tileSize }, normal, color });
            vertices.push_back({ { x0 + tileSize, 0.0f, z0            }, normal, color });

            for (uint16_t i : { 0, 1, 2, 0, 2, 3 }) {
                indices.push_back(base + i);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Шейдеры
// ---------------------------------------------------------------------------
std::vector<char> readShaderFile(const std::string& filename) {
    static const char* searchDirs[] = { "", "../", "../../", "../../../" };

    for (const char* dir : searchDirs) {
        std::ifstream file(std::string(dir) + filename, std::ios::ate | std::ios::binary);
        if (!file.is_open()) continue;

        const size_t fileSize = static_cast<size_t>(file.tellg());
        std::vector<char> buffer(fileSize);
        file.seekg(0);
        file.read(buffer.data(), fileSize);
        return buffer;
    }

    std::cerr << "Failed to open shader file: " << filename
              << " (working directory: " << std::filesystem::current_path().string() << ")\n";
    return {};
}

VkShaderModule createShaderModule(const std::vector<char>& code) {
    if (code.empty() || code.size() % 4 != 0) return VK_NULL_HANDLE;

    const VkShaderModuleCreateInfo createInfo = {
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size(),
        .pCode    = reinterpret_cast<const uint32_t*>(code.data()),
    };

    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device(), &createInfo, nullptr, &module) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return module;
}

// ---------------------------------------------------------------------------
// Дескрипторы
// ---------------------------------------------------------------------------
void writeDescriptor(VkDescriptorSet set, uint32_t binding, VkDescriptorType type, const Buffer& buffer) {
    const VkDescriptorBufferInfo info = {
        .buffer = buffer.buffer,
        .offset = 0,
        .range  = VK_WHOLE_SIZE,
    };

    const VkWriteDescriptorSet write = {
        .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet          = set,
        .dstBinding      = binding,
        .descriptorCount = 1,
        .descriptorType  = type,
        .pBufferInfo     = &info,
    };

    vkUpdateDescriptorSets(device(), 1, &write, 0, nullptr);
}

bool createDescriptorSetLayouts() {
    const VkDescriptorSetLayoutBinding sceneBindings[] = {
        {
            .binding         = 0,
            .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {
            .binding         = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {
            .binding         = 2,
            .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
    };

    const VkDescriptorSetLayoutBinding modelBinding = {
        .binding         = 0,
        .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
    };

    const VkDescriptorSetLayoutCreateInfo sceneInfo = {
        .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(std::size(sceneBindings)),
        .pBindings    = sceneBindings,
    };

    const VkDescriptorSetLayoutCreateInfo modelInfo = {
        .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings    = &modelBinding,
    };

    return vkCreateDescriptorSetLayout(device(), &sceneInfo, nullptr, &sceneSetLayout) == VK_SUCCESS &&
           vkCreateDescriptorSetLayout(device(), &modelInfo, nullptr, &modelSetLayout) == VK_SUCCESS;
}

bool createDescriptorSets() {
    const uint32_t objectCount = static_cast<uint32_t>(objects.size());

    const VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 + objectCount },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 },
    };

    const VkDescriptorPoolCreateInfo poolInfo = {
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets       = 1 + objectCount,
        .poolSizeCount = static_cast<uint32_t>(std::size(poolSizes)),
        .pPoolSizes    = poolSizes,
    };

    if (vkCreateDescriptorPool(device(), &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        return false;
    }

    VkDescriptorSetAllocateInfo allocInfo = {
        .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool     = descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts        = &sceneSetLayout,
    };

    if (vkAllocateDescriptorSets(device(), &allocInfo, &sceneSet) != VK_SUCCESS) {
        return false;
    }

    writeDescriptor(sceneSet, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, sceneUniformBuffer);
    writeDescriptor(sceneSet, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, pointLightBuffer);
    writeDescriptor(sceneSet, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, spotLightBuffer);

    allocInfo.pSetLayouts = &modelSetLayout;

    for (SceneObject& obj : objects) {
        if (vkAllocateDescriptorSets(device(), &allocInfo, &obj.descriptorSet) != VK_SUCCESS) {
            return false;
        }
        writeDescriptor(obj.descriptorSet, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, obj.uniforms);
    }

    return true;
}

// ---------------------------------------------------------------------------
// Графический конвейер
// ---------------------------------------------------------------------------
bool createPipeline() {
    const VkDescriptorSetLayout setLayouts[] = { sceneSetLayout, modelSetLayout };

    const VkPipelineLayoutCreateInfo layoutInfo = {
        .sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = static_cast<uint32_t>(std::size(setLayouts)),
        .pSetLayouts    = setLayouts,
    };

    if (vkCreatePipelineLayout(device(), &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        std::cerr << "Failed to create pipeline layout\n";
        return false;
    }

    VkShaderModule vertModule = createShaderModule(readShaderFile("shaders/shader.vert.spv"));
    VkShaderModule fragModule = createShaderModule(readShaderFile("shaders/shader.frag.spv"));
    if (vertModule == VK_NULL_HANDLE || fragModule == VK_NULL_HANDLE) {
        std::cerr << "Failed to create shader modules\n";
        vkDestroyShaderModule(device(), vertModule, nullptr);
        vkDestroyShaderModule(device(), fragModule, nullptr);
        return false;
    }

    const VkPipelineShaderStageCreateInfo shaderStages[] = {
        {
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertModule,
            .pName  = "main",
        },
        {
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragModule,
            .pName  = "main",
        },
    };

    const VkVertexInputBindingDescription bindingDesc = {
        .binding   = 0,
        .stride    = sizeof(Vertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };

    const VkVertexInputAttributeDescription attribDesc[] = {
        { .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, pos)    },
        { .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, normal) },
        { .location = 2, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, color)  },
    };

    const VkPipelineVertexInputStateCreateInfo vertexInput = {
        .sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount   = 1,
        .pVertexBindingDescriptions      = &bindingDesc,
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(std::size(attribDesc)),
        .pVertexAttributeDescriptions    = attribDesc,
    };

    const VkPipelineInputAssemblyStateCreateInfo inputAssembly = {
        .sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };

    const VkPipelineViewportStateCreateInfo viewportState = {
        .sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount  = 1,
    };

    const VkPipelineRasterizationStateCreateInfo rasterizer = {
        .sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode    = VK_CULL_MODE_BACK_BIT,
        .frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth   = 1.0f,
    };

    const VkPipelineMultisampleStateCreateInfo multisampling = {
        .sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };

    const VkPipelineDepthStencilStateCreateInfo depthStencil = {
        .sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable  = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp   = VK_COMPARE_OP_LESS,
    };

    const VkPipelineColorBlendAttachmentState colorBlendAttachment = {
        .blendEnable    = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };

    const VkPipelineColorBlendStateCreateInfo colorBlending = {
        .sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments    = &colorBlendAttachment,
    };

    const VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

    const VkPipelineDynamicStateCreateInfo dynamicState = {
        .sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = static_cast<uint32_t>(std::size(dynamicStates)),
        .pDynamicStates    = dynamicStates,
    };

    const VkGraphicsPipelineCreateInfo pipelineInfo = {
        .sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount          = static_cast<uint32_t>(std::size(shaderStages)),
        .pStages             = shaderStages,
        .pVertexInputState   = &vertexInput,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState      = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState   = &multisampling,
        .pDepthStencilState  = &depthStencil,
        .pColorBlendState    = &colorBlending,
        .pDynamicState       = &dynamicState,
        .layout              = pipelineLayout,
        .renderPass          = graphics::internal::context.render_pass,
        .subpass             = 0,
        .basePipelineIndex   = -1,
    };

    const VkResult result = vkCreateGraphicsPipelines(device(), VK_NULL_HANDLE, 1, &pipelineInfo,
                                                      nullptr, &pipeline);

    vkDestroyShaderModule(device(), fragModule, nullptr);
    vkDestroyShaderModule(device(), vertModule, nullptr);

    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create graphics pipeline\n";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Сцена по умолчанию
// ---------------------------------------------------------------------------
void createDefaultScene() {
    SceneObject dodecahedron;
    dodecahedron.name = "Dodecahedron";
    dodecahedron.mesh = &dodecahedronMesh;
    dodecahedron.material = {
        .albedo         = glm::vec3(1.0f, 0.7f, 0.2f),
        .vertexColorMix = 0.5f,
        .specular       = glm::vec3(1.0f),
        .shininess      = 64.0f,
    };

    SceneObject floor;
    floor.name = "Floor";
    floor.mesh = &floorMesh;
    floor.position = glm::vec3(0.0f, floorHeight, 0.0f);
    floor.material = {
        .albedo         = glm::vec3(1.0f),
        .vertexColorMix = 1.0f,
        .specular       = glm::vec3(0.3f),
        .shininess      = 16.0f,
    };

    objects = { dodecahedron, floor };

    pointLights = {
        { .position = { -3.0f, 0.5f,  2.0f }, .color = { 1.0f, 0.3f, 0.2f }, .intensity = 8.0f },
        { .position = {  3.0f, 0.5f, -1.0f }, .color = { 0.2f, 0.5f, 1.0f }, .intensity = 8.0f },
    };

    spotLights = {
        { .position = { 0.0f, 5.0f, 3.0f }, .direction = { 0.0f, -1.0f, -0.5f },
          .color = { 1.0f, 1.0f, 0.8f }, .intensity = 25.0f, .innerAngle = 15.0f, .outerAngle = 25.0f },
    };
}

// ---------------------------------------------------------------------------
// Интерфейс
// ---------------------------------------------------------------------------
void drawCameraUI() {
    if (!ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) return;

    if (ImGui::RadioButton("Perspective", camera.perspective)) camera.perspective = true;
    ImGui::SameLine();
    if (ImGui::RadioButton("Orthographic", !camera.perspective)) camera.perspective = false;

    ImGui::DragFloat3("Position##cam", &camera.position.x, 0.05f);
    ImGui::SliderFloat("Yaw", &camera.yaw, -180.0f, 180.0f);
    ImGui::SliderFloat("Pitch", &camera.pitch, -89.0f, 89.0f);

    if (camera.perspective) {
        ImGui::SliderFloat("FOV", &camera.fov, 20.0f, 120.0f);
    } else {
        ImGui::SliderFloat("Ortho size", &camera.orthoSize, 1.0f, 20.0f);
    }

    ImGui::SliderFloat("Move speed", &camera.moveSpeed, 0.5f, 20.0f);
    ImGui::TextDisabled("WASD/QE - move, RMB - look, wheel - FOV");
}

void drawLightsUI() {
    if (ImGui::CollapsingHeader("Ambient & Directional", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ColorEdit3("Ambient color", &ambientColor.x);
        ImGui::SliderFloat("Ambient intensity", &ambientIntensity, 0.0f, 1.0f);
        ImGui::Separator();
        ImGui::SliderFloat3("Direction##dir", &dirLight.direction.x, -1.0f, 1.0f);
        ImGui::ColorEdit3("Color##dir", &dirLight.color.x);
        ImGui::SliderFloat("Intensity##dir", &dirLight.intensity, 0.0f, 3.0f);
    }

    if (ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Orbit point lights", &animateLights);
        if (ImGui::Button("Add point light")) {
            pointLights.push_back({});
        }

        int toRemove = -1;
        for (size_t i = 0; i < pointLights.size(); ++i) {
            PointLight& light = pointLights[i];
            ImGui::PushID(static_cast<int>(i));
            if (ImGui::TreeNode("light", "Point light #%zu", i)) {
                ImGui::DragFloat3("Position", &light.position.x, 0.05f);
                ImGui::ColorEdit3("Color", &light.color.x);
                ImGui::DragFloat("Intensity", &light.intensity, 0.1f, 0.0f, 200.0f);
                if (ImGui::Button("Remove")) toRemove = static_cast<int>(i);
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
        if (toRemove >= 0) pointLights.erase(pointLights.begin() + toRemove);
    }

    if (ImGui::CollapsingHeader("Spot lights", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::Button("Add spot light")) {
            spotLights.push_back({});
        }

        int toRemove = -1;
        for (size_t i = 0; i < spotLights.size(); ++i) {
            SpotLight& light = spotLights[i];
            ImGui::PushID(1000 + static_cast<int>(i));
            if (ImGui::TreeNode("light", "Spot light #%zu", i)) {
                ImGui::Checkbox("Follow camera", &light.followCamera);
                ImGui::BeginDisabled(light.followCamera);
                ImGui::DragFloat3("Position", &light.position.x, 0.05f);
                ImGui::SliderFloat3("Direction", &light.direction.x, -1.0f, 1.0f);
                ImGui::EndDisabled();
                ImGui::ColorEdit3("Color", &light.color.x);
                ImGui::DragFloat("Intensity", &light.intensity, 0.1f, 0.0f, 500.0f);
                ImGui::SliderFloat("Inner angle", &light.innerAngle, 0.0f, 89.0f);
                ImGui::SliderFloat("Outer angle", &light.outerAngle, 0.0f, 89.0f);
                light.outerAngle = std::max(light.outerAngle, light.innerAngle);
                if (ImGui::Button("Remove")) toRemove = static_cast<int>(i);
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
        if (toRemove >= 0) spotLights.erase(spotLights.begin() + toRemove);
    }
}

void drawObjectsUI() {
    if (!ImGui::CollapsingHeader("Objects & Materials", ImGuiTreeNodeFlags_DefaultOpen)) return;

    for (size_t i = 0; i < objects.size(); ++i) {
        SceneObject& obj = objects[i];
        ImGui::PushID(2000 + static_cast<int>(i));
        if (ImGui::TreeNode("obj", "%s", obj.name.c_str())) {
            ImGui::DragFloat3("Position", &obj.position.x, 0.05f);
            ImGui::SliderFloat3("Rotation", &obj.rotation.x, -180.0f, 180.0f);
            ImGui::SliderFloat3("Scale", &obj.scale.x, 0.1f, 5.0f);
            ImGui::Separator();
            ImGui::ColorEdit3("Albedo", &obj.material.albedo.x);
            ImGui::SliderFloat("Vertex color mix", &obj.material.vertexColorMix, 0.0f, 1.0f);
            ImGui::ColorEdit3("Specular color", &obj.material.specular.x);
            ImGui::SliderFloat("Shininess", &obj.material.shininess, 1.0f, 256.0f, "%.1f",
                               ImGuiSliderFlags_Logarithmic);
            ImGui::TreePop();
        }
        ImGui::PopID();
    }

    ImGui::SeparatorText("Dodecahedron animation");
    ImGui::Checkbox("Play animation", &isAnimated);
    ImGui::SliderFloat("Speed", &animationSpeed, 0.0f, 5.0f);
    ImGui::SliderFloat("Orbit radius", &pathRadius, 0.5f, 5.0f);
}

// ---------------------------------------------------------------------------
// Управление камерой
// ---------------------------------------------------------------------------
void updateCameraInput(float dt) {
    const ImGuiIO& io = ImGui::GetIO();

    if (!io.WantCaptureMouse) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
            camera.yaw   += io.MouseDelta.x * camera.sensitivity;
            camera.pitch -= io.MouseDelta.y * camera.sensitivity;
            camera.pitch  = std::clamp(camera.pitch, -89.0f, 89.0f);
            if (camera.yaw > 180.0f)  camera.yaw -= 360.0f;
            if (camera.yaw < -180.0f) camera.yaw += 360.0f;
        }
        camera.fov = std::clamp(camera.fov - io.MouseWheel * 2.0f, 20.0f, 120.0f);
    }

    if (io.WantCaptureKeyboard) return;

    const glm::vec3 up      = glm::vec3(0.0f, 1.0f, 0.0f);
    const glm::vec3 forward = camera.forward();
    const glm::vec3 right   = glm::normalize(glm::cross(forward, up));

    glm::vec3 move(0.0f);
    if (ImGui::IsKeyDown(ImGuiKey_W)) move += forward;
    if (ImGui::IsKeyDown(ImGuiKey_S)) move -= forward;
    if (ImGui::IsKeyDown(ImGuiKey_D)) move += right;
    if (ImGui::IsKeyDown(ImGuiKey_A)) move -= right;
    if (ImGui::IsKeyDown(ImGuiKey_E)) move += up;
    if (ImGui::IsKeyDown(ImGuiKey_Q)) move -= up;

    if (glm::length(move) > 0.0f) {
        const float boost = ImGui::IsKeyDown(ImGuiKey_LeftShift) ? 3.0f : 1.0f;
        camera.position += glm::normalize(move) * camera.moveSpeed * boost * dt;
    }
}

// ---------------------------------------------------------------------------
// Загрузка данных на GPU
// ---------------------------------------------------------------------------
glm::mat4 buildModelMatrix(const SceneObject& obj) {
    glm::mat4 model = glm::translate(glm::mat4(1.0f), obj.position);
    model = glm::rotate(model, glm::radians(obj.rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::rotate(model, glm::radians(obj.rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, glm::radians(obj.rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));
    return glm::scale(model, obj.scale);
}

void uploadSceneData(float aspect) {
    const glm::mat4 view = glm::lookAt(camera.position, camera.position + camera.forward(),
                                       glm::vec3(0.0f, 1.0f, 0.0f));

    glm::mat4 proj;
    if (camera.perspective) {
        proj = glm::perspective(glm::radians(camera.fov), aspect, 0.05f, 200.0f);
    } else {
        const float s = camera.orthoSize;
        proj = glm::ortho(-s * aspect, s * aspect, -s, s, 0.05f, 200.0f);
    }
    proj[1][1] *= -1.0f;

    const SceneUniforms scene = {
        .viewProj          = proj * view,
        .cameraPos         = glm::vec4(camera.position, 1.0f),
        .ambient           = glm::vec4(ambientColor, ambientIntensity),
        .dirLightDirection = glm::vec4(safeNormalize(dirLight.direction, glm::vec3(0.0f, -1.0f, 0.0f)), 0.0f),
        .dirLightColor     = glm::vec4(dirLight.color, dirLight.intensity),
        .lightCounts       = glm::uvec4(pointLights.size(), spotLights.size(), 0, 0),
    };
    uploadBuffer(sceneUniformBuffer, &scene, sizeof(scene));

    std::vector<GpuPointLight> gpuPoints;
    gpuPoints.reserve(pointLights.size());
    for (const PointLight& l : pointLights) {
        gpuPoints.push_back({ glm::vec4(l.position, 1.0f), glm::vec4(l.color, l.intensity) });
    }

    std::vector<GpuSpotLight> gpuSpots;
    gpuSpots.reserve(spotLights.size());
    for (const SpotLight& l : spotLights) {
        const glm::vec3 dir = safeNormalize(l.direction, glm::vec3(0.0f, -1.0f, 0.0f));
        gpuSpots.push_back({
            glm::vec4(l.position, std::cos(glm::radians(l.innerAngle))),
            glm::vec4(dir,        std::cos(glm::radians(l.outerAngle))),
            glm::vec4(l.color,    l.intensity),
        });
    }

    const VkDeviceSize pointSize = sizeof(GpuPointLight) * gpuPoints.size();
    const VkDeviceSize spotSize  = sizeof(GpuSpotLight) * gpuSpots.size();

    if (ensureCapacity(pointLightBuffer, pointSize)) {
        writeDescriptor(sceneSet, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, pointLightBuffer);
    }
    if (ensureCapacity(spotLightBuffer, spotSize)) {
        writeDescriptor(sceneSet, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, spotLightBuffer);
    }

    uploadBuffer(pointLightBuffer, gpuPoints.data(), pointSize);
    uploadBuffer(spotLightBuffer, gpuSpots.data(), spotSize);
}

void uploadObjectData(SceneObject& obj) {
    const glm::mat4 model = buildModelMatrix(obj);

    const ModelUniforms data = {
        .model        = model,
        .normalMatrix = glm::mat4(glm::transpose(glm::inverse(glm::mat3(model)))),
        .albedo       = glm::vec4(obj.material.albedo, obj.material.vertexColorMix),
        .specular     = glm::vec4(obj.material.specular, obj.material.shininess),
    };
    uploadBuffer(obj.uniforms, &data, sizeof(data));
}

} // namespace

// ---------------------------------------------------------------------------
// Инициализация
// ---------------------------------------------------------------------------
static bool createResources() {
    std::vector<Vertex>   vertices;
    std::vector<uint16_t> indices;

    buildDodecahedron(vertices, indices);
    if (!createMesh(vertices, indices, dodecahedronMesh)) return false;

    buildFloor(vertices, indices, 20, 1.0f);
    if (!createMesh(vertices, indices, floorMesh)) return false;

    createDefaultScene();

    if (!createBuffer(sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sceneUniformBuffer) ||
        !createBuffer(sizeof(GpuPointLight) * initialLightCapacity, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, pointLightBuffer) ||
        !createBuffer(sizeof(GpuSpotLight) * initialLightCapacity, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, spotLightBuffer)) {
        return false;
    }

    for (SceneObject& obj : objects) {
        if (!createBuffer(sizeof(ModelUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, obj.uniforms)) {
            return false;
        }
    }

    if (!createDescriptorSetLayouts()) {
        std::cerr << "Failed to create descriptor set layouts\n";
        return false;
    }

    if (!createDescriptorSets()) {
        std::cerr << "Failed to create descriptor sets\n";
        return false;
    }

    return createPipeline();
}

// ---------------------------------------------------------------------------
// Завершение
// ---------------------------------------------------------------------------
void shutdown() {
    vkDeviceWaitIdle(device());

    vkDestroyPipeline(device(), pipeline, nullptr);
    vkDestroyPipelineLayout(device(), pipelineLayout, nullptr);
    vkDestroyDescriptorPool(device(), descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device(), modelSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(device(), sceneSetLayout, nullptr);

    pipeline       = VK_NULL_HANDLE;
    pipelineLayout = VK_NULL_HANDLE;
    descriptorPool = VK_NULL_HANDLE;
    modelSetLayout = VK_NULL_HANDLE;
    sceneSetLayout = VK_NULL_HANDLE;

    for (SceneObject& obj : objects) {
        destroyBuffer(obj.uniforms);
    }
    objects.clear();

    destroyBuffer(sceneUniformBuffer);
    destroyBuffer(pointLightBuffer);
    destroyBuffer(spotLightBuffer);

    destroyMesh(dodecahedronMesh);
    destroyMesh(floorMesh);
}

bool initialize() {
    if (createResources()) return true;

    shutdown();
    return false;
}

// ---------------------------------------------------------------------------
// Обновление (UI + анимация)
// ---------------------------------------------------------------------------
void update(double time) {
    static double lastTime = time;
    const float dt = std::clamp(static_cast<float>(time - lastTime), 0.0f, 0.1f);
    lastTime = time;

    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(360, 680), ImGuiCond_FirstUseEver);
    ImGui::Begin("Lab 2: Camera & Lighting");
    drawCameraUI();
    drawLightsUI();
    drawObjectsUI();
    ImGui::End();

    updateCameraInput(dt);

    if (isAnimated && !objects.empty()) {
        SceneObject& shape = objects[0];
        animationTime += dt * animationSpeed;

        shape.position.x  = std::cos(animationTime) * pathRadius;
        shape.position.z  = std::sin(animationTime) * pathRadius;
        shape.position.y  = std::sin(animationTime * 2.0f) * 0.5f;
        shape.rotation.y += dt * 40.0f * animationSpeed;
        shape.rotation.x += dt * 15.0f * animationSpeed;
        shape.rotation    = glm::mod(shape.rotation + 180.0f, 360.0f) - 180.0f;
    }

    if (animateLights) {
        const glm::mat4 rot = glm::rotate(glm::mat4(1.0f), dt * 0.8f, glm::vec3(0.0f, 1.0f, 0.0f));
        for (PointLight& light : pointLights) {
            light.position = glm::vec3(rot * glm::vec4(light.position, 1.0f));
        }
    }

    for (SpotLight& light : spotLights) {
        if (light.followCamera) {
            light.position  = camera.position;
            light.direction = camera.forward();
        }
    }
}

// ---------------------------------------------------------------------------
// Рендер
// ---------------------------------------------------------------------------
void render(const graphics::internal::FrameData& fd) {
    const VkCommandBuffer cmd    = fd.command_buffer;
    const VkExtent2D      extent = graphics::internal::context.swapchain_extent;

    const VkViewport viewport = {
        .x        = 0.0f,
        .y        = 0.0f,
        .width    = static_cast<float>(extent.width),
        .height   = static_cast<float>(extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    const VkRect2D scissor = { .offset = { 0, 0 }, .extent = extent };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    uploadSceneData(viewport.width / viewport.height);
    for (SceneObject& obj : objects) {
        uploadObjectData(obj);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                            0, 1, &sceneSet, 0, nullptr);

    for (const SceneObject& obj : objects) {
        const VkDeviceSize offset = 0;
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                                1, 1, &obj.descriptorSet, 0, nullptr);
        vkCmdBindVertexBuffers(cmd, 0, 1, &obj.mesh->vertices.buffer, &offset);
        vkCmdBindIndexBuffer(cmd, obj.mesh->indices.buffer, 0, VK_INDEX_TYPE_UINT16);
        vkCmdDrawIndexed(cmd, obj.mesh->indexCount, 1, 0, 0, 0);
    }
}

} // namespace application
