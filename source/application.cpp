#include "application.hpp"
#include "graphics_internal.hpp"

#include <imgui.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <array>
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
    glm::vec2 uv;
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
    glm::vec4 specular;
};

struct SamplingPushConstants {
    glm::vec4 tintA;
    glm::vec4 tintB;
    float     time;
    float     distortion;
    float     layerScale;
    uint32_t  flags;
};

constexpr uint32_t FLAG_ANIMATED_SAMPLING = 1u;
constexpr uint32_t FLAG_COLOR_MODULATION  = 2u;

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

struct Texture {
    std::string     name;
    VkImage         image         = VK_NULL_HANDLE;
    VmaAllocation   allocation    = VK_NULL_HANDLE;
    VkImageView     view          = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
};

struct Material {
    int       texture   = 0;
    glm::vec3 specular  = glm::vec3(0.5f);
    float     shininess = 32.0f;
};

struct SceneObject {
    std::string name;
    Mesh*       mesh     = nullptr;
    glm::vec3   position = glm::vec3(0.0f);
    glm::vec3   rotation = glm::vec3(0.0f);
    glm::vec3   scale    = glm::vec3(1.0f);
    Material    material;
    bool        animatedSampling = false;
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
    glm::vec3 direction = glm::vec3(-0.4f, -1.0f, -0.6f);
    glm::vec3 color     = glm::vec3(1.0f, 0.95f, 0.85f);
    float     intensity = 0.35f;
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
VkPipeline            pipeline          = VK_NULL_HANDLE;
VkPipelineLayout      pipelineLayout    = VK_NULL_HANDLE;
VkDescriptorSetLayout sceneSetLayout    = VK_NULL_HANDLE;
VkDescriptorSetLayout materialSetLayout = VK_NULL_HANDLE;
VkDescriptorPool      descriptorPool    = VK_NULL_HANDLE;
VkDescriptorSet       sceneSet          = VK_NULL_HANDLE;
VkCommandPool         uploadPool        = VK_NULL_HANDLE;
VkSampler             sampler           = VK_NULL_HANDLE;

Buffer       sceneUniformBuffer;
Buffer       pointLightBuffer;
Buffer       spotLightBuffer;
Buffer       modelUniformBuffer;
VkDeviceSize modelUniformStride = 0;

std::vector<Texture> textures;

Mesh dodecahedronMesh;
Mesh floorMesh;
Mesh wallMesh;

std::vector<SceneObject> objects;

Camera                  camera;
DirectionalLight        dirLight;
glm::vec3               ambientColor     = glm::vec3(0.6f, 0.7f, 1.0f);
float                   ambientIntensity = 0.12f;
std::vector<PointLight> pointLights;
std::vector<SpotLight>  spotLights;

bool  isAnimated     = false;
float animationSpeed = 1.0f;
float pathRadius     = 2.0f;
float animationTime  = 0.0f;
bool  animateLights  = false;

float     effectSpeed      = 1.0f;
float     effectTime       = 0.0f;
float     effectDistortion = 0.03f;
float     effectLayerScale = 1.6f;
bool      colorModulation  = true;
glm::vec3 tintA            = glm::vec3(1.0f, 0.55f, 0.25f);
glm::vec3 tintB            = glm::vec3(0.35f, 0.6f, 1.0f);

constexpr size_t   initialLightCapacity = 4;
constexpr float    floorHeight          = -2.5f;
constexpr float    dodecahedronBottom   = 1.61803f;
constexpr VkFormat textureFormat        = VK_FORMAT_R8G8B8A8_SRGB;

VkDevice device() { return graphics::internal::context.device; }
VmaAllocator allocator() { return graphics::internal::context.allocator; }

glm::vec3 safeNormalize(const glm::vec3& v, const glm::vec3& fallback) {
    const float len = glm::length(v);
    return len > 1e-4f ? v / len : fallback;
}

std::filesystem::path findDataPath(const std::string& relative) {
    static const char* searchDirs[] = { "", "../", "../../", "../../../" };
    for (const char* dir : searchDirs) {
        const std::filesystem::path candidate = std::filesystem::path(dir) / relative;
        if (std::filesystem::exists(candidate)) return candidate;
    }
    return {};
}

// ---------------------------------------------------------------------------
// Команды копирования (однократная отправка)
// ---------------------------------------------------------------------------
bool createUploadPool() {
    const VkCommandPoolCreateInfo poolInfo = {
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = graphics::internal::context.graphics_queue_index,
    };
    return vkCreateCommandPool(device(), &poolInfo, nullptr, &uploadPool) == VK_SUCCESS;
}

template <typename Record>
bool immediateSubmit(Record&& record) {
    const VkCommandBufferAllocateInfo allocInfo = {
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = uploadPool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device(), &allocInfo, &cmd) != VK_SUCCESS) return false;

    const VkCommandBufferBeginInfo beginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

    vkBeginCommandBuffer(cmd, &beginInfo);
    record(cmd);
    vkEndCommandBuffer(cmd);

    const VkSubmitInfo submit = {
        .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers    = &cmd,
    };

    const VkQueue queue = graphics::internal::context.graphics_queue;
    VkResult result = vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
    if (result == VK_SUCCESS) result = vkQueueWaitIdle(queue);

    vkFreeCommandBuffers(device(), uploadPool, 1, &cmd);
    return result == VK_SUCCESS;
}

// ---------------------------------------------------------------------------
// Буферы
// ---------------------------------------------------------------------------
bool createHostBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out) {
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
        std::cerr << "Failed to create host buffer\n";
        return false;
    }

    out.mapped = info.pMappedData;
    out.size   = size;
    return true;
}

bool createDeviceBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out) {
    const VkBufferCreateInfo bufferInfo = {
        .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size        = size,
        .usage       = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    const VmaAllocationCreateInfo allocInfo = {
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
    };

    if (vmaCreateBuffer(allocator(), &bufferInfo, &allocInfo,
                        &out.buffer, &out.allocation, nullptr) != VK_SUCCESS) {
        std::cerr << "Failed to create device buffer\n";
        return false;
    }

    out.size = size;
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

bool createStagingBuffer(const void* data, VkDeviceSize size, Buffer& staging) {
    if (!createHostBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, staging)) return false;
    uploadBuffer(staging, data, size);
    return true;
}

bool uploadToDevice(const void* data, VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out) {
    Buffer staging;
    if (!createStagingBuffer(data, size, staging)) return false;

    bool ok = createDeviceBuffer(size, usage, out);
    if (ok) {
        ok = immediateSubmit([&](VkCommandBuffer cmd) {
            const VkBufferCopy region = { .srcOffset = 0, .dstOffset = 0, .size = size };
            vkCmdCopyBuffer(cmd, staging.buffer, out.buffer, 1, &region);
        });
    }

    destroyBuffer(staging);
    return ok;
}

bool ensureCapacity(Buffer& buffer, VkDeviceSize required) {
    if (required <= buffer.size) return false;

    const VkDeviceSize newSize = std::max(buffer.size * 2, required);
    destroyBuffer(buffer);
    createHostBuffer(newSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, buffer);
    return true;
}

bool createMesh(const std::vector<Vertex>& vertices, const std::vector<uint16_t>& indices, Mesh& mesh) {
    if (!uploadToDevice(vertices.data(), sizeof(Vertex) * vertices.size(),
                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, mesh.vertices) ||
        !uploadToDevice(indices.data(), sizeof(uint16_t) * indices.size(),
                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT, mesh.indices)) {
        return false;
    }

    mesh.indexCount = static_cast<uint32_t>(indices.size());
    return true;
}

void destroyMesh(Mesh& mesh) {
    destroyBuffer(mesh.vertices);
    destroyBuffer(mesh.indices);
    mesh.indexCount = 0;
}

// ---------------------------------------------------------------------------
// Текстуры
// ---------------------------------------------------------------------------
void imageBarrier(VkCommandBuffer cmd, VkImage image, uint32_t baseMip, uint32_t mipCount,
                  VkImageLayout oldLayout, VkImageLayout newLayout,
                  VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                  VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) {
    const VkImageMemoryBarrier barrier = {
        .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask       = srcAccess,
        .dstAccessMask       = dstAccess,
        .oldLayout           = oldLayout,
        .newLayout           = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image               = image,
        .subresourceRange    = {
            .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel   = baseMip,
            .levelCount     = mipCount,
            .baseArrayLayer = 0,
            .layerCount     = 1,
        },
    };
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

bool supportsMipmapBlit() {
    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(graphics::internal::context.physical_device, textureFormat, &props);

    const VkFormatFeatureFlags required = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT |
                                          VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    return (props.optimalTilingFeatures & required) == required;
}

bool createTexture(const std::string& name, const uint8_t* pixels, uint32_t width, uint32_t height, Texture& out) {
    const VkDeviceSize size = VkDeviceSize(width) * height * 4;
    const uint32_t mipLevels = supportsMipmapBlit()
        ? static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1
        : 1;

    Buffer staging;
    if (!createStagingBuffer(pixels, size, staging)) return false;

    const VkImageCreateInfo imageInfo = {
        .sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType     = VK_IMAGE_TYPE_2D,
        .format        = textureFormat,
        .extent        = { width, height, 1 },
        .mipLevels     = mipLevels,
        .arrayLayers   = 1,
        .samples       = VK_SAMPLE_COUNT_1_BIT,
        .tiling        = VK_IMAGE_TILING_OPTIMAL,
        .usage         = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                         VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    const VmaAllocationCreateInfo allocInfo = {
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
    };

    if (vmaCreateImage(allocator(), &imageInfo, &allocInfo, &out.image, &out.allocation, nullptr) != VK_SUCCESS) {
        std::cerr << "Failed to create texture image: " << name << "\n";
        destroyBuffer(staging);
        return false;
    }

    const bool uploaded = immediateSubmit([&](VkCommandBuffer cmd) {
        imageBarrier(cmd, out.image, 0, mipLevels,
                     VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     0, VK_ACCESS_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        const VkBufferImageCopy copy = {
            .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
            .imageExtent      = { width, height, 1 },
        };
        vkCmdCopyBufferToImage(cmd, staging.buffer, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

        int32_t mipWidth  = static_cast<int32_t>(width);
        int32_t mipHeight = static_cast<int32_t>(height);

        for (uint32_t i = 1; i < mipLevels; ++i) {
            imageBarrier(cmd, out.image, i - 1, 1,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

            const int32_t nextWidth  = std::max(mipWidth / 2, 1);
            const int32_t nextHeight = std::max(mipHeight / 2, 1);

            const VkImageBlit blit = {
                .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1 },
                .srcOffsets     = { { 0, 0, 0 }, { mipWidth, mipHeight, 1 } },
                .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1 },
                .dstOffsets     = { { 0, 0, 0 }, { nextWidth, nextHeight, 1 } },
            };
            vkCmdBlitImage(cmd, out.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

            imageBarrier(cmd, out.image, i - 1, 1,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

            mipWidth  = nextWidth;
            mipHeight = nextHeight;
        }

        imageBarrier(cmd, out.image, mipLevels - 1, 1,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    });

    destroyBuffer(staging);
    if (!uploaded) return false;

    const VkImageViewCreateInfo viewInfo = {
        .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image            = out.image,
        .viewType         = VK_IMAGE_VIEW_TYPE_2D,
        .format           = textureFormat,
        .subresourceRange = {
            .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel   = 0,
            .levelCount     = mipLevels,
            .baseArrayLayer = 0,
            .layerCount     = 1,
        },
    };

    if (vkCreateImageView(device(), &viewInfo, nullptr, &out.view) != VK_SUCCESS) {
        std::cerr << "Failed to create texture view: " << name << "\n";
        return false;
    }

    out.name = name;
    return true;
}

void destroyTexture(Texture& texture) {
    vkDestroyImageView(device(), texture.view, nullptr);
    if (texture.image != VK_NULL_HANDLE) {
        vmaDestroyImage(allocator(), texture.image, texture.allocation);
    }
    texture = {};
}

bool loadTextures() {
    std::vector<std::filesystem::path> files;

    const std::filesystem::path dir = findDataPath("textures");
    if (!dir.empty()) {
        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tga") {
                files.push_back(entry.path());
            }
        }
        std::sort(files.begin(), files.end());
    }

    for (const auto& file : files) {
        int width = 0, height = 0, channels = 0;
        stbi_uc* pixels = stbi_load(file.string().c_str(), &width, &height, &channels, STBI_rgb_alpha);
        if (pixels == nullptr) {
            std::cerr << "Failed to load " << file.string() << ": " << stbi_failure_reason() << "\n";
            continue;
        }

        Texture texture;
        const bool ok = createTexture(file.filename().string(), pixels, width, height, texture);
        stbi_image_free(pixels);

        if (!ok) return false;
        textures.push_back(texture);
    }

    if (textures.empty()) {
        std::cerr << "No textures found in 'textures' folder, using white fallback\n";
        const uint8_t white[4] = { 255, 255, 255, 255 };
        Texture texture;
        if (!createTexture("white (fallback)", white, 1, 1, texture)) return false;
        textures.push_back(texture);
    }

    return true;
}

int findTexture(const std::string& stem) {
    for (size_t i = 0; i < textures.size(); ++i) {
        if (std::filesystem::path(textures[i].name).stem() == stem) return static_cast<int>(i);
    }
    return 0;
}

bool createSampler() {
    const VkSamplerCreateInfo samplerInfo = {
        .sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter    = VK_FILTER_LINEAR,
        .minFilter    = VK_FILTER_LINEAR,
        .mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .maxLod       = VK_LOD_CLAMP_NONE,
        .borderColor  = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
    };
    return vkCreateSampler(device(), &samplerInfo, nullptr, &sampler) == VK_SUCCESS;
}

// ---------------------------------------------------------------------------
// Геометрия: додекаэдр (плоские нормали, UV на каждую грань)
// ---------------------------------------------------------------------------
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

    struct Face {
        glm::vec3                             normal;
        std::vector<glm::vec3>                corners;
        std::vector<std::array<glm::vec3, 3>> triangles;
    };
    std::vector<Face> faces;

    for (size_t t = 0; t < std::size(tris); t += 3) {
        glm::vec3 a = positions[tris[t + 0]];
        glm::vec3 b = positions[tris[t + 1]];
        glm::vec3 c = positions[tris[t + 2]];

        glm::vec3 n = glm::normalize(glm::cross(b - a, c - a));
        if (glm::dot(n, a + b + c) < 0.0f) {
            std::swap(b, c);
            n = -n;
        }

        auto face = std::find_if(faces.begin(), faces.end(),
                                 [&](const Face& f) { return glm::dot(f.normal, n) > 0.999f; });
        if (face == faces.end()) {
            faces.push_back({ n, {}, {} });
            face = faces.end() - 1;
        }

        face->triangles.push_back({ a, b, c });
        for (const glm::vec3& p : { a, b, c }) {
            const bool known = std::any_of(face->corners.begin(), face->corners.end(),
                                           [&](const glm::vec3& q) { return glm::length(p - q) < 1e-4f; });
            if (!known) face->corners.push_back(p);
        }
    }

    vertices.clear();
    indices.clear();

    for (const Face& face : faces) {
        glm::vec3 center(0.0f);
        for (const glm::vec3& p : face.corners) center += p;
        center /= static_cast<float>(face.corners.size());

        const float     radius = glm::length(face.corners[0] - center);
        const glm::vec3 axisU  = glm::normalize(face.corners[0] - center);
        const glm::vec3 axisV  = glm::cross(face.normal, axisU);

        for (const auto& tri : face.triangles) {
            for (const glm::vec3& p : tri) {
                const glm::vec3 d  = p - center;
                const glm::vec2 uv = glm::vec2(0.5f + 0.5f * glm::dot(d, axisU) / radius,
                                               0.5f - 0.5f * glm::dot(d, axisV) / radius);
                indices.push_back(static_cast<uint16_t>(vertices.size()));
                vertices.push_back({ p, face.normal, uv });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Геометрия: плоскость (пол и стена)
// ---------------------------------------------------------------------------
void buildPlane(std::vector<Vertex>& vertices, std::vector<uint16_t>& indices,
                float width, float depth, float uvPerUnit) {
    const float     hw     = width * 0.5f;
    const float     hd     = depth * 0.5f;
    const glm::vec3 normal = glm::vec3(0.0f, 1.0f, 0.0f);

    vertices = {
        { { -hw, 0.0f, -hd }, normal, { 0.0f,               0.0f               } },
        { { -hw, 0.0f,  hd }, normal, { 0.0f,               depth * uvPerUnit  } },
        { {  hw, 0.0f,  hd }, normal, { width * uvPerUnit,  depth * uvPerUnit  } },
        { {  hw, 0.0f, -hd }, normal, { width * uvPerUnit,  0.0f               } },
    };
    indices = { 0, 1, 2, 0, 2, 3 };
}

// ---------------------------------------------------------------------------
// Шейдеры
// ---------------------------------------------------------------------------
std::vector<char> readShaderFile(const std::string& filename) {
    const std::filesystem::path path = findDataPath(filename);
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (path.empty() || !file.is_open()) {
        std::cerr << "Failed to open shader file: " << filename
                  << " (working directory: " << std::filesystem::current_path().string() << ")\n";
        return {};
    }

    const size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    return buffer;
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
void writeBufferDescriptor(VkDescriptorSet set, uint32_t binding, VkDescriptorType type,
                           const Buffer& buffer, VkDeviceSize range = VK_WHOLE_SIZE) {
    const VkDescriptorBufferInfo info = {
        .buffer = buffer.buffer,
        .offset = 0,
        .range  = range,
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

void writeImageDescriptor(VkDescriptorSet set, uint32_t binding, VkImageView view) {
    const VkDescriptorImageInfo info = {
        .sampler     = sampler,
        .imageView   = view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };

    const VkWriteDescriptorSet write = {
        .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet          = set,
        .dstBinding      = binding,
        .descriptorCount = 1,
        .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo      = &info,
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

    const VkDescriptorSetLayoutBinding materialBindings[] = {
        {
            .binding         = 0,
            .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {
            .binding         = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
    };

    const VkDescriptorSetLayoutCreateInfo sceneInfo = {
        .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(std::size(sceneBindings)),
        .pBindings    = sceneBindings,
    };

    const VkDescriptorSetLayoutCreateInfo materialInfo = {
        .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(std::size(materialBindings)),
        .pBindings    = materialBindings,
    };

    return vkCreateDescriptorSetLayout(device(), &sceneInfo, nullptr, &sceneSetLayout) == VK_SUCCESS &&
           vkCreateDescriptorSetLayout(device(), &materialInfo, nullptr, &materialSetLayout) == VK_SUCCESS;
}

bool createDescriptorSets() {
    const uint32_t textureCount = static_cast<uint32_t>(textures.size());

    const VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1            },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         2            },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, textureCount },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, textureCount },
    };

    const VkDescriptorPoolCreateInfo poolInfo = {
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets       = 1 + textureCount,
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

    writeBufferDescriptor(sceneSet, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, sceneUniformBuffer);
    writeBufferDescriptor(sceneSet, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, pointLightBuffer);
    writeBufferDescriptor(sceneSet, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, spotLightBuffer);

    allocInfo.pSetLayouts = &materialSetLayout;

    for (Texture& texture : textures) {
        if (vkAllocateDescriptorSets(device(), &allocInfo, &texture.descriptorSet) != VK_SUCCESS) {
            return false;
        }
        writeBufferDescriptor(texture.descriptorSet, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                              modelUniformBuffer, sizeof(ModelUniforms));
        writeImageDescriptor(texture.descriptorSet, 1, texture.view);
    }

    return true;
}

// ---------------------------------------------------------------------------
// Графический конвейер
// ---------------------------------------------------------------------------
bool createPipeline() {
    const VkDescriptorSetLayout setLayouts[] = { sceneSetLayout, materialSetLayout };

    const VkPushConstantRange pushRange = {
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset     = 0,
        .size       = sizeof(SamplingPushConstants),
    };

    const VkPipelineLayoutCreateInfo layoutInfo = {
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = static_cast<uint32_t>(std::size(setLayouts)),
        .pSetLayouts            = setLayouts,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pushRange,
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
        { .location = 2, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT,    .offset = offsetof(Vertex, uv)     },
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
    const float smallScale = 0.8f;
    const float smallY     = floorHeight + dodecahedronBottom * smallScale;

    SceneObject lava;
    lava.name = "Dodecahedron (lava)";
    lava.mesh = &dodecahedronMesh;
    lava.material = { .texture = findTexture("lava"), .specular = glm::vec3(0.6f), .shininess = 64.0f };
    lava.animatedSampling = true;

    SceneObject marble;
    marble.name = "Dodecahedron (marble)";
    marble.mesh = &dodecahedronMesh;
    marble.position = glm::vec3(-4.5f, smallY, -3.0f);
    marble.scale = glm::vec3(smallScale);
    marble.material = { .texture = findTexture("marble"), .specular = glm::vec3(1.0f), .shininess = 128.0f };

    SceneObject metal;
    metal.name = "Dodecahedron (metal)";
    metal.mesh = &dodecahedronMesh;
    metal.position = glm::vec3(4.5f, smallY, -3.0f);
    metal.rotation = glm::vec3(0.0f, 30.0f, 0.0f);
    metal.scale = glm::vec3(smallScale);
    metal.material = { .texture = findTexture("metal"), .specular = glm::vec3(1.0f), .shininess = 96.0f };

    SceneObject floor;
    floor.name = "Floor";
    floor.mesh = &floorMesh;
    floor.position = glm::vec3(0.0f, floorHeight, 0.0f);
    floor.material = { .texture = findTexture("stone_tiles"), .specular = glm::vec3(0.2f), .shininess = 16.0f };

    SceneObject wall;
    wall.name = "Wall";
    wall.mesh = &wallMesh;
    wall.position = glm::vec3(0.0f, floorHeight + 4.0f, -10.0f);
    wall.rotation = glm::vec3(90.0f, 0.0f, 0.0f);
    wall.material = { .texture = findTexture("bricks"), .specular = glm::vec3(0.1f), .shininess = 8.0f };

    objects = { lava, marble, metal, floor, wall };

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
    if (!ImGui::CollapsingHeader("Camera")) return;

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
    ImGui::TextDisabled("WASD/QE - move, RMB - look, wheel - zoom");
}

void drawLightsUI() {
    if (ImGui::CollapsingHeader("Ambient & Directional")) {
        ImGui::ColorEdit3("Ambient color", &ambientColor.x);
        ImGui::SliderFloat("Ambient intensity", &ambientIntensity, 0.0f, 1.0f);
        ImGui::Separator();
        ImGui::SliderFloat3("Direction##dir", &dirLight.direction.x, -1.0f, 1.0f);
        ImGui::ColorEdit3("Color##dir", &dirLight.color.x);
        ImGui::SliderFloat("Intensity##dir", &dirLight.intensity, 0.0f, 3.0f);
    }

    if (ImGui::CollapsingHeader("Point lights")) {
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

    if (ImGui::CollapsingHeader("Spot lights")) {
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

            if (ImGui::BeginCombo("Texture", textures[obj.material.texture].name.c_str())) {
                for (int t = 0; t < static_cast<int>(textures.size()); ++t) {
                    const bool selected = t == obj.material.texture;
                    if (ImGui::Selectable(textures[t].name.c_str(), selected)) obj.material.texture = t;
                    if (selected) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }

            ImGui::Checkbox("Animated sampling", &obj.animatedSampling);
            ImGui::ColorEdit3("Specular color", &obj.material.specular.x);
            ImGui::SliderFloat("Shininess", &obj.material.shininess, 1.0f, 256.0f, "%.1f",
                               ImGuiSliderFlags_Logarithmic);
            ImGui::TreePop();
        }
        ImGui::PopID();
    }

    ImGui::SeparatorText("Lava dodecahedron animation");
    ImGui::Checkbox("Play animation", &isAnimated);
    ImGui::SliderFloat("Speed##anim", &animationSpeed, 0.0f, 5.0f);
    ImGui::SliderFloat("Orbit radius", &pathRadius, 0.5f, 5.0f);
}

void drawSamplingUI() {
    if (!ImGui::CollapsingHeader("Animated sampling", ImGuiTreeNodeFlags_DefaultOpen)) return;

    ImGui::SliderFloat("Speed##fx", &effectSpeed, 0.0f, 5.0f);
    ImGui::SliderFloat("Distortion", &effectDistortion, 0.0f, 0.15f);
    ImGui::SliderFloat("Layer scale", &effectLayerScale, 0.25f, 4.0f);
    ImGui::Checkbox("Color modulation", &colorModulation);
    if (colorModulation) {
        ImGui::ColorEdit3("Tint A", &tintA.x);
        ImGui::ColorEdit3("Tint B", &tintB.x);
    }
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
        if (camera.perspective) {
            camera.fov = std::clamp(camera.fov - io.MouseWheel * 2.0f, 20.0f, 120.0f);
        } else {
            camera.orthoSize = std::clamp(camera.orthoSize - io.MouseWheel * 0.5f, 1.0f, 20.0f);
        }
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
        proj = glm::ortho(-s * aspect, s * aspect, -s, s, -100.0f, 100.0f);
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
        writeBufferDescriptor(sceneSet, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, pointLightBuffer);
    }
    if (ensureCapacity(spotLightBuffer, spotSize)) {
        writeBufferDescriptor(sceneSet, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, spotLightBuffer);
    }

    uploadBuffer(pointLightBuffer, gpuPoints.data(), pointSize);
    uploadBuffer(spotLightBuffer, gpuSpots.data(), spotSize);
}

void uploadModelData() {
    auto* base = static_cast<uint8_t*>(modelUniformBuffer.mapped);

    for (size_t i = 0; i < objects.size(); ++i) {
        const SceneObject& obj   = objects[i];
        const glm::mat4    model = buildModelMatrix(obj);

        const ModelUniforms data = {
            .model        = model,
            .normalMatrix = glm::mat4(glm::transpose(glm::inverse(glm::mat3(model)))),
            .specular     = glm::vec4(obj.material.specular, obj.material.shininess),
        };
        std::memcpy(base + i * modelUniformStride, &data, sizeof(data));
    }

    vmaFlushAllocation(allocator(), modelUniformBuffer.allocation, 0, VK_WHOLE_SIZE);
}

// ---------------------------------------------------------------------------
// Создание ресурсов
// ---------------------------------------------------------------------------
bool createResources() {
    if (!createUploadPool() || !createSampler()) {
        std::cerr << "Failed to create upload command pool or sampler\n";
        return false;
    }

    if (!loadTextures()) return false;

    std::vector<Vertex>   vertices;
    std::vector<uint16_t> indices;

    buildDodecahedron(vertices, indices);
    if (!createMesh(vertices, indices, dodecahedronMesh)) return false;

    buildPlane(vertices, indices, 20.0f, 20.0f, 0.25f);
    if (!createMesh(vertices, indices, floorMesh)) return false;

    buildPlane(vertices, indices, 20.0f, 8.0f, 0.25f);
    if (!createMesh(vertices, indices, wallMesh)) return false;

    createDefaultScene();

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(graphics::internal::context.physical_device, &props);
    const VkDeviceSize align = std::max<VkDeviceSize>(props.limits.minUniformBufferOffsetAlignment, 1);
    modelUniformStride = (sizeof(ModelUniforms) + align - 1) / align * align;

    if (!createHostBuffer(sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sceneUniformBuffer) ||
        !createHostBuffer(modelUniformStride * objects.size(), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, modelUniformBuffer) ||
        !createHostBuffer(sizeof(GpuPointLight) * initialLightCapacity, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, pointLightBuffer) ||
        !createHostBuffer(sizeof(GpuSpotLight) * initialLightCapacity, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, spotLightBuffer)) {
        return false;
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

} // namespace

// ---------------------------------------------------------------------------
// Завершение
// ---------------------------------------------------------------------------
void shutdown() {
    vkDeviceWaitIdle(device());

    vkDestroyPipeline(device(), pipeline, nullptr);
    vkDestroyPipelineLayout(device(), pipelineLayout, nullptr);
    vkDestroyDescriptorPool(device(), descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device(), materialSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(device(), sceneSetLayout, nullptr);
    vkDestroySampler(device(), sampler, nullptr);
    vkDestroyCommandPool(device(), uploadPool, nullptr);

    pipeline          = VK_NULL_HANDLE;
    pipelineLayout    = VK_NULL_HANDLE;
    descriptorPool    = VK_NULL_HANDLE;
    materialSetLayout = VK_NULL_HANDLE;
    sceneSetLayout    = VK_NULL_HANDLE;
    sampler           = VK_NULL_HANDLE;
    uploadPool        = VK_NULL_HANDLE;

    for (Texture& texture : textures) {
        destroyTexture(texture);
    }
    textures.clear();
    objects.clear();

    destroyBuffer(sceneUniformBuffer);
    destroyBuffer(modelUniformBuffer);
    destroyBuffer(pointLightBuffer);
    destroyBuffer(spotLightBuffer);

    destroyMesh(dodecahedronMesh);
    destroyMesh(floorMesh);
    destroyMesh(wallMesh);
}

// ---------------------------------------------------------------------------
// Инициализация
// ---------------------------------------------------------------------------
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
    ImGui::Begin("Lab 3: Texture mapping");
    drawCameraUI();
    drawLightsUI();
    drawObjectsUI();
    drawSamplingUI();
    ImGui::End();

    updateCameraInput(dt);

    effectTime += dt * effectSpeed;

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
    uploadModelData();

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                            0, 1, &sceneSet, 0, nullptr);

    for (size_t i = 0; i < objects.size(); ++i) {
        const SceneObject& obj     = objects[i];
        const Texture&     texture = textures[obj.material.texture];
        const uint32_t     offset  = static_cast<uint32_t>(i * modelUniformStride);

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                                1, 1, &texture.descriptorSet, 1, &offset);

        uint32_t flags = 0;
        if (obj.animatedSampling) flags |= FLAG_ANIMATED_SAMPLING;
        if (colorModulation)      flags |= FLAG_COLOR_MODULATION;

        const SamplingPushConstants push = {
            .tintA      = glm::vec4(tintA, 1.0f),
            .tintB      = glm::vec4(tintB, 1.0f),
            .time       = effectTime,
            .distortion = effectDistortion,
            .layerScale = effectLayerScale,
            .flags      = flags,
        };
        vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);

        const VkDeviceSize vertexOffset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &obj.mesh->vertices.buffer, &vertexOffset);
        vkCmdBindIndexBuffer(cmd, obj.mesh->indices.buffer, 0, VK_INDEX_TYPE_UINT16);
        vkCmdDrawIndexed(cmd, obj.mesh->indexCount, 1, 0, 0, 0);
    }
}

} // namespace application
