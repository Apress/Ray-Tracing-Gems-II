/* Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "Common.h"
#include "../shaders/shared.h"
#include <cmath>
#include <unordered_set>
#include <map>

struct Camera
{
    DirectX::XMFLOAT3    position;
    float                aspect = 16.f / 9.f;
    DirectX::XMFLOAT3    up;
    float                fov = 65.f;
    DirectX::XMFLOAT3    right;
    DirectX::XMFLOAT3    forward;

    Camera()
    {
        position = { 0.f, 0.f, 0.f };
        up = { 0.f, 1.f, 0.f };
        right = { 1.f, 0.f, 0.f };
        forward = { 0.f, 0.f, -1.f };
    }
};

// ---[ Geometry ] ------------------------------------------------------------

struct Vertex
{
    DirectX::XMFLOAT3 position;
    DirectX::XMFLOAT3 normal;
    DirectX::XMFLOAT4 tangent;
    DirectX::XMFLOAT2 uv0;

    Vertex& operator=(const Vertex& v)
    {
        position = v.position;
        normal = v.normal;
        tangent = v.tangent;
        uv0 = v.uv0;
        return *this;
    }
};

struct AABB
{
    struct float3
    {
        float x;
        float y;
        float z;
    };

    float3 min;
    float3 max;
};

struct MeshPrimitive
{
    int                  index = -1;
    int                  material = -1;
    bool                 opaque = true;
    AABB                 boundingBox;
    std::vector<Vertex>  vertices;
    std::vector<UINT>    indices;
};

struct Mesh
{
    std::string name = "";
    std::vector<MeshPrimitive> primitives;
};

// ---[ Materials ] -----------------------------------------------------------

struct Material
{
    std::string name = "";
    MaterialData data;
};

struct TextureData {
    UINT8* texels = nullptr;
    int width = 0;
    int height = 0;
    int stride = 0;
    UINT64 texelBytes = 0;
    UINT64 rowPitch = 0;
    bool stbAllocation = false; //< true if texel data was allocated via STB library
};

struct Texture
{
    std::string name = "";
    std::string filepath = "";

    std::vector<TextureData> mips;

    bool doSrgb = false;
    bool isBC7Compressed = false;
    bool isCopy = false; //< Indicates that the texture is copy of another one, so we shouldn't deallocate its data twice
};

// ---[ Scene Graph ] ---------------------------------------------------------

struct Instance
{
    int mesh = -1;
    std::string name = "";
    float transform[3][4] =
    {
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 1.f, 0.f
    };
};

struct Node
{
    int instance = -1;
    int camera = -1;
    std::vector<int> children;
    DirectX::XMFLOAT3 translation = DirectX::XMFLOAT3(0.f, 0.f, 0.f);
    DirectX::XMFLOAT4 rotation = DirectX::XMFLOAT4(0.f, 0.f, 0.f, 1.f);
    DirectX::XMFLOAT3 scale = DirectX::XMFLOAT3(1.f, 1.f, 1.f);
    DirectX::XMMATRIX matrix;
    bool isMatrixProvided = false;
};

struct Scene
{
    UINT numGeometries = 0;
    std::vector<int> roots;
    std::vector<Node> nodes;
    std::vector<Camera> cameras;
    std::vector<Camera> cameraTemplates;
    std::vector<Instance> instances;
    std::vector<Mesh> meshes;
    std::vector<Material> materials;
    std::vector<Texture> textures;
    std::unordered_set<size_t> srgbTextures;
    std::map<size_t, DirectX::XMFLOAT2> textureUVAdjustment;
};