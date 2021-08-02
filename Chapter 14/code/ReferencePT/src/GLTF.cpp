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

#include "GLTF.h"
#include "Textures.h"

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_EXTERNAL_IMAGE
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "tinygltf/tiny_gltf.h"

#include <map>

using namespace DirectX;

namespace GLTF
{

//----------------------------------------------------------------------------------------------------------
// Private Functions
//----------------------------------------------------------------------------------------------------------

/**
* Parse the glTF cameras.
*/
void ParseGLTFCameras(const tinygltf::Model &gltfData, Scene &scene)
{
    for (size_t cameraIndex = 0; cameraIndex < gltfData.cameras.size(); cameraIndex++)
    {
        // Get the glTF camera
        const tinygltf::Camera gltfCamera = gltfData.cameras[cameraIndex];
        if (strcmp(gltfCamera.type.c_str(), "perspective") == 0)
        {
            Camera camera;
            camera.fov = (float)gltfCamera.perspective.yfov * (180.f / XM_PI);

            scene.cameraTemplates.push_back(camera);
        } else {
            // We don't support other camera types, just push default perspective camera to the list
            scene.cameraTemplates.push_back(Camera());
        }
    }
}

/**
* Parse the glTF nodes.
*/
void ParseGLTFNodes(const tinygltf::Model &gltfData, Scene &scene)
{
    // Get the default scene
    const tinygltf::Scene gltfScene = gltfData.scenes[gltfData.defaultScene];

    // Get the indices of the scene's root nodes
    for (size_t rootIndex = 0; rootIndex < gltfScene.nodes.size(); rootIndex++)
    {
        scene.roots.push_back(gltfScene.nodes[rootIndex]);
    }

    // Get all the nodes
    for (size_t nodeIndex = 0; nodeIndex < gltfData.nodes.size(); nodeIndex++)
    {
        // Get the glTF node
        const tinygltf::Node gltfNode = gltfData.nodes[nodeIndex];

        Node node;

        // Get the node's local transform data
        if (gltfNode.matrix.size() > 0) {

            node.matrix = XMMATRIX(gltfNode.matrix[0], gltfNode.matrix[1], gltfNode.matrix[2], gltfNode.matrix[3], 
                gltfNode.matrix[4], gltfNode.matrix[5], gltfNode.matrix[6], gltfNode.matrix[7], 
                gltfNode.matrix[8], gltfNode.matrix[9], gltfNode.matrix[10], gltfNode.matrix[11],
                gltfNode.matrix[12], gltfNode.matrix[13], gltfNode.matrix[14], gltfNode.matrix[15]);

            node.isMatrixProvided = true;
        } else {
            if (gltfNode.translation.size() > 0) node.translation = XMFLOAT3((float)gltfNode.translation[0], (float)gltfNode.translation[1], (float)gltfNode.translation[2]);
            if (gltfNode.rotation.size() > 0) node.rotation = XMFLOAT4((float)gltfNode.rotation[0], (float)gltfNode.rotation[1], (float)gltfNode.rotation[2], (float)gltfNode.rotation[3]);
            if (gltfNode.scale.size() > 0) node.scale = XMFLOAT3((float)gltfNode.scale[0], (float)gltfNode.scale[1], (float)gltfNode.scale[2]);

            node.isMatrixProvided = false;
        }

        // Camera node, remember which camera template should we use
        if (gltfNode.camera != -1)
        {
            node.camera = gltfNode.camera;
        }

        // When at a leaf node, add an instance to the scene (if a mesh exists for it)
        if (gltfNode.children.size() == 0 && gltfNode.mesh != -1)
        {
            // Write the instance data
            Instance instance;
            instance.name = gltfNode.name;
            instance.mesh = gltfNode.mesh;

            node.instance = (int)scene.instances.size();
            scene.instances.push_back(instance);
        }

        // Gather the child node indices
        for (size_t childIndex = 0; childIndex < gltfNode.children.size(); childIndex++)
        {
            node.children.push_back(gltfNode.children[childIndex]);
        }

        // Add the new node to the scene graph
        scene.nodes.push_back(node);
    }
}

/*
* Parse the glTF materials into our format.
*/
void ParseGLTFMaterials(const tinygltf::Model &gltfData, Scene &scene)
{
    for (size_t i = 0; i < gltfData.materials.size(); i++)
    {
        const tinygltf::Material gltfMaterial = gltfData.materials[i];
        const tinygltf::PbrMetallicRoughness pbr = gltfMaterial.pbrMetallicRoughness;

        // Transform glTF material into our material format
        Material material;
        material.name = gltfMaterial.name;
        material.data.doubleSided = (int)gltfMaterial.doubleSided;

        // Albedo and Opacity
        material.data.baseColor = XMFLOAT3((float)pbr.baseColorFactor[0], (float)pbr.baseColorFactor[1], (float)pbr.baseColorFactor[2]);
        material.data.opacity = (float)pbr.baseColorFactor[3];
        material.data.baseColorTexIdx = pbr.baseColorTexture.index;

        // Alpha
        material.data.alphaCutoff = static_cast<float>(gltfMaterial.alphaCutoff);
        if (strcmp(gltfMaterial.alphaMode.c_str(), "OPAQUE") == 0) material.data.alphaMode = ALPHA_MODE_OPAQUE;
        else if (strcmp(gltfMaterial.alphaMode.c_str(), "BLEND") == 0) material.data.alphaMode = ALPHA_MODE_BLEND;
        else if (strcmp(gltfMaterial.alphaMode.c_str(), "MASK") == 0) material.data.alphaMode = ALPHA_MODE_MASK;

        // Roughness and Metallic
        material.data.roughness = (float)pbr.roughnessFactor;
        material.data.metalness = (float)pbr.metallicFactor;
        material.data.roughnessMetalnessTexIdx = pbr.metallicRoughnessTexture.index;

        // Normals
        material.data.normalTexIdx = gltfMaterial.normalTexture.index;

        // Emissive
        material.data.emissive = XMFLOAT3((float)gltfMaterial.emissiveFactor[0], (float)gltfMaterial.emissiveFactor[1], (float)gltfMaterial.emissiveFactor[2]);
        material.data.emissiveTexIdx = gltfMaterial.emissiveTexture.index;

        // Mark base color and emissive textures as sRGB sources. This may overwrite this flag if same texture is has different uses!
        if (pbr.baseColorTexture.index != -1) scene.srgbTextures.insert(pbr.baseColorTexture.index);
        if (gltfMaterial.emissiveTexture.index != -1) scene.srgbTextures.insert(gltfMaterial.emissiveTexture.index);

        scene.materials.push_back(material);
    }

    // If there are no materials, create a default material
    if (scene.materials.size() == 0)
    {
        Material default;
        scene.materials.push_back(default);
    }
}

/**
* Parse glTF textures and load the images.
*/
void ParseGLFTextures(const tinygltf::Model &gltfData, const ConfigInfo &config, Scene &scene)
{
    Textures::Initialize();

    std::map<std::string, size_t> loadedTextures;

    for (size_t textureIndex = 0; textureIndex < gltfData.textures.size(); textureIndex++)
    {
        const tinygltf::Texture gltfTexture = gltfData.textures[textureIndex];

        Texture texture;
        texture.name = gltfTexture.name;

        // Skip this texture if the source image doesn't exist
        if (gltfTexture.source == -1 || gltfData.images.size() <= gltfTexture.source) continue;

        // Construct the texture image filepath
        texture.filepath;
        texture.filepath.append(config.scenePath);
        texture.filepath.append(gltfData.images[gltfTexture.source].uri);

        if (strcmp(texture.name.c_str(), "") == 0)
        {
            texture.name = gltfData.images[gltfTexture.source].uri;
        }

        // Mark texture as sRGB if it's used for color information
        if (scene.srgbTextures.find(textureIndex) != scene.srgbTextures.end()) texture.doSrgb = true;

        if (loadedTextures.find(texture.filepath) != loadedTextures.end()) {
            // This texture was already loaded, copy its data
            const Texture* source = &scene.textures[loadedTextures[texture.filepath]];

            texture.isBC7Compressed = source->isBC7Compressed;
            texture.mips = source->mips;

            texture.isCopy = true;
		} else {

            DirectX::XMFLOAT2 uvAdjustment;

			// Load the texture from disk
			Textures::LoadTexture(texture, uvAdjustment);

            // Remember adjustment of UVs for this texture (if it was resized during load, e.g. to add padding, we need to apply this to vertex UVs)
            if (uvAdjustment.x != 1.0f || uvAdjustment.y != 1.0f) {
                scene.textureUVAdjustment[textureIndex] = uvAdjustment;
            }

			loadedTextures.insert(std::pair<std::string, size_t>(texture.filepath, scene.textures.size()));
		}

        // Add the texture to the scene
        scene.textures.push_back(texture);
    }

    Textures::Cleanup();
}

// Helper to load UV adjustment for given texture and issue a warning if vertex needs multiple different UV adjustments for its textures 
void getUVAdjustment(const int texIdx, Scene& scene, bool& uvAdjustmentNeeded, DirectX::XMFLOAT2& uvAdjustment) {
    if (texIdx != INVALID_ID && scene.textureUVAdjustment.find(texIdx) != scene.textureUVAdjustment.end()) {
        DirectX::XMFLOAT2 newUVAdjustment = scene.textureUVAdjustment[texIdx];

        // Check if UV adjustment for this texture is same as for previously checked textures
        if (uvAdjustmentNeeded && (newUVAdjustment.x != uvAdjustment.x || newUVAdjustment.y != uvAdjustment.y)) {
            // Different UV adjustments are required for different textures on same vertex - revert adjustment to identity
            printf("Different UV adjustments are needed for emissive texture - adjustment won't be applied. Please use textures with sizes which are multiple of 4 to prevent texture distortion.");
            newUVAdjustment = DirectX::XMFLOAT2(1.0f, 1.0f);
        }
        uvAdjustmentNeeded = true;
        uvAdjustment = newUVAdjustment;
    }
}

/**
* Parse the glTF meshes.
*/
void ParseGLTFMeshes(const tinygltf::Model &gltfData, Scene &scene)
{
    // Note: GTLF 2.0's default coordinate system is Right Handed, Y-Up
    // https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#coordinate-system-and-units
    // Meshes are converted from this coordinate system to the chosen coordinate system.

    UINT geometryIndex = 0;
    for (size_t meshIndex = 0; meshIndex < gltfData.meshes.size(); meshIndex++)
    {
        const tinygltf::Mesh gltfMesh = gltfData.meshes[meshIndex];

        Mesh mesh;
        mesh.name = gltfMesh.name;
        for (size_t primitiveIndex = 0; primitiveIndex < gltfMesh.primitives.size(); primitiveIndex++)
        {
            tinygltf::Primitive p = gltfMesh.primitives[primitiveIndex];

            MeshPrimitive m;
            m.index = geometryIndex;
            m.material = p.material;

            // Set to the default material if one is not assigned
            if (m.material == -1) m.material = 0;

            // If the mesh material is blended or masked, it is not opaque
            if (p.material >= 0) {
                if (strcmp(gltfData.materials[m.material].alphaMode.c_str(), "OPAQUE") != 0) m.opaque = false;
            }

            // Get data indices
            int indicesIndex = p.indices;
            int positionIndex = -1;
            int normalIndex = -1;
            int tangentIndex = -1;
            int uv0Index = -1;

            if (p.attributes.count("POSITION") > 0)
            {
                positionIndex = p.attributes["POSITION"];
            }

            if (p.attributes.count("NORMAL") > 0)
            {
                normalIndex = p.attributes["NORMAL"];
            }

            if (p.attributes.count("TANGENT"))
            {
                tangentIndex = p.attributes["TANGENT"];
            }

            if (p.attributes.count("TEXCOORD_0") > 0)
            {
                uv0Index = p.attributes["TEXCOORD_0"];
            }

            // Bounding Box
            if (positionIndex > -1)
            {
                std::vector<double> min = gltfData.accessors[positionIndex].minValues;
                std::vector<double> max = gltfData.accessors[positionIndex].maxValues;

                m.boundingBox.min = { (float)min[0], (float)min[1], (float)min[2] };
                m.boundingBox.max = { (float)max[0], (float)max[1], (float)max[2] };
            }

            // Vertex positions
            tinygltf::Accessor positionAccessor = gltfData.accessors[positionIndex];
            tinygltf::BufferView positionBufferView = gltfData.bufferViews[positionAccessor.bufferView];
            const tinygltf::Buffer& positionBuffer = gltfData.buffers[positionBufferView.buffer];
            const UINT8* positionBufferAddress = positionBuffer.data.data();
            int positionStride = tinygltf::GetComponentSizeInBytes(positionAccessor.componentType) * tinygltf::GetNumComponentsInType(positionAccessor.type);
            assert(positionStride == 12);

            // Vertex indices
            tinygltf::Accessor indexAccessor = gltfData.accessors[indicesIndex];
            tinygltf::BufferView indexBufferView = gltfData.bufferViews[indexAccessor.bufferView];
            const tinygltf::Buffer& indexBuffer = gltfData.buffers[indexBufferView.buffer];
            const UINT8* indexBufferAddress = indexBuffer.data.data();
            int indexStride = tinygltf::GetComponentSizeInBytes(indexAccessor.componentType) * tinygltf::GetNumComponentsInType(indexAccessor.type);
            m.indices.resize(indexAccessor.count);

            // Vertex normals
            tinygltf::Accessor normalAccessor;
            tinygltf::BufferView normalBufferView;
            const UINT8* normalBufferAddress = nullptr;
            int normalStride = -1;
            if (normalIndex > -1)
            {
                normalAccessor = gltfData.accessors[normalIndex];
                normalBufferView = gltfData.bufferViews[normalAccessor.bufferView];
                const tinygltf::Buffer& normalBuffer = gltfData.buffers[normalBufferView.buffer];
                normalBufferAddress = normalBuffer.data.data();
                normalStride = tinygltf::GetComponentSizeInBytes(normalAccessor.componentType) * tinygltf::GetNumComponentsInType(normalAccessor.type);
                assert(normalStride == 12);
            }

            // Vertex tangents
            tinygltf::Accessor tangentAccessor;
            tinygltf::BufferView tangentBufferView;
            const UINT8* tangentBufferAddress = nullptr;
            int tangentStride = -1;
            if (tangentIndex > -1)
            {
                tangentAccessor = gltfData.accessors[tangentIndex];
                tangentBufferView = gltfData.bufferViews[tangentAccessor.bufferView];
                const tinygltf::Buffer& tangentBuffer = gltfData.buffers[tangentBufferView.buffer];
                tangentBufferAddress = tangentBuffer.data.data();
                tangentStride = tinygltf::GetComponentSizeInBytes(tangentAccessor.componentType) * tinygltf::GetNumComponentsInType(tangentAccessor.type);
                assert(tangentStride == 16);
            }

            // Vertex texture coordinates
            tinygltf::Accessor uv0Accessor;
            tinygltf::BufferView uv0BufferView;
            const UINT8* uv0BufferAddress = nullptr;
            int uv0Stride = -1;
            if (uv0Index > -1)
            {
                uv0Accessor = gltfData.accessors[uv0Index];
                uv0BufferView = gltfData.bufferViews[uv0Accessor.bufferView];
                const tinygltf::Buffer& uv0Buffer = gltfData.buffers[uv0BufferView.buffer];
                uv0BufferAddress = uv0Buffer.data.data();
                uv0Stride = tinygltf::GetComponentSizeInBytes(uv0Accessor.componentType) * tinygltf::GetNumComponentsInType(uv0Accessor.type);
                assert(uv0Stride == 8);
            }

            // Figure out how to adjust UVs for this mesh
            const Material& material = scene.materials[m.material];
            DirectX::XMFLOAT2 uvAdjustment = DirectX::XMFLOAT2(1.0f, 1.0f);
            bool uvAdjustmentNeeded = false;

            // Load UV adjustments for all textures used by this mesh and apply adjustment only if it's same for all textures
            getUVAdjustment(material.data.baseColorTexIdx, scene, uvAdjustmentNeeded, uvAdjustment);
            getUVAdjustment(material.data.emissiveTexIdx, scene, uvAdjustmentNeeded, uvAdjustment);
            getUVAdjustment(material.data.roughnessMetalnessTexIdx, scene, uvAdjustmentNeeded, uvAdjustment);
          
            // Get the vertex data
            for (size_t vertexIndex = 0; vertexIndex < positionAccessor.count; vertexIndex++)
            {
                Vertex v;

                {
                    const UINT8* address = positionBufferAddress + positionBufferView.byteOffset + positionAccessor.byteOffset + (vertexIndex * positionStride);
                    memcpy(&v.position, address, positionStride);
                }

                if (normalIndex > -1)
                {
                    const UINT8* address = normalBufferAddress + normalBufferView.byteOffset + normalAccessor.byteOffset + (vertexIndex * normalStride);
                    memcpy(&v.normal, address, normalStride);
                }

                if (tangentIndex > -1)
                {
                    const UINT8* address = tangentBufferAddress + tangentBufferView.byteOffset + tangentAccessor.byteOffset + (vertexIndex * tangentStride);
                    memcpy(&v.tangent, address, tangentStride);
                }

                if (uv0Index > -1)
                {
                    const UINT8* address = uv0BufferAddress + uv0BufferView.byteOffset + uv0Accessor.byteOffset + (vertexIndex * uv0Stride);
                    memcpy(&v.uv0, address, uv0Stride);

                    // Adjust UV coordinates if needed (e.g. due to added padding). Default is no adjustment [1;1]
                    v.uv0.x *= uvAdjustment.x;
                    v.uv0.y *= uvAdjustment.y;
                }

                m.vertices.push_back(v);
            }

            // Get the index data
            // Indices can be either unsigned char, unsigned short, or unsigned long
            // Converting to full precision for easy use on GPU
            const UINT8* baseAddress = indexBufferAddress + indexBufferView.byteOffset + indexAccessor.byteOffset;
            if (indexStride == 1)
            {
                std::vector<UINT8> quarter;
                quarter.resize(indexAccessor.count);

                memcpy(quarter.data(), baseAddress, (indexAccessor.count * indexStride));

                // Convert quarter precision indices to full precision
                for (size_t i = 0; i < indexAccessor.count; i++)
                {
                    m.indices[i] = quarter[i];
                }
            }
            else if (indexStride == 2)
            {
                std::vector<UINT16> half;
                half.resize(indexAccessor.count);

                memcpy(half.data(), baseAddress, (indexAccessor.count * indexStride));

                // Convert half precision indices to full precision
                for (size_t i = 0; i < indexAccessor.count; i++)
                {
                    m.indices[i] = half[i];
                }
            }
            else
            {
                memcpy(m.indices.data(), baseAddress, (indexAccessor.count * indexStride));
            }

            // Add the mesh primitive
            mesh.primitives.push_back(m);

            geometryIndex++;
        }

        scene.meshes.push_back(mesh);
    }

    scene.numGeometries = geometryIndex;
}

/**
* Parse the various data of a GLTF file.
*/
void ParseGLTF(const tinygltf::Model gltfData, const ConfigInfo& config, Scene &scene)
{
    // Parse Cameras
    ParseGLTFCameras(gltfData, scene);

    // Parse Nodes
    ParseGLTFNodes(gltfData, scene);

    // Parse Materials
    ParseGLTFMaterials(gltfData, scene);

    // Parse and Load Textures
    ParseGLFTextures(gltfData, config, scene);

    // Parse Meshes
    ParseGLTFMeshes(gltfData, scene);
}

/**
* Traverse the scene graph and update the instance transforms.
*/
void TraverseScene(size_t nodeIndex, XMMATRIX transform, Scene& scene)
{
	// Get the node
	Node node = scene.nodes[nodeIndex];

    XMMATRIX nodeTransform;
	if (node.isMatrixProvided) {
        nodeTransform = node.matrix;
	} else {
		// Get the node's transforms
		XMFLOAT3 translation = node.translation;
		XMFLOAT4 rotation = node.rotation;
		XMFLOAT3 scale = node.scale;

		// Compose the node's local transform, M = T * R * S
		XMMATRIX t = XMMatrixTranslation(translation.x, translation.y, translation.z);
		XMMATRIX r = XMMatrixRotationQuaternion(XMLoadFloat4(&rotation));
		XMMATRIX s = XMMatrixScaling(scale.x, scale.y, scale.z);
		nodeTransform = XMMatrixMultiply(XMMatrixMultiply(s, r), t);
	}

	// Compose the global transform
	transform = XMMatrixMultiply(nodeTransform, transform);

	// When at a leaf node with a mesh or camera, update their transform
    if (node.children.size() == 0) {
        if (node.instance > -1)
        {
            // Update mesh instance's transform
            Instance* instance = &scene.instances[node.instance];

            // Convert RH to LH system
            XMMATRIX s = XMMatrixScaling(1, 1, -1);
            transform = XMMatrixMultiply(transform, s);

            // Update the instance's transform data
            XMMATRIX transpose = XMMatrixTranspose(transform);
            memcpy(instance->transform, &transpose, sizeof(XMFLOAT4) * 3);
            return;
        } 
        else if (node.camera > -1)
        {
            // Update camera transform
            scene.cameras.push_back(scene.cameraTemplates[node.camera]);
            Camera* instance = &scene.cameras[scene.cameras.size() - 1];

            // Update the camera basis data
            XMStoreFloat3(&scene.cameras[node.camera].right, transform.r[0]);
            XMStoreFloat3(&scene.cameras[node.camera].up, transform.r[1]);
            XMStoreFloat3(&scene.cameras[node.camera].forward, transform.r[2]);

            // Convert RH to LH system
            XMMATRIX s = XMMatrixScaling(1, 1, -1);
            transform = XMMatrixMultiply(transform, s);
            
            // Set camera's origin
            DirectX::XMFLOAT3 position(0.0f , 0.0f, 0.0f);
            XMVECTOR origin = XMVector3Transform(XMLoadFloat3(&position), transform);
            XMStoreFloat3(&scene.cameras[node.camera].position, origin);

            return;
        }
    }

	// Recursively traverse the scene graph
	for (size_t i = 0; i < node.children.size(); i++)
	{
		TraverseScene(node.children[i], transform, scene);
	}
}

//----------------------------------------------------------------------------------------------------------
// Public Functions
//----------------------------------------------------------------------------------------------------------

/**
* Loads and parse a glTF scene.
*/
bool Load(const ConfigInfo& config, Scene &scene)
{
    tinygltf::Model gltfData;
    tinygltf::TinyGLTF gltfLoader;
    std::string err, warn, filepath;

    // Build the path to to GLTF file
    filepath.append(config.scenePath);
    filepath.append(config.sceneFile);

    bool binary = false;
    if (config.sceneFile.find(".glb") != std::string::npos) binary = true;
    else if (config.sceneFile.find(".gltf")) binary = false;
    else return false; // Unknown file format

    // Load the scene
    bool result = false;
    if (binary)
    {
        result = gltfLoader.LoadBinaryFromFile(&gltfData, &err, &warn, filepath);
    }
    else
    {
        result = gltfLoader.LoadASCIIFromFile(&gltfData, &err, &warn, filepath);
    }

    if (!result)
    {
        // An error occurred
        std::string msg = std::string(err.begin(), err.end());
        MessageBox(NULL, std::wstring(msg.begin(), msg.end()).c_str(), L"Error", MB_OK);
        return false;
    }
    else if (warn.length() > 0)
    {
        // Warning
        std::string msg = std::string(warn.begin(), warn.end());
        MessageBox(NULL, std::wstring(msg.begin(), msg.end()).c_str(), L"Warning", MB_OK);
        return false;
    }

    // Parse the GLTF data
    ParseGLTF(gltfData, config, scene);

    // Traverse the scene graph and update instance transforms
    for(size_t rootIndex = 0; rootIndex < scene.roots.size(); rootIndex++)
    {
        XMMATRIX transform = XMMatrixIdentity();
        int nodeIndex = scene.roots[rootIndex];
        TraverseScene(nodeIndex, transform, scene);
    }

    return true;
}

/*
* Releases memory used by the glTF scene.
*/
void Cleanup(Scene &scene)
{
    // Release texture memory
    for (size_t textureIndex = 0; textureIndex < scene.textures.size(); textureIndex++)
    {
        Textures::UnloadTexture(scene.textures[textureIndex]);
    }
    scene.textures.clear();
}

}
