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

// Enable this to compress textures using BC7 via DirectxTex libary (requires DX11)
#define ENABLE_GPU_COMPRESSION 1

#include "Common.h"
#include "Textures.h"
#include "../shaders/shared.h"

#include "stb/stb_image.h"

#if ENABLE_GPU_COMPRESSION
#pragma comment(lib, "d3d11.lib")

#if _DEBUG
#pragma comment(lib, "lib/DirectXTeX/DirectXTex_d.lib")
#else
#pragma comment(lib, "lib/DirectXTeX/DirectXTex.lib")
#endif

#include <d3d11.h>
#include "directxtex/DirectXTex.h"

static ID3D11Device* d3d11Device = nullptr;
using namespace DirectX;
#endif

namespace Textures
{

	//----------------------------------------------------------------------------------------------------------
	// Private Functions
	//----------------------------------------------------------------------------------------------------------

	/**
	* Releases texture data memory using the same method it was allocated
	*/
	void UnloadTextureData(TextureData& textureData) {

		if (textureData.texels) {
			if (textureData.stbAllocation)
				stbi_image_free(textureData.texels);
			else
				delete[] textureData.texels;
		}

		textureData.texels = nullptr;
		textureData.stbAllocation = false;
	}

	/**
	 * Compute the aligned memory required for the texture.
	 * Add texels to the texture if either dimension is not a factor of 4 (required for BC7 compressed formats).
	 */
	bool FormatTexture(TextureData& texture, DirectX::XMFLOAT2& uvAdjustment)
	{
		// BC7 compressed textures require 4x4 texel blocks
		// Add texels to the texture if its original dimensions aren't factors of 4
		if (texture.width % 4 != 0 || texture.height % 4 != 0)
		{
			// Get original row stride
			UINT originalWidth = texture.width;
			UINT rowSize = (texture.width * texture.stride);
			UINT numRows = texture.height;

			// Align the new texture to 4x4
			texture.width = ALIGN(4, texture.width);
			texture.height = ALIGN(4, texture.height);

			UINT alignedRowSize = (texture.width * texture.stride);
			UINT size = alignedRowSize * texture.height;

			// Figure out uv adjustment needed due to padding (we want UVs to point to the area where original unpadded texture is)
			uvAdjustment.x = float(originalWidth) / float(texture.width);
			uvAdjustment.y = float(numRows) / float(texture.height);

			// Copy the original texture into the new one
			size_t offset = 0;
			size_t alignedOffset = 0;
			UINT8* texels = new UINT8[size];

			for (UINT row = 0; row < numRows; row++)
			{
				memcpy(&texels[alignedOffset], &texture.texels[offset], rowSize);
				
				// Fill empty space in the aligned row with border pixel value
				for (UINT i = 0; i < texture.width - originalWidth; i++) {
					memcpy(&texels[alignedOffset + rowSize + (i * UINT(texture.stride))], &texels[alignedOffset + rowSize - texture.stride], texture.stride);
				}

				alignedOffset += alignedRowSize;
				offset += rowSize;
			}
			
			// Copy last row values into new rows
			UINT lastRowOffset = alignedOffset - alignedRowSize;
			for (UINT row = 0; row < texture.height - numRows; row++)
			{
				memcpy(&texels[alignedOffset], &texels[lastRowOffset], alignedRowSize);

				alignedOffset += alignedRowSize;
				offset += rowSize;
			}

			// Release the memory of the original texture
			UnloadTextureData(texture);
			texture.texels = texels;
		}

		// Compute the texture's aligned memory size
		texture.rowPitch = (texture.width * texture.stride);
		texture.texelBytes = (texture.rowPitch * UINT64(texture.height));

		return (texture.texelBytes > 0);
	}

#if ENABLE_GPU_COMPRESSION

	/**
	 * Copy a compressed BC7 texture into our format, aligned for GPU use.
	 */
	bool FormatCompressedTexture(ScratchImage& src, Texture& dst)
	{
		// Get the texture's metadata
		const TexMetadata metadata = src.GetMetadata();

		// Check if the texture's format is supported
		if (metadata.format != DXGI_FORMAT_BC7_UNORM && metadata.format != DXGI_FORMAT_BC7_UNORM_SRGB && metadata.format != DXGI_FORMAT_BC7_TYPELESS)
		{
			std::string msg = "Error: unsupported compressed texture format for: \'" + dst.name + "\' \'" + dst.filepath + "\'\n. Compressed textures must be in BC7 format";
			MessageBox(NULL, std::wstring(msg.begin(), msg.end()).c_str(), L"Error", MB_OK);
			return false;
		}

		// Delete existing texels
		UnloadTexture(dst);

		// Copy each mip level to the texel array
		for (UINT mipIndex = 0; mipIndex < static_cast<UINT>(metadata.mipLevels); mipIndex++)
		{
			const Image* image = src.GetImage(mipIndex, 0, 0);

			TextureData mipData;
			mipData.width = image->width;
			mipData.height = image->height;
			mipData.texelBytes = image->slicePitch;
			mipData.stride = 4;
			mipData.rowPitch = image->rowPitch;

			// Copy texel data
			mipData.texels = new UINT8[mipData.texelBytes];
			memcpy(mipData.texels, image->pixels, image->slicePitch);

			dst.mips.push_back(mipData);
		}

		src.Release();
		return true;
	}

	/**
	 * Covert a R8G8B8A8_UNORM texture to BC7 format and generate mips.
	 */
	bool ProcessTexture(Texture& texture, bool generateMips = true, bool quickBC7 = true)
	{
		// Check if there's anything to compress
		if (texture.mips.empty()) return false;

		TextureData* texData = &texture.mips[0];

		// BC7 textures must be aligned to pixel 4x4 blocks. 
		if (texData->width % 4 != 0 || texData->height % 4 != 0) {
			MessageBox(NULL, L"Size of the input texture must be multiple of 4 for compression to work! Make sure to call FormatTexture() before this function to add necessary padding", L"Error", MB_OK);
			return false;
		}

		Image source = {};
		source.width = texData->width;
		source.height = texData->height;
		source.rowPitch = (texData->width * size_t(texData->stride));
		source.slicePitch = (source.rowPitch * source.height);
		source.format = DXGI_FORMAT_R8G8B8A8_UNORM;
		source.pixels = texData->texels;

		TEX_COMPRESS_FLAGS flags = TEX_COMPRESS_DEFAULT;
		if (quickBC7) flags = TEX_COMPRESS_BC7_QUICK;
		ScratchImage scratchBuffer;

		if (generateMips) {
			ScratchImage mips;
			if (FAILED(DirectX::GenerateMipMaps(source, TEX_FILTER_DEFAULT, 0, mips))) return false;
			if (FAILED(Compress(d3d11Device, mips.GetImages(), mips.GetImageCount(), mips.GetMetadata(), DXGI_FORMAT_BC7_UNORM, flags, 1.f, scratchBuffer))) return false;
			mips.Release();
		} else {
			if (FAILED(Compress(d3d11Device, source, DXGI_FORMAT_BC7_UNORM, flags, 1.f, scratchBuffer))) return false;
		}

		// Format the compressed texture into our format, prepping it for use on the GPU
		texture.isBC7Compressed = true;

		return FormatCompressedTexture(scratchBuffer, texture);
	}
#endif

	//----------------------------------------------------------------------------------------------------------
	// Public Functions
	//----------------------------------------------------------------------------------------------------------

	/**
	* Initialize texture loader before use
	*/
	bool Initialize()
	{
#if ENABLE_GPU_COMPRESSION
		D3D_FEATURE_LEVEL requested = D3D_FEATURE_LEVEL_11_1;
		D3D_FEATURE_LEVEL supported;
		if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, &requested, 1, D3D11_SDK_VERSION, &d3d11Device, &supported, nullptr))) return false;
#endif
		return true;
	}

	/**
	* Load an image from disk using STB library.
	*/
	bool LoadTexture(Texture& texture, DirectX::XMFLOAT2& uvAdjustment)
	{
		// Initialize adjustment to identity
		uvAdjustment.x = 1.0f;
		uvAdjustment.y = 1.0f;

		// Load the texture with stb_image (require 4 component RGBA)
		TextureData textureData;
		textureData.texels = stbi_load(texture.filepath.c_str(), &textureData.width, &textureData.height, &textureData.stride, STBI_rgb_alpha);
		textureData.stride = 4;
		textureData.rowPitch = UINT64(textureData.width) * textureData.stride;
		textureData.texelBytes = textureData.rowPitch * UINT64(textureData.height);
		textureData.stbAllocation = true;

		if (!textureData.texels)
		{
			std::string msg = "Error: failed to load texture: \'";
			msg.append(texture.name);
			msg.append("\'");
			MessageBox(NULL, std::wstring(msg.begin(), msg.end()).c_str(), L"Error", MB_OK);
			return false;
		}

		// Prep the texture for compression and use on the GPU (alignment to 4x4) 
		FormatTexture(textureData, uvAdjustment);

		// Store raw texture data as mip level 0
		texture.mips.push_back(textureData);

#if ENABLE_GPU_COMPRESSION
		// Compress texture on GPU and generate mips
		ProcessTexture(texture);
#endif

		return true;
	}

	/**
	* Releases all memory used by texture
	*/
	void UnloadTexture(Texture& texture) {
		if (!texture.isCopy) {
			for (auto mip : texture.mips) UnloadTextureData(mip);
		}
		texture.mips.clear();
	}

	/**
	* Release resources used by texture loader
	*/
	void Cleanup()
	{
#if ENABLE_GPU_COMPRESSION
		SAFE_RELEASE(d3d11Device);
#endif
	}


}
