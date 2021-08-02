/*
* Copyright (c) 2014-2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "FreeImage.h"
#include "nvh/nvprint.hpp"
#include <filesystem>
#include <iostream>
#include <thread>

//
// This is an extra to the tinyglf, allowing to differ loading
// of images and using FreeImage to load them instead of stb_image
//


namespace tinygltf {

///
/// Converts 96bpp to 128bpp RGBA without clamping.
/// Note that we can't use FreeImage_ConvertToRGBAF() as it clamps to [0,1].
///
static FIBITMAP* ConvertToRGBAF(FIBITMAP* pDib)
{
  const unsigned width  = FreeImage_GetWidth(pDib);
  const unsigned height = FreeImage_GetHeight(pDib);

  auto pNew = FreeImage_AllocateT(FIT_RGBAF, width, height);
  FreeImage_CloneMetadata(pNew, pDib);

  const unsigned src_pitch = FreeImage_GetPitch(pDib);
  const unsigned dst_pitch = FreeImage_GetPitch(pNew);

  const BYTE* src_bits = (BYTE*)FreeImage_GetBits(pDib);
  BYTE*       dst_bits = (BYTE*)FreeImage_GetBits(pNew);

  for(unsigned y = 0; y < height; y++)
  {
    const FIRGBF* src_pixel = (FIRGBF*)src_bits;
    FIRGBAF*      dst_pixel = (FIRGBAF*)dst_bits;

    for(unsigned x = 0; x < width; x++)
    {
      // Convert pixels directly, while adding a "dummy" alpha of 1.0
      dst_pixel[x].red   = src_pixel[x].red;
      dst_pixel[x].green = src_pixel[x].green;
      dst_pixel[x].blue  = src_pixel[x].blue;
      dst_pixel[x].alpha = 1.0F;
    }
    src_bits += src_pitch;
    dst_bits += dst_pitch;
  }
  return pNew;
}

///
/// Local function to put the data of a FreeImage `dib` to the tinygltf::Image
/// Converts all format to RGBA (8bit or 32bit) to be handled by Vulkan
///
static bool LoadFreeImage(Image* image, FIBITMAP* dib)
{
  if(dib == nullptr)
    return false;

  image->width  = FreeImage_GetWidth(dib);
  image->height = FreeImage_GetHeight(dib);

  // Converting always to RGBA, float or 8 bit per channel.
  FREE_IMAGE_TYPE type = FreeImage_GetImageType(dib);
  if(type != FIT_BITMAP)
  {
    if(type == FIT_RGBF)  // HDR image type
    {
      FIBITMAP* ndib = ConvertToRGBAF(dib);
      assert(ndib != nullptr);
      FreeImage_Unload(dib);
      dib = ndib;
    }
    else
    {
      // PNG (16bit, 64bit)
      FIBITMAP* ndib = FreeImage_ConvertToType(dib, FIT_BITMAP);
      assert(ndib != nullptr);
      FreeImage_Unload(dib);
      dib              = ndib;
      unsigned int bpp = FreeImage_GetBPP(dib);
      if(bpp != 32)
      {
        FIBITMAP* ndib = FreeImage_ConvertTo32Bits(dib);
        assert(ndib != nullptr);
        FreeImage_Unload(dib);
        dib = ndib;
      }
    }
  }
  else
  {
    // JPG, PNG, BMP (24bit, 32bit)
    unsigned int bpp = FreeImage_GetBPP(dib);
    if(bpp != 32)  // RGB -> RGBA
    {
      FIBITMAP* ndib = FreeImage_ConvertTo32Bits(dib);
      assert(ndib != nullptr);
      FreeImage_Unload(dib);
      dib = ndib;
    }
  }

  // Image is upside down to what is usually expected
  FreeImage_FlipVertical(dib);

  if(type == FIT_RGBF || type == FIT_RGBAF)
    image->pixel_type = TINYGLTF_COMPONENT_TYPE_FLOAT;
  else
    image->pixel_type = TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;

  unsigned int bpp = FreeImage_GetBPP(dib);
  image->component = 4;                       // Always RGBA
  image->bits      = bpp / image->component;  // bits per component

  // Copy the pixel data to the image
  size_t Bpc       = image->bits / 8;  // Byte per component
  size_t data_size = static_cast<size_t>(image->width * image->height * image->component) * Bpc;
  BYTE*  pData     = FreeImage_GetBits(dib);
  image->image.resize(data_size);
  std::copy(pData, pData + data_size, image->image.begin());
  FreeImage_Unload(dib);

  return true;
}


///
/// Loading all external images using FreeImage.
/// Need to set tinygltf::TinyGLTF::SetImageLoader(nullptr, nullptr)
/// Then call this function after loading the scene
///
static void loadExternalImages(Model* model, const std::string& gltf_filename)
{
  if(model->images.empty())
    return;

  namespace fs = std::filesystem;
  auto basedir = fs::path(gltf_filename).parent_path();

  // Loading all images from disk sequentially
  std::vector<FIBITMAP*> all_fib;
  for(auto& img : model->images)
  {
    std::string       img_uri = fs::path(basedir / img.uri).string();
    FREE_IMAGE_FORMAT fif     = FreeImage_GetFileType(img_uri.c_str(), 0);

    if(fif == FIF_UNKNOWN || FreeImage_FIFSupportsReading(fif) == false)
    {
      LOGE("Couldn't load: %s \n", img_uri.c_str());
      all_fib.emplace_back(nullptr);
      continue;
    }

    int flags{0};
    // Import JPEGs with full accuracy (to get identical results as with the default settings of the JPEG library).
    if(fif == FIF_JPEG)
      flags |= JPEG_ACCURATE;

    all_fib.emplace_back(FreeImage_Load(fif, img_uri.c_str(), flags));
  }

  // Converting in parallel the images to an appropriate format
  std::vector<std::thread> tasks;
  for(size_t i = 0; i < all_fib.size(); i++)
  {
    tasks.emplace_back(std::thread([&, i]() { LoadFreeImage(&model->images[i], all_fib[i]); }));
  }
  // Waiting for all threads
  for(auto& t : tasks)
    t.join();
}

///
/// Override the tinygltf image loader and leave it to Iray
/// Need to set tinygltf::TinyGLTF::SetImageLoader(&tinygltf::LoadFreeImageData, nullptr)
/// This image is loaded in memory and the data arrives in `bytes`
///
static bool LoadFreeImageData(Image*               image,
                              const int            image_idx,
                              std::string*         err,
                              std::string*         warn,
                              int                  req_width,
                              int                  req_height,
                              const unsigned char* bytes,
                              int                  size,
                              void*                user_data)
{
  auto*             stream = FreeImage_OpenMemory((BYTE*)bytes, size);
  FREE_IMAGE_FORMAT fif    = FreeImage_GetFileTypeFromMemory(stream, 0);

  int flags{0};
  if(fif == FIF_JPEG)
    flags |= JPEG_ACCURATE;

  FIBITMAP* dib = FreeImage_LoadFromMemory(fif, stream, flags);
  return LoadFreeImage(image, dib);
}


}  // namespace tinygltf
