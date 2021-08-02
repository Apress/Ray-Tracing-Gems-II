/* Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifdef MEMORY_LEAKS_CHECK
#   pragma message("build will Check for Memory Leaks!")
#   define _CRTDBG_MAP_ALLOC
#   include <stdlib.h>
#   include <crtdbg.h>
#endif

#include<iostream>
#include <string.h>

#include <stdio.h>
#include <assert.h>
#include "nv_dds.h"

using namespace std;
using namespace nv_dds;

inline unsigned int comps2internalfmt(unsigned int i)
{
    switch(i)
    {
    case 1:
        return RED; // LUMINANCE is not Core OpenGL
    case 2:
        return RG8;
    case 3:
        return RGB8;
    case 4:
        return RGBA8;
    }
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
// CDDSImage public functions

///////////////////////////////////////////////////////////////////////////////
// default constructor
CDDSImage::CDDSImage()
  : m_format(0),
    m_components(0),
    m_type(TextureNone),
    m_valid(false)
{
}

CDDSImage::~CDDSImage()
{
}

void CDDSImage::create_textureFlat(unsigned int format, unsigned int components, const CTexture &baseImage)
{
    assert(format != 0);
    assert(components != 0);
    assert(baseImage.get_depth() == 1);

    // remove any existing images
    clear();
    
    m_format = format;
    m_components = components;
    m_internal_format = comps2internalfmt(m_components);
    m_type = TextureFlat;

    m_images.push_back(baseImage);

    m_valid = true;
}

void CDDSImage::create_texture3D(unsigned int format, unsigned int components, const CTexture &baseImage)
{
    assert(format != 0);
    assert(components != 0);
    assert(baseImage.get_depth() > 1);

    // remove any existing images
    clear();

    m_format = format;
    m_components = components;
    m_internal_format = comps2internalfmt(m_components);
    m_type = Texture3D;

    m_images.push_back(baseImage);

    m_valid = true;
}

inline bool same_size(const CTexture &a, const CTexture &b)
{
    if (a.get_width() != b.get_width())
        return false;
    if (a.get_height() != b.get_height())
        return false;
    if (a.get_depth() != b.get_depth())
        return false;

    return true;
}

void CDDSImage::create_textureCubemap(unsigned int format, unsigned int components,
                                      const CTexture &positiveX, const CTexture &negativeX, 
                                      const CTexture &positiveY, const CTexture &negativeY, 
                                      const CTexture &positiveZ, const CTexture &negativeZ)
{
    assert(format != 0);
    assert(components != 0);
    assert(positiveX.get_depth() == 1);

    // verify that all dimensions are the same 
    assert(same_size(positiveX, negativeX));
    assert(same_size(positiveX, positiveY));
    assert(same_size(positiveX, negativeY));
    assert(same_size(positiveX, positiveZ));
    assert(same_size(positiveX, negativeZ));

    // remove any existing images
    clear();

    m_format = format;
    m_components = components;
    m_internal_format = comps2internalfmt(m_components);
    m_type = TextureCubemap;

    m_images.push_back(positiveX);
    m_images.push_back(negativeX);
    m_images.push_back(positiveY);
    m_images.push_back(negativeY);
    m_images.push_back(positiveZ);
    m_images.push_back(negativeZ);

    m_valid = true;
}

///////////////////////////////////////////////////////////////////////////////
// loads DDS image
//
// filename - fully qualified name of DDS image
// flipImage - specifies whether image is flipped on load, default is true
bool CDDSImage::load(string filename, bool flipImage, bool RGB2RGBA)
{
    assert(filename.length() != 0);

    std::cout << "File name has length" << std::endl;
    
    // clear any previously loaded images
    clear();
    
    // open file
    FILE *fp = fopen(filename.c_str(), "rb");
    if (fp == NULL)
        return false;

    std::cout << "Found a file" << std::endl;

    // read in file marker, make sure its a DDS file
    char filecode[4];
    fread(filecode, 1, 4, fp);
    if (strncmp(filecode, "DDS ", 4) != 0)
    {
        fclose(fp);
        return false;
    }

    std::cout << "Its a DDS file" << std::endl;

    // read in DDS header
    DDS_HEADER ddsh;
    fread(&ddsh, sizeof(DDS_HEADER), 1, fp);

    swap_endian(&ddsh.dwSize);
    swap_endian(&ddsh.dwFlags);
    swap_endian(&ddsh.dwHeight);
    swap_endian(&ddsh.dwWidth);
    swap_endian(&ddsh.dwPitchOrLinearSize);
    swap_endian(&ddsh.dwMipMapCount);
    swap_endian(&ddsh.ddspf.dwSize);
    swap_endian(&ddsh.ddspf.dwFlags);
    swap_endian(&ddsh.ddspf.dwFourCC);
    swap_endian(&ddsh.ddspf.dwRGBBitCount);
    swap_endian(&ddsh.dwCaps1);
    swap_endian(&ddsh.dwCaps2);

    // default to flat texture type (1D, 2D, or rectangle)
    m_type = TextureFlat;

    // check if image is a cubemap
    if (ddsh.dwCaps2 & DDSF_CUBEMAP){
        m_type = TextureCubemap;
    }

    // check if image is a volume texture
    if ((ddsh.dwCaps2 & DDSF_VOLUME) && (ddsh.dwDepth > 0)){
        m_type = Texture3D;
    }


    // figure out what the image format is
    if (ddsh.ddspf.dwFlags & DDSF_FOURCC) 
    {
        switch(ddsh.ddspf.dwFourCC)
        {
            case FOURCC_DXT1:
                m_format = COMPRESSED_RGBA_S3TC_DXT1_EXT;
                m_components = 3;
                m_internal_format = m_format;
                break;
            case FOURCC_DXT3:
                m_format = COMPRESSED_RGBA_S3TC_DXT3_EXT;
                m_components = 4;
                m_internal_format = m_format;
                break;
            case FOURCC_DXT5:
                m_format = COMPRESSED_RGBA_S3TC_DXT5_EXT;
                m_components = 4;
                m_internal_format = m_format;
                break;
            default:
                fclose(fp);
                return false;
        }
    }
    else if (ddsh.ddspf.dwFlags == DDSF_RGBA && ddsh.ddspf.dwRGBBitCount == 32)
    {
        m_format = BGRA_EXT; 
        m_components = 4;
        m_internal_format = comps2internalfmt(m_components);
    }
    else if (ddsh.ddspf.dwFlags == DDSF_RGB  && ddsh.ddspf.dwRGBBitCount == 32)
    {
        m_format = BGRA_EXT; 
        m_components = 4;
        m_internal_format = comps2internalfmt(m_components);
    }
    else if (ddsh.ddspf.dwFlags == DDSF_RGB  && ddsh.ddspf.dwRGBBitCount == 24)
    {
        m_format = BGR_EXT; 
        m_components = 3;
        m_internal_format = comps2internalfmt(m_components);
    }
	else if (ddsh.ddspf.dwRGBBitCount == 8)
	{
		m_format = LUMINANCE; 
		m_components = 1;
        m_internal_format = comps2internalfmt(m_components);
	}
    else 
    {
        fclose(fp);
        return false;
    }
    
    // store primary surface width/height/depth
    unsigned int width, height, depth;
    width = ddsh.dwWidth;
    height = ddsh.dwHeight;
    depth = clamp_size(ddsh.dwDepth);   // set to 1 if 0
    
    // use correct size calculation function depending on whether image is 
    // compressed
    unsigned int (CDDSImage::*sizefunc)(unsigned int, unsigned int);
    sizefunc = (is_compressed() ? &CDDSImage::size_dxtc : &CDDSImage::size_rgb);

    // load all surfaces for the image (6 surfaces for cubemaps)
    for (unsigned int n = 0; n < (unsigned int)(m_type == TextureCubemap ? 6 : 1); n++)
    {
        // add empty texture object
        m_images.push_back(CTexture());

        // get reference to newly added texture object
        CTexture &img = m_images[n];
        
        // calculate amount of data to load and surface size
        unsigned int size = (this->*sizefunc)(width, height)*depth;
        unsigned char *pixels = NULL;
        // special case where we might have to pad each component with non-used Alpha
        bool bRGB2RGBA = false;
        if(RGB2RGBA && (m_format == BGR_EXT) )
        {
            bRGB2RGBA = true;
            m_format = BGRA_EXT;
            m_internal_format = RGBA8;
            m_components = 4; // RGBX8
            unsigned int szRGB = size / 3;
            // calculate again the amount of data to load and surface size
            size = (this->*sizefunc)(width, height)*depth;
            // load surface
            pixels = new unsigned char[size];
            fread(pixels, 1, szRGB*3, fp);
            for(int i=szRGB-1; i > 0; i--)
            {
                pixels[(i*4)+3] = 0;
                pixels[(i*4)+2] = pixels[(i*3)+2];
                pixels[(i*4)+1] = pixels[(i*3)+1];
                pixels[(i*4)+0] = pixels[(i*3)+0];
            }
        } else {
            // load surface
            pixels = new unsigned char[size];
            fread(pixels, 1, size, fp);
        }

        img.create(width, height, depth, size, pixels);
        
        delete [] pixels;

        if (flipImage) flip(img);
        
        unsigned int w = clamp_size(width >> 1);
        unsigned int h = clamp_size(height >> 1);
        unsigned int d = clamp_size(depth >> 1); 

        // store number of mipmaps
        unsigned int numMipmaps = ddsh.dwMipMapCount;

        // number of mipmaps in file includes main surface so decrease count 
        // by one
        if (numMipmaps != 0)
            numMipmaps--;

        // load all mipmaps for current surface
        for (unsigned int i = 0; i < numMipmaps && (w || h); i++)
        {
            // add empty surface
            img.add_mipmap(CSurface());

            // get reference to newly added mipmap
            CSurface &mipmap = img.get_mipmap(i);

            // calculate mipmap size
            size = (this->*sizefunc)(w, h)*d;
            unsigned char *pixels = NULL;

            // special case where we might have to pad each component with non-used Alpha
            if(bRGB2RGBA)
            {
                unsigned int szRGB = size / 3;
                // calculate again the amount of data to load and surface size
                size = (this->*sizefunc)(width, height)*depth;
                // load surface
                pixels = new unsigned char[size];
                fread(pixels, 1, szRGB*3, fp);
                for(int i=szRGB-1; i > 0; i--)
                {
                    pixels[(i*4)+3] = 0;
                    pixels[(i*4)+2] = pixels[(i*3)+2];
                    pixels[(i*4)+1] = pixels[(i*3)+1];
                    pixels[(i*4)+0] = pixels[(i*3)+0];
                }
            } else {
                // load surface
                pixels = new unsigned char[size];
                fread(pixels, 1, size, fp);
            }

            mipmap.create(w, h, d, size, pixels);
            
            delete [] pixels;

            if (flipImage) flip(mipmap);

            // shrink to next power of 2
            w = clamp_size(w >> 1);
            h = clamp_size(h >> 1);
            d = clamp_size(d >> 1); 
        }
    }

    // swap cubemaps on y axis (since image is flipped in OGL)
    if (m_type == TextureCubemap && flipImage)
    {
        CTexture tmp;
        tmp = m_images[3];
        m_images[3] = m_images[2];
        m_images[2] = tmp;
    }
    
    fclose(fp);

    m_valid = true;

    return true;
}

void CDDSImage::write_texture(const CTexture &texture, FILE *fp)
{
    assert(get_num_mipmaps() == texture.get_num_mipmaps());
    
    fwrite(texture, 1, texture.get_size(), fp);
    
    for (unsigned int i = 0; i < texture.get_num_mipmaps(); i++)
    {
        const CSurface &mipmap = texture.get_mipmap(i);
        fwrite(mipmap, 1, mipmap.get_size(), fp);
    }
}

bool CDDSImage::save(std::string filename, bool flipImage)
{
    assert(m_valid);
    assert(m_type != TextureNone);

    DDS_HEADER ddsh;
    unsigned int headerSize = sizeof(DDS_HEADER);
    memset(&ddsh, 0, headerSize);
    ddsh.dwSize = headerSize;
    ddsh.dwFlags = DDSF_CAPS | DDSF_WIDTH | DDSF_HEIGHT | DDSF_PIXELFORMAT;
    ddsh.dwHeight = get_height();
    ddsh.dwWidth = get_width();

    if (is_compressed())
    {
        ddsh.dwFlags |= DDSF_LINEARSIZE;
        ddsh.dwPitchOrLinearSize = get_size();
    }
    else
    {
        ddsh.dwFlags |= DDSF_PITCH;
        ddsh.dwPitchOrLinearSize = get_dword_aligned_linesize(get_width(), m_components * 8);
    }
    
    if (m_type == Texture3D)
    {
        ddsh.dwFlags |= DDSF_DEPTH;
        ddsh.dwDepth = get_depth();
    }

    if (get_num_mipmaps() > 0)
    {
        ddsh.dwFlags |= DDSF_MIPMAPCOUNT;
        ddsh.dwMipMapCount = get_num_mipmaps() + 1;
    }

    ddsh.ddspf.dwSize = sizeof(DDS_PIXELFORMAT);

    if (is_compressed())
    {
        ddsh.ddspf.dwFlags = DDSF_FOURCC;
        
        if (m_format == COMPRESSED_RGBA_S3TC_DXT1_EXT)
            ddsh.ddspf.dwFourCC = FOURCC_DXT1;
        if (m_format == COMPRESSED_RGBA_S3TC_DXT3_EXT)
            ddsh.ddspf.dwFourCC = FOURCC_DXT3;
        if (m_format == COMPRESSED_RGBA_S3TC_DXT5_EXT)
            ddsh.ddspf.dwFourCC = FOURCC_DXT5;
    }
    else
    {
        ddsh.ddspf.dwFlags = (m_components == 4) ? DDSF_RGBA : DDSF_RGB;
        ddsh.ddspf.dwRGBBitCount = m_components * 8;
        ddsh.ddspf.dwRBitMask = 0x00ff0000;
        ddsh.ddspf.dwGBitMask = 0x0000ff00;
        ddsh.ddspf.dwBBitMask = 0x000000ff;
 
        if (m_components == 4)
        {
            ddsh.ddspf.dwFlags |= DDSF_ALPHAPIXELS;
            ddsh.ddspf.dwABitMask = 0xff000000;
        }
    }
    
    ddsh.dwCaps1 = DDSF_TEXTURE;
    
    if (m_type == TextureCubemap)
    {
        ddsh.dwCaps1 |= DDSF_COMPLEX;
        ddsh.dwCaps2 = DDSF_CUBEMAP | DDSF_CUBEMAP_ALL_FACES;
    }

    if (m_type == Texture3D)
    {
        ddsh.dwCaps1 |= DDSF_COMPLEX;
        ddsh.dwCaps2 = DDSF_VOLUME;
    }

    if (get_num_mipmaps() > 0)
        ddsh.dwCaps1 |= DDSF_COMPLEX | DDSF_MIPMAP;

    // open file
    FILE *fp = fopen(filename.c_str(), "wb");
    if (fp == NULL)
        return false;

    // write file header
    fwrite("DDS ", 1, 4, fp);
    
    // write dds header
    fwrite(&ddsh, 1, sizeof(DDS_HEADER), fp);

    if (m_type != TextureCubemap)
    {
        CTexture tex = m_images[0];
        if (flipImage) flip_texture(tex);
        write_texture(tex, fp);
    }
    else
    {
        assert(m_images.size() == 6);

        for (unsigned int i = 0; i < m_images.size(); i++)
        {
            CTexture cubeFace;

            if (i == 2) 
                cubeFace = m_images[3];
            else if (i == 3) 
                cubeFace = m_images[2];
            else 
                cubeFace = m_images[i];

            if (flipImage) flip_texture(cubeFace);
            write_texture(cubeFace, fp);
        }
    }

    fclose(fp);
    return true;
}

///////////////////////////////////////////////////////////////////////////////
// free image memory
void CDDSImage::clear()
{
    m_components = 0;
    m_format = 0;
    m_type = TextureNone;
    m_valid = false;

    m_images.clear();
}


///////////////////////////////////////////////////////////////////////////////
// clamps input size to [1-size]
inline unsigned int CDDSImage::clamp_size(unsigned int size)
{
    if (size <= 0)
        size = 1;

    return size;
}

///////////////////////////////////////////////////////////////////////////////
// CDDSImage private functions
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// calculates size of DXTC texture in bytes
inline unsigned int CDDSImage::size_dxtc(unsigned int width, unsigned int height)
{
    return ((width+3)/4)*((height+3)/4)*
        (m_format == COMPRESSED_RGBA_S3TC_DXT1_EXT ? 8 : 16);   
}

///////////////////////////////////////////////////////////////////////////////
// calculates size of uncompressed RGB texture in bytes
inline unsigned int CDDSImage::size_rgb(unsigned int width, unsigned int height)
{
    return width*height*m_components;
}

///////////////////////////////////////////////////////////////////////////////
// Swap the bytes in a 32 bit value
inline void CDDSImage::swap_endian(void *val)
{
#ifdef MACOS
    unsigned int *ival = (unsigned int *)val;

    *ival = ((*ival >> 24) & 0x000000ff) |
            ((*ival >>  8) & 0x0000ff00) |
            ((*ival <<  8) & 0x00ff0000) |
            ((*ival << 24) & 0xff000000);
#endif
}

///////////////////////////////////////////////////////////////////////////////
// flip image around X axis
void CDDSImage::flip(CSurface &surface)
{
    unsigned int linesize;
    unsigned int offset;

    if (!is_compressed())
    {
        assert(surface.get_depth() > 0);

        unsigned int imagesize = surface.get_size()/surface.get_depth();
        linesize = imagesize / surface.get_height();

        for (unsigned int n = 0; n < surface.get_depth(); n++)
        {
            offset = imagesize*n;
            unsigned char *top = (unsigned char*)surface + offset;
            unsigned char *bottom = top + (imagesize-linesize);
    
            for (unsigned int i = 0; i < (surface.get_height() >> 1); i++)
            {
                swap(bottom, top, linesize);

                top += linesize;
                bottom -= linesize;
            }
        }
    }
    else
    {
        void (CDDSImage::*flipblocks)(DXTColBlock*, unsigned int);
        unsigned int xblocks = surface.get_width() / 4;
        unsigned int yblocks = surface.get_height() / 4;
        unsigned int blocksize;

        switch (m_format)
        {
            case COMPRESSED_RGBA_S3TC_DXT1_EXT: 
                blocksize = 8;
                flipblocks = &CDDSImage::flip_blocks_dxtc1; 
                break;
            case COMPRESSED_RGBA_S3TC_DXT3_EXT: 
                blocksize = 16;
                flipblocks = &CDDSImage::flip_blocks_dxtc3; 
                break;
            case COMPRESSED_RGBA_S3TC_DXT5_EXT: 
                blocksize = 16;
                flipblocks = &CDDSImage::flip_blocks_dxtc5; 
                break;
            default:
                return;
        }

        linesize = xblocks * blocksize;

		for (unsigned int j = 0; j < ((yblocks >> 1) > (unsigned int)1 ? yblocks >> 1: (unsigned int)1); j++)
        {
            DXTColBlock *top = (DXTColBlock*)((unsigned char*)surface + j * linesize);
            DXTColBlock *bottom = (DXTColBlock*)((unsigned char*)surface + (((yblocks-j)-1) * linesize));

            if (top == bottom)
            {
                (this->*flipblocks)(top, xblocks);
                break;
            }

            (this->*flipblocks)(top, xblocks);
            (this->*flipblocks)(bottom, xblocks);

            swap(bottom, top, linesize);
        }
    }
}    

void CDDSImage::flip_texture(CTexture &texture)
{
    flip(texture);
    
    for (unsigned int i = 0; i < texture.get_num_mipmaps(); i++)
    {
        flip(texture.get_mipmap(i));
    }
}

///////////////////////////////////////////////////////////////////////////////
// swap to sections of memory
void CDDSImage::swap(void *byte1, void *byte2, unsigned int size)
{
    unsigned char *tmp = new unsigned char[size];

    memcpy(tmp, byte1, size);
    memcpy(byte1, byte2, size);
    memcpy(byte2, tmp, size);

    delete [] tmp;
}

///////////////////////////////////////////////////////////////////////////////
// flip a DXT1 color block
void CDDSImage::flip_blocks_dxtc1(DXTColBlock *line, unsigned int numBlocks)
{
    DXTColBlock *curblock = line;

    for (unsigned int i = 0; i < numBlocks; i++)
    {
        swap(&curblock->row[0], &curblock->row[3], sizeof(unsigned char));
        swap(&curblock->row[1], &curblock->row[2], sizeof(unsigned char));

        curblock++;
    }
}

///////////////////////////////////////////////////////////////////////////////
// flip a DXT3 color block
void CDDSImage::flip_blocks_dxtc3(DXTColBlock *line, unsigned int numBlocks)
{
    DXTColBlock *curblock = line;
    DXT3AlphaBlock *alphablock;

    for (unsigned int i = 0; i < numBlocks; i++)
    {
        alphablock = (DXT3AlphaBlock*)curblock;

        swap(&alphablock->row[0], &alphablock->row[3], sizeof(unsigned short));
        swap(&alphablock->row[1], &alphablock->row[2], sizeof(unsigned short));

        curblock++;

        swap(&curblock->row[0], &curblock->row[3], sizeof(unsigned char));
        swap(&curblock->row[1], &curblock->row[2], sizeof(unsigned char));

        curblock++;
    }
}

///////////////////////////////////////////////////////////////////////////////
// flip a DXT5 alpha block
void CDDSImage::flip_dxt5_alpha(DXT5AlphaBlock *block)
{
    unsigned char gBits[4][4];
    
    const uint32_t mask = 0x00000007;          // bits = 00 00 01 11
    uint32_t bits = 0;
    memcpy(&bits, &block->row[0], sizeof(unsigned char) * 3);

    gBits[0][0] = (unsigned char)(bits & mask);
    bits >>= 3;
    gBits[0][1] = (unsigned char)(bits & mask);
    bits >>= 3;
    gBits[0][2] = (unsigned char)(bits & mask);
    bits >>= 3;
    gBits[0][3] = (unsigned char)(bits & mask);
    bits >>= 3;
    gBits[1][0] = (unsigned char)(bits & mask);
    bits >>= 3;
    gBits[1][1] = (unsigned char)(bits & mask);
    bits >>= 3;
    gBits[1][2] = (unsigned char)(bits & mask);
    bits >>= 3;
    gBits[1][3] = (unsigned char)(bits & mask);

    bits = 0;
    memcpy(&bits, &block->row[3], sizeof(unsigned char) * 3);

    gBits[2][0] = (unsigned char)(bits & mask);
    bits >>= 3;
    gBits[2][1] = (unsigned char)(bits & mask);
    bits >>= 3;
    gBits[2][2] = (unsigned char)(bits & mask);
    bits >>= 3;
    gBits[2][3] = (unsigned char)(bits & mask);
    bits >>= 3;
    gBits[3][0] = (unsigned char)(bits & mask);
    bits >>= 3;
    gBits[3][1] = (unsigned char)(bits & mask);
    bits >>= 3;
    gBits[3][2] = (unsigned char)(bits & mask);
    bits >>= 3;
    gBits[3][3] = (unsigned char)(bits & mask);

    // clear existing alpha bits
    memset(block->row, 0, sizeof(unsigned char) * 6);

    uint32_t *pBits = ((uint32_t*) &(block->row[0]));

    *pBits = *pBits | (gBits[3][0] << 0);
    *pBits = *pBits | (gBits[3][1] << 3);
    *pBits = *pBits | (gBits[3][2] << 6);
    *pBits = *pBits | (gBits[3][3] << 9);

    *pBits = *pBits | (gBits[2][0] << 12);
    *pBits = *pBits | (gBits[2][1] << 15);
    *pBits = *pBits | (gBits[2][2] << 18);
    *pBits = *pBits | (gBits[2][3] << 21);

    pBits = ((uint32_t*) &(block->row[3]));

    *pBits = *pBits | (gBits[1][0] << 0);
    *pBits = *pBits | (gBits[1][1] << 3);
    *pBits = *pBits | (gBits[1][2] << 6);
    *pBits = *pBits | (gBits[1][3] << 9);

    *pBits = *pBits | (gBits[0][0] << 12);
    *pBits = *pBits | (gBits[0][1] << 15);
    *pBits = *pBits | (gBits[0][2] << 18);
    *pBits = *pBits | (gBits[0][3] << 21);
}

///////////////////////////////////////////////////////////////////////////////
// flip a DXT5 color block
void CDDSImage::flip_blocks_dxtc5(DXTColBlock *line, unsigned int numBlocks)
{
    DXTColBlock *curblock = line;
    DXT5AlphaBlock *alphablock;
    
    for (unsigned int i = 0; i < numBlocks; i++)
    {
        alphablock = (DXT5AlphaBlock*)curblock;
        
        flip_dxt5_alpha(alphablock);

        curblock++;

        swap(&curblock->row[0], &curblock->row[3], sizeof(unsigned char));
        swap(&curblock->row[1], &curblock->row[2], sizeof(unsigned char));

        curblock++;
    }
}

///////////////////////////////////////////////////////////////////////////////
// CTexture implementation
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// default constructor
CTexture::CTexture()
  : CSurface()  // initialize base class part
{
}

///////////////////////////////////////////////////////////////////////////////
// creates an empty texture
CTexture::CTexture(unsigned int w, unsigned int h, unsigned int d, unsigned int imgsize, const unsigned char *pixels)
  : CSurface(w, h, d, imgsize, pixels)  // initialize base class part
{
}

CTexture::~CTexture()
{
}

///////////////////////////////////////////////////////////////////////////////
// copy constructor
CTexture::CTexture(const CTexture &copy)
  : CSurface(copy)
{
    for (unsigned int i = 0; i < copy.get_num_mipmaps(); i++)
        m_mipmaps.push_back(copy.get_mipmap(i));
}

///////////////////////////////////////////////////////////////////////////////
// assignment operator
CTexture &CTexture::operator= (const CTexture &rhs)
{
    if (this != &rhs)
    {
        CSurface::operator = (rhs);

        m_mipmaps.clear();
        for (unsigned int i = 0; i < rhs.get_num_mipmaps(); i++)
            m_mipmaps.push_back(rhs.get_mipmap(i));
    }

    return *this;
}

void CTexture::create(unsigned int w, unsigned int h, unsigned int d, unsigned int imgsize, const unsigned char *pixels)
{
    CSurface::create(w, h, d, imgsize, pixels);

    m_mipmaps.clear();
}

void CTexture::clear()
{
    CSurface::clear();

    m_mipmaps.clear();
}

///////////////////////////////////////////////////////////////////////////////
// CSurface implementation
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// default constructor
CSurface::CSurface()
  : m_width(0),
    m_height(0),
    m_depth(0),
    m_size(0),
    m_pixels(NULL)
{
}

///////////////////////////////////////////////////////////////////////////////
// creates an empty image
CSurface::CSurface(unsigned int w, unsigned int h, unsigned int d, unsigned int imgsize, const unsigned char *pixels)
  : m_width(0),
    m_height(0),
    m_depth(0),
    m_size(0),
    m_pixels(NULL)
{
    create(w, h, d, imgsize, pixels);
}

///////////////////////////////////////////////////////////////////////////////
// copy constructor
CSurface::CSurface(const CSurface &copy)
  : m_width(0),
    m_height(0),
    m_depth(0),
    m_size(0),
    m_pixels(NULL)
{
    if (copy.get_size() != 0)
    {
        m_size = copy.get_size();
        m_width = copy.get_width();
        m_height = copy.get_height();
        m_depth = copy.get_depth();

        m_pixels = new unsigned char[m_size];
        memcpy(m_pixels, copy, m_size);
    }
}

///////////////////////////////////////////////////////////////////////////////
// assignment operator
CSurface &CSurface::operator= (const CSurface &rhs)
{
    if (this != &rhs)
    {
        clear();

        if (rhs.get_size())
        {
            m_size = rhs.get_size();
            m_width = rhs.get_width();
            m_height = rhs.get_height();
            m_depth = rhs.get_depth();

            m_pixels = new unsigned char[m_size];
            memcpy(m_pixels, rhs, m_size);
        }
    }

    return *this;
}

///////////////////////////////////////////////////////////////////////////////
// clean up image memory
CSurface::~CSurface()
{
    clear();
}

///////////////////////////////////////////////////////////////////////////////
// returns a pointer to image
CSurface::operator unsigned char*() const
{ 
    return m_pixels; 
}

///////////////////////////////////////////////////////////////////////////////
// creates an empty image
void CSurface::create(unsigned int w, unsigned int h, unsigned int d, unsigned int imgsize, const unsigned char *pixels)
{
    assert(w != 0);
    assert(h != 0);
    assert(d != 0);
    assert(imgsize != 0);
    assert(pixels);

    clear();

    m_width = w;
    m_height = h;
    m_depth = d;
    m_size = imgsize;
    m_pixels = new unsigned char[imgsize];
    memcpy(m_pixels, pixels, imgsize);
}

///////////////////////////////////////////////////////////////////////////////
// free surface memory
void CSurface::clear()
{
    if (m_pixels != NULL)
    {
        delete [] m_pixels;
        m_pixels = NULL;
    }
}
