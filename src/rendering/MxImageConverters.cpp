/*
    This file is part of Magnum.

    Copyright © 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019
              Vladimír Vondruš <mosra@centrum.cz>

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
*/

#include <Magnum/Trade/Trade.h>

#include <csetjmp>
#include <Corrade/Containers/Array.h>
#include <Corrade/Utility/ConfigurationGroup.h>
#include <Magnum/ImageView.h>
#include <Magnum/PixelFormat.h>
#include <Corrade/Utility/Debug.h>
#include <rendering/MxImageConverters.h>
#include <MagnumPlugins/TgaImageConverter/TgaImageConverter.h>
#include <string.h>
#include <tuple>
#include <carbon.h>

/* On Windows we need to circumvent conflicting definition of INT32 in
   <windows.h> (included from OpenGL headers). Problem with libjpeg-tubo only,
   libjpeg solves that already somehow. */
#ifdef CORRADE_TARGET_WINDOWS
#define XMD_H
#endif
#include <jpeglib.h>

using namespace Magnum;
using namespace Magnum::Trade;
using namespace Corrade;
using namespace Corrade::Utility;

// Converts RGBA to RGB (removing the alpha values) to prepare to send data to
// libjpeg. This converts one row of data in rgba with the given width in
// pixels the the given rgb destination buffer (which should have enough space
// reserved for the final data).
static void stripAlpha(const unsigned char* rgba, int pixel_width, unsigned char* rgb)
{
  for (int x = 0; x < pixel_width; x++) {
    const unsigned char* pixel_in = &rgba[x * 4];
    unsigned char* pixel_out = &rgb[x * 3];
    pixel_out[0] = pixel_in[0];
    pixel_out[1] = pixel_in[1];
    pixel_out[2] = pixel_in[2];
  }
}


Corrade::Containers::Array<char> convertImageDataToJpeg(const Magnum::ImageView2D& image, int jpegQuality) {
    static_assert(BITS_IN_JSAMPLE == 8, "Only 8-bit JPEG is supported");

    int components;
    J_COLOR_SPACE colorSpace;

    switch(image.format()) {
        case PixelFormat::R8Unorm:
            components = 1;
            colorSpace = JCS_GRAYSCALE;
            break;
        case PixelFormat::RGB8Unorm:
            components = 3;
            colorSpace = JCS_RGB;
            break;
        case PixelFormat::RGBA8Unorm:
            components = 3;
            colorSpace = JCS_RGB;
            Log(LOG_DEBUG) << "convertImageDataToJpeg ignoring alpha channel";
            break;
        default:
            Error() << "Trade::JpegImageConverter::exportToData(): unsupported pixel format" << image.format();
            return nullptr;
    }

    /* Initialize structures. Needs to be before the setjmp crap in order to
       avoid leaks on error. */
    jpeg_compress_struct info;

    struct DestinationManager {
        jpeg_destination_mgr jpegDestinationManager;
        std::string output;
    } destinationManager;

    Containers::Array<JSAMPROW> rows;
    Containers::Array<char> data;

    /* Fugly error handling stuff */
    /** @todo Get rid of this crap */
    struct ErrorManager {
        jpeg_error_mgr jpegErrorManager;
        std::jmp_buf setjmpBuffer;
        char message[JMSG_LENGTH_MAX]{};
    } errorManager;

    info.err = jpeg_std_error(&errorManager.jpegErrorManager);
    errorManager.jpegErrorManager.error_exit = [](j_common_ptr info) {
        auto& errorManager = *reinterpret_cast<ErrorManager*>(info->err);
        info->err->format_message(info, errorManager.message);
        std::longjmp(errorManager.setjmpBuffer, 1);
    };
    if(setjmp(errorManager.setjmpBuffer)) {
        Error{} << "convertImageDataToJpeg: error:" << errorManager.message;
        jpeg_destroy_compress(&info);
        return nullptr;
    }

    /* Create the compression structure */
    jpeg_create_compress(&info);
    info.dest = reinterpret_cast<jpeg_destination_mgr*>(&destinationManager);
    info.dest->init_destination = [](j_compress_ptr info) {
        auto& destinationManager = *reinterpret_cast<DestinationManager*>(info->dest);
        destinationManager.output.resize(1); /* It crashes if the buffer has zero free space */
        info->dest->next_output_byte = reinterpret_cast<JSAMPLE*>(&destinationManager.output[0]);
        info->dest->free_in_buffer = destinationManager.output.size()/sizeof(JSAMPLE);
    };
    info.dest->term_destination = [](j_compress_ptr info) {
        auto& destinationManager = *reinterpret_cast<DestinationManager*>(info->dest);
        destinationManager.output.resize(destinationManager.output.size() - info->dest->free_in_buffer);
    };
    info.dest->empty_output_buffer = [](j_compress_ptr info) -> boolean {
        auto& destinationManager = *reinterpret_cast<DestinationManager*>(info->dest);
        const std::size_t oldSize = destinationManager.output.size();
        destinationManager.output.resize(oldSize*2); /* Double capacity each time it is exceeded */
        info->dest->next_output_byte = reinterpret_cast<JSAMPLE*>(&destinationManager.output[0] + oldSize);
        info->dest->free_in_buffer = (destinationManager.output.size() - oldSize)/sizeof(JSAMPLE);
        return boolean(true);
    };

    /* Fill the info structure */
    info.image_width = image.size().x();
    info.image_height = image.size().y();
    info.input_components = components;
    info.in_color_space = colorSpace;

    jpeg_set_defaults(&info);
    jpeg_set_quality(&info, jpegQuality, boolean(true));
    jpeg_start_compress(&info, boolean(true));

    /* Data properties */
    Math::Vector2<std::size_t> offset, dataSize;
    std::tie(offset, dataSize) = image.dataProperties();

    // check if need to strip alpha
    if(image.format() == PixelFormat::RGBA8Unorm) {

        unsigned char *rgb = (unsigned char*)alloca(3 * sizeof(unsigned char) * image.size().x());

        while(info.next_scanline < info.image_height) {
            const unsigned char* rgba = (unsigned char*)(image.data() + offset.sum() + (image.size().y() - info.next_scanline - 1)*dataSize.x());
            stripAlpha(rgba, image.size().x(), rgb);
            jpeg_write_scanlines(&info, &rgb, 1);
        }
    } else {
        while(info.next_scanline < info.image_height) {
            /* libJPEG HAVE YOU EVER HEARD ABOUT CONST ARGUMENTS?! IT'S NOT 1978
               ANYMORE */
            JSAMPROW row = reinterpret_cast<JSAMPROW>(const_cast<char*>(image.data() + offset.sum() + (image.size().y() - info.next_scanline - 1)*dataSize.x()));
            jpeg_write_scanlines(&info, &row, 1);
        }
    }

    jpeg_finish_compress(&info);
    jpeg_destroy_compress(&info);

    /* Copy the string into the output array (I would kill for having std::string::release()) */
    Containers::Array<char> fileData{destinationManager.output.size()};
    std::copy(destinationManager.output.begin(), destinationManager.output.end(), fileData.data());
    return fileData;
}


Corrade::Containers::Array<char> convertImageDataToTGA(const Magnum::ImageView2D& image) {
    
    TgaImageConverter conv;
    
    return conv.exportToData(image);
}
