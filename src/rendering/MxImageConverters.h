#ifndef Magnum_Trade_JpegImageConverter_h
#define Magnum_Trade_JpegImageConverter_h
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

/** @file
 * @brief Class @ref Magnum::Trade::JpegImageConverter
 */

#include <Magnum/Magnum.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/Image.h>
#include <Magnum/ImageView.h>

/**
@brief JPEG image converter plugin

Creates JPEG (`*.jpg`, `*.jpe`, `*.jpeg`) files from images with format
@ref PixelFormat::R8Unorm or @ref PixelFormat::RGB8Unorm. Images in
@ref PixelFormat::RGBA8Unorm are supported only if you use libJPEG Turbo
instead of vanilla libJPEG and the alpha channel gets ignored (with a warning
printed to the console). @ref PixelFormat::RG8Unorm can't be easily supported,
see @ref StbImageConverter for an alternative with a possibility to export RG
images as a grayscale JPEG.

This plugin depends on the @ref Trade and [libJPEG](http://libjpeg.sourceforge.net/)
libraries and is built if `WITH_JPEGIMAGECONVERTER` is enabled when building
Magnum Plugins. To use as a dynamic plugin, you need to load the
@cpp "JpegImageConverter" @ce plugin from `MAGNUM_PLUGINS_IMAGECONVERTER_DIR`.
To use as a static plugin or as a dependency of another plugin with CMake, you
need to request the `JpegImageConverter` component of the `MagnumPlugins`
package and link to the `MagnumPlugins::JpegImageConverter` target. See
@ref building-plugins, @ref cmake-plugins and @ref plugins for more
information.

@m_class{m-block m-success}

@thirdparty This plugin makes use of the [libJPEG](http://ijg.org/) library,
    released under a custom @m_class{m-label m-success} **Libjpeg license**
    ([license text](https://jpegclub.org/reference/libjpeg-license/)). It
    requires attribution for public use. Note that this plugin can be built
    against any other compatible and possibly differently-licensed libJPEG
    implementation as well.

@section Trade-JpegImageConverter-implementations libJPEG implementations

While some systems (such as macOS) still ship only with the vanilla libJPEG,
you can get a much better performance and/or quality/size ratios by using other
implementations:

-   [libjpeg-turbo](https://libjpeg-turbo.org/), optimized for compression and
    decompression speed, though not necessarily the best quality/size ratio
-   [MozJPEG](https://github.com/mozilla/mozjpeg), optimized for quality/size
    ratio, though generally much slower than libjpeg-turbo

@section Trade-JpegImageConverter-configuration Plugin-specific config

It's possible to tune various output options through @ref configuration(). See
below for all options and their default values:

@snippet MagnumPlugins/JpegImageConverter/JpegImageConverter.conf config

*/


/**
 * jpegQuality shall construct JPEG quantization tables for the given quality setting.
 * The quality value ranges from 0..100. If "force_baseline" is TRUE, the computed quantization
 * table entries are limited to 1..255 for JPEG baseline compatibility.
 */
Corrade::Containers::Array<char> convertImageDataToJpeg(const Magnum::ImageView2D& image, int jpegQuality = 100);


#endif
