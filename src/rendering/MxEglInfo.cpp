/*
 * eglinfo - like glxinfo but for EGL
 *
 * Brian Paul
 * 11 March 2005
 *
 * Copyright (C) 2005  Brian Paul   All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * BRIAN PAUL BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
 * AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "MxGlInfo.h"
#include "CConvert.hpp"

#include <sstream>
#include <iomanip>

#ifdef MX_LINUX

#define EGL_EGLEXT_PROTOTYPES

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_CONFIGS 1000
#define MAX_MODES 1000
#define MAX_SCREENS 10

/* These are X visual types, so if you're running eglinfo under
 * something not X, they probably don't make sense. */
static const char *vnames[] = { "SG", "GS", "SC", "PC", "TC", "DC" };

/**
 * Print table of all available configurations.
 */
static void PrintConfigs(std::stringstream& ss, EGLDisplay d)
{
   EGLConfig configs[MAX_CONFIGS];
   EGLint numConfigs, i;

   eglGetConfigs(d, configs, MAX_CONFIGS, &numConfigs);

   ss << "Configurations:\n";
   ss << "     bf lv colorbuffer dp st  ms    vis   cav bi  renderable  supported\n";
   ss << "  id sz  l  r  g  b  a th cl ns b    id   eat nd gl es es2 vg surfaces \n";
   ss << "---------------------------------------------------------------------\n";
   for (i = 0; i < numConfigs; i++) {
      EGLint id, size, level;
      EGLint red, green, blue, alpha;
      EGLint depth, stencil;
      EGLint renderable, surfaces;
      EGLint vid, vtype, caveat, bindRgb, bindRgba;
      EGLint samples, sampleBuffers;
      char surfString[100] = "";

      eglGetConfigAttrib(d, configs[i], EGL_CONFIG_ID, &id);
      eglGetConfigAttrib(d, configs[i], EGL_BUFFER_SIZE, &size);
      eglGetConfigAttrib(d, configs[i], EGL_LEVEL, &level);

      eglGetConfigAttrib(d, configs[i], EGL_RED_SIZE, &red);
      eglGetConfigAttrib(d, configs[i], EGL_GREEN_SIZE, &green);
      eglGetConfigAttrib(d, configs[i], EGL_BLUE_SIZE, &blue);
      eglGetConfigAttrib(d, configs[i], EGL_ALPHA_SIZE, &alpha);
      eglGetConfigAttrib(d, configs[i], EGL_DEPTH_SIZE, &depth);
      eglGetConfigAttrib(d, configs[i], EGL_STENCIL_SIZE, &stencil);
      eglGetConfigAttrib(d, configs[i], EGL_NATIVE_VISUAL_ID, &vid);
      eglGetConfigAttrib(d, configs[i], EGL_NATIVE_VISUAL_TYPE, &vtype);

      eglGetConfigAttrib(d, configs[i], EGL_CONFIG_CAVEAT, &caveat);
      eglGetConfigAttrib(d, configs[i], EGL_BIND_TO_TEXTURE_RGB, &bindRgb);
      eglGetConfigAttrib(d, configs[i], EGL_BIND_TO_TEXTURE_RGBA, &bindRgba);
      eglGetConfigAttrib(d, configs[i], EGL_RENDERABLE_TYPE, &renderable);
      eglGetConfigAttrib(d, configs[i], EGL_SURFACE_TYPE, &surfaces);

      eglGetConfigAttrib(d, configs[i], EGL_SAMPLES, &samples);
      eglGetConfigAttrib(d, configs[i], EGL_SAMPLE_BUFFERS, &sampleBuffers);

      if (surfaces & EGL_WINDOW_BIT)
         strcat(surfString, "win,");
      if (surfaces & EGL_PBUFFER_BIT)
         strcat(surfString, "pb,");
      if (surfaces & EGL_PIXMAP_BIT)
         strcat(surfString, "pix,");
      if (surfaces & EGL_STREAM_BIT_KHR)
         strcat(surfString, "str,");
      if (strlen(surfString) > 0)
         surfString[strlen(surfString) - 1] = 0;

      //             1   2   3   4   5   6   7   8   9  10 11     1213
      //printf("0x%02x %2d %2d %2d %2d %2d %2d %2d %2d %2d%2d 0x%02x%s ",
      //       id, size, level,
      //       red, green, blue, alpha,
      //       depth, stencil,
      //       samples, sampleBuffers, vid, vtype < 6 ? vnames[vtype] : "--");
      //printf("  %c  %c  %c  %c  %c   %c %s\n",
      //       (caveat != EGL_NONE) ? 'y' : ' ',
      //       (bindRgba) ? 'a' : (bindRgb) ? 'y' : ' ',
      //       (renderable & EGL_OPENGL_BIT) ? 'y' : ' ',
      //       (renderable & EGL_OPENGL_ES_BIT) ? 'y' : ' ',
      //       (renderable & EGL_OPENGL_ES2_BIT) ? 'y' : ' ',
      //       (renderable & EGL_OPENVG_BIT) ? 'y' : ' ',
      //       surfString);


      ss << "0x" << std::setfill('0') << std::setw(2) << std::right << std::hex << id << " " // 1
	 << std::setfill(' ') << std::dec
	 << std::setw(2) << size    << ' '           // 2
	 << std::setw(2) << level   << ' '           // 3
	 << std::setw(2) << red     << ' '           // 4
	 << std::setw(2) << green   << ' '           // 5
	 << std::setw(2) << blue    << ' '           // 6
	 << std::setw(2) << alpha   << ' '           // 7
	 << std::setw(2) << depth   << ' '           // 8
	 << std::setw(2) << stencil << ' '           // 9
	 << std::setw(2) << samples                  // 10
	 << std::setw(2) << sampleBuffers << ' '     // 11
	 << "0x" << std::setfill('0') << std::setw(2) << std::right << std::hex << vid  // 12
	 << (vtype < 6 ? vnames[vtype] : "--");
      
      //printf("  %c  %c  %c  %c  %c   %c %s\n",
      ss << ' ' << (char)((caveat != EGL_NONE) ? 'y' : ' ')
	 << ' ' << (char)((bindRgba) ? 'a' : (bindRgb) ? 'y' : ' ')
	 << ' ' << (char)((renderable & EGL_OPENGL_BIT) ? 'y' : ' ')
	 << ' ' << (char)((renderable & EGL_OPENGL_ES_BIT) ? 'y' : ' ')
	 << ' ' << (char)((renderable & EGL_OPENGL_ES2_BIT) ? 'y' : ' ')
	 << ' ' << (char)((renderable & EGL_OPENVG_BIT) ? 'y' : ' ')
	 << ' ' << (const char*)surfString
	 << std::endl;
   }
}


static void PrintExtensions(std::stringstream& ss, EGLDisplay d)
{
   const char *extensions, *p, *end, *next;
   int column;

   ss << (d == EGL_NO_DISPLAY ? "EGL client extensions string:" :
	                        "EGL extensions string:");

   extensions = eglQueryString(d, EGL_EXTENSIONS);
   if (!extensions)
      return;

   column = 0;
   end = extensions + strlen(extensions);

   for (p = extensions; p < end; p = next + 1) {
      next = strchr(p, ' ');
      if (next == NULL)
         next = end;

      if (column > 0 && column + next - p + 1 > 70) {
	 ss << "\n";
	 column = 0;
      }
      if (column == 0)
	 ss << "    ";
      else
	 ss << " ";
      column += next - p + 1;

      //printf("%.*s", (int) (next - p), p);
      ss << std::string(p, (int)(next - p));

      p = next + 1;
   }

   if (column > 0)
     ss << "\n";
}

static int doOneDisplay(std::stringstream &ss, EGLDisplay d, const char *name)
{
   int maj, min;

   //printf("%s:\n", name);
   ss << name << "\n";
   if (!eglInitialize(d, &maj, &min)) {
      ss << "eglinfo: eglInitialize failed\n";
      return 1;
   }

   ss << "EGL API version: " << maj << "." << min << "\n";
   ss << "EGL vendor string: " << eglQueryString(d, EGL_VENDOR) << "\n";
   ss << "EGL version string: " <<  eglQueryString(d, EGL_VERSION) << "\n";
#ifdef EGL_VERSION_1_2
   ss << "EGL client APIs: " << eglQueryString(d, EGL_CLIENT_APIS) << "\n";
#endif

   PrintExtensions(ss, d);

   PrintConfigs(ss, d);
   ss << "\n";
   return 0;
}

std::string print_eglinfo()
{
   std::stringstream ss;
   int ret;

   PrintExtensions(ss, EGL_NO_DISPLAY);
   ss << "\n";

   ret = doOneDisplay(ss, eglGetDisplay(EGL_DEFAULT_DISPLAY), "Default display");

   return ss.str();
}

#else

#endif

PyObject *Mx_EglInfo(PyObject *args, PyObject *kwds) {

#ifdef MX_LINUX
  std::string info = print_eglinfo();
#else
  
  std::string info = "Not a Linux system, no EGL";
#endif
  
  PyObject *result = carbon::cast(info);

  return result;
}


