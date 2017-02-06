/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	FileJPEG.cpp
*
******************************************************************/

#include <x3d/FileJPEG.h>

#ifdef CX3D_SUPPORT_JPEG

#include <stdio.h>
#include <ctype.h>
extern "C" {
#ifdef WIN32
#include <cdjpeg.h>
#else
#include <jpeglib.h>
#endif
}

using namespace CyberX3D;

#ifdef WIN32
static const char * const cdjpeg_message_table[] = {
#include "cderror.h"
  NULL
};
#endif

METHODDEF(void) ErrorExit(j_common_ptr cinfo)
{
  (*cinfo->err->output_message) (cinfo);
}

FileJPEG::FileJPEG(const char *filename)
{	
	imgBuffer = NULL;
	width = height = 0;
	
	load(filename);
}

bool FileJPEG::load(const char *filename)
{	
	imgBuffer = NULL;
	width = height = 0;

	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;

	/* Initialize the JPEG decompression object with default error handling. */
	jpeg_std_error(&jerr);
	jerr.error_exit = ErrorExit;
	cinfo.err = &jerr;
	jpeg_create_decompress(&cinfo);

	/* Add some application-specific error messages (from cderror.h) */
#ifdef WIN32
	jerr.addon_message_table = cdjpeg_message_table;
	jerr.first_addon_message = JMSG_FIRSTADDONCODE;
	jerr.last_addon_message = JMSG_LASTADDONCODE;
#endif

	FILE *fp = fopen(filename, "rb");
	if (!fp) 
		return false;

	/* Specify data source for decompression */
	jpeg_stdio_src(&cinfo, fp);

	/* Read file header, set default decompression parameters */
	jpeg_read_header(&cinfo, TRUE);

//	if (cinfo.err->msg_code != 0)
//		return false;

	width = cinfo.image_width;
	height = cinfo.image_height;
	imgBuffer = new RGBColor24[width*height];

	/* Start decompressor */
	jpeg_start_decompress(&cinfo);

	/* Process data */
	unsigned char	**buffer = new unsigned char *[1]; 
	int				scanline = 0;

	while (cinfo.output_scanline < cinfo.output_height) {
		buffer[0] = (unsigned char *)imgBuffer[width*scanline];
		jpeg_read_scanlines(&cinfo, buffer, 1);
		scanline++;
	}

	delete [] buffer;

	jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);

	/* Close files, if we opened them */
	if (fp)
		fclose(fp);

	return true;
}

FileJPEG::~FileJPEG()
{
	if (imgBuffer)
		delete []imgBuffer;
}

#endif
