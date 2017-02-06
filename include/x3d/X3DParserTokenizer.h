/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	X3DParserTokenizer.h
*
******************************************************************/

#include <x3d/CyberX3DConfig.h>

#ifdef CX3D_SUPPORT_X3D

#ifndef _CX3D_X3DPARSERTOKENIZER_H_
#define _CX3D_X3DPARSERTOKENIZER_H_

#include <xercesc/util/XMLStringTokenizer.hpp>

namespace CyberX3D {

class X3DParserTokenizer : public xercesc::XMLStringTokenizer {

public:

	X3DParserTokenizer(const XMLCh *const srcStr);
	virtual ~X3DParserTokenizer();

};

}

#endif

#endif
