/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	UrlFile.h
*
******************************************************************/

#ifndef _CX3D_URLFILE_H_
#define _CX3D_URLFILE_H_

#include <x3d/JavaVM.h>
#include <x3d/StringUtil.h>

#if defined(CX3D_SUPPORT_URL)

namespace CyberX3D {

class UrlFile : public JavaVM {

	static jclass		mUrlGetStreamClassID;
	static jmethodID	mUrlGetStreamInitMethodID;
	static jmethodID	mUrlGetStreamGetStreamMethodID;
	static jobject		mUrlGetStreamObject;		

	String				*mUrl;	
	String				*mUrlString;

public:

	UrlFile();
	virtual ~UrlFile();
	void	initialize();
	void	setUrl(char *urlString);
	const char	*getUrl() const;
	bool	getStream(const char *urlString) const;
	char	*getOutputFilename();
	bool	deleteOutputFilename();
};

}

#endif

#endif
