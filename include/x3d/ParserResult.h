/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	ParserResult.h
*
******************************************************************/

#ifndef _CX3D_PARSERRESULT_H_
#define _CX3D_PARSERRESULT_H_

#include <x3d/StringUtil.h>

namespace CyberX3D {

class ParserResult {

	int						mErrorLineNumber;
	String					mErrorMsg;
	String					mErrorToken;
	String					mErrorLineString;
	bool					mIsOK;

public:

	ParserResult()
	{
	}

	virtual ~ParserResult()
	{
	}

	void init()
	{
		setParserResult(false);
		setErrorLineNumber(0);
		setErrorMessage("");
		setErrorToken("");
		setErrorLineString(""); 
	}

	void setParserResult(bool bOK) { 
		mIsOK = bOK; 
	}
	bool getParserResult() const { 
		return mIsOK; 
	}
	bool isOK(void) const {
		return getParserResult(); 
	}

	void setErrorLineNumber(int n) { 
		mErrorLineNumber = n; 
	}
	int	getErrorLineNumber(void) const{
		return mErrorLineNumber; 
	}

	void setErrorMessage(const char *msg) {
		mErrorMsg.setValue(msg); 
	}
	const char *getErrorMessage(void) const { 
		return mErrorMsg.getValue(); 
	}

	void setErrorToken(const char *error) {
		mErrorToken.setValue(error); 
	}
	const char *getErrorToken(void) const { 
		return mErrorToken.getValue(); 
	}

	void setErrorLineString(const char *error) { 
		mErrorLineString.setValue(error); 
	}
	const char *getErrorLineString(void) const {
		return mErrorLineString.getValue(); 
	}
};

}

#endif


