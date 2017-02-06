/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	SFString.h
*
******************************************************************/

#ifndef _CX3D_SFSTRING_H_
#define _CX3D_SFSTRING_H_

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x3d/Field.h>
#include <x3d/StringUtil.h>

namespace CyberX3D {

class SFString : public Field {

	static	int	mInit;

	String	mValue;

public:

	SFString();
	SFString(const char *value);
	SFString(SFString *value);

	void InitializeJavaIDs();

	virtual ~SFString();

	void setValue(const char *value);
	void setValue(SFString *value);
	const char *getValue() const;

	int getValueCount() const
	{
		return 1;
	}

	////////////////////////////////////////////////
	//	Output
	////////////////////////////////////////////////

	friend std::ostream& operator<<(std::ostream &s, const SFString &string);
	friend std::ostream& operator<<(std::ostream &s, const SFString *string);

	////////////////////////////////////////////////
	//	String
	////////////////////////////////////////////////

	const char *getValue(char *buffer, int bufferLen) const;
	const char *getValue(String &buffer) const;

	////////////////////////////////////////////////
	//	Compare
	////////////////////////////////////////////////

	bool equals(Field *field) const;

	////////////////////////////////////////////////
	//	Java
	////////////////////////////////////////////////

#if defined(CX3D_SUPPORT_JSAI)

private:

	static jclass		mFieldClassID;
	static jclass		mConstFieldClassID;

	static jmethodID	mInitMethodID;
	static jmethodID	mSetValueMethodID;
	static jmethodID	mGetValueMethodID;
	static jmethodID	mSetNameMethodID;

	static jmethodID	mConstInitMethodID;
	static jmethodID	mConstSetValueMethodID;
	static jmethodID	mConstGetValueMethodID;
	static jmethodID	mConstSetNameMethodID;

public:

	void		setJavaIDs();

	jclass		getFieldID()				{return mFieldClassID;}
	jclass		getConstFieldID()			{return mConstFieldClassID;}

	jmethodID	getInitMethodID()			{return mInitMethodID;}
	jmethodID	getSetValueMethodID()		{return mSetValueMethodID;}
	jmethodID	getGetValueMethodID()		{return mGetValueMethodID;}
	jmethodID	getSetNameMethodID()		{return mSetNameMethodID;}

	jmethodID	getConstInitMethodID()		{return mConstInitMethodID;}
	jmethodID	getConstSetValueMethodID()	{return mConstSetValueMethodID;}
	jmethodID	getConstGetValueMethodID()	{return mConstGetValueMethodID;}
	jmethodID	getConstSetNameMethodID()	{return mConstSetNameMethodID;}

	jobject toJavaObject(int bConstField = 0);
	void setValue(jobject field, int bConstField = 0);
	void getValue(jobject field, int bConstField = 0);

#endif

};

}

#endif
