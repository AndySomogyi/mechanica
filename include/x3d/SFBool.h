/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	SFBool.h
*
******************************************************************/

#ifndef _CX3D_SFBOOL_H_
#define _CX3D_SFBOOL_H_

#include <iostream>
#include <stdio.h>
#include <x3d/Field.h>

namespace CyberX3D {

class SFBool : public Field {

	static	int	mInit;

	bool mValue; 

public:

	SFBool();
	SFBool(bool value);
	SFBool(int value);
	SFBool(const char *value);
	SFBool(SFBool *value);

	void InitializeJavaIDs();

	void setValue(bool value);
	void setValue(int value);
	void setValue(SFBool *value);
	bool getValue() const;

	////////////////////////////////////////////////
	//	Output
	////////////////////////////////////////////////

	friend std::ostream& operator<<(std::ostream &s, const SFBool &value);
	friend std::ostream& operator<<(std::ostream &s, const SFBool *value);

	////////////////////////////////////////////////
	//	String
	////////////////////////////////////////////////

	void setValue(const char *value);
	const char *getValue(char *buffer, int bufferLen) const;

	int getValueCount() const 
	{
		return 1;
	}

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
