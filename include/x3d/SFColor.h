/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	SFColor.h
*
******************************************************************/

#ifndef _CX3D_SFCOLOR_H_
#define _CX3D_SFCOLOR_H_

#include <iostream>
#include <stdio.h>
#include <x3d/Field.h>

namespace CyberX3D {

class SFColor : public Field {

	static	int	mInit;

	float mValue[3]; 

public:

	SFColor();
	SFColor(float r, float g, float b);
	SFColor(float value[]);
	SFColor(SFColor *color);

	void InitializeJavaIDs();

	////////////////////////////////////////////////
	//	get value
	////////////////////////////////////////////////

	void getValue(float value[]) const;
	const float *getValue() const;
	float getRed() const;
	float getGreen() const;
	float getBlue() const;

	int getValueCount() const 
	{
		return 3;
	}

	////////////////////////////////////////////////
	//	set value
	////////////////////////////////////////////////

	void setValue(float r, float g, float b);
	void setValue(float value[]);
	void setValue(SFColor *color);

	////////////////////////////////////////////////
	//	add value
	////////////////////////////////////////////////

	void add(float x, float y, float z);
	void add(const float value[]);
	void add(SFColor value);

	////////////////////////////////////////////////
	//	sub value
	////////////////////////////////////////////////

	void sub(float x, float y, float z);
	void sub(const float value[]);
	void sub(SFColor value);

	////////////////////////////////////////////////
	//	Output
	////////////////////////////////////////////////

	friend std::ostream& operator<<(std::ostream &s, const SFColor &vector);
	friend std::ostream& operator<<(std::ostream &s, const SFColor *vector);

	////////////////////////////////////////////////
	//	String
	////////////////////////////////////////////////

	void setValue(const char *value);
	const char *getValue(char *buffer, int bufferLen) const;

	////////////////////////////////////////////////
	//	scale
	////////////////////////////////////////////////

	void scale(float scale);

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
	static jmethodID	mGetRedMethodID;
	static jmethodID	mGetGreenMethodID;
	static jmethodID	mGetBlueMethodID;
	static jmethodID	mSetValueMethodID;
	static jmethodID	mSetNameMethodID;

	static jmethodID	mConstInitMethodID;
	static jmethodID	mConstGetRedMethodID;
	static jmethodID	mConstGetGreenMethodID;
	static jmethodID	mConstGetBlueMethodID;
	static jmethodID	mConstSetValueMethodID;
	static jmethodID	mConstSetNameMethodID;

public:

	void		setJavaIDs();

	jclass		getFieldID()				{return mFieldClassID;}
	jclass		getConstFieldID()			{return mConstFieldClassID;}

	jmethodID	getInitMethodID()			{return mInitMethodID;}
	jmethodID	getGetRedMethodID()			{return mGetRedMethodID;}
	jmethodID	getGetGreenMethodID()			{return mGetGreenMethodID;}
	jmethodID	getGetBlueMethodID()			{return mGetBlueMethodID;}
	jmethodID	getSetValueMethodID()		{return mSetValueMethodID;}
	jmethodID	getSetNameMethodID()		{return mSetNameMethodID;}

	jmethodID	getConstInitMethodID()		{return mConstInitMethodID;}
	jmethodID	getConstGetRedMethodID()		{return mConstGetRedMethodID;}
	jmethodID	getConstGetGreenMethodID()		{return mConstGetGreenMethodID;}
	jmethodID	getConstGetBlueMethodID()		{return mConstGetBlueMethodID;}
	jmethodID	getConstSetValueMethodID()	{return mConstSetValueMethodID;}
	jmethodID	getConstSetNameMethodID()	{return mConstSetNameMethodID;}

	jobject toJavaObject(int bConstField = 0);
	void setValue(jobject field, int bConstField = 0);
	void getValue(jobject field, int bConstField = 0);

#endif
};

}

#endif
