/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	SFVec2f.h
*
******************************************************************/

#ifndef _CX3D_SFVEC2F_H_
#define _CX3D_SFVEC2F_H_

#include <iostream>
#include <math.h>
#include <stdio.h>
#include <x3d/Field.h>

namespace CyberX3D {

class SFVec2f : public Field {

	static	int	mInit;

	float mValue[2]; 

public:

	SFVec2f();
	SFVec2f(float x, float y);
	SFVec2f(const float value[]);
	SFVec2f(SFVec2f *value);

	void InitializeJavaIDs();

	////////////////////////////////////////////////
	//	get value
	////////////////////////////////////////////////

	void getValue(float value[]) const;
	const float *getValue() const;
	float getX() const;
	float getY() const;

	int getValueCount() const 
	{
		return 2;
	}

	////////////////////////////////////////////////
	//	set value
	////////////////////////////////////////////////

	void setValue(float x, float y);
	void setValue(const float value[]);
	void setValue(SFVec2f *vector);
	void setX(float x);
	void setY(float y);

	////////////////////////////////////////////////
	//	add value
	////////////////////////////////////////////////

	void add(float x, float y);
	void add(const float value[]);
	void add(SFVec2f value);
	void translate(float x, float y);
	void translate(const float value[]);
	void translate(SFVec2f value);

	////////////////////////////////////////////////
	//	sub value
	////////////////////////////////////////////////

	void sub(float x, float y);
	void sub(const float value[]);
	void sub(SFVec2f value);

	////////////////////////////////////////////////
	//	scale
	////////////////////////////////////////////////

	void scale(float value);
	void scale(float xscale, float yscale);
	void scale(const float value[2]);

	////////////////////////////////////////////////
	//	invert
	////////////////////////////////////////////////

	void invert();

	////////////////////////////////////////////////
	//	scalar
	////////////////////////////////////////////////

	float getScalar() const;

	////////////////////////////////////////////////
	//	normalize
	////////////////////////////////////////////////

	void normalize();

	////////////////////////////////////////////////
	//	Output
	////////////////////////////////////////////////

	friend std::ostream& operator<<(std::ostream &s, const SFVec2f &vector);
	friend std::ostream& operator<<(std::ostream &s, const SFVec2f *vector);

	////////////////////////////////////////////////
	//	String
	////////////////////////////////////////////////

	void setValue(const char *value);
	const char *getValue(char *buffer, int bufferLen) const;

	////////////////////////////////////////////////
	//	Compare
	////////////////////////////////////////////////

	bool equals(Field *field) const;
	bool equals(const float value[2]) const;
	bool equals(float x, float y) const;

	////////////////////////////////////////////////
	//	Java
	////////////////////////////////////////////////

#if defined(CX3D_SUPPORT_JSAI)

private:

	static jclass		mFieldClassID;
	static jclass		mConstFieldClassID;

	static jmethodID	mInitMethodID;
	static jmethodID	mGetXMethodID;
	static jmethodID	mGetYMethodID;
	static jmethodID	mSetValueMethodID;
	static jmethodID	mSetNameMethodID;

	static jmethodID	mConstInitMethodID;
	static jmethodID	mConstGetXMethodID;
	static jmethodID	mConstGetYMethodID;
	static jmethodID	mConstSetValueMethodID;
	static jmethodID	mConstSetNameMethodID;

public:

	void		setJavaIDs();

	jclass		getFieldID()				{return mFieldClassID;}
	jclass		getConstFieldID()			{return mConstFieldClassID;}

	jmethodID	getInitMethodID()			{return mInitMethodID;}
	jmethodID	getGetXMethodID()			{return mGetXMethodID;}
	jmethodID	getGetYMethodID()			{return mGetYMethodID;}
	jmethodID	getSetValueMethodID()		{return mSetValueMethodID;}
	jmethodID	getSetNameMethodID()		{return mSetNameMethodID;}

	jmethodID	getConstInitMethodID()		{return mConstInitMethodID;}
	jmethodID	getConstGetXMethodID()		{return mConstGetXMethodID;}
	jmethodID	getConstGetYMethodID()		{return mConstGetYMethodID;}
	jmethodID	getConstSetValueMethodID()	{return mConstSetValueMethodID;}
	jmethodID	getConstSetNameMethodID()	{return mConstSetNameMethodID;}

	jobject toJavaObject(int bConstField = 0);
	void setValue(jobject field, int bConstField = 0);
	void getValue(jobject field, int bConstField = 0);

#endif

};

}

#endif
