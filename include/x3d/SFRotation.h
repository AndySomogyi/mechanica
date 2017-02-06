/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	SFRotation.h
*
******************************************************************/

#ifndef _CX3D_SFROTATION_H_
#define _CX3D_SFROTATION_H_

#include <iostream>
#include <x3d/SFVec3f.h>
#include <x3d/Field.h>

namespace CyberX3D {

class SFMatrix;

class SFRotation : public Field {

	static	int	mInit;

	SFVec3f		mVector;
	float		mAngle;

public:

	SFRotation();
	SFRotation(float x, float y, float z, float rot);
	SFRotation(float vector[3], float rot);
	SFRotation(float value[]);
	SFRotation(SFRotation *value);
	SFRotation(SFMatrix *matrix);

	void InitializeJavaIDs();

	////////////////////////////////////////////////
	//	set value
	////////////////////////////////////////////////

	void setValue(float x, float y, float z, float rot);
	void setValue(float value[]);
	void setValue(float value[], float angle);
	void setValue(SFRotation *rotation);
	void setValue(SFMatrix *matrix);
	
	////////////////////////////////////////////////
	//	get value
	////////////////////////////////////////////////

	void getValue(float value[]) const;
	void getVector(float vector[]) const;
	float getX() const;
	float getY() const;
	float getZ() const;
	float getAngle() const;

	int getValueCount() const 
	{
		return 4;
	}

	////////////////////////////////////////////////
	//	add 
	////////////////////////////////////////////////

	void add(SFRotation *rot);
	void add(float rotationValue[]);
	void add(float x, float y, float z, float rot);

	////////////////////////////////////////////////
	//	multi 
	////////////////////////////////////////////////

	void multi(double vector[]);
	void multi(float vector[]);
	void multi(float *x, float *y, float *z);
	void multi(SFVec3f *vector);

	////////////////////////////////////////////////
	//	convert
	////////////////////////////////////////////////

	void getSFMatrix(SFMatrix *matrix) const;

	////////////////////////////////////////////////
	//	invert
	////////////////////////////////////////////////

	void invert();

	////////////////////////////////////////////////
	//	Output
	////////////////////////////////////////////////

	friend std::ostream& operator<<(std::ostream &s, const SFRotation &rotation);
	friend std::ostream& operator<<(std::ostream &s, const SFRotation *rotation);

	////////////////////////////////////////////////
	//	String
	////////////////////////////////////////////////

	void setValue(const char *value);
	const char *getValue(char *buffer, int bufferLen) const;

	////////////////////////////////////////////////
	//	initialize
	////////////////////////////////////////////////

	void initValue();

	////////////////////////////////////////////////
	//	Compare
	////////////////////////////////////////////////

	bool equals(Field *field) const;
	bool equals(float value[4]) const;
	bool equals(float x, float y, float z, float angle) const;

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
	static jmethodID	mGetZMethodID;
	static jmethodID	mGetAngleMethodID;
	static jmethodID	mSetValueMethodID;
	static jmethodID	mSetNameMethodID;

	static jmethodID	mConstInitMethodID;
	static jmethodID	mConstGetXMethodID;
	static jmethodID	mConstGetYMethodID;
	static jmethodID	mConstGetZMethodID;
	static jmethodID	mConstGetAngleMethodID;
	static jmethodID	mConstSetValueMethodID;
	static jmethodID	mConstSetNameMethodID;

public:

	void		setJavaIDs();

	jclass		getFieldID()				{return mFieldClassID;}
	jclass		getConstFieldID()			{return mConstFieldClassID;}

	jmethodID	getInitMethodID()			{return mInitMethodID;}
	jmethodID	getGetXMethodID()			{return mGetXMethodID;}
	jmethodID	getGetYMethodID()			{return mGetYMethodID;}
	jmethodID	getGetZMethodID()			{return mGetZMethodID;}
	jmethodID	getGetAngleMethodID()		{return mGetAngleMethodID;}
	jmethodID	getSetValueMethodID()		{return mSetValueMethodID;}
	jmethodID	getSetNameMethodID()		{return mSetNameMethodID;}

	jmethodID	getConstInitMethodID()		{return mConstInitMethodID;}
	jmethodID	getConstGetXMethodID()		{return mConstGetXMethodID;}
	jmethodID	getConstGetYMethodID()		{return mConstGetYMethodID;}
	jmethodID	getConstGetZMethodID()		{return mConstGetZMethodID;}
	jmethodID	getConstGetAngleMethodID()	{return mConstGetAngleMethodID;}
	jmethodID	getConstSetValueMethodID()	{return mConstSetValueMethodID;}
	jmethodID	getConstSetNameMethodID()	{return mConstSetNameMethodID;}

	jobject toJavaObject(int bConstField = 0);
	void setValue(jobject field, int bConstField = 0);
	void getValue(jobject field, int bConstField = 0);

#endif

};

}

#endif
