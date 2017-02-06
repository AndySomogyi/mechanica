/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	MFInt32.h
*
******************************************************************/

#ifndef _CX3D_MFINT32_H_
#define _CX3D_MFINT32_H_

#include <x3d/MField.h>
#include <x3d/SFInt32.h>

namespace CyberX3D {

class MFInt32 : public MField {

	static	int	mInit;

public:

	MFInt32();

	void InitializeJavaIDs();

	void addValue(int value);
	void addValue(SFInt32 *sfvalue);
	void addValue(const char *value);

	void insertValue(int index, int value);

	int get1Value(int index) const;
	void set1Value(int index, int value);

	void setValue(MField *mfield);
	void setValue(MFInt32 *values);
	void setValue(int size, int values[]);

	int getValueCount() const 
	{
		return 1;
	}

	////////////////////////////////////////////////
	//	Output
	////////////////////////////////////////////////

	void outputContext(std::ostream& printStream, const char *indentString) const;

	////////////////////////////////////////////////
	//	Java
	////////////////////////////////////////////////

#if defined(CX3D_SUPPORT_JSAI)

private:

	static jclass		mFieldClassID;
	static jclass		mConstFieldClassID;

	static jmethodID	mInitMethodID;
	static jmethodID	mGetSizeMethodID;
	static jmethodID	mClearMethodID;
	static jmethodID	mDeleteMethodID;
	static jmethodID	mAddValueMethodID;
	static jmethodID	mInsertValueMethodID;
	static jmethodID	mSet1ValueMethodID;
	static jmethodID	mGet1ValueMethodID;
	static jmethodID	mSetNameMethodID;

	static jmethodID	mConstInitMethodID;
	static jmethodID	mConstGetSizeMethodID;
	static jmethodID	mConstClearMethodID;
	static jmethodID	mConstDeleteMethodID;
	static jmethodID	mConstAddValueMethodID;
	static jmethodID	mConstInsertValueMethodID;
	static jmethodID	mConstSet1ValueMethodID;
	static jmethodID	mConstGet1ValueMethodID;
	static jmethodID	mConstSetNameMethodID;

public:

	void		setJavaIDs();

	jclass		getFieldID()				{return mFieldClassID;}
	jclass		getConstFieldID()			{return mConstFieldClassID;}

	jmethodID	getInitMethodID()			{return mInitMethodID;}
	jmethodID	getGetSizeMethodID()		{return mGetSizeMethodID;}
	jmethodID	getClearMethodID()			{return mClearMethodID;}
	jmethodID	getDeleteMethodID()			{return mDeleteMethodID;}
	jmethodID	getAddValueMethodID()		{return mAddValueMethodID;}
	jmethodID	getInsertValueMethodID()	{return mInsertValueMethodID;}
	jmethodID	getSet1ValueMethodID()		{return mSet1ValueMethodID;}
	jmethodID	getGet1ValueMethodID()		{return mGet1ValueMethodID;}
	jmethodID	getSetNameMethodID()		{return mSetNameMethodID;}

	jmethodID	getConstInitMethodID()			{return mConstInitMethodID;}
	jmethodID	getConstGetSizeMethodID()		{return mConstGetSizeMethodID;}
	jmethodID	getConstClearMethodID()			{return mConstClearMethodID;}
	jmethodID	getConstDeleteMethodID()		{return mConstDeleteMethodID;}
	jmethodID	getConstAddValueMethodID()		{return mConstAddValueMethodID;}
	jmethodID	getConstInsertValueMethodID()	{return mConstInsertValueMethodID;}
	jmethodID	getConstSet1ValueMethodID()		{return mConstSet1ValueMethodID;}
	jmethodID	getConstGet1ValueMethodID()		{return mConstGet1ValueMethodID;}
	jmethodID	getConstSetNameMethodID()		{return mConstSetNameMethodID;}

	jobject toJavaObject(int bConstField = 0);
	void setValue(jobject field, int bConstField = 0);
	void getValue(jobject field, int bConstField = 0);

#endif
};

}

#endif
