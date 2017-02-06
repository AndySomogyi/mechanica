/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	Field.h
*
******************************************************************/

#ifndef _CX3D_FIELD_H_
#define _CX3D_FIELD_H_

#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <x3d/StringUtil.h>
#include <x3d/JavaVM.h>

#ifdef CX3D_SUPPORT_OLDCPP
#include <OldCpp.h>
#endif

#if defined(WIN32)
#define snprintf _snprintf
#endif

namespace CyberX3D {

enum {
fieldTypeNone,
fieldTypeSFBool,
fieldTypeSFFloat,
fieldTypeSFDouble,
fieldTypeSFInt32,
fieldTypeSFVec2f,
fieldTypeSFVec3f,
fieldTypeSFVec2d,
fieldTypeSFVec3d,
fieldTypeSFString,
fieldTypeSFColor,
fieldTypeSFTime,
fieldTypeSFRotation,
fieldTypeSFImage,
fieldTypeSFNode,
fieldTypeSFChar,
fieldTypeMFBool,
fieldTypeMFFloat,
fieldTypeMFDouble,
fieldTypeMFInt32,
fieldTypeMFVec2f,
fieldTypeMFVec3f,
fieldTypeMFVec2d,
fieldTypeMFVec3d,
fieldTypeMFString,
fieldTypeMFColor,
fieldTypeMFTime,
fieldTypeMFRotation,
fieldTypeMFImage,
fieldTypeMFNode,
fieldTypeMFChar,
fieldTypeXMLElement,
fieldTypeMaxNum,
};

class	SFBool;
class	SFFloat;
class	SFInt32;
class	SFVec2f;
class	SFVec3f;
class	SFString;
class	SFColor;
class	SFTime;
class	SFRotation;
//class	SFNode;
class	MFFloat;
class	MFInt32;
class	MFVec2f;
class	MFVec3f;
class	MFString;
class	MFColor;
class	MFTime;
class	MFRotation;
//class	MFNode;

#define	eventInStripString		"set_"
#define eventOutStripString		"_changed"

#define JAVAOBJECT_FIELD		0
#define JAVAOBJECT_CONSTFIELD	1

#define FIELD_MIN_BUFFERSIZE	256

#if defined(CX3D_SUPPORT_JSAI)
class Field : public CJavaVM {
#else
class Field {
#endif

	String	mName;
	int		mType;

public:

	Field() {
		mType = fieldTypeNone;
	}	

	virtual ~Field() {
	}	

	const char *getTypeName() const;

	void setType(int type) {
		mType = type;
	}

	void setType(const char *type);

	int getType() const {
		return mType;
	}

	void setName(const char *name) {
		mName.setValue(name);
	}

	const char *getName() const {
		return mName.getValue();
	}

	friend std::ostream& operator<<(std::ostream &s, const Field &value);
	friend std::ostream& operator<<(std::ostream &s, const Field *value);

	////////////////////////////////////////////////
	//	String
	////////////////////////////////////////////////

	virtual void setValue(const char *value) = 0;
	virtual const char *getValue(char *buffer, int bufferLen = -1) const = 0;
  virtual const char *getValueu(char *buffer, int bufferLen = -1) const;
	virtual int getValueCount() const = 0;

	virtual const char *getValue(String &strBuffer) const {
		char buffer[FIELD_MIN_BUFFERSIZE];
		getValue(buffer, FIELD_MIN_BUFFERSIZE);
		strBuffer.setValue(buffer);
		return strBuffer.getValue();
	}

  virtual const char *getValueu(String &strBuffer) const {
    char buffer[FIELD_MIN_BUFFERSIZE];
    getValueu(buffer, FIELD_MIN_BUFFERSIZE);
    strBuffer.setValue(buffer);
    return strBuffer.getValue();
  }

	////////////////////////////////////////////////
	//	XML String
	////////////////////////////////////////////////

	virtual const char *toXMLString() const {
		static String strBuffer;
		return getValue(strBuffer);
	}

	////////////////////////////////////////////////
	//	Compare
	////////////////////////////////////////////////

	virtual bool equals(Field *field) const {
		return false;
	}

	////////////////////////////////////////////////
	//	isSF*
	////////////////////////////////////////////////

	bool isSFBool() const
	{
		return (fieldTypeSFBool == mType) ? true : false;
	}

	bool isSFFloat() const
	{
		return (fieldTypeSFFloat == mType) ? true : false;
	}

	bool isSFInt32() const
	{
		return (fieldTypeSFInt32 == mType) ? true : false;
	}

	bool isSFVec2f() const
	{
		return (fieldTypeSFVec2f == mType) ? true : false;
	}

	bool isSFVec3f() const
	{
		return (fieldTypeSFVec3f == mType) ? true : false;
	}

	bool isSFString() const
	{
		return (fieldTypeSFString == mType) ? true : false;
	}

	bool isSFColor() const
	{
		return (fieldTypeSFColor == mType) ? true : false;
	}

	bool isSFTime() const
	{
		return (fieldTypeSFTime == mType) ? true : false;
	}

	bool isSFRotatioin() const
	{
		return (fieldTypeSFRotation == mType) ? true : false;
	}

	bool isSFImage() const
	{
		return (fieldTypeSFImage == mType) ? true : false;
	}

	bool isSFNode() const
	{
		return (fieldTypeSFNode == mType) ? true : false;
	}

	////////////////////////////////////////////////
	//	isMF*
	////////////////////////////////////////////////

	bool isMFFloat() const
	{
		return (fieldTypeMFFloat == mType) ? true : false;
	}

	bool isMFInt32() const
	{
		return (fieldTypeMFInt32 == mType) ? true : false;
	}

	bool isMFVec2f() const
	{
		return (fieldTypeMFVec2f == mType) ? true : false;
	}

	bool isMFVec3f() const
	{
		return (fieldTypeMFVec3f == mType) ? true : false;
	}

	bool isMFString() const
	{
		return (fieldTypeMFString == mType) ? true : false;
	}

	bool isMFColor() const
	{
		return (fieldTypeMFColor == mType) ? true : false;
	}

	bool isMFTime() const
	{
		return (fieldTypeMFTime == mType) ? true : false;
	}

	bool isMFRotatioin() const
	{
		return (fieldTypeMFRotation == mType) ? true : false;
	}

	bool isMFNode() const
	{
		return (fieldTypeMFNode == mType) ? true : false;
	}

	bool isXMLElement() const
	{
		return (fieldTypeXMLElement == mType) ? true : false;
	}

	////////////////////////////////////////////////
	//	is(MF|SF)Field
	////////////////////////////////////////////////

	bool isSField() const
	{
		return !isMField();
	}

	bool isMField() const
	{
		if (isMFFloat())
			return true;
		if (isMFInt32())
			return true;
		if (isMFVec2f())
			return true;
		if (isMFVec3f())
			return true;
		if (isMFString())
			return true;
		if (isMFColor())
			return true;
		if (isMFTime())
			return true;
		if (isMFRotatioin())
			return true;
		if (isMFNode())
			return true;
		return false;
	}

	bool isSingleValueMField() const
	{
		if (isMFFloat())
			return true;
		if (isMFInt32())
			return true;
		if (isMFNode())
			return true;
		if (isMFString())
			return true;
		if (isMFTime())
			return true;
		return false;
	}

	////////////////////////////////////////////////
	//	Java
	////////////////////////////////////////////////

#if defined(CX3D_SUPPORT_JSAI)
	virtual jobject toJavaObject(int bConstField = 0) {
		assert(0);
		return NULL;
	};
	virtual void setValue(jobject field, int bConstField = 0) {
		assert(0);
	};
	virtual void getValue(jobject field, int bConstField = 0) {
		assert(0);
	};
#endif
};

}

#endif
