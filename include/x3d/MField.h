/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	MFiled.h
*
******************************************************************/

#ifndef _CX3D_MFIELD_H_
#define _CX3D_MFIELD_H_

#include <iostream>

#include <x3d/Field.h>
#include <x3d/Vector.h>

namespace CyberX3D {

class MField : public Field {

	Vector<Field>	mFieldVector;

public:
	
	MField();
	virtual ~MField();

	int getSize() const;

	int size() const;

	void add(Field *object);

	void insert(Field *object, int index);
	void insert(int index, Field *object);
	void setObject(int index, Field *object);
	void replace(int index, Field *object);

	void clear();

	void remove(int index);

	void removeLastObject();

	void removeFirstObject();

	Field *lastObject() const;
	Field *firstObject() const;
	Field *getObject(int index) const;

	void copy(MField *srcMField);

	virtual void addValue(const char *value) = 0;
	void setValue(const char *buffer);
	const char *getValue(char *buffer, int bufferLen) const;
	const char *getValue(String &buffer) const;

	virtual int getValueCount() const = 0;
	virtual void setValue(MField *mfield) = 0;
	virtual void outputContext(std::ostream& printStream, const char *indentString) const = 0;

	void outputContext(std::ostream& printStream, const char *indentString1, const char *indentString2) const;

	////////////////////////////////////////////////
	//	Java
	////////////////////////////////////////////////

#if defined(CX3D_SUPPORT_JSAI)

	virtual jobject toJavaObject(int bConstField = 0) = 0;
	virtual void setValue(jobject field, int bConstField = 0) = 0;
	virtual void getValue(jobject field, int bConstField = 0) = 0;

#endif
};

}

#endif
