/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	PROTO.h
*
******************************************************************/

#ifndef _CX3D_PROTO_H_
#define _CX3D_PROTO_H_

#include <x3d/LinkedList.h>
#include <x3d/StringUtil.h>
#include <x3d/Field.h>
#include <x3d/Vector.h>

namespace CyberX3D {

class PROTO : public LinkedListNode<PROTO> {
	String				mName;
	String				mString;
	Vector<Field>		mDefaultFieldVector;
	Vector<Field>		mFieldVector;
public:

	PROTO(const char *name, const char *string, const char *fieldString);
	virtual ~PROTO(void);

	void		setName(const char *name);
	const char		*getName(void) const;

	void		setString(const char *string);
	const char		*getString() const;
	void		getString(String &returnString) const;

	void		addDefaultField(Field *field);
	void		addField(Field *field);

	int			getNDefaultFields() const;
	int			getNFields() const;

	Field		*getDefaultField(int n) const;
	Field		*getField(int n) const;

	void		addFieldValues(const char *fieldString, int bDefaultField);
	void		addDefaultFields(const char *fieldString);
	void		addFields(const char *fieldString);
	void		deleteDefaultFields(void);
	void		deleteFields(void);

	Field		*getField(const char *name) const;
	int			getFieldType(const char *name) const;
};

}

#endif


