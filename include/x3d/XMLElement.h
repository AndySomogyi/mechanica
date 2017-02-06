/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	XMLElement.h
*
******************************************************************/

#ifndef _CX3D_XMLELEMENT_H_
#define _CX3D_XMLELEMENT_H_

#include <iostream>
#include <x3d/Field.h>
#include <x3d/StringUtil.h>

namespace CyberX3D {

class XMLElement : public Field {

	static	int	mInit;

	String	mValue;

public:

	XMLElement();
	XMLElement(const char *value);
	XMLElement(XMLElement *value);

	virtual ~XMLElement();

	void setValue(const char *value);
	void setValue(XMLElement *value);
	const char *getValue() const;

	int getValueCount() const
	{
		return 1;
	}

	////////////////////////////////////////////////
	//	Output
	////////////////////////////////////////////////

	friend std::ostream& operator<<(std::ostream &s, const XMLElement &string);
	friend std::ostream& operator<<(std::ostream &s, const XMLElement *string);

	////////////////////////////////////////////////
	//	String
	////////////////////////////////////////////////

	const char *getValue(char *buffer, int bufferLen) const;
	const char *getValue(String &buffer) const;

	////////////////////////////////////////////////
	//	Compare
	////////////////////////////////////////////////

	bool equals(Field *field) const;

};

}

#endif
