/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	XMLElement.cpp
*	03/23/04
*	- Thanks for Joerg Scheurich aka MUFTI <rusmufti@helpdesk.rus.uni-stuttgart.de>
*	- Added to include stdio.h for IRIX platform.
*
******************************************************************/

#include <stdio.h>
#include <x3d/XMLElement.h>

using namespace CyberX3D;

XMLElement::XMLElement() 
{
	setType(fieldTypeXMLElement);
	setValue((char *)NULL);
}

XMLElement::XMLElement(const char *value) 
{
	setType(fieldTypeXMLElement);
	setValue(value);
}

XMLElement::XMLElement(XMLElement *value) 
{
	setType(fieldTypeXMLElement);
	setValue(value);
}

XMLElement::~XMLElement() 
{
}

void XMLElement::setValue(const char *value) 
{
	mValue.setValue(value);
}

void XMLElement::setValue(XMLElement *value) 
{
	mValue.setValue(value->getValue());
}

const char *XMLElement::getValue() const
{
	return mValue.getValue();
}

////////////////////////////////////////////////
//	Output
////////////////////////////////////////////////

std::ostream& operator<<(std::ostream &s, const XMLElement &string) 
{
	if (string.getValue())
		return s << "\"" << string.getValue() << "\"";
	else
		return s << "\"" << "\"";
}

std::ostream& operator<<(std::ostream &s, const XMLElement *string) 
{
	if (string->getValue())
		return s << "\"" << string->getValue() << "\"";
	else
		return s << "\"" << "\"";
}

////////////////////////////////////////////////
//	String
////////////////////////////////////////////////

const char *XMLElement::getValue(String &buffer) const
{
	buffer.setValue(getValue());
	return buffer.getValue();
}

const char *XMLElement::getValue(char *buffer, int bufferLen) const
{
	snprintf(buffer, bufferLen, "%s", getValue());
	return buffer;
}

////////////////////////////////////////////////
//	Compare
////////////////////////////////////////////////

bool XMLElement::equals(Field *field) const
{
	XMLElement *stringField = (XMLElement *)field;
	if (!getValue() && !stringField->getValue())
		return true;
	if (getValue() && stringField->getValue())
		return (!strcmp(getValue(), stringField->getValue()) ? true : false);
	else
		return false;
}
