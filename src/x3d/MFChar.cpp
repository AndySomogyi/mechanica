/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	MFChar.cpp
*
******************************************************************/

#include <x3d/MFChar.h>

using namespace CyberX3D;

MFChar::MFChar() 
{
	setType(fieldTypeMFChar);
}

void MFChar::addValue(char value) 
{
	SFChar *sfvalue = new SFChar(value);
	add(sfvalue);
}

void MFChar::addValue(SFChar *sfvalue) 
{
	add(sfvalue);
}

void MFChar::addValue(const char *value) 
{
	SFChar *field = new SFChar();
	field->setValue(value);
	add(field);
}

void MFChar::insertValue(int index, char value) 
{
	SFChar *sfvalue = new SFChar(value);
	insert(sfvalue, index);
}

char MFChar::get1Value(int index) const
{
	SFChar *sfvalue = (SFChar *)getObject(index);
	if (sfvalue)
		return sfvalue->getValue();
	return ' ';
}

void MFChar::set1Value(int index, char value) 
{
	SFChar *sfvalue = (SFChar *)getObject(index);
	if (sfvalue)
		sfvalue->setValue(value);
}

void MFChar::setValue(MFChar *values)
{
	clear();

	int size = values->getSize();
	for (int n=0; n<size; n++) {
		addValue(values->get1Value(n));
	}
}

void MFChar::setValue(MField *mfield)
{
	if (mfield->getType() == fieldTypeMFChar)
		setValue((MFChar *)mfield);
}

void MFChar::setValue(int size, char values[])
{
	clear();

	for (int n=0; n<size; n++)
		addValue(values[n]);
}

////////////////////////////////////////////////
//	Output
////////////////////////////////////////////////

void MFChar::outputContext(std::ostream& printStream, const char *indentString) const
{
	for (int n=0; n<getSize(); n++) {
		if (n < getSize()-1)
			printStream << indentString << get1Value(n) << "," << std::endl;
		else	
			printStream << indentString << get1Value(n) << std::endl;
	}
}
