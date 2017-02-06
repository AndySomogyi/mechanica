/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	MFBool.cpp
*
******************************************************************/

#include <x3d/MFBool.h>

using namespace CyberX3D;

MFBool::MFBool() 
{
	setType(fieldTypeMFBool);
}

void MFBool::addValue(bool value) 
{
	SFBool *sfvalue = new SFBool(value);
	add(sfvalue);
}

void MFBool::addValue(SFBool *sfvalue) 
{
	add(sfvalue);
}

void MFBool::addValue(const char *value) 
{
	SFBool *field = new SFBool();
	field->setValue(value);
	add(field);
}

void MFBool::insertValue(int index, bool value) 
{
	SFBool *sfvalue = new SFBool(value);
	insert(sfvalue, index);
}

bool MFBool::get1Value(int index) const
{
	SFBool *sfvalue = (SFBool *)getObject(index);
	if (sfvalue)
		return sfvalue->getValue();
	else
		return 0.0f;
}

void MFBool::set1Value(int index, bool value) 
{
	SFBool *sfvalue = (SFBool *)getObject(index);
	if (sfvalue)
		sfvalue->setValue(value);
}

void MFBool::setValue(MFBool *values)
{
	clear();

	int size = values->getSize();
	for (int n=0; n<size; n++) {
		addValue(values->get1Value(n));
	}
}

void MFBool::setValue(MField *mfield)
{
	if (mfield->getType() == fieldTypeMFBool)
		setValue((MFBool *)mfield);
}

void MFBool::setValue(int size, bool values[])
{
	clear();

	for (int n=0; n<size; n++)
		addValue(values[n]);
}

////////////////////////////////////////////////
//	Output
////////////////////////////////////////////////

void MFBool::outputContext(std::ostream& printStream, const char *indentString) const
{
	for (int n=0; n<getSize(); n++) {
		if (n < getSize()-1)
			printStream << indentString << get1Value(n) << "," << std::endl;
		else	
			printStream << indentString << get1Value(n) << std::endl;
	}
}
