/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	MFDouble.cpp
*
******************************************************************/

#include <x3d/MFDouble.h>

using namespace CyberX3D;

MFDouble::MFDouble() 
{
	setType(fieldTypeMFDouble);
}

void MFDouble::addValue(double value) 
{
	SFDouble *sfvalue = new SFDouble(value);
	add(sfvalue);
}

void MFDouble::addValue(SFDouble *sfvalue) 
{
	add(sfvalue);
}

void MFDouble::addValue(const char *value) 
{
	SFDouble *field = new SFDouble();
	field->setValue(value);
	add(field);
}

void MFDouble::insertValue(int index, double value) 
{
	SFDouble *sfvalue = new SFDouble(value);
	insert(sfvalue, index);
}

double MFDouble::get1Value(int index) const
{
	SFDouble *sfvalue = (SFDouble *)getObject(index);
	if (sfvalue)
		return sfvalue->getValue();
	else
		return 0.0f;
}

void MFDouble::set1Value(int index, double value) 
{
	SFDouble *sfvalue = (SFDouble *)getObject(index);
	if (sfvalue)
		sfvalue->setValue(value);
}

void MFDouble::setValue(MFDouble *values)
{
	clear();

	int size = values->getSize();
	for (int n=0; n<size; n++) {
		addValue(values->get1Value(n));
	}
}

void MFDouble::setValue(MField *mfield)
{
	if (mfield->getType() == fieldTypeMFDouble)
		setValue((MFDouble *)mfield);
}

void MFDouble::setValue(int size, double values[])
{
	clear();

	for (int n=0; n<size; n++)
		addValue(values[n]);
}

////////////////////////////////////////////////
//	Output
////////////////////////////////////////////////

void MFDouble::outputContext(std::ostream& printStream, const char *indentString) const
{
	for (int n=0; n<getSize(); n++) {
		if (n < getSize()-1)
			printStream << indentString << get1Value(n) << "," << std::endl;
		else	
			printStream << indentString << get1Value(n) << std::endl;
	}
}
