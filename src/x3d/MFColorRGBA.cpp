/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	MFColorRGBA.cpp
*
******************************************************************/

#include <x3d/MFColorRGBA.h>

using namespace CyberX3D;

MFColorRGBA::MFColorRGBA() 
{
	setType(fieldTypeMFColor);
	InitializeJavaIDs();
}

void MFColorRGBA::InitializeJavaIDs() 
{
#if defined(CX3D_SUPPORT_JSAI)
	setJavaIDs();
#endif
}

void MFColorRGBA::addValue(float r, float g, float b, float a) 
{
	SFColorRGBA *color = new SFColorRGBA(r, g, b, a);
	add(color);
}

void MFColorRGBA::addValue(float value[]) 
{
	SFColorRGBA *color = new SFColorRGBA(value);
	add(color);
}

void MFColorRGBA::addValue(SFColorRGBA *color) 
{
	add(color);
}

void MFColorRGBA::addValue(const char *value) 
{
	SFColorRGBA *field = new SFColorRGBA();
	field->setValue(value);
	add(field);
}

void MFColorRGBA::insertValue(int index, float r, float g, float b, float a) 
{
	SFColorRGBA *color = new SFColorRGBA(r, g, b, a);
	insert(color, index);
}

void MFColorRGBA::insertValue(int index, float value[]) 
{
	SFColorRGBA *color = new SFColorRGBA(value);
	insert(color, index);
}

void MFColorRGBA::insertValue(int index, SFColorRGBA *color) 
{
	insert(color, index);
}

void MFColorRGBA::get1Value(int index, float value[]) const
{
	SFColorRGBA *color = (SFColorRGBA *)getObject(index);
	if (color)
		color->getValue(value);
}

void MFColorRGBA::set1Value(int index, float value[]) 
{
	SFColorRGBA *color = (SFColorRGBA *)getObject(index);
	if (color)
		color->setValue(value);
}

void MFColorRGBA::set1Value(int index, float r, float g, float b, float a) 
{
	SFColorRGBA *color = (SFColorRGBA *)getObject(index);
	if (color)
		color->setValue(r, g, b, a);
}

void MFColorRGBA::setValue(MFColorRGBA *colors)
{
	clear();

	float value[4];
	int size = colors->getSize();
	for (int n=0; n<size; n++) {
		colors->get1Value(n, value);
		addValue(value);
	}
}

void MFColorRGBA::setValue(MField *mfield)
{
	if (mfield->getType() == fieldTypeMFColor)
		setValue((MFColorRGBA *)mfield);
}

void MFColorRGBA::setValue(int size, float colors[][4])
{
	clear();

	for (int n=0; n<size; n++)
		addValue(colors[n]);
}

////////////////////////////////////////////////
//	Output
////////////////////////////////////////////////

void MFColorRGBA::outputContext(std::ostream& printStream, const char *indentString) const
{
	float value[3];
	for (int n=0; n<getSize(); n++) {
		get1Value(n, value);
		if (n < getSize()-1)
			printStream << indentString << value[0] << " " << value[1] << " " << value[2] << "," << value[3] << "," << std::endl;
		else	
			printStream << indentString << value[0] << " " << value[1] << " " << value[2] << " " << value[3] << std::endl;
	}
}
