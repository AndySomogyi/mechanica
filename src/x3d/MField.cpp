/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	MField.cpp
*
******************************************************************/

#include <x3d/MField.h>

using namespace CyberX3D;

MField::MField() 
{
}

MField::~MField() 
{
}

int MField::getSize() const
{
	return mFieldVector.size();
}

int MField::size() const
{
	return mFieldVector.size();
}

void MField::add(Field *object) 
{
	mFieldVector.addElement(object);
}

void MField::insert(int index, Field *object) 
{
	mFieldVector.insertElementAt(object, index);
}

void MField::insert(Field *object, int index) 
{
	mFieldVector.insertElementAt(object, index);
}

void MField::clear() 
{
	mFieldVector.removeAllElements();
}

void MField::remove(int index) 
{
	mFieldVector.removeElementAt(index);
}

void MField::removeLastObject() 
{
	int eleSize = getSize();
	mFieldVector.removeElementAt(eleSize-1);
}

void MField::removeFirstObject() 
{
	mFieldVector.removeElementAt(0);
}

Field *MField::lastObject() const
{
	return (Field *)mFieldVector.lastElement();
}

Field *MField::firstObject() const
{
	return (Field *)mFieldVector.firstElement();
}

Field *MField::getObject(int index) const
{
	return (Field *)mFieldVector.elementAt(index);
}

void MField::setObject(int index, Field *object) 
{
	mFieldVector.setElementAt(object, index);
}

void MField::replace(int index, Field *object) 
{
	setObject(index, object); 
}

void MField::copy(MField *srcMField) 
{
	clear();
	for (int n=0; n<srcMField->getSize(); n++) {
		add(srcMField->getObject(n));
	}
}

void MField::outputContext(std::ostream& printStream, const char *indentString1, const char *indentString2) const
{
	char *indentString = new char[strlen(indentString1)+strlen(indentString2)+1];
	strcpy(indentString, indentString1);
	strcat(indentString, indentString2);
	outputContext(printStream, indentString);
	delete indentString;
}

////////////////////////////////////////////////
//	MField::setValue
////////////////////////////////////////////////

void MField::setValue(const char *buffer)
{
	const char *bp = buffer;
	char value[128];
	int nSize = getSize();
	for (int n=0; n<nSize; n++) {
		int l=0;
		while (bp[l] != ',' && bp[l] != '\0')
			l++;
		if (bp[l] == '\0')
			return;
		strncpy(value, bp, l); 
		Field *field = getObject(n);
		field->setValue(value);
		bp += l;
	}
}

////////////////////////////////////////////////
//	MField::getValue
////////////////////////////////////////////////

const char *MField::getValue(String &mfBuffer) const
{
	mfBuffer.clear();
	
	int		nSize = getSize();
	String value;

	for (int n=0, j = 0; n<nSize; n++, j++) {
		const Field *field = getObject(n);
    field->getValueu(value);
		int nString = mfBuffer.length();
		if (0 < nString)
			mfBuffer.appendu(", ");
		mfBuffer.appendu(value.getValue());
    if (j >=32)
      {
      mfBuffer.appendu("\n");
      j = 0;
      }
	}

	return mfBuffer.getValue();
}

const char *MField::getValue(char *buffer, int bufferLen) const
{
	buffer[0] = '\0';
	
	int		nString = 0;
	int		nSize = getSize();
	char	value[FIELD_MIN_BUFFERSIZE];

	for (int n=0; n<nSize; n++) {
		const Field *field = getObject(n);
		field->getValue(value);
		int l = strlen(value);
		if ((nString + l + 2) < bufferLen) {
			if (0 < nString)
				strcat(buffer, ", ");
			strcat(buffer, value);
			if (0 < nString)
				nString += (l + 2);
			else
				nString += l;
		}
		else
			break;
	}

	return buffer;
}

