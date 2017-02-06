/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	StringUtil.cpp
*
******************************************************************/

#include <cstdlib>
#include <cstring>
#include <ctype.h>
#include <x3d/StringUtil.h>

using namespace CyberX3D;

////////////////////////////////////////////////////
// STL Version
////////////////////////////////////////////////////

#ifndef NO_USE_STL

String::String() 
{
}

String::String(const char *value) 
{
	setValue(value);
}

String::String(const char *value, int offset, int count) 
{
	setValue(value, offset, count);
}

String::~String() 
{
}

void String::setValue(const char *value) 
{
	clear();
	if (!value)
		return;
	if (strlen(value) <= 0)
		return;
	mValue = value;
}

void String::setValue(const char *value, int offset, int count) 
{ 
	if (!value && (int)strlen(value) < (offset + count)) 
		return;
	std::string orgValue = value;
	mValue = orgValue.substr(offset, count);
}

void String::append(const char *value) 
{ 
	if (!value)
		return;
	if (strlen(value) <= 0)
		return;
	mValue.append(value);
}

const char *String::getValue() const
{
	return (char *)mValue.c_str();
}

void String::clear() 
{
	mValue = "";
}

int String::length() const
{
	return mValue.length();
}

char String::charAt(int  index) const
{
	return mValue[index];
}

#endif

////////////////////////////////////////////////////
// Normal Version
////////////////////////////////////////////////////

#ifdef NO_USE_STL

String::String() 
{
	mValue = NULL;
}

String::String(const char *value) 
{
	mValue = NULL;
	setValue(value);
}

String::String(const char *value, int offset, int count) 
{
	mValue = NULL;
	setValue(value, offset, count);
}

String::~String() 
{
	clear();
}

void String::setValue(const char *value)
{
	clear();
	if (!value)
		return;
	if (strlen(value) <= 0)
		return;
	mValue = new char[strlen(value)+1];
	strcpy(mValue, value);
}

void String::setValue(const char *value, int offset, int count) 
{ 
	clear();
	if (!value && (int)strlen(value) < (offset + count)) 
		return;
	mValue = new char[count+1];
	strncpy(mValue, &value[offset], count);
}

const char *String::getValue() const
{
	return mValue;
}

void String::append(const char *value) 
{ 
	if (!value)
		return;
	if (strlen(value) <= 0)
		return;
	strcat(mValue, value);
}

void String::clear() 
{
	delete[] mValue;
	mValue = NULL;
}

int String::length() const
{
	if (!mValue)
		return 0;
	return strlen(mValue);
}

char String::charAt(int  index) const
{
	return mValue[index];
}

#endif

////////////////////////////////////////////////////
// Common
////////////////////////////////////////////////////

void String::append(char c) 
{ 
	char value[2];
	value[0] = c;
	value[1] = '\0';
	append(value);
}

int String::compareTo(const char *anotherString) const 
{
	const char *value = getValue();
	if (!value || !anotherString)
		return -1;
	return strcmp(value, anotherString);
}

void String::copyValueOf(const char *data) 
{
	if (!data)
		return;
	strcpy((char *)data, getValue());
}

void String::copyValueOf(const char  data[], int  offset, int count) 
{
	if (!data)
		return;
	const char *value = getValue();
	strncpy((char *)data, &value[offset], count);
}

int String::startsWith(const char *prefix) const
{
	if (!prefix)
		return -1;
	return regionMatches(0, prefix, 0, strlen(prefix));
}

int String::endsWith(const char *suffix) const
{
	if (!suffix)
		return -1;
	return regionMatches(length()-strlen(suffix), suffix, 0, strlen(suffix));
}

int String::regionMatchesIgnoreCase(int toffset, const char *other, int ooffset, int len) const
{
	if (!other)
		return -1;
	
	int n;

	int value1Len = length();
	char *value1 = new char[value1Len+1]; 
	strcpy(value1, getValue());
	for (n=0; n<value1Len; n++)
		value1[n] = (char)toupper(value1[n]);

	int value2Len = strlen(other);
	char *value2 = new char[value2Len+1]; 
	strcpy(value2, other);
	for (n=0; n<value2Len; n++)
		value2[n] = (char)toupper(value2[n]);
		
	int ret = regionMatches(toffset, other, ooffset, len);

	delete value1;
	delete value2;

	return ret;
}

int String::regionMatches(int toffset, const char *other, int ooffset, int len) const 
{
	const char *value = getValue();
	if (!value || !other)
		return -1;
	if (length() < toffset)
		return -1;
	if ((int)strlen(other) < ooffset + len)
		return -1;
	if (toffset<0 || ooffset<0)
		return -1;
	return strncmp(&value[toffset], &other[ooffset], len);
}

////////////////////////////////////////////////////
// operator
////////////////////////////////////////////////////

String &String::operator=(String &other)
{
	if (this == &other)
		return *this;
	setValue(other.getValue());
	return *this;
};

String &String::operator=(const char *value)
{
	setValue(value);
	return *this;
};

String &String::operator+(String &other)
{
	if (this == &other)
		return *this;
	append(other.getValue());
	return *this;
};

String &String::operator+(const char *value)
{
	append(value);
	return *this;
};

std::ostream& CyberX3D::operator<<(std::ostream &s, const String &value) 
{
	return s << value.getValue();
}

std::ostream& CyberX3D::operator<<(std::ostream &s, const String *value) 
{
	return s << value->getValue();
}

////////////////////////////////////////////////////
// Util
////////////////////////////////////////////////////

bool CyberX3D::HasString(const char *value)
{
	if (0 < StringLength(value))
		return true;
	return false;
}

int CyberX3D::StringLength(const char *value)
{
	if (value == NULL)
		return 0;
	return strlen(value);
}
////////////////////////////////////////////////////
void String::appendu(const char *value)
  {
#ifndef NO_USE_STL
  if (!value)
		return;
	if (strlen(value) <= 0)
		return;
  mValue.append(value);
#else
  if (!value)
		return;
	if (strlen(value) <= 0)
		return;
  if(mValue == NULL)
    {
    mValue = new char[strlen(value)+1];
    strcpy(mValue, value);
    }
  else
    {
    int nmValLen = strlen(mValue);
    int nValLen = strlen(value);
    char *pcTemp = new char[nmValLen + 1];
    strcpy(pcTemp, mValue);
    delete[] mValue;
    mValue = new char[nmValLen + nValLen + 1];
    strcpy(mValue, pcTemp);
    strcat(mValue, value);

    delete[]pcTemp;
    pcTemp = NULL;

    }
#endif
  }