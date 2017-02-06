/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	StringUtil.h
*
******************************************************************/

#ifndef _CX3D_STRINGUTIL_H_
#define _CX3D_STRINGUTIL_H_

#ifndef NO_USE_STL
#include <string>
#endif
#include <iostream>

namespace CyberX3D {

////////////////////////////////////////////////////////////////////////////////
// String
////////////////////////////////////////////////////////////////////////////////

class  String
{

#ifndef NO_USE_STL
	std::string	mValue;
#else
	char	*mValue;
#endif

public:

	String();
	String(const char *value);
	String(const char *value, int offset, int count); 

	virtual ~String();

	String &operator=(String &value);
	String &operator=(const char *value);
	String &operator+(String &value);
	String &operator+(const char *value);
	friend std::ostream& operator<<(std::ostream &s, const String &value);
	friend std::ostream& operator<<(std::ostream &s, const String *value);

	void setValue(const char *value);
	void setValue(const char *value, int offset, int count); 
	void append(const char *value);
	void append(const char c);
  void appendu(const char *value);

	const char *getValue() const;
	void clear();

	int length() const;

	char charAt(int  index) const;

	int compareTo(const char *anotherString) const;
	int compareToIgnoreCase(const char *anotherString) const;

	void concat(const char *str);
	void copyValueOf(const char *data);
	void copyValueOf(const char  *data, int  offset, int count);

	int regionMatches(int toffset, const char *other, int ooffset, int len) const;
	int regionMatchesIgnoreCase(int toffset, const char *other, int ooffset, int len) const;

	int startsWith(const char *prefix) const;
	int endsWith(const char *suffix) const;
};

////////////////////////////////////////////////////////////////////////////////
// String Functions
////////////////////////////////////////////////////////////////////////////////

bool HasString(const char *value);
int StringLength(const char *value);

}

#endif
