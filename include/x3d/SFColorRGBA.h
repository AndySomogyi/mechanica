/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	SFColorRGBA.h
*
******************************************************************/

#ifndef _CX3D_SFCOLORRGBA_H_
#define _CX3D_SFCOLORRGBA_H_

#include <iostream>
#include <stdio.h>
#include <x3d/Field.h>

namespace CyberX3D {

class SFColorRGBA : public Field {

	static	int	mInit;

	float mValue[3]; 

public:

	SFColorRGBA();
	SFColorRGBA(float r, float g, float b, float a);
	SFColorRGBA(float value[]);
	SFColorRGBA(SFColorRGBA *color);

	void InitializeJavaIDs();

	////////////////////////////////////////////////
	//	get value
	////////////////////////////////////////////////

	void getValue(float value[]) const;
	const float *getValue() const;
	float getRed() const;
	float getGreen() const;
	float getBlue() const;
	float getAlpha() const;

	int getValueCount() const 
	{
		return 4;
	}

	////////////////////////////////////////////////
	//	set value
	////////////////////////////////////////////////

	void setValue(float r, float g, float b, float a);
	void setValue(const float value[]);
	void setValue(SFColorRGBA *color);

	////////////////////////////////////////////////
	//	add value
	////////////////////////////////////////////////

	void add(float x, float y, float z, float a);
	void add(const float value[]);
	void add(SFColorRGBA value);

	////////////////////////////////////////////////
	//	sub value
	////////////////////////////////////////////////

	void sub(float x, float y, float z, float a);
	void sub(const float value[]);
	void sub(SFColorRGBA value);

	////////////////////////////////////////////////
	//	Output
	////////////////////////////////////////////////

	friend std::ostream& operator<<(std::ostream &s, const SFColorRGBA &vector);
	friend std::ostream& operator<<(std::ostream &s, const SFColorRGBA *vector);

	////////////////////////////////////////////////
	//	String
	////////////////////////////////////////////////

	void setValue(const char *value);
	const char *getValue(char *buffer, int bufferLen) const;


	////////////////////////////////////////////////
	//	scale
	////////////////////////////////////////////////

	void scale(float scale);

	////////////////////////////////////////////////
	//	Compare
	////////////////////////////////////////////////

	bool equals(Field *field) const;
};

}

#endif
