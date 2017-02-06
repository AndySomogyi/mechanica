/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	SFVec2d.h
*
******************************************************************/

#ifndef _CX3D_SFVEC2D_H_
#define _CX3D_SFVEC2D_H_

#include <iostream>
#include <math.h>
#include <stdio.h>
#include <x3d/Field.h>

namespace CyberX3D {

class SFVec2d : public Field {

	double mValue[2]; 

public:

	SFVec2d();
	SFVec2d(double x, double y);
	SFVec2d(const double value[]);
	SFVec2d(SFVec2d *value);

	////////////////////////////////////////////////
	//	get value
	////////////////////////////////////////////////

	void getValue(double value[]) const;
	const double *getValue() const;
	double getX() const;
	double getY() const;

	int getValueCount() const 
	{
		return 2;
	}

	////////////////////////////////////////////////
	//	set value
	////////////////////////////////////////////////

	void setValue(double x, double y);
	void setValue(const double value[]);
	void setValue(SFVec2d *vector);
	void setX(double x);
	void setY(double y);

	////////////////////////////////////////////////
	//	add value
	////////////////////////////////////////////////

	void add(double x, double y);
	void add(const double value[]);
	void add(SFVec2d value);
	void translate(double x, double y);
	void translate(const double value[]);
	void translate(SFVec2d value);

	////////////////////////////////////////////////
	//	sub value
	////////////////////////////////////////////////

	void sub(double x, double y);
	void sub(const double value[]);
	void sub(SFVec2d value);

	////////////////////////////////////////////////
	//	scale
	////////////////////////////////////////////////

	void scale(double value);
	void scale(double xscale, double yscale);
	void scale(const double value[2]);

	////////////////////////////////////////////////
	//	invert
	////////////////////////////////////////////////

	void invert();

	////////////////////////////////////////////////
	//	scalar
	////////////////////////////////////////////////

	double getScalar() const;

	////////////////////////////////////////////////
	//	normalize
	////////////////////////////////////////////////

	void normalize();

	////////////////////////////////////////////////
	//	Output
	////////////////////////////////////////////////

	friend std::ostream& operator<<(std::ostream &s, const SFVec2d &vector);
	friend std::ostream& operator<<(std::ostream &s, const SFVec2d *vector);

	////////////////////////////////////////////////
	//	String
	////////////////////////////////////////////////

	void setValue(const char *value);
	const char *getValue(char *buffer, int bufferLen) const;

	////////////////////////////////////////////////
	//	Compare
	////////////////////////////////////////////////

	bool equals(Field *field) const;
	bool equals(const double value[2]) const;
	bool equals(double x, double y) const;

};

}

#endif
