/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	SFVec3d.h
*
******************************************************************/

#ifndef _CX3D_SFVEC3D_H_
#define _CX3D_SFVEC3D_H_

#include <iostream>
#include <math.h>
#include <stdio.h>
#include <x3d/Field.h>

namespace CyberX3D {

class SFRotation;

class SFVec3d : public Field {

	double	mValue[3]; 

public:

	SFVec3d();
	SFVec3d(double x, double y, double z);
	SFVec3d(const double value[]);
	SFVec3d(SFVec3d *value);

	////////////////////////////////////////////////
	//	get value
	////////////////////////////////////////////////

	void getValue(double value[]) const;
	const double *getValue() const;
	double getX() const;
	double getY() const;
	double getZ() const;

	int getValueCount() const 
	{
		return 3;
	}

	////////////////////////////////////////////////
	//	set value
	////////////////////////////////////////////////

	void setValue(double x, double y, double z);
	void setValue(const double value[]);
	void setValue(SFVec3d *vector);
	void setX(double x);
	void setY(double y);
	void setZ(double z);

	////////////////////////////////////////////////
	//	add value
	////////////////////////////////////////////////

	void add(double x, double y, double z);
	void add(const double value[]);
	void add(SFVec3d value);
	void translate(double x, double y, double z);
	void translate(const double value[]);
	void translate(SFVec3d value);

	////////////////////////////////////////////////
	//	sub value
	////////////////////////////////////////////////

	void sub(double x, double y, double z);
	void sub(const double value[]);
	void sub(SFVec3d value);

	////////////////////////////////////////////////
	//	scale
	////////////////////////////////////////////////

	void scale(double value);
	void scale(double xscale, double yscale, double zscale);
	void scale(const double value[3]);

	////////////////////////////////////////////////
	//	rotate
	////////////////////////////////////////////////

	void rotate(SFRotation *rotation);
	void rotate(double x, double y, double z, double angle);
	void rotate(const double value[3]);

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
	//	Overload
	////////////////////////////////////////////////

	friend std::ostream& operator<<(std::ostream &s, const SFVec3d &vector);
	friend std::ostream& operator<<(std::ostream &s, const SFVec3d *vector);

	////////////////////////////////////////////////
	//	String
	////////////////////////////////////////////////

	void setValue(const char *value);
	const char *getValue(char *buffer, int bufferLen) const;

	////////////////////////////////////////////////
	//	Compare
	////////////////////////////////////////////////

	bool equals(Field *field) const;
	bool equals(const double value[3]) const;
	bool equals(double x, double y, double z) const;

};

}

#endif
