/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	SFVec3d.cpp
*
******************************************************************/

#include <stdio.h>
#include <x3d/SFVec3d.h>
#include <x3d/SFRotation.h>

using namespace CyberX3D;

SFVec3d::SFVec3d() 
{
	setType(fieldTypeSFVec3d);
	setValue(0.0, 0.0, 0.0);
}

SFVec3d::SFVec3d(double x, double y, double z) 
{
	setType(fieldTypeSFVec3d);
	setValue(x, y, z);
}

SFVec3d::SFVec3d(const double value[]) 
{
	setType(fieldTypeSFVec3d);
	setValue(value);
}

SFVec3d::SFVec3d(SFVec3d *value) 
{
	setType(fieldTypeSFVec3d);
	setValue(value);
}

////////////////////////////////////////////////
//	get value
////////////////////////////////////////////////

void SFVec3d::getValue(double value[]) const
{
	value[0] = mValue[0];
	value[1] = mValue[1];
	value[2] = mValue[2];
}

const double *SFVec3d::getValue() const
{
	return mValue;
}

double SFVec3d::getX() const
{
	return mValue[0];
}

double SFVec3d::getY() const
{
	return mValue[1];
}

double SFVec3d::getZ() const
{
	return mValue[2];
}

////////////////////////////////////////////////
//	set value
////////////////////////////////////////////////

void SFVec3d::setValue(double x, double y, double z) 
{
	mValue[0] = x;
	mValue[1] = y;
	mValue[2] = z;
}

void SFVec3d::setValue(const double value[]) 
{
	mValue[0] = value[0];
	mValue[1] = value[1];
	mValue[2] = value[2];
}

void SFVec3d::setValue(SFVec3d *vector) 
{
	setValue(vector->getX(), vector->getY(), vector->getZ());
}

void SFVec3d::setX(double x) 
{
	setValue(x, getY(), getZ());
}

void SFVec3d::setY(double y) 
{
	setValue(getX(), y, getZ());
}

void SFVec3d::setZ(double z) 
{
	setValue(getX(), getY(), z);
}

////////////////////////////////////////////////
//	add value
////////////////////////////////////////////////

void SFVec3d::add(double x, double y, double z) 
{
	mValue[0] += x;
	mValue[1] += y;
	mValue[2] += z;
}

void SFVec3d::add(const double value[]) 
{
	mValue[0] += value[0];
	mValue[1] += value[1];
	mValue[2] += value[2];
}

void SFVec3d::add(SFVec3d value) 
{
	add(value.getValue());
}

void SFVec3d::translate(double x, double y, double z) 
{
	add(x, y, z);
}

void SFVec3d::translate(const double value[]) 
{
	add(value);
}

void SFVec3d::translate(SFVec3d value) 
{
	add(value);
}

////////////////////////////////////////////////
//	sub value
////////////////////////////////////////////////

void SFVec3d::sub(double x, double y, double z) 
{
	mValue[0] -= x;
	mValue[1] -= y;
	mValue[2] -= z;
}

void SFVec3d::sub(const double value[]) 
{
	mValue[0] -= value[0];
	mValue[1] -= value[1];
	mValue[2] -= value[2];
}

void SFVec3d::sub(SFVec3d value) 
{
	sub(value.getValue());
}

////////////////////////////////////////////////
//	scale
////////////////////////////////////////////////

void SFVec3d::scale(double value) 
{
	mValue[0] *= value;
	mValue[1] *= value;
	mValue[2] *= value;
}

void SFVec3d::scale(double xscale, double yscale, double zscale) 
{
	mValue[0] *= xscale;
	mValue[1] *= yscale;
	mValue[2] *= zscale;
}

void SFVec3d::scale(const double value[3]) 
{
	scale(value[0], value[1], value[2]);
}

////////////////////////////////////////////////
//	rotate
////////////////////////////////////////////////

void SFVec3d::rotate(SFRotation *rotation) 
{
	rotation->multi(mValue);
}

void SFVec3d::rotate(double x, double y, double z, double angle) 
{
	SFRotation rotation(x, y, z, angle);
	rotate(&rotation);
}

void SFVec3d::rotate(const double value[3]) 
{
	rotate(value[0], value[1], value[2], value[3]);
}

////////////////////////////////////////////////
//	invert
////////////////////////////////////////////////

void SFVec3d::invert() 
{
	mValue[0] = -mValue[0];
	mValue[1] = -mValue[1];
	mValue[2] = -mValue[2];
}

////////////////////////////////////////////////
//	scalar
////////////////////////////////////////////////

double SFVec3d::getScalar() const
{
	return (double)sqrt(mValue[0]*mValue[0]+mValue[1]*mValue[1]+mValue[2]*mValue[2]);
}

////////////////////////////////////////////////
//	normalize
////////////////////////////////////////////////

void SFVec3d::normalize()
{
	double scale = getScalar();
	if (scale != 0.0f) {
		mValue[0] /= scale;
		mValue[1] /= scale;
		mValue[2] /= scale;
	}
}

////////////////////////////////////////////////
//	String
////////////////////////////////////////////////

void SFVec3d::setValue(const char *value) 
{
	if (!value)
		return;
	double	x, y, z;
	if (sscanf(value,"%f %f %f", &x, &y, &z) == 3) 
		setValue(x, y, z);
}

const char *SFVec3d::getValue(char *buffer, int bufferLen) const
{
	snprintf(buffer, bufferLen, "%g %g %g", getX(), getY(), getZ());
	return buffer;
}

////////////////////////////////////////////////
//	Compare
////////////////////////////////////////////////

bool SFVec3d::equals(Field *field) const
{
	SFVec3d *vector = (SFVec3d *)field;
	if (getX() == vector->getX() && getY() == vector->getY() && getZ() == vector->getZ())
		return true;
	else
		return false;
}

bool SFVec3d::equals(const double value[3]) const
{
	SFVec3d vector(value);
	return equals(&vector);
}

bool SFVec3d::equals(double x, double y, double z) const
{
	SFVec3d vector(x, y, z);
	return equals(&vector);
}


////////////////////////////////////////////////
//	Overload
////////////////////////////////////////////////

std::ostream& CyberX3D::operator<<(std::ostream &s, const SFVec3d &vector) 
{
	return s << vector.getX() << " " << vector.getY() << " " << vector.getZ();
}

std::ostream& CyberX3D::operator<<(std::ostream &s, const SFVec3d *vector) 
{
	return s << vector->getX() << " " << vector->getY() << " " << vector->getZ();
}
