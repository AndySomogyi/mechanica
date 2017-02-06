/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	BoundingBox2D.cpp
*
*	Revisions:
*
*		12/02/02
*			- The first release.
*
******************************************************************/

#include <float.h>
#include <math.h>
#include <x3d/BoundingBox2D.h>

using namespace CyberX3D;

BoundingBox2D::BoundingBox2D()
{
	initialize();
}

BoundingBox2D::BoundingBox2D(BoundingBox2D *bbox)
{
	set(bbox);
}

BoundingBox2D::BoundingBox2D(float center[2], float size[2])
{
	set(center, size);
}

BoundingBox2D::~BoundingBox2D()
{
}

void BoundingBox2D::initialize()
{
	setMinPosition(FLT_MAX, FLT_MAX);
	setMaxPosition(FLT_MIN, FLT_MIN);
	setNPoints(0);
}

void BoundingBox2D::addPoint(float point[2])
{
	for (int n=0; n<2; n++) {
		if (point[n] < mMinPosition[n])
			mMinPosition[n] = point[n];
		if (mMaxPosition[n] < point[n]) 
			mMaxPosition[n] = point[n];
	}
	setNPoints(getNPoints()+1);
}

void BoundingBox2D::addPoint(float x, float y)
{
	float point[] = {x, y};
	addPoint(point);
}

void BoundingBox2D::addBoundingBox2D(float center[2], float size[2])
{
	float	point[2];
	for (int n=0; n<4; n++) {
		point[0] = (n < 4)			? center[0] - size[0] : center[0] + size[0];
		point[1] = (n % 2)			? center[1] - size[1] : center[1] + size[1];
		addPoint(point);
	}
}

void BoundingBox2D::addBoundingBox2D(BoundingBox2D *bbox)
{
	float	center[2];
	float	size[2];
	bbox->getCenter(center);
	bbox->getSize(size);
	addBoundingBox2D(center, size);
}

void BoundingBox2D::setNPoints(int npoints)
{
	mNPoints = npoints;
}

int BoundingBox2D::getNPoints() const
{
	return mNPoints;
}

void BoundingBox2D::setMinPosition(float x, float y)
{
	mMinPosition[0] = x;
	mMinPosition[1] = y;
}

void BoundingBox2D::setMaxPosition(float x, float y)
{
	mMaxPosition[0] = x;
	mMaxPosition[1] = y;
}

void BoundingBox2D::setMinPosition(float pos[2])
{
	setMinPosition(pos[0], pos[1]);
}

void BoundingBox2D::setMaxPosition(float pos[2])
{
	setMaxPosition(pos[0], pos[1]);
}

void BoundingBox2D::getMinPosition(float pos[2]) const
{
	pos[0] = mMinPosition[0];
	pos[1] = mMinPosition[1];
}

void BoundingBox2D::getMaxPosition(float pos[2]) const
{
	pos[0] = mMaxPosition[0];
	pos[1] = mMaxPosition[1];
}

void BoundingBox2D::set(float center[2], float size[2])
{
	for (int n=0; n<2; n++) {
		mMinPosition[n] = center[n] - size[n];
		mMaxPosition[n] = center[n] + size[n];;
	}
	setNPoints(1);
}

void BoundingBox2D::set(BoundingBox2D *bbox)
{
	float	center[2];
	float	size[2];
	bbox->getCenter(center);
	bbox->getSize(size);
	set(center, size);
}

void BoundingBox2D::getCenter(float center[2]) const
{
	if (0 < getNPoints()) {
		center[0] = (mMaxPosition[0] + mMinPosition[0]) / 2.0f;
		center[1] = (mMaxPosition[1] + mMinPosition[1]) / 2.0f;
	}
	else {
		center[0] = 0.0f;
		center[1] = 0.0f;
	}
}

void BoundingBox2D::getSize(float size[2]) const
{
	if (0 < getNPoints()) {
		size[0] = (float)fabs(mMaxPosition[0] - mMinPosition[0]) / 2.0f;
		size[1] = (float)fabs(mMaxPosition[1] - mMinPosition[1]) / 2.0f;
	}
	else {
		size[0] = -1.0f;
		size[1] = -1.0f;
	}
}
