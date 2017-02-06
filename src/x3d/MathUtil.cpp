/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	MathUtil.cpp
*
******************************************************************/

#include <math.h>
#include <x3d/MathUtil.h>

using namespace CyberX3D;

////////////////////////////////////////////////////////////
//	VectorNormalize
////////////////////////////////////////////////////////////

void CyberX3D::VectorNormalize(float vector[3])
{
	float mag = (float)sqrt(vector[0]*vector[0] + vector[1]*vector[1] + vector[2]*vector[2]);
	if (mag != 0.0f) {
		vector[0] /= mag;
		vector[1] /= mag;
		vector[2] /= mag;
	}
	else {
		vector[0] = 0.0f;
		vector[1] = 0.0f;
		vector[2] = 1.0f;
	}
}

////////////////////////////////////////////////////////////
//	VectorGetLength
////////////////////////////////////////////////////////////

float CyberX3D::VectorGetLength(float vector[3])
{
	return (float)sqrt(vector[0]*vector[0]+vector[1]*vector[1]+vector[2]*vector[2]);
}

////////////////////////////////////////////////////////////
//	VectorInverse
////////////////////////////////////////////////////////////

void CyberX3D::VectorInverse(float vector[3])
{
	vector[0] = -vector[0];
	vector[1] = -vector[1];
	vector[2] = -vector[2];
}

////////////////////////////////////////////////////////////
//	VectorGenerateFromTwoPoints
////////////////////////////////////////////////////////////

void CyberX3D::VectorGetDirection(
float	point1[3], 
float	point2[3],
float	vector[3])
{
	vector[0] = point1[0] - point2[0];
	vector[1] = point1[1] - point2[1];
	vector[2] = point1[2] - point2[2];
	VectorNormalize(vector);
}

////////////////////////////////////////////////////////////
//	VectorGetPlaneVector
////////////////////////////////////////////////////////////

void CyberX3D::VectorGetCross(
float	vector1[3], 
float	vector2[3],
float	result[3]) 
{
	result[0] = vector1[1]*vector2[2] - vector1[2]*vector2[1];
	result[1] = vector1[2]*vector2[0] - vector1[0]*vector2[2];
	result[2] = vector1[0]*vector2[1] - vector1[1]*vector2[0];
	VectorNormalize(result);
}

////////////////////////////////////////////////////////////
//	VectorGetCrossProduct
////////////////////////////////////////////////////////////

float CyberX3D::VectorGetDot(
float	vector1[3], 
float	vector2[3])
{
	return vector1[0]*vector2[0] + vector1[1]*vector2[1] + vector1[2]*vector2[2];
}

////////////////////////////////////////////////////////////
//	GetAngle
////////////////////////////////////////////////////////////

float CyberX3D::VectorGetAngle(
float vector1[3], 
float vector2[3] )
{
	float angle;
	angle = VectorGetDot(vector1, vector2) / (VectorGetLength(vector1) * VectorGetLength(vector2));
	angle = (float)acos(angle);
	return angle;
}

////////////////////////////////////////////////////////////
//	IsVectorEquals
////////////////////////////////////////////////////////////

bool CyberX3D::VectorEquals(
float	vector1[3], 
float	vector2[3])
{
	if (vector1[0] == vector2[0] && vector1[1] == vector2[1] && vector1[2] == vector2[2])
		return true;
	return false;
}

////////////////////////////////////////////////////////////
//	GetNormalFromVertices
////////////////////////////////////////////////////////////

void CyberX3D::GetNormalFromVertices(float point[3][3], float resultVector[3])
{
	resultVector[0] = (point[1][1] - point[0][1])*(point[2][2] - point[1][2]) - (point[1][2] - point[0][2])*(point[2][1] - point[1][1]);
	resultVector[1] = (point[1][2] - point[0][2])*(point[2][0] - point[1][0]) - (point[1][0] - point[0][0])*(point[2][2] - point[1][2]);
	resultVector[2] = (point[1][0] - point[0][0])*(point[2][1] - point[1][1]) - (point[1][1] - point[0][1])*(point[2][0] - point[1][0]);
	VectorNormalize(resultVector);
}

////////////////////////////////////////////////////////////
//	GetPlaneVectorFromTwoVectors
////////////////////////////////////////////////////////////

void CyberX3D::GetPlaneVectorFromTwoVectors(float vector1[3], float vector2[3], float resultVector[3])
{
	VectorGetCross(vector1, vector2, resultVector);
}
