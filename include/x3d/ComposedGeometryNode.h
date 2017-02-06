/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	ComposedGeometryNode.h
*
******************************************************************/

#ifndef _CX3D_COMPOSEDGEOMETRYNODE_H_
#define _CX3D_COMPOSEDGEOMETRYNODE_H_

#include <x3d/Geometry3DNode.h>
#include <x3d/NormalNode.h>
#include <x3d/ColorNode.h>
#include <x3d/CoordinateNode.h>
#include <x3d/TextureCoordinateNode.h>

namespace CyberX3D {

class ComposedGeometryNode : public Geometry3DNode 
{
	SFBool *ccwField;
	SFBool *colorPerVertexField;
	SFBool *normalPerVertexField;
	SFBool *solidField;

	SFNode *colorField;
	SFNode *coordField;
	SFNode *normalField;
	SFNode *texCoordField;
	
public:

	ComposedGeometryNode();
	virtual ~ComposedGeometryNode();

	////////////////////////////////////////////////
	//	Color
	////////////////////////////////////////////////

	SFNode *getColorField() const;
	
	////////////////////////////////////////////////
	//	Coord
	////////////////////////////////////////////////

	SFNode *getCoordField() const;

	////////////////////////////////////////////////
	//	Normal
	////////////////////////////////////////////////

	SFNode *getNormalField() const;
	
	////////////////////////////////////////////////
	//	texCoord
	////////////////////////////////////////////////

	SFNode *getTexCoordField() const;
	
	////////////////////////////////////////////////
	//	CCW
	////////////////////////////////////////////////

	SFBool *getCCWField() const;
	
	void setCCW(bool value);
	void setCCW(int value);
	bool getCCW() const;

	////////////////////////////////////////////////
	//	ColorPerVertex
	////////////////////////////////////////////////

	SFBool *getColorPerVertexField() const;
	
	void setColorPerVertex(bool value);
	void setColorPerVertex(int value);
	bool getColorPerVertex() const;

	////////////////////////////////////////////////
	//	NormalPerVertex
	////////////////////////////////////////////////
	
	SFBool *getNormalPerVertexField() const;

	void setNormalPerVertex(bool value);
	void setNormalPerVertex(int value);
	bool getNormalPerVertex() const;

	////////////////////////////////////////////////
	//	Solid
	////////////////////////////////////////////////

	SFBool *getSolidField() const;
	
	void setSolid(bool value);
	void setSolid(int value);
	bool getSolid() const;

	////////////////////////////////////////////////
	//	BoundingBox
	////////////////////////////////////////////////

	void recomputeBoundingBox();
};

}

#endif

