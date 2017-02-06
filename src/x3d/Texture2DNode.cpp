/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File: Texture2DNode.cpp
*
*	Revisions:
*
*	12/02/02
*		- The first revision.
*
******************************************************************/

#include <x3d/SceneGraph.h>
#include <x3d/Texture2DNode.h>
#include <x3d/Graphic3D.h>

using namespace CyberX3D;

////////////////////////////////////////////////
//	Texture2DNode::Texture2DNode
////////////////////////////////////////////////

Texture2DNode::Texture2DNode() 
{
	///////////////////////////
	// Field 
	///////////////////////////

	// repeatS field
	repeatSField = new SFBool(true);
	addField(repeatSFieldString, repeatSField);

	// repeatT field
	repeatTField = new SFBool(true);
	addField(repeatTFieldString, repeatTField);

#ifdef CX3D_SUPPORT_OPENGL
	///////////////////////////
	// Private Field 
	///////////////////////////

	// texture name field
	texNameField = new SFInt32(0);
	texNameField->setName(textureNamePrivateFieldString);
	addPrivateField(texNameField);

	// texture name field
	hasTransColorField = new SFBool(false);
	hasTransColorField->setName(hasTransparencyColorPrivateFieldString);
	addPrivateField(hasTransColorField);

#endif
}

////////////////////////////////////////////////
//	Texture2DNode::~Texture2DNode
////////////////////////////////////////////////

Texture2DNode::~Texture2DNode() 
{
}

////////////////////////////////////////////////
//	RepeatS
////////////////////////////////////////////////

SFBool *Texture2DNode::getRepeatSField() const
{
	if (isInstanceNode() == false)
		return repeatSField;
	return (SFBool *)getField(repeatSFieldString);
}
	
void Texture2DNode::setRepeatS(bool value) 
{
	getRepeatSField()->setValue(value);
}

void Texture2DNode::setRepeatS(int value) 
{
	setRepeatS(value ? true : false);
}

bool Texture2DNode::getRepeatS() const
{
	return getRepeatSField()->getValue();
}

////////////////////////////////////////////////
//	RepeatT
////////////////////////////////////////////////

SFBool *Texture2DNode::getRepeatTField() const
{
	if (isInstanceNode() == false)
		return repeatTField;
	return (SFBool *)getField(repeatTFieldString);
}
	
void Texture2DNode::setRepeatT(bool value) 
{
	getRepeatTField()->setValue(value);
}

void Texture2DNode::setRepeatT(int value) 
{
	setRepeatT(value ? true : false);
}

bool Texture2DNode::getRepeatT() const
{
	return getRepeatTField()->getValue();
}

////////////////////////////////////////////////
//	TextureName
////////////////////////////////////////////////

#ifdef CX3D_SUPPORT_OPENGL

SFInt32 *Texture2DNode::getTextureNameField() const
{
	if (isInstanceNode() == false)
		return texNameField;
	return (SFInt32 *)getPrivateField(textureNamePrivateFieldString);
}

void Texture2DNode::setTextureName(unsigned int n) 
{
	getTextureNameField()->setValue((int)n);
}

unsigned int Texture2DNode::getTextureName() const
{
	return (unsigned int)getTextureNameField()->getValue();
} 

#endif

////////////////////////////////////////////////
//	Transparency
////////////////////////////////////////////////

#ifdef CX3D_SUPPORT_OPENGL

SFBool *Texture2DNode::getHasTransparencyColorField() const
{
	if (isInstanceNode() == false)
		return hasTransColorField;
	return (SFBool *)getPrivateField(hasTransparencyColorPrivateFieldString);
}

void Texture2DNode::setHasTransparencyColor(bool value) 
{
	getHasTransparencyColorField()->setValue(value);
}

bool Texture2DNode::hasTransparencyColor() const
{
	return getHasTransparencyColorField()->getValue();
} 

#endif

////////////////////////////////////////////////
//	Texture2DNode::getTextureSize
////////////////////////////////////////////////

#ifdef CX3D_SUPPORT_OPENGL
int CyberX3D::GetOpenGLTextureSize(int size) 
{
	int n = 1;
	while ((1 << n) <= size)
		n++;

	return (1 << (n-1));
}
#endif
