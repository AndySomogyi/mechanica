/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	Field.cpp
*
******************************************************************/

#include <assert.h>
#include <x3d/VRML97Fields.h>

using namespace CyberX3D;

static char fieldTypeString[][16] = {
"None",
"SFBool",
"SFFloat",
"SFDouble",
"SFInt32",
"SFVec2f",
"SFVec3f",
"SFVec2d",
"SFVec3d",
"SFString",
"SFColor",
"SFTime",
"SFRotation",
"SFImage",
"SFNode",
"SFChar",
"MFBool",
"MFFloat",
"MFDouble",
"MFInt32",
"MFVec2f",
"MFVec3f",
"MFVec2d",
"MFVec3d",
"MFString",
"MFColor",
"MFTime",
"MFRotation",
"MFImage",
"MFNode",
"MFChar",
};

////////////////////////////////////////////////////////////
//	Field::getTypeName
////////////////////////////////////////////////////////////

const char *Field::getTypeName() const {
	if (0 < getType() && getType() < fieldTypeMaxNum)
		return fieldTypeString[getType()];
	return NULL;
}

////////////////////////////////////////////////////////////
//	Field::setTypeName
////////////////////////////////////////////////////////////

void Field::setType(const char *type) {

	if (!type || strlen(type) == 0) {
		setType(fieldTypeNone);
		return;
	}

	for (int n=1; n<fieldTypeMaxNum; n++) {
		if (strcmp(fieldTypeString[n], type) == 0) {
			setType(n);
			return;
		}
	}
	setType(fieldTypeNone);
}
////////////////////////////////////////////////////////////
const char *Field::getValueu(char *buffer, int bufferLen) const
{
  return 0;
}
////////////////////////////////////////////////////////////
//	Field::operator
////////////////////////////////////////////////////////////

std::ostream& operator<<(std::ostream &s, const Field &value) {
	switch (value.getType()) {
	case fieldTypeSFBool		: return s << ((SFBool &)value);
	case fieldTypeSFColor		: return s << ((SFColor &)value);
	case fieldTypeSFFloat		: return s << ((SFFloat &)value);
	case fieldTypeSFInt32		: return s << ((SFInt32 &)value);
	case fieldTypeSFRotation	: return s << ((SFRotation &)value);
	case fieldTypeSFString		: return s << ((SFString &)value);
	case fieldTypeSFTime		: return s << ((SFTime &)value);
  	case fieldTypeSFVec2f		: return s << ((SFVec2f &)value);
  	case fieldTypeSFVec3f		: return s << ((SFVec3f &)value);
//	case fieldTypeSFNode		: return s << (SFNode &)value;
/*
	case fieldTypeMFColor		: return s << (MFColor &)value;
	case fieldTypeMFFloat		: return s << (MFFloat &)value;
	case fieldTypeMFInt32		: return s << (MFInt32 &)value;
	case fieldTypeMFRotation	: return s << (MFRotation &)value;
	case fieldTypeMFString		: return s << (MFString &)value;
	case fieldTypeMFTime		: return s << (MFTime &)value;
  	case fieldTypeMFVec2f		: return s << (MFVec2f &)value;
  	case fieldTypeMFVec3f		: return s << (MFVec3f &)value;
//	case fieldTypeMFNode		: return s << (MFNode &)value;
*/
	}
	return s;
}

std::ostream& CyberX3D::operator<<(std::ostream &s, const Field *value) {
	switch (value->getType()) {
	case fieldTypeSFBool		: return s << (SFBool *)value;
	case fieldTypeSFColor		: return s << (SFColor *)value;
	case fieldTypeSFFloat		: return s << (SFFloat *)value;
	case fieldTypeSFInt32		: return s << (SFInt32 *)value;
	case fieldTypeSFRotation	: return s << (SFRotation *)value;
	case fieldTypeSFString		: return s << (SFString *)value;
	case fieldTypeSFTime		: return s << (SFTime *)value;
 	case fieldTypeSFVec2f		: return s << (SFVec2f *)value;
  	case fieldTypeSFVec3f		: return s << (SFVec3f *)value;
//	  	case fieldTypeSFNode		: return s << (SFNode *)value;
	case fieldTypeMFColor		: return s << (MFColor *)value;
	case fieldTypeMFFloat		: return s << (MFFloat *)value;
	case fieldTypeMFInt32		: return s << (MFInt32 *)value;
	case fieldTypeMFRotation	: return s << (MFRotation *)value;
	case fieldTypeMFString		: return s << (MFString *)value;
	case fieldTypeMFTime		: return s << (MFTime *)value;
  	case fieldTypeMFVec2f		: return s << (MFVec2f *)value;
  	case fieldTypeMFVec3f		: return s << (MFVec3f *)value;
//	  	case fieldTypeMFNode		: return s << (MFNode *)value;
	}
	return s;
}

