/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	VRML97ParserFunc.cpp
*
******************************************************************/

#include <x3d/VRML97Parser.h>
#include <x3d/VRML97ParserFunc.h>
#include <x3d/NodeType.h>

using namespace CyberX3D;

/******************************************************************
*	AddSF* 
******************************************************************/

void CyberX3D::AddSFColor(float color[3])
{
    switch (ParserGetCurrentNodeType()) {
	case VRML97_COLOR:
		{
			((ColorNode *)ParserGetCurrentNode())->addColor(color);
		}
		break;
    case VRML97_BACKGROUND_GROUNDCOLOR:
		{
			BackgroundNode *bg = (BackgroundNode *)ParserGetCurrentNode();
			bg->addGroundColor(color);
		}		
	    break;
    case VRML97_BACKGROUND_SKYCOLOR:
		{
			BackgroundNode *bg = (BackgroundNode *)ParserGetCurrentNode();
			bg->addSkyColor(color);
		}		
	    break;
	case VRML97_INTERPOLATOR_KEYVALUE:
		switch (ParserGetParentNodeType()) {
		case VRML97_COLORINTERPOLATOR:
			{
				ColorInterpolatorNode *colorInterp = (ColorInterpolatorNode *)ParserGetCurrentNode();
				colorInterp->addKeyValue(color);
			}
			break;
		}
		break;
    }
}

void CyberX3D::AddSFRotation(float rotation[4])
{	
    switch (ParserGetCurrentNodeType()) {
	case VRML97_EXTRUSION_ORIENTATION:
		{
			ExtrusionNode *ex = (ExtrusionNode *)ParserGetCurrentNode();
			ex->addOrientation(rotation);
		}
		break;
	case VRML97_INTERPOLATOR_KEYVALUE:
		switch (ParserGetParentNodeType()) {
		case VRML97_ORIENTATIONINTERPOLATOR:
			{
				OrientationInterpolatorNode *oriInterp = (OrientationInterpolatorNode *)ParserGetCurrentNode();
				oriInterp->addKeyValue(rotation);
			}
			break;
		}
	}
}

void CyberX3D::AddSFVec3f(float vector[3])
{	
    switch (ParserGetCurrentNodeType()) {
	case VRML97_NORMAL:
		{
			((NormalNode *)ParserGetCurrentNode())->addVector(vector);
		}
	    break;
	case VRML97_COORDINATE:
		{
			((CoordinateNode *)ParserGetCurrentNode())->addPoint(vector);
		}
		break;
	case VRML97_INTERPOLATOR_KEYVALUE:
		switch (ParserGetParentNodeType()) {
		case VRML97_COORDINATEINTERPOLATOR:
			{
				CoordinateInterpolatorNode *coordInterp = (CoordinateInterpolatorNode *)ParserGetCurrentNode();
				coordInterp->addKeyValue(vector);
			}
			break;
		case VRML97_NORMALINTERPOLATOR:
			{
				NormalInterpolatorNode *normInterp = (NormalInterpolatorNode *)ParserGetCurrentNode();
				normInterp->addKeyValue(vector);
			}
			break;
		case VRML97_POSITIONINTERPOLATOR:
			{
				PositionInterpolatorNode *posInterp = (PositionInterpolatorNode *)ParserGetCurrentNode();
				posInterp->addKeyValue(vector);
			}
			break;
		}
		break;
	case VRML97_EXTRUSION_SPINE:
		{
			ExtrusionNode *ex = (ExtrusionNode *)ParserGetCurrentNode();
			ex->addSpine(vector);
		}
		break;
	}
}

void CyberX3D::AddSFVec2f(float vector[2])
{	
	switch (ParserGetCurrentNodeType()) {
	case VRML97_TEXTURECOODINATE:
		{
			((TextureCoordinateNode *)ParserGetCurrentNode())->addPoint(vector);
		}
	    break;
	case VRML97_EXTRUSION_CROSSSECTION:
		{
			ExtrusionNode *ex = (ExtrusionNode *)ParserGetCurrentNode();
			ex->addCrossSection(vector);
		}
		break;
	case VRML97_EXTRUSION_SCALE:
		{
			ExtrusionNode *ex = (ExtrusionNode *)ParserGetCurrentNode();
			ex->addScale(vector);
		}
		break;
	}
}

void CyberX3D::AddSFInt32(int	value)
{	
    switch (ParserGetParentNodeType()) {
    case VRML97_INDEXEDFACESET:
		{
			IndexedFaceSetNode *idxFaceSet = (IndexedFaceSetNode *)ParserGetCurrentNode();
			switch (ParserGetCurrentNodeType()) {
			case VRML97_COLOR_INDEX:
				idxFaceSet->addColorIndex(value); break;
			case VRML97_COORDINATE_INDEX:
				idxFaceSet->addCoordIndex(value); break;
			case VRML97_NORMAL_INDEX:
				idxFaceSet->addNormalIndex(value); break;
			case VRML97_TEXTURECOODINATE_INDEX:
				idxFaceSet->addTexCoordIndex(value); break;
			}
		}
	    break;
    case VRML97_INDEXEDLINESET:
		{
			IndexedLineSetNode *idxLineSet = (IndexedLineSetNode *)ParserGetCurrentNode();
			switch (ParserGetCurrentNodeType()) {
			case VRML97_COLOR_INDEX:
				idxLineSet->addColorIndex(value); break;
			case VRML97_COORDINATE_INDEX:
				idxLineSet->addCoordIndex(value); break;
			}
		}		
		break;
    case VRML97_PIXELTEXTURE:
		{
			PixelTextureNode *pixTexture = (PixelTextureNode *)ParserGetCurrentNode();
			switch (ParserGetCurrentNodeType()) {
			case VRML97_PIXELTEXTURE_IMAGE:
				pixTexture->addImage(value); break;
			}
		}	
		break;
    }

}

void CyberX3D::AddSFFloat(float value)
{	
    switch (ParserGetCurrentNodeType()) {
	case VRML97_ELEVATIONGRID_HEIGHT:
		{
			ElevationGridNode *elev = (ElevationGridNode *)ParserGetCurrentNode();
			elev->addHeight(value);
		}
		break;
    case VRML97_BACKGROUND_GROUNDANGLE:
		{
			BackgroundNode *bg = (BackgroundNode *)ParserGetCurrentNode();
			bg->addGroundAngle(value);
		}		
	    break;
    case VRML97_BACKGROUND_SKYANGLE:
		{
			BackgroundNode *bg = (BackgroundNode *)ParserGetCurrentNode();
			bg->addSkyAngle(value);
		}		
	    break;
	case VRML97_INTERPOLATOR_KEY:
		switch (ParserGetParentNodeType()) {
		case VRML97_COLORINTERPOLATOR:
			{
				ColorInterpolatorNode *colorInterp = (ColorInterpolatorNode *)ParserGetCurrentNode();
				colorInterp->addKey(value);
			}
			break;
		case VRML97_COORDINATEINTERPOLATOR:
			{
				CoordinateInterpolatorNode *coordInterp = (CoordinateInterpolatorNode *)ParserGetCurrentNode();
				coordInterp->addKey(value);
			}
			break;
		case VRML97_NORMALINTERPOLATOR:
			{
				NormalInterpolatorNode *normInterp = (NormalInterpolatorNode *)ParserGetCurrentNode();
				normInterp->addKey(value);
			}
			break;
		case VRML97_ORIENTATIONINTERPOLATOR:
			{
				OrientationInterpolatorNode *oriInterp = (OrientationInterpolatorNode *)ParserGetCurrentNode();
				oriInterp->addKey(value);
			}
			break;
		case VRML97_POSITIONINTERPOLATOR:
			{
				PositionInterpolatorNode *posInterp = (PositionInterpolatorNode *)ParserGetCurrentNode();
				posInterp->addKey(value);
			}
			break;
		case VRML97_SCALARINTERPOLATOR:
			{
				ScalarInterpolatorNode *scalarInterp = (ScalarInterpolatorNode *)ParserGetCurrentNode();
				scalarInterp->addKey(value);
			}
			break;
		}
		break;
	case VRML97_INTERPOLATOR_KEYVALUE:
		switch (ParserGetParentNodeType()) {
		case VRML97_SCALARINTERPOLATOR:
			{
				ScalarInterpolatorNode *scalarInterp = (ScalarInterpolatorNode *)ParserGetCurrentNode();
				scalarInterp->addKeyValue(value);
			}
			break;
		}
		break;
	case VRML97_LOD_RANGE:
		{
			((LODNode *)ParserGetCurrentNode())->addRange(value);
		}
		break;
	case VRML97_NAVIGATIONINFO_AVATARSIZE:
		{
			NavigationInfoNode *navInfo = (NavigationInfoNode *)ParserGetCurrentNode();
			navInfo->addAvatarSize(value);
		}
		break;
	case VRML97_TEXT_LENGTH:
		{
			TextNode *text = (TextNode *)ParserGetCurrentNode();
			text->addLength(value);
		}
		break;
    }
}


void CyberX3D::AddSFString(const char *string)
{	
	switch (ParserGetCurrentNodeType()) {
	case VRML97_ANCHOR_PARAMETER:
		{
			((AnchorNode *)ParserGetCurrentNode())->addParameter(string);
		}
		break;
	case VRML97_ANCHOR_URL:
		{
			((AnchorNode *)ParserGetCurrentNode())->addUrl(string);
		}
		break;
	case VRML97_INLINE_URL:
		{
			((InlineNode *)ParserGetCurrentNode())->addUrl(string);
		}
		break;
	case VRML97_AUDIOCLIP_URL:
		{
			AudioClipNode *aclip = (AudioClipNode *)ParserGetCurrentNode();
			aclip->addUrl(string);
		}
		break;
	case VRML97_BACKGROUND_BACKURL:
		{
			BackgroundNode *bg = (BackgroundNode *)ParserGetCurrentNode();
			bg->addBackUrl(string);
		}
		break;
	case VRML97_BACKGROUND_BOTTOMURL:
		{
			BackgroundNode *bg = (BackgroundNode *)ParserGetCurrentNode();
			bg->addBottomUrl(string);
		}
		break;
	case VRML97_BACKGROUND_FRONTURL:
		{
			BackgroundNode *bg = (BackgroundNode *)ParserGetCurrentNode();
			bg->addFrontUrl(string);
		}
		break;
	case VRML97_BACKGROUND_LEFTURL:
		{
			BackgroundNode *bg = (BackgroundNode *)ParserGetCurrentNode();
			bg->addLeftUrl(string);
		}
		break;
	case VRML97_BACKGROUND_RIGHTURL:
		{
			BackgroundNode *bg = (BackgroundNode *)ParserGetCurrentNode();
			bg->addRightUrl(string);
		}
		break;
	case VRML97_BACKGROUND_TOPURL:
		{
			BackgroundNode *bg = (BackgroundNode *)ParserGetCurrentNode();
			bg->addTopUrl(string);
		}
		break;
	case VRML97_FONTSTYLE_JUSTIFY:
		{
			FontStyleNode *fs = (FontStyleNode *)ParserGetCurrentNode();
			fs->addJustify(string);
		}
		break;
	case VRML97_IMAGETEXTURE_URL:
		{
			ImageTextureNode *image = (ImageTextureNode *)ParserGetCurrentNode();
			image->addUrl(string);
		}
		break;
	case VRML97_MOVIETEXTURE_URL:
		{
			MovieTextureNode *image = (MovieTextureNode *)ParserGetCurrentNode();
			image->addUrl(string);
		}
		break;
	case VRML97_NAVIGATIONINFO_TYPE:
		{
			NavigationInfoNode *navInfo = (NavigationInfoNode *)ParserGetCurrentNode();
			navInfo->addType(string);
		}
		break;
	case VRML97_SCRIPT_URL:
		{
			ScriptNode *script = (ScriptNode *)ParserGetCurrentNode();
			script->addUrl(string);
		}
		break;
	case VRML97_TEXT_STRING:
		{
			TextNode *text = (TextNode *)ParserGetCurrentNode();
			text->addString(string);
		}
		break;
	case VRML97_WORLDINFO_INFO:
		{
			WorldInfoNode *worldInfo = (WorldInfoNode *)ParserGetCurrentNode();
			worldInfo->addInfo(string);
		}
		break;
	}
}

/******************************************************************
*	DEF action
******************************************************************/

void CyberX3D::AddDEFInfo(
const char *name,
const char *string)
{
	((VRML97Parser *)GetParserObject())->addDEF(name, string);
}

const char *CyberX3D::GetDEFSrting(
const char *name)
{
	return ((VRML97Parser *)GetParserObject())->getDEFString(name);
}

/******************************************************************
*	PROTO action
******************************************************************/

PROTO *CyberX3D::AddPROTOInfo(
const char *name,
const char *string,
const char *fieldString)
{
	PROTO *proto = new PROTO(name, string, fieldString);
	((VRML97Parser *)GetParserObject())->addPROTO(proto);
	return proto;
}

PROTO *CyberX3D::IsPROTOName(
const char *name)
{
	return ((VRML97Parser *)GetParserObject())->getPROTO(name);
}

/******************************************************************
*	Node Name
******************************************************************/

#define	MAX_DEFNAME	512

static char	gDEFName[MAX_DEFNAME];

void CyberX3D::SetDEFName(const char *name)
{
	((VRML97Parser *)GetParserObject())->setDefName(name);
}

const char *CyberX3D::GetDEFName(void)
{
	const char *defName = ((VRML97Parser *)GetParserObject())->getDefName();
	if (defName)
		strcpy(gDEFName, defName);
	SetDEFName(NULL);
	if (defName)
		return gDEFName;
	return NULL;
}
