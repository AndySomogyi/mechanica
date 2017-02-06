/******************************************************************
*
*	CyberVRML97 for C++
*
*	Copyright (C) Satoshi Konno 1996-2002
*
*	File: vrml.y
*
*	Revisions:
*
*	11/20/02
*		- Rong Wang <wangr@acusoft.com>
*		- Changed SF*List syntax to support illegal extra commas, [x, x, x, ,-1...]
*	01/07/04
*		- Simon Goodall <sg02r@ecs.soton.ac.uk>
*		- Added GeometryInfoNode() to set Material nodes in root.
*
******************************************************************/

%union {
int		ival;
float	fval;
char	*sval;
}

%token <ival> NUMBER
%token <fval> FLOAT
%token <sval> STRING NAME

%token ANCHOR APPEARANCE AUDIOCLIP BACKGROUND BILLBOARD BOX COLLISION COLOR
%token COLOR_INTERP COORDINATE COORDINATE_INTERP CYLINDER_SENSOR NULL_STRING
%token CONE CUBE CYLINDER DIRECTIONALLIGHT FONTSTYLE ERROR EXTRUSION
%token ELEVATION_GRID FOG INLINE MOVIE_TEXTURE NAVIGATION_INFO PIXEL_TEXTURE
%token GROUP INDEXEDFACESET INDEXEDLINESET S_INFO LOD MATERIAL NORMAL
%token POSITION_INTERP PROXIMITY_SENSOR SCALAR_INTERP SCRIPT SHAPE SOUND SPOTLIGHT
%token SPHERE_SENSOR TEXT TEXTURE_COORDINATE TEXTURE_TRANSFORM TIME_SENSOR SWITCH
%token TOUCH_SENSOR VIEWPOINT VISIBILITY_SENSOR WORLD_INFO NORMAL_INTERP ORIENTATION_INTERP
%token POINTLIGHT POINTSET SPHERE PLANE_SENSOR TRANSFORM

%token S_CHILDREN S_PARAMETER S_URL S_MATERIAL S_TEXTURETRANSFORM S_TEXTURE S_LOOP
%token S_STARTTIME S_STOPTIME S_GROUNDANGLE S_GROUNDCOLOR S_SPEED S_AVATAR_SIZE
%token S_BACKURL S_BOTTOMURL S_FRONTURL S_LEFTURL S_RIGHTURL S_TOPURL S_SKYANGLE S_SKYCOLOR 
%token S_AXIS_OF_ROTATION S_COLLIDE S_COLLIDETIME S_PROXY S_SIDE S_AUTO_OFFSET S_DISK_ANGLE
%token S_ENABLED S_MAX_ANGLE S_MIN_ANGLE S_OFFSET S_BBOXSIZE S_BBOXCENTER S_VISIBILITY_LIMIT
%token S_AMBIENT_INTENSITY S_NORMAL S_TEXCOORD S_CCW S_COLOR_PER_VERTEX S_CREASE_ANGLE
%token S_NORMAL_PER_VERTEX S_XDIMENSION S_XSPACING S_ZDIMENSION S_ZSPACING S_BEGIN_CAP
%token S_CROSS_SECTION S_END_CAP S_SPINE S_FOG_TYPE S_VISIBILITY_RANGE S_HORIZONTAL S_JUSTIFY 
%token S_LANGUAGE S_LEFT2RIGHT S_TOP2BOTTOM IMAGE_TEXTURE S_SOLID S_KEY S_KEYVALUE 
%token S_REPEAT_S S_REPEAT_T S_CONVEX S_BOTTOM S_PICTH S_COORD S_COLOR_INDEX S_COORD_INDEX S_NORMAL_INDEX
%token S_MAX_POSITION S_MIN_POSITION S_ATTENUATION S_APPEARANCE S_GEOMETRY S_DIRECT_OUTPUT
%token S_MUST_EVALUATE S_MAX_BACK S_MIN_BACK S_MAX_FRONT S_MIN_FRONT S_PRIORITY S_SOURCE S_SPATIALIZE
%token S_BERM_WIDTH S_CHOICE S_WHICHCHOICE S_FONTSTYLE S_LENGTH S_MAX_EXTENT S_ROTATION S_SCALE
%token S_CYCLE_INTERVAL S_FIELD_OF_VIEW S_JUMP S_TITLE S_TEXCOORD_INDEX S_HEADLIGHT
%token S_TOP S_BOTTOMRADIUS S_HEIGHT S_POINT S_STRING S_SPACING S_SCALE S_HEADLIGHT S_TYPE
%token S_RADIUS S_ON S_INTENSITY S_COLOR S_DIRECTION S_SIZE S_FAMILY S_STYLE S_RANGE
%token S_CENTER S_TRANSLATION S_LEVEL S_DIFFUSECOLOR S_SPECULARCOLOR S_EMISSIVECOLOR S_SHININESS
%token S_TRANSPARENCY S_VECTOR S_POSITION S_ORIENTATION S_LOCATION S_ROTATION 
%token S_CUTOFFANGLE S_WHICHCHILD S_IMAGE S_SCALEORIENTATION S_DESCRIPTION  
 
%token SFBOOL SFFLOAT SFINT32 SFTIME SFROTATION SFNODE SFCOLOR SFIMAGE SFSTRING SFVEC2F SFVEC3F
%token MFBOOL MFFLOAT MFINT32 MFTIME MFROTATION MFNODE MFCOLOR MFIMAGE MFSTRING MFVEC2F MFVEC3F
%token FIELD EVENTIN EVENTOUT USE

%token S_VALUE_CHANGED

%type <fval> SFFloat SFTime
%type <ival> SFBool SFInt32
%type <sval> SFString 

%start Vrml

%{

#include <stdio.h>
#include <stdlib.h>

#ifndef __GNUC__
#define alloca	malloc
#endif

#include <x3d/SceneGraph.h>
#include <x3d/NodeType.h>
#include <x3d/VRML97Parser.h>
#include <x3d/VRML97ParserFunc.h>

using namespace CyberX3D;

static float gColor[3];
static float gVec2f[2];
static float gVec3f[3];
static float gRotation[4];
static int gWidth;
static int gHeight;
static int gComponents;

#define YYINITDEPTH  (1024 * 64)
#define	YYMAXDEPTH	(YYINITDEPTH * 128)

int yyerror(char *s);
int yyparse(void);
int yylex(void);

%} 

%%

Vrml
	: VrmlNodes
	| error		{YYABORT;}
	| ERROR		{YYABORT;}
	;

VrmlNodes
	: SFNode VrmlNodes
	|
	;

GroupingNode
	: Anchor
	| Billboard
	| Collision
	| Group
	| Inline
	| Lod
	| Switch
	| Transform
	;

InterpolatorNode
	: ColorInterp
	| CoordinateInterp
	| NormalInterp
	| OrientationInterp
	| PositionInterp
	| ScalarInterp
	;

SensorNode
	: CylinderSensor
	| PlaneSensor
	| SphereSensor
	| ProximitySensor
	| TimeSensor
	| TouchSensor
	| VisibilitySensor
	;

GeometryNode
	: Box
	| Cone
	| Cylinder
	| ElevationGrid
	| Extrusion
	| IdxFaceset
	| IdxLineset
	| Pointset
	| Sphere
	| Text
	;

GeometryInfoNode
	: Color
	| Coordinate
	| Normal
	| TexCoordinate
	| Appearance
	| Material
	;

LightNode
	: DirLight
	| SpotLight
	| PointLight
	;

CommonNode
	: AudioClip
	| LightNode
	| Script
	| Shape
	| Sound
	| WorldInfo
	;

BindableNode
	: Background
	| Fog
	| NavigationInfo
	| Viewpoint
	;

SFNode
	: CommonNode
	| BindableNode
	| FontStyle
	| InterpolatorNode
	| SensorNode
	| GroupingNode
	| GeometryInfoNode
	| USE
	;

SFInt32
	: NUMBER
		{
			AddSFInt32($1);
		}
	;

SFBool
	: NUMBER
	;

SFString
	: STRING
		{
			AddSFString($1);
		}
	;

SFFloat
	: FLOAT
		{
			AddSFFloat($1);
		}
	| NUMBER
		{
			$$ = (float)$1;
			AddSFFloat((float)$1);
		}
	;

SFTime
	: FLOAT
	| NUMBER {$$ = (float)$1;}
	;

SFColor
	: SFFloat SFFloat SFFloat 
	    {
			gColor[0] = $1;
			gColor[1] = $2;
			gColor[2] = $3;
			AddSFColor(gColor);
	    }
	;

SFRotation
	: SFFloat SFFloat SFFloat SFFloat 
	    {
			gRotation[0] = $1;
			gRotation[1] = $2;
			gRotation[2] = $3;
			gRotation[3] = $4;
			AddSFRotation(gRotation);
		}
	;

SFImageList
	: SFInt32 SFImageList {}
	|
	;


SFImageHeader
	: NUMBER NUMBER NUMBER
	    {
			gWidth = $1;
			gHeight = $2;
			gComponents = $3;
	    }
	;

SFImage
	: '[' SFImageHeader SFImageList ']'
	;

SFVec2f
	: SFFloat SFFloat 
	    {
			gVec2f[0] = $1;
			gVec2f[1] = $2;
			AddSFVec2f(gVec2f);
		}
	;

SFVec3f
	: SFFloat SFFloat SFFloat
		{
			gVec3f[0] = $1;
			gVec3f[1] = $2;
			gVec3f[2] = $3;
			AddSFVec3f(gVec3f);
		}
	;

SFColorList
	: SFColor
	| SFColorList SFColor
	;

MFColor
	: SFColor 
	| '['  ']'
	| '[' SFColorList ']'
	;

SFInt32List
	: SFInt32 {}
	|SFInt32List SFInt32 {}
	;

MFInt32
	: SFInt32 {}
	| '[' ']' {}
	| '[' SFInt32List ']' {}
	; 


SFFloatList
	: SFFloat {}
	| SFFloatList SFFloat {}
	;

MFFloat
	: SFFloat {}
	| '['  ']' {}
	| '[' SFFloatList ']' {}
	; 

SFStringList
	: SFString {}
	| SFStringList SFString {}
	;

MFString
	: SFString {}
	| '['  ']' {}
	| '[' SFStringList ']' {}
	; 

SFVec2fList
	: SFVec2f
	| SFVec2fList SFVec2f
	;

MFVec2f
	: SFVec2f
	| '[' ']'
	| '[' SFVec2fList ']'
	; 

SFVec3fList
	: SFVec3f
	| SFVec3fList SFVec3f
	;

MFVec3f
	: SFVec3f
	| '[' ']'
	| '[' SFVec3fList ']'
	; 

SFRotationList
	: SFRotation
	| SFRotationList SFRotation
	;

MFRotation
	: SFRotation
	| '[' ']'
	| '[' SFRotationList ']'
	; 

NodeBegin
	: '{'
	;

NodeEnd
	: '}'
	;

/******************************************************************
*
*	Anchor
*
******************************************************************/

AnchorElements	
	: AnchorElement AnchorElements
	|
	;

AnchorElementParameterBegin 
	: S_PARAMETER
		{
			ParserPushNode(VRML97_ANCHOR_PARAMETER, ParserGetCurrentNode());
		}
	;

AnchorElementURLBegin 
	: S_URL	
		{
			ParserPushNode(VRML97_ANCHOR_URL, ParserGetCurrentNode());
		}
	;

bboxCenter
	: S_BBOXCENTER	SFVec3f
		{
			((AnchorNode *)ParserGetCurrentNode())->setBoundingBoxCenter(gVec3f);
		}
	;

bboxSize
	: S_BBOXSIZE	SFVec3f
		{
			((AnchorNode *)ParserGetCurrentNode())->setBoundingBoxSize(gVec3f);
		}
	;

AnchorElement 
	: children
	| S_DESCRIPTION	SFString
		{
			((AnchorNode *)ParserGetCurrentNode())->setDescription($2);
		}

	| AnchorElementParameterBegin MFString 
		{
			ParserPopNode();
		}
	| AnchorElementURLBegin MFString
		{
			ParserPopNode();
		}
	| bboxCenter
	| bboxSize
	;

AnchorBegin
	: ANCHOR 
		{   
			AnchorNode	*anchor = new AnchorNode();
			anchor->setName(GetDEFName());
			ParserAddNode(anchor);
			ParserPushNode(VRML97_ANCHOR, anchor);
		}	
	;

Anchor
	: AnchorBegin NodeBegin AnchorElements NodeEnd
		{
			AnchorNode *anchor = (AnchorNode *)ParserGetCurrentNode();
			anchor->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*	Appearance
*
******************************************************************/

AppearanceNodes
	: AppearanceNode AppearanceNodes
	|
	;

AppearanceNode
	: S_MATERIAL NULL_STRING
	| S_MATERIAL Material
	| S_MATERIAL USE
	| S_TEXTURE NULL_STRING
	| S_TEXTURE ImageTexture
	| S_TEXTURE MovieTexture
	| S_TEXTURE PixelTexture
	| S_TEXTURE USE
	| S_TEXTURETRANSFORM NULL_STRING
	| S_TEXTURETRANSFORM TexTransform
	| S_TEXTURETRANSFORM USE
	;
	
AppearanceBegin
	: APPEARANCE  
		{
			AppearanceNode	*appearance = new AppearanceNode();
			appearance->setName(GetDEFName());
			ParserAddNode(appearance);
			ParserPushNode(VRML97_APPEARANCE, appearance);
		}
		;

Appearance
	:  AppearanceBegin NodeBegin AppearanceNodes NodeEnd
		{
			AppearanceNode	*appearance = (AppearanceNode *)ParserGetCurrentNode();
			appearance->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*	Audio Clip
*
******************************************************************/

AudioClipElements
	: AudioClipElement AudioClipElements
	|
	;

AudioClipURL
	: S_URL
		{
			ParserPushNode(VRML97_AUDIOCLIP_URL, ParserGetCurrentNode());
		}
	;

AudioClipElement
	: S_DESCRIPTION			SFString
		{
			((AudioClipNode *)ParserGetCurrentNode())->setDescription($2);
		}
	| S_LOOP					SFBool
		{
			((AudioClipNode *)ParserGetCurrentNode())->setLoop($2);
		}
	| S_PICTH					SFFloat
		{
			((AudioClipNode *)ParserGetCurrentNode())->setPitch($2);
		}
	| S_STARTTIME				SFTime
		{
			((AudioClipNode *)ParserGetCurrentNode())->setStartTime($2);
		}
	| S_STOPTIME				SFTime
		{
			((AudioClipNode *)ParserGetCurrentNode())->setStopTime($2);
		}
	| AudioClipURL	MFString
		{
			ParserPopNode();
		}
	;

AudioClipBegin
	: AUDIOCLIP 
		{
			AudioClipNode	*audioClip = new AudioClipNode();
			audioClip->setName(GetDEFName());
			ParserAddNode(audioClip);
			ParserPushNode(VRML97_AUDIOCLIP, audioClip);
		}
	;

AudioClip
	: AudioClipBegin NodeBegin AudioClipElements NodeEnd
		{
			AudioClipNode *audioClip = (AudioClipNode *)ParserGetCurrentNode();
			audioClip->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*	Background
*
******************************************************************/

BackGroundElements
	: BackGroundElement BackGroundElements
	|
	;

BackGroundBackURL
	: S_BACKURL
		{
			ParserPushNode(VRML97_BACKGROUND_BACKURL, ParserGetCurrentNode());
		}
	;

BackGroundBottomURL
	: S_BOTTOMURL
		{
			ParserPushNode(VRML97_BACKGROUND_BOTTOMURL, ParserGetCurrentNode());
		}
	;

BackGroundFrontURL
	: S_FRONTURL
		{
			ParserPushNode(VRML97_BACKGROUND_FRONTURL, ParserGetCurrentNode());
		}
	;

BackGroundLeftURL
	: S_LEFTURL	
		{
			ParserPushNode(VRML97_BACKGROUND_LEFTURL, ParserGetCurrentNode());
		}
	;

BackGroundRightURL
	: S_RIGHTURL
		{
			ParserPushNode(VRML97_BACKGROUND_RIGHTURL, ParserGetCurrentNode());
		}
	;

BackGroundTopURL
	: S_TOPURL
		{
			ParserPushNode(VRML97_BACKGROUND_TOPURL, ParserGetCurrentNode());
		}
	;

BackGroundGroundAngle
	: S_GROUNDANGLE
		{
			ParserPushNode(VRML97_BACKGROUND_GROUNDANGLE, ParserGetCurrentNode());
		}
	;

BackGroundGroundColor
	: S_GROUNDCOLOR
		{
			ParserPushNode(VRML97_BACKGROUND_GROUNDCOLOR, ParserGetCurrentNode());
		}
	;

BackGroundSkyAngle
	: S_SKYANGLE
		{
			ParserPushNode(VRML97_BACKGROUND_SKYANGLE, ParserGetCurrentNode());
		}
	;

BackGroundSkyColor
	: S_SKYCOLOR
		{
			ParserPushNode(VRML97_BACKGROUND_SKYCOLOR, ParserGetCurrentNode());
		}
	;

BackGroundElement
	: BackGroundGroundAngle	MFFloat
		{
			ParserPopNode();
		}
	| BackGroundGroundColor	MFColor
		{
			ParserPopNode();
		}
	| BackGroundBackURL	MFString
		{
			ParserPopNode();
		}
	| BackGroundBottomURL	MFString
		{
			ParserPopNode();
		}
	| BackGroundFrontURL	MFString
		{
			ParserPopNode();
		}
	| BackGroundLeftURL	MFString
		{
			ParserPopNode();
		}
	| BackGroundRightURL	MFString
		{
			ParserPopNode();
		}
	| BackGroundTopURL		MFString
		{
			ParserPopNode();
		}
	| BackGroundSkyAngle	MFFloat
		{
			ParserPopNode();
		}
	| BackGroundSkyColor	MFColor
		{
			ParserPopNode();
		}
	;

BackgroundBegin
	: BACKGROUND 
		{
			BackgroundNode *bg = new BackgroundNode();
			bg->setName(GetDEFName());
			ParserAddNode(bg);
			ParserPushNode(VRML97_BACKGROUND, bg);
		}
	;

Background
	: BackgroundBegin NodeBegin BackGroundElements NodeEnd
		{
			BackgroundNode *bg = (BackgroundNode *)ParserGetCurrentNode();
			bg->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*	Billboard
*
******************************************************************/

BillboardElements
	: BillboardElement BillboardElements
	|
	;

BillboardElement
	: children
	| S_AXIS_OF_ROTATION	SFVec3f
		{
			((BillboardNode *)ParserGetCurrentNode())->setAxisOfRotation(gVec3f);
		}
	| bboxCenter
	| bboxSize
	;

BillboardBegin
	: BILLBOARD 
		{   
			BillboardNode *billboard = new BillboardNode();
			billboard->setName(GetDEFName());
			ParserAddNode(billboard);
			ParserPushNode(VRML97_BILLBOARD, billboard);
		}	
	;

Billboard
	: BillboardBegin NodeBegin BillboardElements NodeEnd
		{
			BillboardNode *billboard = (BillboardNode *)ParserGetCurrentNode();
			billboard->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*	Box
*
******************************************************************/

BoxElements
	: BoxElement BoxElements
	|
	;

BoxElement
	: S_SIZE SFVec3f
		{
			((BoxNode *)ParserGetCurrentNode())->setSize(gVec3f);
		}
	;

BoxBegin
	: BOX 
		{
			BoxNode *box = new BoxNode();
			box->setName(GetDEFName());
			ParserAddNode(box);
			ParserPushNode(VRML97_BOX, box);
		}
	;

Box					
	: BoxBegin NodeBegin BoxElements NodeEnd
		{
			BoxNode *box = (BoxNode *)ParserGetCurrentNode();
			box->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*	Children
*
******************************************************************/

childrenElements
	: SFNode childrenElements
	|
	;

children
	: S_CHILDREN '[' childrenElements ']'
	| S_CHILDREN SFNode
	;

/******************************************************************
*
*	Collision
*
******************************************************************/

CollisionElements
	: CollisionElement CollisionElements
	|
	;

CollisionElementProxyBegin
	: S_PROXY
		{
			ParserPushNode(VRML97_COLLISION_PROXY, ParserGetCurrentNode());
		}
	;

CollisionElement
	: children
	| S_COLLIDE						SFBool
		{
			((CollisionNode *)ParserGetCurrentNode())->setCollide($2);
		}
	| bboxCenter
	| bboxSize
	| S_PROXY USE
	| S_PROXY NULL_STRING
	| CollisionElementProxyBegin	SFNode
		{
			ParserPopNode();							
		}
	;

CollisionBegin
	: COLLISION 
		{   
			CollisionNode *collision = new CollisionNode();
			collision->setName(GetDEFName());
			ParserAddNode(collision);
			ParserPushNode(VRML97_BOX, collision);
		}	
	;

Collision
	: CollisionBegin NodeBegin CollisionElements NodeEnd
		{
			CollisionNode *collision = (CollisionNode *)ParserGetCurrentNode();
			collision->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*	Color
*
******************************************************************/

ColorElements
	: ColorElement ColorElements
	|
	;

ColorElement
	: S_COLOR MFColor 				
	;

ColorBegin
	: COLOR  
		{
			ColorNode *color = new ColorNode();
			color->setName(GetDEFName());
			ParserAddNode(color);
			ParserPushNode(VRML97_COLOR, color);
		}
	;

Color
	: ColorBegin NodeBegin ColorElements NodeEnd
		{
			ColorNode *color = (ColorNode *)ParserGetCurrentNode();
			color->initialize();
			ParserPopNode();
		}
	;
		                                                                                                                                                                                                                                                                                          
/******************************************************************
*
*	ColorInterpolator
*
******************************************************************/

ColorInterpElements
	: ColorInterpElement ColorInterpElements
	|
	;

InterpolateKey
	: S_KEY
		{
			ParserPushNode(VRML97_INTERPOLATOR_KEY, ParserGetCurrentNode());
		}
	;

InterporlateKeyValue
	: S_KEYVALUE
		{
			ParserPushNode(VRML97_INTERPOLATOR_KEYVALUE, ParserGetCurrentNode());
		}
	;

ColorInterpElement
	: InterpolateKey		MFFloat
		{
			ParserPopNode();
		}
	| InterporlateKeyValue	MFColor
		{
			ParserPopNode();
		}
	;

ColorInterpBegin
	: COLOR_INTERP  
		{
			ColorInterpolatorNode *colInterp = new ColorInterpolatorNode();
			colInterp->setName(GetDEFName());
			ParserAddNode(colInterp);
			ParserPushNode(VRML97_COLORINTERPOLATOR, colInterp);
		}
	;

ColorInterp
	: ColorInterpBegin NodeBegin ColorInterpElements NodeEnd
		{
			ColorInterpolatorNode *colInterp = (ColorInterpolatorNode *)ParserGetCurrentNode();
			colInterp->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*   Cone
*
******************************************************************/

ConeElements
	: ConeElement ConeElements
	|
	;

ConeElement
	: S_SIDE			SFBool
		{
			((ConeNode *)ParserGetCurrentNode())->setSide($2);
		}
	| S_BOTTOM		SFBool
		{
			((ConeNode *)ParserGetCurrentNode())->setBottom($2);
		}
	| S_BOTTOMRADIUS	SFFloat
		{
			((ConeNode *)ParserGetCurrentNode())->setBottomRadius($2);
		}
	| S_HEIGHT		SFFloat
		{
			((ConeNode *)ParserGetCurrentNode())->setHeight($2);
		}
	;

ConeBegin
	: CONE 
		{
			ConeNode *cone = new ConeNode();
			cone->setName(GetDEFName());
			ParserAddNode(cone);
			ParserPushNode(VRML97_CONE, cone);
		}
	;

Cone
	: ConeBegin NodeBegin ConeElements NodeEnd
		{
			ConeNode *cone = (ConeNode *)ParserGetCurrentNode();
			cone->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*   Coordinate
*
******************************************************************/

CoordinateElements
	:  S_POINT	MFVec3f
	|
	;

CoordinateBegin
	: COORDINATE 
		{
			CoordinateNode *coord = new CoordinateNode();
			coord->setName(GetDEFName());
			ParserAddNode(coord);
			ParserPushNode(VRML97_COORDINATE, coord);
		}
	;

Coordinate
	: CoordinateBegin NodeBegin CoordinateElements NodeEnd
		{
			CoordinateNode *coord = (CoordinateNode *)ParserGetCurrentNode();
			coord->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*	ColorInterpolator
*
******************************************************************/

CoordinateInterpElements
	: CoordinateInterpElement CoordinateInterpElements
	|
	;

CoordinateInterpElement
	: InterpolateKey		MFFloat
		{
			ParserPopNode();
		}
	| InterporlateKeyValue	MFVec3f
		{
			ParserPopNode();
		}
	;

CoordinateInterpBegin
	: COORDINATE_INTERP  
		{
			CoordinateInterpolatorNode *coordInterp = new CoordinateInterpolatorNode();
			coordInterp->setName(GetDEFName());
			ParserAddNode(coordInterp);
			ParserPushNode(VRML97_COORDINATEINTERPOLATOR, coordInterp);
		}
	;

CoordinateInterp
	: CoordinateInterpBegin NodeBegin CoordinateInterpElements NodeEnd
		{
			CoordinateInterpolatorNode *coordInterp = (CoordinateInterpolatorNode *)ParserGetCurrentNode();
			coordInterp->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*   Cylinder
*
******************************************************************/

CylinderElements		
	: CylinderElement CylinderElements
	|
	;

CylinderElement
	: S_SIDE		SFBool
		{
			((CylinderNode *)ParserGetCurrentNode())->setSide($2);
		}
	| S_BOTTOM		SFBool
		{
			((CylinderNode *)ParserGetCurrentNode())->setBottom($2);
		}
	| S_TOP		SFBool
		{
			((CylinderNode *)ParserGetCurrentNode())->setTop($2);
		}
	| S_RADIUS		SFFloat
		{
			((CylinderNode *)ParserGetCurrentNode())->setRadius($2);
		}
	| S_HEIGHT		SFFloat
		{
			((CylinderNode *)ParserGetCurrentNode())->setHeight($2);
		}
	;

CylinderBegin
	: CYLINDER  
		{
			CylinderNode *cylinder = new CylinderNode();
			cylinder->setName(GetDEFName());
			ParserAddNode(cylinder);
			ParserPushNode(VRML97_CYLINDER, cylinder);
		}
	;

Cylinder
	: CylinderBegin NodeBegin CylinderElements NodeEnd
		{
			CylinderNode *cylinder = (CylinderNode *)ParserGetCurrentNode();
			cylinder->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*   CylinderSensor
*
******************************************************************/

CylinderSensorElements
	: CylinderSensorElement CylinderSensorElements
	|
	;

CylinderSensorElement
	: S_AUTO_OFFSET			SFBool
		{
			((CylinderSensorNode *)ParserGetCurrentNode())->setAutoOffset($2);
		}
	| S_DISK_ANGLE			SFFloat
		{
			((CylinderSensorNode *)ParserGetCurrentNode())->setDiskAngle($2);
		}
	| S_ENABLED				SFBool
		{
			((CylinderSensorNode *)ParserGetCurrentNode())->setEnabled($2);
		}
	| S_MAX_ANGLE				SFFloat
		{
			((CylinderSensorNode *)ParserGetCurrentNode())->setMaxAngle($2);
		}
	| S_MIN_ANGLE				SFFloat
		{
			((CylinderSensorNode *)ParserGetCurrentNode())->setMinAngle($2);
		}
	| S_OFFSET				SFFloat
		{
			((CylinderSensorNode *)ParserGetCurrentNode())->setOffset($2);
		}
	;


CylinderSensorBegin
	: CYLINDER_SENSOR 
		{
			CylinderSensorNode *cysensor = new CylinderSensorNode();
			cysensor->setName(GetDEFName());
			ParserAddNode(cysensor);
			ParserPushNode(VRML97_CYLINDERSENSOR, cysensor);
		}
	;

CylinderSensor
	: CylinderSensorBegin NodeBegin CylinderSensorElements NodeEnd
		{
			CylinderSensorNode *cysensor = (CylinderSensorNode *)ParserGetCurrentNode();
			cysensor->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*   Directional Light
*
******************************************************************/

DirLightElements		
	: DirLightElement DirLightElements
	|
	;

DirLightElement
	: S_ON				SFBool
		{
			((DirectionalLightNode *)ParserGetCurrentNode())->setOn($2);
		}
	| S_INTENSITY			SFFloat
		{
			((DirectionalLightNode *)ParserGetCurrentNode())->setIntensity($2);
		}
	| S_COLOR		SFColor
		{
			((DirectionalLightNode *)ParserGetCurrentNode())->setColor(gColor);
		}
	| S_DIRECTION			SFVec3f
		{
			((DirectionalLightNode *)ParserGetCurrentNode())->setDirection(gVec3f);
		}
	| S_AMBIENT_INTENSITY	SFFloat
		{
			((DirectionalLightNode *)ParserGetCurrentNode())->setAmbientIntensity($2);
		}
	;

DirLightBegin			
	: DIRECTIONALLIGHT 
		{
			DirectionalLightNode *dirLight = new DirectionalLightNode();
			dirLight->setName(GetDEFName());
			ParserAddNode(dirLight);
			ParserPushNode(VRML97_DIRECTIONALLIGHT, dirLight);
		}
	;

DirLight
	: DirLightBegin NodeBegin DirLightElements NodeEnd
		{
			DirectionalLightNode *dirLight = (DirectionalLightNode *)ParserGetCurrentNode();
			dirLight->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*   ElevationGrid
*
******************************************************************/

ElevationGridElements
	: ElevationGridElement ElevationGridElements
	|
	;

ElevationGridHeight
	: S_HEIGHT
		{
			ParserPushNode(VRML97_ELEVATIONGRID_HEIGHT, ParserGetCurrentNode());
		}
	;


ElevationGridElement
	: S_COLOR					NULL_STRING
	| S_COLOR					Color
	| S_COLOR					USE
	| S_NORMAL					NULL_STRING
	| S_NORMAL					Normal
	| S_NORMAL					USE
	| S_TEXCOORD				NULL_STRING
	| S_TEXCOORD				TexCoordinate
	| S_TEXCOORD				USE
	| ElevationGridHeight		MFFloat
		{
			ParserPopNode();
		}
	| S_CCW 				SFBool
		{
			((ElevationGridNode *)ParserGetCurrentNode())->setCCW($2);
		}
	| S_CREASE_ANGLE		SFFloat
		{
			((ElevationGridNode *)ParserGetCurrentNode())->setCreaseAngle($2);
		}
	| S_SOLID				SFBool
		{
			((ElevationGridNode *)ParserGetCurrentNode())->setSolid($2);
		}
	| S_COLOR_PER_VERTEX	SFBool
		{
			((ElevationGridNode *)ParserGetCurrentNode())->setColorPerVertex($2);
		}
	| S_NORMAL_PER_VERTEX	SFBool
		{
			((ElevationGridNode *)ParserGetCurrentNode())->setNormalPerVertex($2);
		}
	| S_XDIMENSION		SFInt32
		{
			((ElevationGridNode *)ParserGetCurrentNode())->setXDimension($2);
		}
	| S_XSPACING			SFFloat
		{
			((ElevationGridNode *)ParserGetCurrentNode())->setXSpacing($2);
		}
	| S_ZDIMENSION		SFInt32
		{
			((ElevationGridNode *)ParserGetCurrentNode())->setZDimension($2);
		}
	| S_ZSPACING			SFFloat
		{
			((ElevationGridNode *)ParserGetCurrentNode())->setZSpacing($2);
		}
	;

ElevationGridBegin
	: ELEVATION_GRID 
		{
			ElevationGridNode *elev = new ElevationGridNode();
			elev->setName(GetDEFName());
			ParserAddNode(elev);
			ParserPushNode(VRML97_ELEVATIONGRID, elev);
		}
	;

ElevationGrid
	: ElevationGridBegin NodeBegin ElevationGridElements NodeEnd
		{
			ElevationGridNode *elev = (ElevationGridNode *)ParserGetCurrentNode();
			elev->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*   Extrusion
*
******************************************************************/

ExtrusionElements
	: ExtrusionElement ExtrusionElements
	|
	;

ExtrusionCrossSection
	: S_CROSS_SECTION
		{
			ParserPushNode(VRML97_EXTRUSION_CROSSSECTION, ParserGetCurrentNode());
		}
	;

ExtrusionOrientation
	: S_ORIENTATION
		{
			ParserPushNode(VRML97_EXTRUSION_ORIENTATION, ParserGetCurrentNode());
		}
	;

ExtrusionScale
	: S_SCALE
		{
			ParserPushNode(VRML97_EXTRUSION_SCALE, ParserGetCurrentNode());
		}
	;

ExtrusionSpine
	: S_SPINE
		{
			ParserPushNode(VRML97_EXTRUSION_SPINE, ParserGetCurrentNode());
		}
	;

ExtrusionElement
	: S_BEGIN_CAP			SFBool
		{
			((ExtrusionNode *)ParserGetCurrentNode())->setBeginCap($2);
		}
	| S_CCW					SFBool 
		{
			((ExtrusionNode *)ParserGetCurrentNode())->setCCW($2);
		}
	| S_CONVEX				SFBool
		{
			((ExtrusionNode *)ParserGetCurrentNode())->setConvex($2);
		}
	| S_CREASE_ANGLE		SFFloat
		{
			((ExtrusionNode *)ParserGetCurrentNode())->setCreaseAngle($2);
		}
	| S_SOLID				SFBool
		{
			((ExtrusionNode *)ParserGetCurrentNode())->setSolid($2);
		}
	| ExtrusionCrossSection MFVec2f
		{
			ParserPopNode();
		}
	| S_END_CAP			SFBool
		{
			((ExtrusionNode *)ParserGetCurrentNode())->setEndCap($2);
		}
	| ExtrusionOrientation	MFRotation
		{
			ParserPopNode();
		}
	| ExtrusionScale MFVec2f
		{
			ParserPopNode();
		}
	| ExtrusionSpine MFVec3f
		{
			ParserPopNode();
		}
	;

ExtrusionBegin
	: EXTRUSION  
		{
			ExtrusionNode *ex = new ExtrusionNode();
			ex->setName(GetDEFName());
			ParserAddNode(ex);
			ParserPushNode(VRML97_EXTRUSION, ex);
		}
	;

Extrusion
	: ExtrusionBegin NodeBegin ExtrusionElements NodeEnd
		{
			ExtrusionNode *ex = (ExtrusionNode *)ParserGetCurrentNode();
			ex->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*   Fog
*
******************************************************************/

FogElements
	: FogElement FogElements
	|
	;

FogElement
	: S_COLOR		SFColor
		{
			((FogNode *)ParserGetCurrentNode())->setColor(gColor);
		}
	| S_FOG_TYPE			SFString
		{
			((FogNode *)ParserGetCurrentNode())->setFogType($2);
		}
	| S_VISIBILITY_RANGE			SFFloat
		{
			((FogNode *)ParserGetCurrentNode())->setVisibilityRange($2);
		}
	;

FogBegin
	: FOG  
		{
			FogNode *fog= new FogNode();
			fog->setName(GetDEFName());
			ParserAddNode(fog);
			ParserPushNode(VRML97_FOG, fog);
		}
	;

Fog
	: FogBegin NodeBegin FogElements NodeEnd
		{
			FogNode *fog= (FogNode *)ParserGetCurrentNode();
			fog->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*   Font Style 
*
******************************************************************/

FontStyleElements
	: FontStyleElement FontStyleElements
	|
	;

FontStyleJustify
	: S_JUSTIFY 
		{
			ParserPushNode(VRML97_FONTSTYLE_JUSTIFY, ParserGetCurrentNode());
		}
	;

FontStyleElement
	: S_FAMILY		SFString
		{
			((FontStyleNode *)ParserGetCurrentNode())->setFamily($2);
		}
	| S_HORIZONTAL	SFBool
		{
			((FontStyleNode *)ParserGetCurrentNode())->setHorizontal($2);
		}
	| FontStyleJustify		MFString
		{
			ParserPopNode();
		}
	| S_LANGUAGE	SFString
		{
			((FontStyleNode *)ParserGetCurrentNode())->setLanguage($2);
		}
	| S_LEFT2RIGHT	SFBool
		{
			((FontStyleNode *)ParserGetCurrentNode())->setLeftToRight($2);
		}
	| S_SIZE		SFFloat
		{
			((FontStyleNode *)ParserGetCurrentNode())->setSize($2);
		}
	| S_SPACING		SFFloat
		{
			((FontStyleNode *)ParserGetCurrentNode())->setSpacing($2);
		}
	| S_STYLE			SFString
		{
			((FontStyleNode *)ParserGetCurrentNode())->setStyle($2);
		}
	| S_TOP2BOTTOM	SFBool
		{
			((FontStyleNode *)ParserGetCurrentNode())->setTopToBottom($2);
		}
	;

FontStyleBegin	
	: FONTSTYLE NodeBegin
		{
			FontStyleNode *fs = new FontStyleNode();
			fs->setName(GetDEFName());
			ParserAddNode(fs);
			ParserPushNode(VRML97_FONTSTYLE, fs);
		}
	;

FontStyle		
	: FontStyleBegin FontStyleElements NodeEnd
		{
			FontStyleNode *fs = (FontStyleNode *)ParserGetCurrentNode();
			fs->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*   Group
*
******************************************************************/

GroupElements
	: GroupElement GroupElements
	|
	;

GroupElement
	: children
	| bboxCenter
	| bboxSize
	;

GroupBegin
	: GROUP 
		{   
			GroupNode *group = new GroupNode();
			group->setName(GetDEFName());
			ParserAddNode(group);
			ParserPushNode(VRML97_GROUP, group);
		}	
	;

Group
	: GroupBegin NodeBegin GroupElements NodeEnd
		{
			GroupNode *group = (GroupNode *)ParserGetCurrentNode();
			group->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*   ImageTexture
*
******************************************************************/

ImgTexElements
	: ImgTexElement ImgTexElements
	|
	;

ImgTexURL
	: S_URL
		{
			ParserPushNode(VRML97_IMAGETEXTURE_URL, ParserGetCurrentNode());
		}
	;

ImgTexElement
	: ImgTexURL	MFString
		{
			ParserPopNode();
		}
	| S_REPEAT_S			SFBool
		{
			((ImageTextureNode *)ParserGetCurrentNode())->setRepeatS($2);
		}
	| S_REPEAT_T			SFBool
		{
			((ImageTextureNode *)ParserGetCurrentNode())->setRepeatT($2);
		}
	;

ImageTextureBegin
	: IMAGE_TEXTURE 
		{
			ImageTextureNode *imgTexture = new ImageTextureNode();
			imgTexture->setName(GetDEFName());
			ParserAddNode(imgTexture);
			ParserPushNode(VRML97_IMAGETEXTURE, imgTexture);
		}
	;

ImageTexture
	: ImageTextureBegin NodeBegin ImgTexElements NodeEnd
		{
			ImageTextureNode *imgTexture = (ImageTextureNode *)ParserGetCurrentNode();
			imgTexture->initialize();
			ParserPopNode();
		} 
	;

/******************************************************************
*
*   Indexed Face set
*
******************************************************************/

IdxFacesetElements
	: IdxFacesetElement IdxFacesetElements
	|
	;

ColorIndex	
	: S_COLOR_INDEX
		{
			ParserPushNode(VRML97_COLOR_INDEX, ParserGetCurrentNode());
		}
	;

CoordIndex	
	: S_COORD_INDEX
		{
			ParserPushNode(VRML97_COORDINATE_INDEX, ParserGetCurrentNode());
		}
	;

NormalIndex
	: S_NORMAL_INDEX
		{
			ParserPushNode(VRML97_NORMAL_INDEX, ParserGetCurrentNode());
		}
	;

TextureIndex
	: S_TEXCOORD_INDEX
	    {
			ParserPushNode(VRML97_TEXTURECOODINATE_INDEX, ParserGetCurrentNode());
		}
	;

IdxFacesetElement
	: S_COLOR			NULL_STRING
	| S_COLOR			Color
	| S_COLOR			USE
	| S_COORD			NULL_STRING
	| S_COORD			Coordinate
	| S_COORD			USE
	| S_NORMAL			NULL_STRING
	| S_NORMAL			Normal
	| S_NORMAL			USE
	| S_TEXCOORD		NULL_STRING
	| S_TEXCOORD		TexCoordinate
	| S_TEXCOORD		USE
	| S_CCW				SFBool
		{
			((IndexedFaceSetNode *)ParserGetCurrentNode())->setCCW($2);
		}
	| S_CONVEX			SFBool
		{
			((IndexedFaceSetNode *)ParserGetCurrentNode())->setConvex($2);
		}
	| S_SOLID			SFBool
		{
			((IndexedFaceSetNode *)ParserGetCurrentNode())->setSolid($2);
		}
	| S_CREASE_ANGLE	SFFloat
		{
			((IndexedFaceSetNode *)ParserGetCurrentNode())->setCreaseAngle($2);
		}
	| ColorIndex	MFInt32
		{
			ParserPopNode();
		}
	| S_COLOR_PER_VERTEX	SFBool
		{
			((IndexedFaceSetNode *)ParserGetCurrentNode())->setColorPerVertex($2);
		}
	| CoordIndex	MFInt32
		{
			ParserPopNode();
		}
	| NormalIndex		MFInt32
		{
			ParserPopNode();
		}
	| TextureIndex		MFInt32
		{
			ParserPopNode();
		}
	| S_NORMAL_PER_VERTEX	SFBool
		{
			((IndexedFaceSetNode *)ParserGetCurrentNode())->setNormalPerVertex($2);
		}
	;

IdxFacesetBegin
	: INDEXEDFACESET  
		{
			IndexedFaceSetNode	*idxFaceset = new IndexedFaceSetNode();
			idxFaceset->setName(GetDEFName());
			ParserAddNode(idxFaceset);
			ParserPushNode(VRML97_INDEXEDFACESET, idxFaceset);
		}
	;

IdxFaceset
	: IdxFacesetBegin NodeBegin IdxFacesetElements NodeEnd
		{
			IndexedFaceSetNode *idxFaceset = (IndexedFaceSetNode *)ParserGetCurrentNode();
			idxFaceset->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*   Indexed Line set
*
******************************************************************/

IdxLinesetElements
	: IdxLinesetElement IdxLinesetElements
	|
	;

IdxLinesetElement
	: S_COLOR				NULL_STRING
	| S_COLOR				Color
	| S_COLOR				USE
	| S_COORD				NULL_STRING
	| S_COORD				Coordinate
	| S_COORD				USE
	| S_COLOR_PER_VERTEX	SFBool
		{
			((IndexedLineSetNode *)ParserGetCurrentNode())->setColorPerVertex($2);
		}
	| ColorIndex		MFInt32
		{
			ParserPopNode();
		}
	| CoordIndex		MFInt32
		{
			ParserPopNode();
		}
	;

IdxLinesetBegin	
	: INDEXEDLINESET NodeBegin 
		{
			IndexedLineSetNode	*idxLineset = new IndexedLineSetNode();
			idxLineset->setName(GetDEFName());
			ParserAddNode(idxLineset);
			ParserPushNode(VRML97_INDEXEDLINESET, idxLineset);
		}
	;

IdxLineset		
	: IdxLinesetBegin IdxLinesetElements NodeEnd
		{
			IndexedLineSetNode *idxLineset = (IndexedLineSetNode *)ParserGetCurrentNode();
			idxLineset->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*   Inline
*
******************************************************************/

InlineElements		
	: InlineElement InlineElements
	|
	;

InlineURL 
	: S_URL	
		{
			ParserPushNode(VRML97_INLINE_URL, ParserGetCurrentNode());
		}
	;

InlineElement
	: InlineURL	MFString
		{
			ParserPopNode();
		}
	| bboxCenter
	| bboxSize
	;

InlineBegin
	: INLINE
		{   
			InlineNode *inlineNode = new InlineNode();
			inlineNode->setName(GetDEFName());
			ParserAddNode(inlineNode);
			ParserPushNode(VRML97_INLINE, inlineNode);
		}	
	;

Inline
	: InlineBegin NodeBegin InlineElements NodeEnd
		{
			InlineNode *inlineNode = (InlineNode *)ParserGetCurrentNode();
			//inlineNode->initialize();
			ParserPopNode();
		}
	;

/************************************************************
*
*   LOD
*
************************************************************/

LodElements		
	: LodElement LodElements
	|
	;

LodRange
	:  S_RANGE
		{
			ParserPushNode(VRML97_LOD_RANGE, ParserGetCurrentNode());
		}
	;


LodLevel
	: S_LEVEL
		{
			ParserPushNode(VRML97_LOD_LEVEL, ParserGetCurrentNode());
		}
	;

LodElement
	: LodRange	    MFFloat
		{
			ParserPopNode();							
		}
	| S_CENTER			SFVec3f
		{
			((LODNode *)ParserGetCurrentNode())->setCenter(gVec3f);
		}
	| LodLevel	SFNode
		{
			ParserPopNode();							
		}
	| LodLevel		'[' VrmlNodes ']'
		{
			ParserPopNode();							
		}
	;

LodBegin
	: LOD
		{   
			LODNode	*lod = new LODNode();
			lod->setName(GetDEFName());
			ParserAddNode(lod);
			ParserPushNode(VRML97_INLINE, lod);
		}	
	;

Lod				
	: LodBegin NodeBegin LodElements NodeEnd
		{
			LODNode	*lod = (LODNode *)ParserGetCurrentNode();
			lod->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*   Material
*
******************************************************************/

MaterialElements	
	: MaterialElement MaterialElements
	|
	;

MaterialElement	
	: S_AMBIENT_INTENSITY	SFFloat
		{
			((MaterialNode *)ParserGetCurrentNode())->setAmbientIntensity($2);
		}
	| S_DIFFUSECOLOR		SFColor
		{
			((MaterialNode *)ParserGetCurrentNode())->setDiffuseColor(gColor);
		}
	| S_EMISSIVECOLOR		SFColor
		{
			((MaterialNode *)ParserGetCurrentNode())->setEmissiveColor(gColor);
		}
	| S_SHININESS			SFFloat
		{
			((MaterialNode *)ParserGetCurrentNode())->setShininess($2);
		}
	| S_SPECULARCOLOR		SFColor
		{
			((MaterialNode *)ParserGetCurrentNode())->setSpecularColor(gColor);
		}
	| S_TRANSPARENCY		SFFloat
		{
			((MaterialNode *)ParserGetCurrentNode())->setTransparency($2);
		}
	;

MaterialBegin	
	: MATERIAL 
		{
			MaterialNode *material = new MaterialNode();
			material->setName(GetDEFName());
			ParserAddNode(material);
			ParserPushNode(VRML97_MATERIAL, material);
		}
	;

Material
	: MaterialBegin NodeBegin MaterialElements NodeEnd
		{
			MaterialNode *material = (MaterialNode *)ParserGetCurrentNode();
			material->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*	MovieTexture
*
******************************************************************/

MovieTextureElements
	: MovieTextureElement MovieTextureElements
	|
	;

MovieTextureURL
	: S_URL
		{
			ParserPushNode(VRML97_MOVIETEXTURE_URL, ParserGetCurrentNode());
		}
	;

MovieTextureElement	
	: S_LOOP				SFBool
		{
			((MovieTextureNode *)ParserGetCurrentNode())->setLoop($2);
		}
	| S_SPEED				SFFloat
		{
			((MovieTextureNode *)ParserGetCurrentNode())->setSpeed($2);
		}
	| S_STARTTIME			SFTime
		{
			((MovieTextureNode *)ParserGetCurrentNode())->setStartTime($2);
		}
	| S_STOPTIME			SFTime
		{
			((MovieTextureNode *)ParserGetCurrentNode())->setStopTime($2);
		}
	| MovieTextureURL MFString
		{
			ParserPopNode();
		}
	| S_REPEAT_S			SFBool
		{
			((MovieTextureNode *)ParserGetCurrentNode())->setRepeatS($2);
		}
	| S_REPEAT_T			SFBool
		{
			((MovieTextureNode *)ParserGetCurrentNode())->setRepeatT($2);
		}
	;

MovieTextureBegin
	: MOVIE_TEXTURE  
		{
			MovieTextureNode *movieTexture = new MovieTextureNode();
			movieTexture->setName(GetDEFName());
			ParserAddNode(movieTexture);
			ParserPushNode(VRML97_MOVIETEXTURE, movieTexture);
		}
	;

MovieTexture
	: MovieTextureBegin NodeBegin MovieTextureElements NodeEnd
		{
			MovieTextureNode *movieTexture = (MovieTextureNode *)ParserGetCurrentNode();
			movieTexture->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*	Navigation Info
*
******************************************************************/

NavigationInfoElements
	: NavigationInfoElement NavigationInfoElements
	|
	;

NavigationInfoAvatarSize
	: S_AVATAR_SIZE
		{
			ParserPushNode(VRML97_NAVIGATIONINFO_AVATARSIZE, ParserGetCurrentNode());
		}
	;

NavigationInfoType
	: S_TYPE
		{
			ParserPushNode(VRML97_NAVIGATIONINFO_TYPE, ParserGetCurrentNode());
		}
	;

NavigationInfoElement
	: NavigationInfoAvatarSize	MFFloat
		{
			ParserPopNode();
		}
	| S_HEADLIGHT						SFBool
		{
			((NavigationInfoNode *)ParserGetCurrentNode())->setHeadlight($2);
		}
	| S_SPEED							SFFloat
		{
			((NavigationInfoNode *)ParserGetCurrentNode())->setSpeed($2);
		}
	| NavigationInfoType		MFString
		{
			ParserPopNode();
		}
	| S_VISIBILITY_LIMIT				SFFloat
		{
			((NavigationInfoNode *)ParserGetCurrentNode())->setVisibilityLimit($2);
		}
	;

NavigationInfoBegin
	: NAVIGATION_INFO
		{
			NavigationInfoNode *navInfo = new NavigationInfoNode();
			navInfo->setName(GetDEFName());
			ParserAddNode(navInfo);
			ParserPushNode(VRML97_NAVIGATIONINFO, navInfo);
		}
	;

NavigationInfo		
	: NavigationInfoBegin NodeBegin NavigationInfoElements NodeEnd
		{
			NavigationInfoNode *navInfo = (NavigationInfoNode *)ParserGetCurrentNode();
			navInfo->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*   Normal
*
******************************************************************/

NormalElements	
	: NormalElement NormalElements
	|
	;

NormalElement
	: S_VECTOR	MFVec3f
	;

NormalBegin
	: NORMAL  
		{
			NormalNode *normal = new NormalNode();
			normal->setName(GetDEFName());
			ParserAddNode(normal);
			ParserPushNode(VRML97_NORMAL, normal);
		}
	;

Normal
	: NormalBegin NodeBegin NormalElements NodeEnd
		{
			NormalNode *normal = (NormalNode *)ParserGetCurrentNode();
			normal->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*	Normal Interpolator
*
******************************************************************/

NormalInterpElements	
	: NormalInterpElement NormalInterpElements
	|
	;

NormalInterpElement
	: InterpolateKey			MFFloat
		{
			ParserPopNode();
		}
	| InterporlateKeyValue		MFVec3f
		{
			ParserPopNode();
		}
	| S_VALUE_CHANGED			SFVec3f
		{
		}
	;

NormalInterpBegin
	: NORMAL_INTERP
		{
			NormalInterpolatorNode *normInterp = new NormalInterpolatorNode();
			normInterp->setName(GetDEFName());
			ParserAddNode(normInterp);
			ParserPushNode(VRML97_NORMALINTERPOLATOR, normInterp);
		}
	;

NormalInterp
	: NormalInterpBegin NodeBegin	NormalInterpElements NodeEnd
		{
			NormalInterpolatorNode *normInterp = (NormalInterpolatorNode *)ParserGetCurrentNode();
			normInterp->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*	Orientation Interpolator
*
******************************************************************/

OrientationInterpElements
	: OrientationInterpElement OrientationInterpElements
	|
	;

OrientationInterpElement
	: InterpolateKey			MFFloat
		{
			ParserPopNode();
		}
	| InterporlateKeyValue		MFRotation
		{
			ParserPopNode();
		}
	| S_VALUE_CHANGED			SFRotation
		{
		}
	;

OrientationInterpBegin
	: ORIENTATION_INTERP
		{
			OrientationInterpolatorNode *oriInterp = new OrientationInterpolatorNode();
			oriInterp->setName(GetDEFName());
			ParserAddNode(oriInterp);
			ParserPushNode(VRML97_ORIENTATIONINTERPOLATOR, oriInterp);
		}
	;

OrientationInterp
	: OrientationInterpBegin NodeBegin OrientationInterpElements NodeEnd
		{
			OrientationInterpolatorNode *oriInterp = (OrientationInterpolatorNode *)ParserGetCurrentNode();
			oriInterp->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*	Pixel Texture
*
******************************************************************/

PixelTextureElements
	: PixelTextureElement PixelTextureElements
	|
	;

PixelTextureImage
	: S_IMAGE  
		{
			ParserPushNode(VRML97_PIXELTEXTURE_IMAGE, ParserGetCurrentNode());
		}
	;

PixelTextureElement
	: PixelTextureImage	'[' SFImageList ']'
		{
			ParserPopNode();
		}
	| S_REPEAT_S		SFBool
		{
			((PixelTextureNode *)ParserGetCurrentNode())->setRepeatS($2);
		}
	| S_REPEAT_T		SFBool
		{
			((PixelTextureNode *)ParserGetCurrentNode())->setRepeatT($2);
		}
	;

PixelTextureBegin
	: PIXEL_TEXTURE 
		{
			PixelTextureNode *pixTexture = new PixelTextureNode();
			pixTexture->setName(GetDEFName());
			ParserAddNode(pixTexture);
			ParserPushNode(VRML97_PIXELTEXTURE, pixTexture);
		}
	;

PixelTexture		
	: PixelTextureBegin NodeBegin PixelTextureElements NodeEnd
		{
			PixelTextureNode *pixTexture = (PixelTextureNode *)ParserGetCurrentNode();
			pixTexture->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*	Plane Sensor
*
******************************************************************/

PlaneSensorElements
	: PlaneSensorElement PlaneSensorElements
	|
	;

PlaneSensorElement
	: S_AUTO_OFFSET	SFBool
		{
			((PlaneSensorNode *)ParserGetCurrentNode())->setAutoOffset($2);
		}
	| S_ENABLED		SFBool
		{
			((PlaneSensorNode *)ParserGetCurrentNode())->setEnabled($2);
		}
	| S_MAX_POSITION	SFVec2f
		{
			((PlaneSensorNode *)ParserGetCurrentNode())->setMaxPosition(gVec2f);
		}
	| S_MIN_POSITION	SFVec2f
		{
			((PlaneSensorNode *)ParserGetCurrentNode())->setMinPosition(gVec2f);
		}
	| S_OFFSET		SFVec3f
		{
			((PlaneSensorNode *)ParserGetCurrentNode())->setOffset(gVec3f);
		}
	;

PlaneSensorBegin
	: PLANE_SENSOR
		{
			PlaneSensorNode *psensor = new PlaneSensorNode();
			psensor->setName(GetDEFName());
			ParserAddNode(psensor);
			ParserPushNode(VRML97_PLANESENSOR, psensor);
		}
	;

PlaneSensor
	: PlaneSensorBegin NodeBegin PlaneSensorElements NodeEnd
		{
			PlaneSensorNode *psensor = (PlaneSensorNode *)ParserGetCurrentNode();
			psensor->initialize();
			ParserPopNode();
		}
	;


/******************************************************************
*
*   Point Light
*
******************************************************************/

PointLightNodes
	: PointLightNode PointLightNodes
	|
	;

PointLightNode
	: S_AMBIENT_INTENSITY	SFFloat
		{
			((PointLightNode *)ParserGetCurrentNode())->setAmbientIntensity($2);
		}
	| S_ATTENUATION		SFVec3f
		{
			((PointLightNode *)ParserGetCurrentNode())->setAttenuation(gVec3f);
		}
	| S_COLOR		SFColor
		{
			((PointLightNode *)ParserGetCurrentNode())->setColor(gColor);
		}
	| S_INTENSITY	SFFloat
		{
			((PointLightNode *)ParserGetCurrentNode())->setIntensity($2);
		}
	| S_LOCATION	SFVec3f
		{
			((PointLightNode *)ParserGetCurrentNode())->setLocation(gVec3f);
		}
	| S_ON		SFBool
		{
			((PointLightNode *)ParserGetCurrentNode())->setOn($2);
		}
	| S_RADIUS	SFFloat
		{
			((PointLightNode *)ParserGetCurrentNode())->setRadius($2);
		}
	;

PointLightBegin	
	: POINTLIGHT  
		{
			PointLightNode *pointLight = new PointLightNode();
			pointLight->setName(GetDEFName());
			ParserAddNode(pointLight);
			ParserPushNode(VRML97_POINTLIGHT, pointLight);
		}
	;

PointLight
	: PointLightBegin NodeBegin PointLightNodes NodeEnd
		{
			PointLightNode *pointLight = (PointLightNode *)ParserGetCurrentNode();
			pointLight->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*   Point set
*
******************************************************************/

PointsetElements		
	: PointsetElement PointsetElements
	|
	;

PointsetElement
	: S_COLOR	NULL_STRING
	| S_COLOR	Color
	| S_COLOR	USE
	| S_COORD	NULL_STRING
	| S_COORD	Coordinate
	| S_COORD	USE
	;


PointsetBegin
	: POINTSET
		{
			PointSetNode *pset = new PointSetNode();
			pset->setName(GetDEFName());
			ParserAddNode(pset);
			ParserPushNode(VRML97_POINTSET, pset);
		}
	;

Pointset	
	: PointsetBegin NodeBegin PointsetElements NodeEnd
		{
			PointSetNode *pset = (PointSetNode *)ParserGetCurrentNode();
			pset->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*	Position Interpolator
*
******************************************************************/

PositionInterpElements
	: PositionInterpElement PositionInterpElements
	|
	;

PositionInterpElement
	: InterpolateKey			MFFloat
		{
			ParserPopNode();
		}
	| InterporlateKeyValue		MFVec3f
		{
			ParserPopNode();
		}
	| S_VALUE_CHANGED			SFVec3f
		{
		}
	;

PositionInterpBegin
	: POSITION_INTERP
		{
			PositionInterpolatorNode *posInterp = new PositionInterpolatorNode();
			posInterp->setName(GetDEFName());
			ParserAddNode(posInterp);
			ParserPushNode(VRML97_POSITIONINTERPOLATOR, posInterp);
		}
	;

PositionInterp
	: PositionInterpBegin NodeBegin PositionInterpElements NodeEnd
		{
			PositionInterpolatorNode *posInterp = (PositionInterpolatorNode *)ParserGetCurrentNode();
			posInterp->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*	Proximity Sensor
*
******************************************************************/

ProximitySensorElements
	: ProximitySensorElement ProximitySensorElements
	|
	;

ProximitySensorElement
	: S_CENTER		SFVec3f
		{
			((ProximitySensorNode *)ParserGetCurrentNode())->setCenter(gVec3f);
		}
	| S_SIZE		SFVec3f
		{
			((ProximitySensorNode *)ParserGetCurrentNode())->setSize(gVec3f);
		}
	| S_ENABLED		SFBool
		{
			((ProximitySensorNode *)ParserGetCurrentNode())->setEnabled($2);
		}
	;

ProximitySensorBegin
	: PROXIMITY_SENSOR
		{
			ProximitySensorNode *psensor = new ProximitySensorNode();
			psensor->setName(GetDEFName());
			ParserAddNode(psensor);
			ParserPushNode(VRML97_PROXIMITYSENSOR, psensor);
		}
	;

ProximitySensor		
	: ProximitySensorBegin NodeBegin ProximitySensorElements NodeEnd
		{
			ProximitySensorNode *psensor = (ProximitySensorNode *)ParserGetCurrentNode();
			psensor->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*	Scalar Interpolator
*
******************************************************************/

ScalarInterpElements	
	: ScalarInterpElement ScalarInterpElements
	|
	;

ScalarInterpElement
	: InterpolateKey			MFFloat
		{
			ParserPopNode();
		}
	| InterporlateKeyValue		MFFloat
		{
			ParserPopNode();
		}
	| S_VALUE_CHANGED			SFVec2f
		{
		}
	;

ScalarInterpBegin
	: SCALAR_INTERP
		{
			ScalarInterpolatorNode *scalarInterp = new ScalarInterpolatorNode();
			scalarInterp->setName(GetDEFName());
			ParserAddNode(scalarInterp);
			ParserPushNode(VRML97_SCALARINTERPOLATOR, scalarInterp);
		}
	;

ScalarInterp
	: ScalarInterpBegin NodeBegin ScalarInterpElements NodeEnd
		{
			ScalarInterpolatorNode *scalarInterp = (ScalarInterpolatorNode *)ParserGetCurrentNode();
			scalarInterp->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*	Script
*
******************************************************************/

ScriptElements
	: ScriptElement ScriptElements
	|
	;

ScriptURL
	: S_URL
		{
			ParserPushNode(VRML97_SCRIPT_URL, ParserGetCurrentNode());
		}
	;

ScriptElement
	: ScriptURL	MFString
		{
			ParserPopNode();
		}
	| S_DIRECT_OUTPUT		SFBool
		{
			((ScriptNode *)ParserGetCurrentNode())->setDirectOutput($2);
		}
	| S_MUST_EVALUATE		SFBool
		{
			((ScriptNode *)ParserGetCurrentNode())->setMustEvaluate($2);
		}

	/*********************************************************
	*	eventIn (SFNode)
	*********************************************************/
	
	| EVENTIN	SFBOOL		NAME
		{
			SFBool *value = new SFBool();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn($3, value);
			delete[] $3;
		}
	| EVENTIN	SFFLOAT		NAME
		{
			SFFloat *value = new SFFloat();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn($3, value);
			delete[] $3;
		}
	| EVENTIN	SFINT32		NAME
		{
			SFInt32 *value = new SFInt32();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn($3, value);
			delete[] $3;
		}
	| EVENTIN	SFTIME		NAME
		{
			SFTime *value = new SFTime();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn($3, value);
			delete[] $3;
		}
	| EVENTIN	SFROTATION	NAME
		{
			SFRotation *value = new SFRotation();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn($3, value);
			delete[] $3;
		}
/* 
	| EVENTIN	SFNODE		NAME
		{
			Node *value = new Node();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn($3, value);
			delete[] $3;
		}
*/
	| EVENTIN	SFCOLOR		NAME
		{
			SFColor *value = new SFColor();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn($3, value);
			delete[] $3;
		}
	| EVENTIN	SFIMAGE		NAME
		{
			SFImage *value = new SFImage();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn($3, value);
			delete[] $3;
		}
	| EVENTIN	SFSTRING	NAME
		{
			SFString *value = new SFString();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn($3, value);
			delete[] $3;
		}
	| EVENTIN	SFVEC2F		NAME
		{
			SFVec2f *value = new SFVec2f();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn($3, value);
			delete[] $3;
		}
	| EVENTIN	SFVEC3F		NAME
		{
			SFVec3f *value = new SFVec3f();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn($3, value);
			delete[] $3;
		}

	/*********************************************************
	*	eventIn (MFNode)
	*********************************************************/
	
	| EVENTIN	MFFLOAT		NAME
		{
			MFFloat *value = new MFFloat();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn($3, value);
			delete[] $3;
		}
	| EVENTIN	MFINT32		NAME
		{
			MFInt32 *value = new MFInt32();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn($3, value);
			delete[] $3;
		}
	| EVENTIN	MFTIME		NAME
		{
			MFTime *value = new MFTime();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn($3, value);
			delete[] $3;
		}
	| EVENTIN	MFROTATION	NAME
		{
			MFRotation *value = new MFRotation();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn($3, value);
			delete[] $3;
		}
/* 
	| EVENTIN	MFNODE		NAME
		{
			Node *value = new Node();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn($3, value);
			delete[] $3;
		}
*/
	| EVENTIN	MFCOLOR		NAME
		{
			MFColor *value = new MFColor();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn($3, value);
			delete[] $3;
		}
	| EVENTIN	MFSTRING	NAME
		{
			MFString *value = new MFString();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn($3, value);
			delete[] $3;
		}
	| EVENTIN	MFVEC2F		NAME
		{
			MFVec2f *value = new MFVec2f();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn($3, value);
			delete[] $3;
		}
	| EVENTIN	MFVEC3F		NAME
		{
			MFVec3f *value = new MFVec3f();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn($3, value);
			delete[] $3;
		}

	/*********************************************************
	*	eventOut (SFNode)
	*********************************************************/
	
	| EVENTOUT	SFBOOL		NAME
		{
			SFBool *value = new SFBool();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut($3, value);
			delete[] $3;
		}
	| EVENTOUT	SFFLOAT		NAME
		{
			SFFloat *value = new SFFloat();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut($3, value);
			delete[] $3;
		}
	| EVENTOUT	SFINT32		NAME
		{
			SFInt32 *value = new SFInt32();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut($3, value);
			delete[] $3;
		}
	| EVENTOUT	SFTIME		NAME
		{
			SFTime *value = new SFTime();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut($3, value);
			delete[] $3;
		}
	| EVENTOUT	SFROTATION	NAME
		{
			SFRotation *value = new SFRotation();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut($3, value);
			delete[] $3;
		}
/* 
	| EVENTOUT	SFNODE		NAME
		{
			Node *value = new Node();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut($3, value);
			delete[] $3;
		}
*/
	| EVENTOUT	SFCOLOR		NAME
		{
			SFColor *value = new SFColor();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut($3, value);
			delete[] $3;
		}
	| EVENTOUT	SFIMAGE		NAME
		{
			SFImage *value = new SFImage();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut($3, value);
			delete[] $3;
		}
	| EVENTOUT	SFSTRING	NAME
		{
			SFString *value = new SFString();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut($3, value);
			delete[] $3;
		}
	| EVENTOUT	SFVEC2F		NAME
		{
			SFVec2f *value = new SFVec2f();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut($3, value);
			delete[] $3;
		}
	| EVENTOUT	SFVEC3F		NAME
		{
			SFVec3f *value = new SFVec3f();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut($3, value);
			delete[] $3;
		}

	/*********************************************************
	*	eventOut (MFNode)
	*********************************************************/
	
	| EVENTOUT	MFFLOAT		NAME
		{
			MFFloat *value = new MFFloat();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut($3, value);
			delete[] $3;
		}
	| EVENTOUT	MFINT32		NAME
		{
			MFInt32 *value = new MFInt32();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut($3, value);
			delete[] $3;
		}
	| EVENTOUT	MFTIME		NAME
		{
			MFTime *value = new MFTime();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut($3, value);
			delete[] $3;
		}
	| EVENTOUT	MFROTATION	NAME
		{
			MFRotation *value = new MFRotation();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut($3, value);
			delete[] $3;
		}
/* 
	| EVENTOUT	MFNODE		NAME
		{
			Node *value = new Node();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut($3, value);
			delete[] $3;
		}
*/
	| EVENTOUT	MFCOLOR		NAME
		{
			MFColor *value = new MFColor();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut($3, value);
			delete[] $3;
		}
	| EVENTOUT	MFSTRING	NAME
		{
			MFString *value = new MFString();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut($3, value);
			delete[] $3;
		}
	| EVENTOUT	MFVEC2F		NAME
		{
			MFVec2f *value = new MFVec2f();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut($3, value);
			delete[] $3;
		}
	| EVENTOUT	MFVEC3F		NAME
		{
			MFVec3f *value = new MFVec3f();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut($3, value);
			delete[] $3;
		}

	/*********************************************************
	*	field (SFNode)
	*********************************************************/
	
	| FIELD	SFBOOL		NAME	SFBool
		{
			SFBool *value = new SFBool($4);
			((ScriptNode *)ParserGetCurrentNode())->addField($3, value);
			delete[] $3;
		}
	| FIELD	SFFLOAT		NAME	SFFloat
		{
			SFFloat *value = new SFFloat($4);
			((ScriptNode *)ParserGetCurrentNode())->addField($3, value);
			delete[] $3;
		}
	| FIELD	SFINT32		NAME	SFInt32
		{
			SFInt32 *value = new SFInt32($4);
			((ScriptNode *)ParserGetCurrentNode())->addField($3, value);
			delete[] $3;
		}
	| FIELD	SFTIME		NAME	SFTime
		{
			SFTime *value = new SFTime($4);
			((ScriptNode *)ParserGetCurrentNode())->addField($3, value);
			delete[] $3;
		}
	| FIELD	SFROTATION	NAME	SFRotation
		{
			SFRotation *value = new SFRotation(gRotation);
			((ScriptNode *)ParserGetCurrentNode())->addField($3, value);
			delete[] $3;
		}
 
	| FIELD	SFNODE		NAME	NULL_STRING
		{
			SFNode *value = new SFNode();
			((ScriptNode *)ParserGetCurrentNode())->addField($3, value);
			delete[] $3;
		}

	| FIELD	SFNODE		NAME	USE		NAME
		{
			Node *node = GetParserObject()->findNode($5);
			SFNode *value = new SFNode(node);
			((ScriptNode *)ParserGetCurrentNode())->addField($3, value);
			delete[] $3; delete[] $5;
		}

	| FIELD	SFCOLOR		NAME	SFColor
		{
			SFColor *value = new SFColor(gColor);
			((ScriptNode *)ParserGetCurrentNode())->addField($3, value);
			delete[] $3;
		}
/*
	| FIELD	SFIMAGE		NAME	SFImage
		{
			SFImage *value = new SFImage($4);
			((ScriptNode *)ParserGetCurrentNode())->addField($3, value);
			delete[] $3;
		}
*/
	| FIELD	SFSTRING	NAME	SFString
		{
			SFString *value = new SFString($4);
			((ScriptNode *)ParserGetCurrentNode())->addField($3, value);
			delete[] $3;
		}
	| FIELD	SFVEC2F		NAME	SFVec2f
		{
			SFVec2f *value = new SFVec2f(gVec2f);
			((ScriptNode *)ParserGetCurrentNode())->addField($3, value);
			delete[] $3;
		}
	| FIELD	SFVEC3F		NAME	SFVec3f
		{
			SFVec3f *value = new SFVec3f(gVec3f);
			((ScriptNode *)ParserGetCurrentNode())->addField($3, value);
			delete[] $3;
		}

	;

ScriptBegin
	: SCRIPT
		{
			ScriptNode *script = new ScriptNode();
			script->setName(GetDEFName());
			ParserAddNode(script);
			ParserPushNode(VRML97_SCRIPT, script);
		}
	;

Script
	: ScriptBegin NodeBegin ScriptElements NodeEnd
		{
			ScriptNode *script = (ScriptNode *)ParserGetCurrentNode();
			script->initialize();
			ParserPopNode();
		}
	;
		

/******************************************************************
*
*	Shape
*
******************************************************************/

SharpElements
	: SharpElement SharpElements
	|
	;

SharpElement
	: S_APPEARANCE		NULL_STRING
	| S_APPEARANCE		Appearance
	| S_APPEARANCE		USE
	| S_GEOMETRY		NULL_STRING
	| S_GEOMETRY		GeometryNode
	| S_GEOMETRY		USE
	;

ShapeBegin
	: SHAPE  
		{
			ShapeNode *shape = new ShapeNode();
			shape->setName(GetDEFName());
			ParserAddNode(shape);
			ParserPushNode(VRML97_SHAPE, shape);
		}
	;

Shape
	: ShapeBegin NodeBegin SharpElements NodeEnd
		{
			ShapeNode *shape = (ShapeNode *)ParserGetCurrentNode();
			shape->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*	Sound
*
******************************************************************/

SoundElements
	: SoundElement SoundElements
	|
	;

SoundElement
	: S_DIRECTION			SFVec3f
		{
			((SoundNode *)ParserGetCurrentNode())->setDirection(gVec3f);
		}
	| S_INTENSITY			SFFloat
		{
			((SoundNode *)ParserGetCurrentNode())->setIntensity($2);
		}
	| S_LOCATION			SFVec3f
		{
			((SoundNode *)ParserGetCurrentNode())->setLocation(gVec3f);
		}
	| S_MAX_BACK			SFFloat
		{
			((SoundNode *)ParserGetCurrentNode())->setMinBack($2);
		}
	| S_MAX_FRONT			SFFloat
		{
			((SoundNode *)ParserGetCurrentNode())->setMaxFront($2);
		}
	| S_MIN_BACK			SFFloat
		{
			((SoundNode *)ParserGetCurrentNode())->setMinBack($2);
		}
	| S_MIN_FRONT			SFFloat
		{
			((SoundNode *)ParserGetCurrentNode())->setMinFront($2);
		}
	| S_PRIORITY			SFFloat
		{
			((SoundNode *)ParserGetCurrentNode())->setPriority($2);
		}
	| S_SOURCE			NULL_STRING
	| S_SOURCE			AudioClip
	| S_SOURCE			MovieTexture
	| S_SOURCE			USE
	| S_SPATIALIZE		SFBool
		{
			((SoundNode *)ParserGetCurrentNode())->setSpatialize($2);
		}
	;

SoundBegin
	: SOUND
		{
			SoundNode *sound = new SoundNode();
			sound->setName(GetDEFName());
			ParserAddNode(sound);
			ParserPushNode(VRML97_SOUND, sound);
		}
	;

Sound
	: SoundBegin NodeBegin SoundElements NodeEnd
		{
			SoundNode *sound = (SoundNode *)ParserGetCurrentNode();
			sound->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*   Sphere
*
******************************************************************/

SphereElements
	: SphereElement SphereElements
	|
	;

SphereElement
	: S_RADIUS	SFFloat
		{
			((SphereNode *)ParserGetCurrentNode())->setRadius($2);
		}
	;

SphereBegin	
	: SPHERE  
		{
			SphereNode *sphere = new SphereNode();
			sphere->setName(GetDEFName());
			ParserAddNode(sphere);
			ParserPushNode(VRML97_SPHERE, sphere);
		}
	;

Sphere
	: SphereBegin NodeBegin SphereElements NodeEnd
		{
			SphereNode *sphere = (SphereNode *)ParserGetCurrentNode();
			sphere->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*	Spehere Sensor
*
******************************************************************/

SphereSensorElements
	: SphereSensorElement SphereSensorElements
	|
	;

SphereSensorElement
	: S_AUTO_OFFSET	SFBool
		{
			((SphereSensorNode *)ParserGetCurrentNode())->setAutoOffset($2);
		}
	| S_ENABLED		SFBool
		{
			((SphereSensorNode *)ParserGetCurrentNode())->setEnabled($2);
		}
	| S_OFFSET		SFRotation
		{
			((SphereSensorNode *)ParserGetCurrentNode())->setOffset(gRotation);
		}
	;

SphereSensorBegin
	: SPHERE_SENSOR
		{
			SphereSensorNode *spsensor = new SphereSensorNode();
			spsensor->setName(GetDEFName());
			ParserAddNode(spsensor);
			ParserPushNode(VRML97_SPHERESENSOR, spsensor);
		}
	;

SphereSensor
	: SphereSensorBegin NodeBegin SphereSensorElements NodeEnd
		{
			SphereSensorNode *spsensor = (SphereSensorNode *)ParserGetCurrentNode();
			spsensor->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*   Spot Light
*
******************************************************************/

SpotLightElements	
	: SpotLightElement SpotLightElements
	|
	;

SpotLightElement	
	: S_AMBIENT_INTENSITY	SFFloat
		{
			((SpotLightNode *)ParserGetCurrentNode())->setAmbientIntensity($2);
		}
	| S_ATTENUATION		SFVec3f
		{
			((SpotLightNode *)ParserGetCurrentNode())->setAttenuation(gVec3f);
		}
	| S_BERM_WIDTH		SFFloat
		{
			((SpotLightNode *)ParserGetCurrentNode())->setBeamWidth($2);
		}
	| S_COLOR		SFColor
		{
			((SpotLightNode *)ParserGetCurrentNode())->setColor(gColor);
		}
	| S_CUTOFFANGLE		SFFloat
		{
			((SpotLightNode *)ParserGetCurrentNode())->setCutOffAngle($2);
		}
	| S_DIRECTION			SFVec3f
		{
			((SpotLightNode *)ParserGetCurrentNode())->setDirection(gVec3f);
		}
	| S_INTENSITY			SFFloat
		{
			((SpotLightNode *)ParserGetCurrentNode())->setIntensity($2);
		}
	| S_LOCATION			SFVec3f
		{
			((SpotLightNode *)ParserGetCurrentNode())->setLocation(gVec3f);
		}
	| S_ON				SFBool
		{
			((SpotLightNode *)ParserGetCurrentNode())->setOn($2);
		}
	| S_RADIUS			SFFloat
		{
			((SpotLightNode *)ParserGetCurrentNode())->setRadius($2);
		}
	;

SpotLightBegin
	: SPOTLIGHT 
		{
			SpotLightNode *spotLight = new SpotLightNode();
			spotLight->setName(GetDEFName());
			ParserAddNode(spotLight);
			ParserPushNode(VRML97_SPOTLIGHT, spotLight);
		}
	;

SpotLight		
	: SpotLightBegin NodeBegin SpotLightElements NodeEnd
		{
			SpotLightNode *spotLight = (SpotLightNode *)ParserGetCurrentNode();
			spotLight->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*   Switch
*
******************************************************************/

SwitchElements	
	: SwitchElement SwitchElements
	|
	;

SwitchChoice
	: S_CHOICE
		{
			ParserPushNode(VRML97_SWITCH_CHOICE, ParserGetCurrentNode());
		}
	;

SwitchElement
	: SwitchChoice	SFNode
		{
			ParserPopNode();							
		}
	| SwitchChoice '[' VrmlNodes ']'
		{
			ParserPopNode();							
		}
	| S_WHICHCHOICE	SFInt32
		{
			((SwitchNode *)ParserGetCurrentNode())->setWhichChoice($2);
		}
	;


SwitchBegin
	: SWITCH
		{   
			SwitchNode *switchNode = new SwitchNode();
			switchNode->setName(GetDEFName());
			ParserAddNode(switchNode);
			ParserPushNode(VRML97_SWITCH, switchNode);
		}	
	;

Switch			
	: SwitchBegin NodeBegin SwitchElements NodeEnd
		{
			SwitchNode *switchNode = (SwitchNode *)ParserGetCurrentNode();
			switchNode->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*   Text
*
******************************************************************/

TextElements
	: TextElement TextElements
	|
	;

TextString
	: S_STRING
		{
			ParserPushNode(VRML97_TEXT_STRING, ParserGetCurrentNode());
		}
	;

TextLength
	: S_LENGTH
		{
			ParserPushNode(VRML97_TEXT_LENGTH, ParserGetCurrentNode());
		}
	;

TextElement
	: TextString	MFString
		{
			ParserPopNode();
		}
	| S_FONTSTYLE	NULL_STRING
	| S_FONTSTYLE	FontStyle
	| S_FONTSTYLE	USE
	| TextLength	MFFloat
		{
			ParserPopNode();
		}
	| S_MAX_EXTENT	SFFloat
		{
			((TextNode *)ParserGetCurrentNode())->setMaxExtent($2);
		}
	;


TextBegin
	: TEXT
		{
			TextNode *text = new TextNode();
			text->setName(GetDEFName());
			ParserAddNode(text);
			ParserPushNode(VRML97_TEXT, text);
		}
	;

Text
	: TextBegin NodeBegin TextElements NodeEnd
		{
			TextNode *text = (TextNode *)ParserGetCurrentNode();
			text->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*   TexCoordinate
*
******************************************************************/

TexCoordElements	
	: TexCoordElement TexCoordElements
	|
	;

TexCoordElement
	: S_POINT			MFVec2f
	;


TexCoordBegin
	: TEXTURE_COORDINATE  
		{
			TextureCoordinateNode *texCoord = new TextureCoordinateNode();
			texCoord->setName(GetDEFName());
			ParserAddNode(texCoord);
			ParserPushNode(VRML97_TEXTURECOODINATE, texCoord);
		}
	;

TexCoordinate
	: TexCoordBegin NodeBegin TexCoordElements NodeEnd
		{
			TextureCoordinateNode *texCoord = (TextureCoordinateNode *)ParserGetCurrentNode();
			texCoord->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*   TextureTransform
*
******************************************************************/

TextureTransformElements
	: TextureTransformElement TextureTransformElements
	|
	;

TextureTransformElement
	: S_CENTER			SFVec2f
		{
			((TextureTransformNode *)ParserGetCurrentNode())->setCenter(gVec2f);
		}
	| S_ROTATION		SFFloat
		{
			((TextureTransformNode *)ParserGetCurrentNode())->setRotation($2);
		}
	| S_SCALE			SFVec2f
		{
			((TextureTransformNode *)ParserGetCurrentNode())->setScale(gVec2f);
		}
	| S_TRANSLATION		SFVec2f
		{
			((TextureTransformNode *)ParserGetCurrentNode())->setTranslation(gVec2f);
		}
	;


TexTransformBegin
	: TEXTURE_TRANSFORM 
		{
			TextureTransformNode *textureTransform = new TextureTransformNode();
			textureTransform->setName(GetDEFName());
			ParserAddNode(textureTransform);
			ParserPushNode(VRML97_TEXTURETRANSFORM, textureTransform);
		}
	;

TexTransform
	: TexTransformBegin NodeBegin TextureTransformElements NodeEnd
		{
			TextureTransformNode *textureTransform = (TextureTransformNode *)ParserGetCurrentNode();
			textureTransform->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*   TimeSensor
*
******************************************************************/

TimeSensorElements	
	: TimeSensorElement TimeSensorElements
	|
	;

TimeSensorElement
	: S_CYCLE_INTERVAL	SFTime
		{
			((TimeSensorNode *)ParserGetCurrentNode())->setCycleInterval($2);
		}
	| S_ENABLED			SFBool
		{
			((TimeSensorNode *)ParserGetCurrentNode())->setEnabled($2);
		}
	| S_LOOP			SFBool
		{
			((TimeSensorNode *)ParserGetCurrentNode())->setLoop($2);
		}
	| S_STARTTIME		SFTime
		{
			((TimeSensorNode *)ParserGetCurrentNode())->setStartTime($2);
		}
	| S_STOPTIME		SFTime
		{
			((TimeSensorNode *)ParserGetCurrentNode())->setStopTime($2);
		}
	;


TimeSensorBegin
	: TIME_SENSOR
		{
			TimeSensorNode *tsensor = new TimeSensorNode();
			tsensor->setName(GetDEFName());
			ParserAddNode(tsensor);
			ParserPushNode(VRML97_TIMESENSOR, tsensor);
		}
	;

TimeSensor
	: TimeSensorBegin NodeBegin TimeSensorElements NodeEnd
		{
			TimeSensorNode *tsensor = (TimeSensorNode *)ParserGetCurrentNode();
			tsensor->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*   TouchSensor
*
******************************************************************/

TouchSensorElements	
	: TouchSensorElement TouchSensorElements
	|
	;

TouchSensorElement
	: S_ENABLED			SFBool
		{
			((TouchSensorNode *)ParserGetCurrentNode())->setEnabled($2);
		}
	;

TouchSensorBegin
	: TOUCH_SENSOR
		{
			TouchSensorNode *touchSensor = new TouchSensorNode();
			touchSensor->setName(GetDEFName());
			ParserAddNode(touchSensor);
			ParserPushNode(VRML97_TOUCHSENSOR, touchSensor);
		}
	;

TouchSensor
	: TouchSensorBegin NodeBegin TouchSensorElements NodeEnd
		{
			TouchSensorNode *touchSensor = (TouchSensorNode *)ParserGetCurrentNode();
			touchSensor->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*  Transform
*
******************************************************************/

TransformElements	
	: TransformElement TransformElements
	|
	;

TransformElement 
	: children
	| S_CENTER			SFVec3f
		{
			((TransformNode *)ParserGetCurrentNode())->setCenter(gVec3f);
		}
	| S_ROTATION		SFRotation
		{
			((TransformNode *)ParserGetCurrentNode())->setRotation(gRotation);
		}
	| S_SCALE			SFVec3f
		{
			((TransformNode *)ParserGetCurrentNode())->setScale(gVec3f);
		}
	| S_SCALEORIENTATION	SFRotation
	    {
			((TransformNode *)ParserGetCurrentNode())->setScaleOrientation(gRotation);
		}
	| S_TRANSLATION		SFVec3f
		{
			((TransformNode *)ParserGetCurrentNode())->setTranslation(gVec3f);
		}
	| bboxCenter
	| bboxSize
	;

TransformBegin
	: TRANSFORM 
		{
			TransformNode *transform = new TransformNode();
			transform->setName(GetDEFName());
			ParserAddNode(transform);
			ParserPushNode(VRML97_TRANSFORM, transform);
		}
	;

Transform
	: TransformBegin NodeBegin TransformElements NodeEnd
		{
			TransformNode *transform = (TransformNode *)ParserGetCurrentNode();
			transform->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*	Viewpoint
*
******************************************************************/

ViewpointElements		
	: ViewpointElement ViewpointElements
	|
	;

ViewpointElement
	: S_FIELD_OF_VIEW			SFFloat
		{
			((ViewpointNode *)ParserGetCurrentNode())->setFieldOfView($2);
		}
	| S_JUMP					SFBool
		{
			((ViewpointNode *)ParserGetCurrentNode())->setJump($2);
		}
	| S_ORIENTATION			SFRotation
		{
			((ViewpointNode *)ParserGetCurrentNode())->setOrientation(gRotation);
		}
	| S_POSITION				SFVec3f
		{
			((ViewpointNode *)ParserGetCurrentNode())->setPosition(gVec3f);
		}
	| S_DESCRIPTION			SFString
		{
			((ViewpointNode *)ParserGetCurrentNode())->setDescription($2);
		}
	;

ViewpointBegin
	: VIEWPOINT 
		{
			ViewpointNode *viewpoint = new ViewpointNode();
			viewpoint->setName(GetDEFName());
			ParserAddNode(viewpoint);
			ParserPushNode(VRML97_VIEWPOINT, viewpoint);
		}
	;

Viewpoint 	
	: ViewpointBegin NodeBegin ViewpointElements NodeEnd
		{
			ViewpointNode *viewpoint = (ViewpointNode *)ParserGetCurrentNode();
			viewpoint->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*	VisibilitySensor
*
******************************************************************/

VisibilitySensors
	: VisibilitySensor VisibilitySensors
	|
	;

VisibilitySensor		
	: S_CENTER				SFVec3f
		{
			((VisibilitySensorNode *)ParserGetCurrentNode())->setCenter(gVec3f);
		}
	| S_ENABLED				SFBool
		{
			((VisibilitySensorNode *)ParserGetCurrentNode())->setEnabled($2);
		}
	| S_SIZE				SFVec3f
		{
			((VisibilitySensorNode *)ParserGetCurrentNode())->setSize(gVec3f);
		}
	;

VisibilitySensorBegine
	: VISIBILITY_SENSOR
		{
			VisibilitySensorNode *vsensor = new VisibilitySensorNode();
			vsensor->setName(GetDEFName());
			ParserAddNode(vsensor);
			ParserPushNode(VRML97_VISIBILITYSENSOR, vsensor);
		}
	;

VisibilitySensor	
	: VisibilitySensorBegine NodeBegin VisibilitySensors NodeEnd
		{
			VisibilitySensorNode *vsensor = (VisibilitySensorNode *)ParserGetCurrentNode();
			vsensor->initialize();
			ParserPopNode();
		}
	;

/******************************************************************
*
*	WorldInfo
*
******************************************************************/

WorldInfoElements		
	: WorldInfoElement WorldInfoElements
	|
	;

WorldInfoInfo
	: S_INFO
		{
			ParserPushNode(VRML97_WORLDINFO_INFO, ParserGetCurrentNode());
		}
	;

WorldInfoElement
	: WorldInfoInfo	MFString
		{
			ParserPopNode();
		}
	| S_TITLE					SFString
		{
			((WorldInfoNode *)ParserGetCurrentNode())->setTitle($2);
		}
	;

WorldInfoBegin
	: WORLD_INFO 
		{
			WorldInfoNode *worldInfo = new WorldInfoNode();
			worldInfo->setName(GetDEFName());
			ParserAddNode(worldInfo);
			ParserPushNode(VRML97_WORLDINFO, worldInfo);
		}
	;

WorldInfo			
	: WorldInfoBegin NodeBegin WorldInfoElements NodeEnd
		{
			WorldInfoNode *worldInfo = (WorldInfoNode *)ParserGetCurrentNode();
			worldInfo->initialize();
			ParserPopNode();
		}
	;

%%
