%{
/******************************************************************
*
*	CyberVRML97 for C++
*
*	Copyright (C) Satoshi Konno 1996-2002
*
*	File:	vrml.l
*
******************************************************************/
 
#include <stdio.h>
#include <string.h>
#include <string>
#include <sstream>
#include <x3d/VRML97Parser.h>
#include <x3d/VRML97ParserFunc.h>
#include <x3d/VRML97.tab.h>

using namespace CyberX3D;

static const char *in_buffer = 0;
static int in_size = 0;
static int	nCurrentLine = 1;
static std::string lineBuffer;
static std::string name;
static void (*callbackFn)(int nLine, void *info);
static void *callbackFnInfo;

#define input	yyinput
#define output	yyoutput

%}

%start	SCRIPTNODE
%option interactive
%option always-interactive

%%

-?[0-9]+								{ yylval.ival = atoi(yytext); return NUMBER;}
-?[0-9]*\.?[0-9]*([eE][+-]?[0-9]+)?		{ yylval.fval = (float)atof(yytext); return FLOAT;}
0x[0-9A-Fa-f]+							{ sscanf(yytext, "0x%x", &yylval.ival); return NUMBER; }

"}"					{ BEGIN INITIAL;	return '}'; }
"{"					{ return '{'; }
"["					{ return '['; }
"]"					{ return ']'; }

[,\r \t]+				;
#.*\n				{ CurrentLineIncrement();}
#.*					{ }
\n.*				{ lineBuffer = (yytext+1); CurrentLineIncrement(); yyless(1);}
\"[^\"]*\"			{ yytext[yyleng-1] = '\0'; yylval.sval = yytext+1;  return STRING; }

TRUE				{ yylval.ival = 1; return NUMBER; }
FALSE				{ yylval.ival = 0; return NUMBER; }
NULL				{ return NULL_STRING; }

<INITIAL>Anchor				{ return ANCHOR; }
<INITIAL>Appearance			{ return APPEARANCE; }
<INITIAL>AudioClip			{ return AUDIOCLIP; }
<INITIAL>Background			{ return BACKGROUND; }
<INITIAL>Billboard			{ return BILLBOARD; }
<INITIAL>Box				{ return BOX; }
<INITIAL>Collision			{ return COLLISION; }
<INITIAL>Color				{ return COLOR; }
<INITIAL>ColorInterpolator	{ return COLOR_INTERP; }
<INITIAL>Cone				{ return CONE; }
<INITIAL>Coordinate			{ return COORDINATE; }
<INITIAL>CoordinateInterpolator	{ return COORDINATE_INTERP; }
<INITIAL>Cylinder			{ return CYLINDER; }
<INITIAL>CylinderSensor		{ return CYLINDER_SENSOR; }
<INITIAL>DirectionalLight	{ return DIRECTIONALLIGHT; }
<INITIAL>ElevationGrid		{ return ELEVATION_GRID; }
<INITIAL>Extrusion			{ return EXTRUSION; }
<INITIAL>Fog				{ return FOG; }
<INITIAL>FontStyle			{ return FONTSTYLE; }
<INITIAL>Group				{ return GROUP; }
<INITIAL>ImageTexture		{ return IMAGE_TEXTURE; }
<INITIAL>IndexedFaceSet		{ return INDEXEDFACESET; }
<INITIAL>IndexedLineSet		{ return INDEXEDLINESET; }
<INITIAL>Inline				{ return INLINE; }
<INITIAL>LOD				{ return LOD; }
<INITIAL>Material			{ return MATERIAL; }
<INITIAL>MovieTexture		{ return MOVIE_TEXTURE; }
<INITIAL>NavigationInfo		{ return NAVIGATION_INFO; }
<INITIAL>Normal				{ return NORMAL; }
<INITIAL>NormalInterpolator	{ return NORMAL_INTERP; }
<INITIAL>OFF				{ yylval.ival = 1; return NUMBER; }
<INITIAL>ON					{ yylval.ival = 0; return NUMBER; }
<INITIAL>OrientationInterpolator	{ return ORIENTATION_INTERP; }
<INITIAL>PixelTexture		{ return PIXEL_TEXTURE; }
<INITIAL>PlaneSensor		{ return PLANE_SENSOR; }
<INITIAL>PointLight			{ return POINTLIGHT; }
<INITIAL>PointSet			{ return POINTSET; }
<INITIAL>PositionInterpolator	{ return POSITION_INTERP; }
<INITIAL>ProximitySensor	{ return PROXIMITY_SENSOR; }
<INITIAL>ScalarInterpolator	{ return SCALAR_INTERP; }
<INITIAL>Script				{ BEGIN SCRIPTNODE;	return SCRIPT; }
<INITIAL>Shape				{ return SHAPE; }
<INITIAL>Sound				{ return SOUND; }
<INITIAL>Sphere				{ return SPHERE; }
<INITIAL>SphereSensor		{ return SPHERE_SENSOR; }
<INITIAL>SpotLight			{ return SPOTLIGHT; }
<INITIAL>Switch				{ return SWITCH; }
<INITIAL>Text				{ return TEXT; }
<INITIAL>TextureCoordinate	{ return TEXTURE_COORDINATE; }
<INITIAL>TextureTransform	{ return TEXTURE_TRANSFORM; }
<INITIAL>TimeSensor			{ return TIME_SENSOR; }
<INITIAL>TouchSensor		{ return TOUCH_SENSOR; }
<INITIAL>Transform			{ return TRANSFORM; }
<INITIAL>Viewpoint			{ return VIEWPOINT; }
<INITIAL>VisibilitySensor	{ return VISIBILITY_SENSOR; }
<INITIAL>WorldInfo			{ return WORLD_INFO; }

<INITIAL>ambientIntensity	{ return S_AMBIENT_INTENSITY; }
<INITIAL>appearance			{ return S_APPEARANCE; }
<INITIAL>attenuation		{ return S_ATTENUATION; }
<INITIAL>autoOffset			{ return S_AUTO_OFFSET; }
<INITIAL>avatarSize			{ return S_AVATAR_SIZE; }
<INITIAL>axisOfRotation		{ return S_AXIS_OF_ROTATION;}
<INITIAL>backUrl			{ return S_BACKURL; }
<INITIAL>bboxCenter			{ return S_BBOXCENTER; }
<INITIAL>bboxSize			{ return S_BBOXSIZE; }
<INITIAL>beamWidth			{ return S_BERM_WIDTH; }
<INITIAL>beginCap			{ return S_BEGIN_CAP; }
<INITIAL>bottom				{ return S_BOTTOM; }
<INITIAL>bottomRadius		{ return S_BOTTOMRADIUS; }
<INITIAL>bottomUrl			{ return S_BOTTOMURL; }
<INITIAL>ccw				{ return S_CCW; }
<INITIAL>center				{ return S_CENTER; }
<INITIAL>children			{ return S_CHILDREN; }
<INITIAL>choice				{ return S_CHOICE; }
<INITIAL>collide			{ return S_COLLIDE; }
<INITIAL>collideTime		{ return S_COLLIDETIME;}
<INITIAL>color				{ return S_COLOR; }
<INITIAL>colorIndex			{ return S_COLOR_INDEX; }
<INITIAL>colorPerVertex		{ return S_COLOR_PER_VERTEX; }
<INITIAL>convex				{ return S_CONVEX; }
<INITIAL>coord				{ return S_COORD; }
<INITIAL>coordIndex			{ return S_COORD_INDEX; }
<INITIAL>creaseAngle		{ return S_CREASE_ANGLE;}
<INITIAL>crossSection		{ return S_CROSS_SECTION; }
<INITIAL>cutOffAngle		{ return S_CUTOFFANGLE; }
<INITIAL>cycleInterval		{ return S_CYCLE_INTERVAL; }
<INITIAL>description		{ return S_DESCRIPTION; }
<INITIAL>diffuseColor		{ return S_DIFFUSECOLOR; }
<INITIAL>direction			{ return S_DIRECTION; }
<INITIAL>diskAngle			{ return S_DISK_ANGLE; }
<INITIAL>emissiveColor		{ return S_EMISSIVECOLOR; }
<INITIAL>enabled			{ return S_ENABLED; }
<INITIAL>endCap				{ return S_END_CAP; }
<INITIAL>family				{ return S_FAMILY; }
<INITIAL>fieldOfView		{ return S_FIELD_OF_VIEW; }
<INITIAL>fogType			{ return S_FOG_TYPE; }
<INITIAL>fontStyle			{ return S_FONTSTYLE; }
<INITIAL>frontUrl			{ return S_FRONTURL; }
<INITIAL>geometry			{ return S_GEOMETRY; }
<INITIAL>groundAngle		{ return S_GROUNDANGLE; }
<INITIAL>groundColor		{ return S_GROUNDCOLOR; }
<INITIAL>headlight			{ return S_HEADLIGHT; }
<INITIAL>height				{ return S_HEIGHT; }
<INITIAL>horizontal			{ return S_HORIZONTAL; }
<INITIAL>image				{ return S_IMAGE; }
<INITIAL>info				{ return S_INFO; }
<INITIAL>intensity			{ return S_INTENSITY; }
<INITIAL>jump				{ return S_JUMP; }
<INITIAL>justify			{ return S_JUSTIFY; }
<INITIAL>key				{ return S_KEY; }
<INITIAL>keyValue			{ return S_KEYVALUE; }
<INITIAL>language			{ return S_LANGUAGE; }
<INITIAL>leftToRight		{ return S_LEFT2RIGHT; }
<INITIAL>leftUrl			{ return S_LEFTURL; }
<INITIAL>length				{ return S_LENGTH; }
<INITIAL>level				{ return S_LEVEL; }
<INITIAL>location			{ return S_LOCATION; }
<INITIAL>loop				{ return S_LOOP; }
<INITIAL>material			{ return S_MATERIAL; }
<INITIAL>maxAngle			{ return S_MAX_ANGLE; }
<INITIAL>maxBack			{ return S_MAX_BACK; }
<INITIAL>maxExtent			{ return S_MAX_EXTENT; }
<INITIAL>maxFront			{ return S_MAX_FRONT; }
<INITIAL>maxPosition		{ return S_MAX_POSITION; }
<INITIAL>minAngle			{ return S_MIN_ANGLE; }
<INITIAL>minBack			{ return S_MIN_BACK; }
<INITIAL>minFront			{ return S_MIN_FRONT; }
<INITIAL>minPosition		{ return S_MIN_POSITION; }
<INITIAL>normal				{ return S_NORMAL; }
<INITIAL>normalIndex		{ return S_NORMAL_INDEX; }
<INITIAL>normalPerVertex	{ return S_NORMAL_PER_VERTEX; }
<INITIAL>offset				{ return S_OFFSET;}
<INITIAL>on					{ return S_ON; }
<INITIAL>orientation		{ return S_ORIENTATION; }
<INITIAL>parameter			{ return S_PARAMETER; }
<INITIAL>pitch				{ return S_PICTH; }
<INITIAL>point				{ return S_POINT; }
<INITIAL>position			{ return S_POSITION; }
<INITIAL>priority			{ return S_PRIORITY; }
<INITIAL>proxy				{ return S_PROXY; }
<INITIAL>radius				{ return S_RADIUS; }
<INITIAL>range				{ return S_RANGE; }
<INITIAL>repeatS			{ return S_REPEAT_S; }
<INITIAL>repeatT			{ return S_REPEAT_T; }
<INITIAL>rightUrl			{ return S_RIGHTURL; }
<INITIAL>rotation			{ return S_ROTATION; }
<INITIAL>scale				{ return S_SCALE; }
<INITIAL>scaleOrientation	{ return S_SCALEORIENTATION; }
<INITIAL>shininess			{ return S_SHININESS; }
<INITIAL>side				{ return S_SIDE; }
<INITIAL>size				{ return S_SIZE; }
<INITIAL>skyAngle			{ return S_SKYANGLE; }
<INITIAL>skyColor			{ return S_SKYCOLOR; }
<INITIAL>solid				{ return S_SOLID;}
<INITIAL>source				{ return S_SOURCE; }
<INITIAL>spacing			{ return S_SPACING; }
<INITIAL>spatialize			{ return S_SPATIALIZE; }
<INITIAL>specularColor		{ return S_SPECULARCOLOR; }
<INITIAL>speed				{ return S_SPEED; }
<INITIAL>spine				{ return S_SPINE; }
<INITIAL>startTime			{ return S_STARTTIME; }
<INITIAL>stopTime			{ return S_STOPTIME; }
<INITIAL>string				{ return S_STRING; }
<INITIAL>style				{ return S_STYLE; }
<INITIAL>texCoord			{ return S_TEXCOORD; }
<INITIAL>texCoordIndex		{ return S_TEXCOORD_INDEX;}
<INITIAL>texture			{ return S_TEXTURE; }
<INITIAL>textureTransform	{ return S_TEXTURETRANSFORM; }
<INITIAL>title				{ return S_TITLE; }
<INITIAL>top				{ return S_TOP; }
<INITIAL>topToBottom		{ return S_TOP2BOTTOM; }
<INITIAL>topUrl				{ return S_TOPURL; }
<INITIAL>translation		{ return S_TRANSLATION; }
<INITIAL>transparency		{ return S_TRANSPARENCY; }
<INITIAL>type				{ return S_TYPE; }
<INITIAL>url				{ return S_URL; }
<INITIAL>vector				{ return S_VECTOR; }
<INITIAL>visibilityLimit	{ return S_VISIBILITY_LIMIT; }
<INITIAL>visibilityRange	{ return S_VISIBILITY_RANGE; }
<INITIAL>whichChild			{ return S_WHICHCHILD; }
<INITIAL>whichChoice		{ return S_WHICHCHOICE; }
<INITIAL>xDimension			{ return S_XDIMENSION; }
<INITIAL>xSpacing			{ return S_XSPACING; }
<INITIAL>zDimension			{ return S_ZDIMENSION; }
<INITIAL>zSpacing			{ return S_ZSPACING; }

<INITIAL>value_changed		{ return S_VALUE_CHANGED; }

<INITIAL>DEF	{
	std::ostringstream defBuf;
	int c = input();
	while (c != '{' && c != EOF) {
		defBuf << (char)(c);
		c = input();
	} 

	char	defName[256];
	char	nodeName[256];
	const std::string &defStr = defBuf.str();
	const char *defCStr = defStr.c_str();
	sscanf(defStr.c_str(), "%255s %255s", defName, nodeName);

	std::string unputbuf;
	unputbuf.append(nodeName);
	unputbuf.append(" {");

	SetDEFName(defName);
	UnputString(unputbuf.c_str());
}

<INITIAL>USE	{
	std::ostringstream useBuf;
	int c = 0;
	do {
		c = input();
		if (c == '\n')
			nCurrentLine++;
	} while (c == '\t' || c == ' ');

	while (('0' <= c && c <= '9') || ('A' <= c && c <= 'Z') || ('a' <= c && c <= 'z') || c == '_' || c == '-' || c == '+') {
		useBuf << (char)(c);
		c = input();
		if (c == '\n')
			nCurrentLine++;
	}
	unput(c);

	std::string useStr = useBuf.str();
#ifdef UNUSE_DEF_NODE
	Node *defNode = GetParserObject()->findLastNode(useStr.c_str());
	Node *useParentNode = GetCurrentNodeObject();
	if (defNode) {	
		if (useParentNode == NULL || (useParentNode != defNode && useParentNode->isAncestorNode(defNode) == false)) {
			Node *instanceNode = defNode->createInstanceNode();
			if (instanceNode)
				ParserAddNode(instanceNode);
		}
	}
	return USE;
#else
	Node *node = GetParserObject()->findLastNode(useStr.c_str());
	if (node) {
		Node *defNode = node->createDEFNode();
		ParserAddNode(defNode);
	}
	return USE;
#endif
}

<INITIAL>PROTO	{
	std::ostringstream protoNameBuf;
	int c = 0;
	do {
		c = input();
		protoNameBuf << (char)(c);
		if (c == '\n')
			nCurrentLine++;
	} while (c != '[' && c != EOF);

	std::string protoNameStr = protoNameBuf.str();
	char *protoName = new char[protoNameStr.length()+1];
	sscanf(protoNameStr.c_str(), "%s", protoName);

	int bigBracket = 1;

	std::ostringstream fieldBuf;
	do {
		c = input();
		if (c == ']') {
			bigBracket--;
			if (bigBracket <= 0)
				break;
		}
		if (c == '[')
			bigBracket++;
		if (c == '\n')
			nCurrentLine++;
		fieldBuf << (char)(c);
	} while (c != EOF);
	std::string fieldString = fieldBuf.str();

	while (c != '{' && c != EOF) {
		c = input();
		if (c == '\n')
			nCurrentLine++;
	}

	std::ostringstream protoDefBuf;
	do {
		c = input();
		if (c != '}')protoDefBuf << (char)(c);
		if (c == '\n')
			nCurrentLine++;
	} while (c != '{' && c != '}' && c != EOF);

	if (c != '}') {
	
		int inputFlag = 1;
		do {
			c = input();
	
			switch (c) {
			case '{' : 
				{
					inputFlag++;
				}
				break;
			case '}' : 
				{
					inputFlag--;
				}
				break;
			case '\n' :
				{ 
					nCurrentLine++;
				}
				break;
			case '#'	: /* sval comment */
				{
					do {
						c = input();
					} while (c != '\n');
					protoDefBuf << '\n';
					nCurrentLine++;
				}
			}
	
			protoDefBuf << (char)(c);

		} while (0 < inputFlag && c != EOF);

		do {
			c = input();
			if (c == '\n')
				nCurrentLine++;
		} while (c != '}' && c != EOF);

	}

	std::string protoDefSet = protoDefBuf.str();
	PROTO *proto = AddPROTOInfo(protoName, protoDefSet.c_str(), fieldString.c_str());

	delete []protoName;
}

<INITIAL>ROUTE {
	int c = 0;
	std::ostringstream routeBuf;
	do {
		c = input();
		if (c == '\n')
			CurrentLineIncrement();
		else
			routeBuf << (char)(c);
	} while (c != '\n' && c != EOF);
	std::string routeStr = routeBuf.str();
	ParserAddRouteInfo(routeStr.c_str());
}

<INITIAL>[_a-zA-Z][_\+\-a-zA-Z0-9]* {

	PROTO *proto = IsPROTOName(yytext);

	if (proto) {
		int c;
		c = 0;
		do {
			c = input();
			if (c == '\n')
				nCurrentLine++;
		} while (c != '{' && c != EOF);

		String readBuffer;
		int smallBracket = 1;
		do {
			c = input();
			if (c == '}') {
				smallBracket--;
				if (smallBracket <= 0)
					break;
			}
			if (c == '{')
				smallBracket++;
			readBuffer.append(c);
			if (c == '\n')
				nCurrentLine++;
		} while (c != EOF);

		const char *readValue = readBuffer.getValue();
		proto->addFields(readValue);

//		std::cout << "==== read value ===="<< std::endl;
//		std::cout << readValue << std::endl;

		String replaceBuffer;
		proto->getString(replaceBuffer);
		const char *replaceValue = replaceBuffer.getValue();
		UnputString(replaceValue);

//		std::cout << "==== replace value ===="<< std::endl;
//		std::cout << replaceValue << std::endl;

		proto->deleteFields();
	}
	else {
		name = yytext;
		yylval.sval = (char *)name.c_str();	
		return NAME;
	}
}

<SCRIPTNODE>directOutput	{ return S_DIRECT_OUTPUT; }
<SCRIPTNODE>mustEvaluate	{ return S_MUST_EVALUATE; }
<SCRIPTNODE>url				{ return S_URL; }

<SCRIPTNODE>field			{ return FIELD; }
<SCRIPTNODE>eventIn			{ return EVENTIN; }
<SCRIPTNODE>eventOut		{ return EVENTOUT; }

<SCRIPTNODE>SFBool			{ return SFBOOL; }
<SCRIPTNODE>SFFloat			{ return SFFLOAT; }
<SCRIPTNODE>SFInt32			{ return SFINT32; }
<SCRIPTNODE>SFTime			{ return SFTIME; }
<SCRIPTNODE>SFRotation		{ return SFROTATION; }
<SCRIPTNODE>SFNode			{ return SFNODE; }
<SCRIPTNODE>SFColor			{ return SFCOLOR; }
<SCRIPTNODE>SFString		{ return SFSTRING; }
<SCRIPTNODE>SFVec2f			{ return SFVEC2F; }
<SCRIPTNODE>SFVec3f			{ return SFVEC3F; }

<SCRIPTNODE>MFBool			{ return MFBOOL; }
<SCRIPTNODE>MFFloat			{ return MFFLOAT; }
<SCRIPTNODE>MFInt32			{ return MFINT32; }
<SCRIPTNODE>MFTime			{ return MFTIME; }
<SCRIPTNODE>MFRotation		{ return MFROTATION; }
<SCRIPTNODE>MFNode			{ return MFNODE; }
<SCRIPTNODE>MFColor			{ return MFCOLOR; }
<SCRIPTNODE>MFImage			{ return MFIMAGE; }
<SCRIPTNODE>MFString		{ return MFSTRING; }
<SCRIPTNODE>MFVec2f			{ return MFVEC2F; }
<SCRIPTNODE>MFVec3f			{ return MFVEC3F; }

<SCRIPTNODE>USE				{ return USE; }

<SCRIPTNODE>[_a-zA-Z][_\+\-a-zA-Z0-9]* {
//	char *name = strdup(yytext);
	char *name = new char[strlen(yytext)+1];
	strcpy(name, yytext);
	yylval.sval = name;	
	return	NAME;
}

%%

int yywrap()
{
  return 1;
} 

int yyerror(char *errmsg) 
{ 
	ParserResult *presult = GetParserResultObject();
	presult->setErrorLineNumber(GetCurrentLineNumber());
	presult->setErrorMessage(errmsg);
	presult->setErrorToken(yytext);
	presult->setErrorLineString(GetErrorLineString());	
	return 0;
}

void CyberX3D::CurrentLineIncrement()
{
	nCurrentLine++;
	if (callbackFn) 
		callbackFn(nCurrentLine, callbackFnInfo);
}

void CyberX3D::MakeLexerBuffers(int lexBufferSize, int lineBufferSize)
{
	YY_BUFFER_STATE buffer = yy_create_buffer(yyin, lexBufferSize);
	yypush_buffer_state(buffer);
}

void CyberX3D::DeleteLexerBuffers(void)
{
	yy_delete_buffer(YY_CURRENT_BUFFER  );
	YY_CURRENT_BUFFER_LVALUE = NULL;
	yypop_buffer_state();
}

void CyberX3D::SetLexCallbackFn(void (*func)(int nLine, void *info), void *fnInfo)
{
	callbackFn = func;
	callbackFnInfo = fnInfo;
}

int CyberX3D::UnputString(const char *pBegin)
{
	int nOutputLine = 0;

	if (!pBegin && !strlen(pBegin))
		return nOutputLine;

	const char *pEnd = pBegin + strlen(pBegin);
	while(pEnd > pBegin) {
		unput(*--pEnd);
		if (*pEnd == '\n')
			nOutputLine++;
	}
	return nOutputLine;
}

void CyberX3D::SetInputFile(FILE *fp)
{
  nCurrentLine = 1;
  yyrestart(fp);
}

int CyberX3D::GetCurrentLineNumber(void)
{
  return nCurrentLine;
}

const char *CyberX3D::GetErrorLineString(void)
{
  return lineBuffer.c_str();
}


void CyberX3D::SetInputBuffer(const char *str, int size) {
  nCurrentLine = 1;
  in_buffer = str;
  in_size = size;
}

void CyberX3D::MakeLexerStringBuffers(int lexBufferSize, int lineBufferSize)
{
        YY_BUFFER_STATE buffer = yy_scan_bytes(in_buffer, in_size);
	yypush_buffer_state(buffer);
}

void CyberX3D::DeleteLexerStringBuffers(void)
{
	yy_delete_buffer(YY_CURRENT_BUFFER  );
	YY_CURRENT_BUFFER_LVALUE = NULL;
	yypop_buffer_state();
}

