/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	ParserFunc.cpp
*
******************************************************************/

#include <x3d/Parser.h>
#include <x3d/ParserFunc.h>
#include <x3d/NodeType.h>
#include <x3d/X3DNodes.h>

using namespace CyberX3D;

/******************************************************************
*	ParserResultObject
******************************************************************/

static ParserResult *gParserResult;

void CyberX3D::SetParserResultObject(ParserResult *result)
{
	gParserResult = result;
}

ParserResult *CyberX3D::GetParserResultObject()
{
	return gParserResult;
}

/******************************************************************
*	ParserObject
******************************************************************/

static LinkedList<Parser> gParserList;

void CyberX3D::PushParserObject(Parser *parser)
{
	gParserList.addNode(parser);
}

void CyberX3D::PopParserObject()
{
	Parser *lastNode = gParserList.getLastNode(); 
	lastNode->remove();
}

Parser *CyberX3D::GetParserObject()
{
	return gParserList.getLastNode(); 
}

int CyberX3D::ParserGetCurrentNodeType(void)
{
	return GetParserObject()->getCurrentNodeType();
}

int CyberX3D::ParserGetParentNodeType(void)
{
	return GetParserObject()->getPrevNodeType();
}

Node *CyberX3D::ParserGetCurrentNode(void)
{
	return GetParserObject()->getCurrentNode();
}

void CyberX3D::ParserPushNode(int parserType, Node *node)
{
	GetParserObject()->pushNode(node, parserType);
}

void CyberX3D::ParserPushNode(Node *node)
{
	GetParserObject()->pushNode(node, UNKNOWN_NODE);
}

void CyberX3D::ParserPopNode(void)
{
	GetParserObject()->popNode();
}

void CyberX3D::ParserAddNode(Node *node)
{
	GetParserObject()->addNode(node, 0);
}

/******************************************************************
*	AddRouteInfo
******************************************************************/

#define ROUTE_STRING_MAX	2048

static char targetNodeName[ROUTE_STRING_MAX];
static char sourceNodeName[ROUTE_STRING_MAX];
static char targetNodeTypeName[ROUTE_STRING_MAX];
static char sourceNodeTypeName[ROUTE_STRING_MAX];

void CyberX3D::ParserAddRouteInfo(const char *string)
{
	if (!string || !strlen(string))
		return;

	// Copy string so we can modify it
	char *buffer = new char[strlen(string) + 1];
	strcpy(buffer, string);

	for (int n=0; n<(int)strlen(buffer); n++) {
		if (buffer[n] == '.')
			buffer[n] = ' ';
	}

	sscanf(buffer, "%s %s TO %s %s", sourceNodeName, sourceNodeTypeName, targetNodeName, targetNodeTypeName);

	delete [] buffer;

	GetParserObject()->addRoute(sourceNodeName, sourceNodeTypeName, targetNodeName, targetNodeTypeName);
}


/******************************************************************
*	ParserObject
******************************************************************/

Node *CyberX3D::CreateNode(int nodeType)
{
	switch (nodeType) {
		case ANCHOR_NODE:
			return new AnchorNode();
		case APPEARANCE_NODE:
			return new AppearanceNode();
		case AUDIOCLIP_NODE:
			return new AudioClipNode();
		case BACKGROUND_NODE:
			return new BackgroundNode();
		case BILLBOARD_NODE:
			return new BillboardNode();
		case BOX_NODE:
			return new BoxNode();
		case COLLISION_NODE:
			return new CollisionNode();
		case COLORINTERPOLATOR_NODE:
			return new ColorInterpolatorNode();
		case COLOR_NODE:
			return new ColorNode();
		case CONE_NODE:
			return new ConeNode();
		case COORDINATEINTERPOLATOR_NODE:
			return new CoordinateInterpolatorNode();
		case COORDINATE_NODE:
			return new CoordinateNode();
		case CYLINDER_NODE:
			return new CylinderNode();
		case CYLINDERSENSOR_NODE:
			return new CylinderSensorNode();
		//case DEF_NODE: //"DEF":
		case DIRECTIONALLIGHT_NODE:
			return new DirectionalLightNode();
		case ELEVATIONGRID_NODE:
			return new ElevationGridNode();
		case EXTRUSION_NODE:
			return new ExtrusionNode();
		case FOG_NODE:
			return new FogNode();
		case FONTSTYLE_NODE:
			return new FontStyleNode();
		case GROUP_NODE:
			return new GroupNode();
		case IMAGETEXTURE_NODE:
			return new ImageTextureNode();
		case INDEXEDFACESET_NODE:
			return new IndexedFaceSetNode();
		case INDEXEDLINESET_NODE:
			return new IndexedLineSetNode();
		case INLINE_NODE:
			return new InlineNode();
		case LOD_NODE:
			return new LODNode();
		case MATERIAL_NODE:
			return new MaterialNode();
		case MOVIETEXTURE_NODE:
			return new MovieTextureNode();
		case NAVIGATIONINFO_NODE:
			return new NavigationInfoNode();
		case NORMALINTERPOLATOR_NODE:
			return new NormalInterpolatorNode();
		case NORMAL_NODE:
			return new NormalNode();
		case ORIENTATIONINTERPOLATOR_NODE:
			return new OrientationInterpolatorNode();
		case PIXELTEXTURE_NODE:
			return new PixelTextureNode();
		case PLANESENSOR_NODE:
			return new PlaneSensorNode();
		case POINTLIGHT_NODE:
			return new PointLightNode();
		case POINTSET_NODE:
			return new PointSetNode();
		case POSITIONINTERPOLATOR_NODE:
			return new PositionInterpolatorNode();
		case PROXIMITYSENSOR_NODE:
			return new ProximitySensorNode();
		//case ROOT_NODE: //"ROOT":
		case SCALARINTERPOLATOR_NODE:
			return new ScalarInterpolatorNode();
		case SCRIPT_NODE:
			return new ScriptNode();
		case SHAPE_NODE:
			return new ShapeNode();
		case SOUND_NODE:
			return new SoundNode();
		case SPHERE_NODE:
			return new SphereNode();
		case SPHERESENSOR_NODE:
			return new SphereSensorNode();
		case SPOTLIGHT_NODE:
			return new SpotLightNode();
		case SWITCH_NODE:
			return new SwitchNode();
		case TEXT_NODE:
			return new TextNode();
		case TEXTURECOORDINATE_NODE:
			return new TextureCoordinateNode();
		case TEXTURETRANSFORM_NODE:
			return new TextureTransformNode();
		case TIMESENSOR_NODE:
			return new TimeSensorNode();
		case TOUCHSENSOR_NODE:
			return new TouchSensorNode();
		case TRANSFORM_NODE:
			return new TransformNode();
		case VIEWPOINT_NODE:
			return new ViewpointNode();
		case VISIBILITYSENSOR_NODE:
			return new VisibilitySensorNode();
		case WORLDINFO_NODE:
			return new WorldInfoNode();
	}

	return NULL;
}

Node *CyberX3D::CreateX3DNode(int nodeType)
{
	Node *node = CreateNode(nodeType);
	if (node != NULL)
		return node;

	switch (nodeType) {
		// Scene (X3D)
		case SCENE_NODE:
			return new SceneNode();
		// 9. Networking component (X3D)
		case LOADSENSOR_NODE:
			return new LoadSensorNode();
		// 10. Grouping component (X3D)
		case STATICGROUP_NODE:
			return new StaticGroupNode();
		// 11. Rendering component (X3D)
		case COLORRGBA_NODE:
			return new ColorRGBANode();
		case TRIANGLESET_NODE:  
			return new TriangleSetNode();
		case TRIANGLEFANSET_NODE:
			return new TriangleFanSetNode();
		case TRIANGLESTRIPSET_NODE:
			return new TriangleStripSetNode();
		// 12. Shape component (X3D)
		case FILLPROPERTIES_NODE:
			return new FillPropertiesNode();
		case LINEPROPERTIES_NODE:
			return new LinePropertiesNode();
		// 14. Geometry2D component (X3D)
		case ARC2D_NODE:
			return new Arc2DNode();
		case ARCCLOSE2D_NODE:
			return new ArcClose2DNode();
		case CIRCLE2D_NODE:
			return new Circle2DNode();
		case DISK2D_NODE:
			return new Disk2DNode();
		case POLYLINE2D_NODE:
			return new Polyline2DNode();
		case POLYPOINT2D_NODE:
			return new Polypoint2DNode();
		case RECTANGLE2D_NODE:
			return new Rectangle2DNode();
		case TRIANGLESET2D_NODE:
			return new TriangleSet2DNode();
		// 18. Texturing component (x3D)
		case MULTITEXTURE_NODE:
			return new MultiTextureNode();
		case MULTITEXTURECOORD_NODE:
			return new MultiTextureCoordinateNode();
		case MULTITEXTURETRANSFORM_NODE:
			return new MultiTextureTransformNode();
		case TEXCOORDGEN_NODE:
			return new TextureCoordinateGeneratorNode();
		// 19. Interpolation component (X3D)
		case COORDINATEINTERPOLATOR2D_NODE:
			return new CoordinateInterpolator2DNode();
		case POSITIONINTERPOLATOR2D_NODE:
			return new PositionInterpolator2DNode();
		// 21. Key device sensor component (X3D)
		case KEYSENSOR_NODE:
			return new KeySensorNode();
		case STRINGSENSOR_NODE:
			return new StringSensorNode();
		// 30. Event Utilities component (X3D)
		case BOOLEANFILTER_NODE:
			return new BooleanFilterNode();
		case BOOLEANTOGGLE_NODE:
			return new BooleanToggleNode();
		case BOOLEANTRIGGER_NODE:
			return new BooleanTriggerNode();
		case BOOLEANSEQUENCER_NODE:
			return new BooleanSequencerNode();
		case INTEGERTRIGGER_NODE:
			return new IntegerTriggerNode();
		case INTEGERSEQUENCER_NODE:
			return new IntegerSequencerNode();
		case TIMETRIGGER_NODE:
			return new TimeTriggerNode();
		// Deprecated components (X3D)
		case NODESEQUENCER_NODE:
			return new NodeSequencerNode();
		case SHAPE2D_NODE:
			return new Shape2DNode();
		case BOOLEANTIMETRIGGER_NODE:
				return new BooleanTimeTriggerNode();
		case TRANSFORM2D_NODE:
			return new Transform2DNode();
		// Route (X3D)
		case ROUTE_NODE:
			return new RouteNode();
	}

	return new XMLNode();
}
