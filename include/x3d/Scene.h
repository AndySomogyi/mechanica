/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	Scene.h
*
******************************************************************/

#ifndef _CX3D_SCENE_H_
#define _CX3D_SCENE_H_

#include <x3d/NodeList.h>
#include <x3d/Route.h>
#include <x3d/X3DNodes.h>

namespace CyberX3D {

class Scene {

	NodeList					mNodeList;
	LinkedList<Route>	mRouteList;

public:

	Scene();
	virtual ~Scene();

	NodeList *getNodeList() {
		return &mNodeList;		
	}

	Node *getRootNode() const {
		return (Node *)mNodeList.getRootNode();		
	}

	Node *getNodes() const {
		return (Node *)mNodeList.getNodes();		
	}

	///////////////////////////////////////////////
	//	Load
	///////////////////////////////////////////////

	void clearNodeList() {
		mNodeList.deleteNodes();		
	}

	void clearRouteList() {
		mRouteList.deleteNodes();		
	}

	///////////////////////////////////////////////
	//	Praser action
	///////////////////////////////////////////////

	void addNode(Node *node, bool initialize = true);
	void addNodeAtFirst(Node *node, bool initialize = true);

	void moveNode(Node *node);
	void moveNodeAtFirst(Node *node);

	////////////////////////////////////////////////
	//	find node
	////////////////////////////////////////////////

	Node *findNode(const char *name)const ;
	Node *findNode(const int type)const ;
	Node *findNode(const int type, const char *name)const ;

	Node *findLastNode(const char *name)const ;
	Node *findDEFNode(const char *name)const ;

	bool hasNode(Node *targetNode)const ;

	///////////////////////////////////////////////
	//	ROUTE
	///////////////////////////////////////////////

	Route *getRoutes()const ;
	Route *getRoute(Node *eventOutNode, Field *eventOutField, Node *eventInNode, Field *eventInField)const ;
	void addRoute(Route *route);
	Route *addRoute(const char *eventOutNodeName, const char *eventOutFieldName, const char *eventInNodeName, const char *eventInFieldName);
	Route *addRoute(Node *eventOutNode, Field *eventOutField, Node *eventInNode, Field *eventInField);
	void deleteRoute(Node *eventOutNode, Field *eventOutField, Node *eventInNode, Field *eventInField);
	void deleteRoutes(Node *node);
	void deleteEventInFieldRoutes(Node *node, Field *field);
	void deleteEventOutFieldRoutes(Node *node, Field *field);
	void deleteRoutes(Node *node, Field *field);
	void deleteRoute(Route *deleteRoute);
	void removeRoute(Node *eventOutNode, Field *eventOutField, Node *eventInNode, Field *eventInField);
	void removeRoutes(Node *node);
	void removeEventInFieldRoutes(Node *node, Field *field);
	void removeEventOutFieldRoutes(Node *node, Field *field);
	void removeRoutes(Node *node, Field *field);
	void removeRoute(Route *removeRoute);

	////////////////////////////////////////////////
	//	find*Node
	////////////////////////////////////////////////

	GroupingNode *findGroupingNode() const {
		for (Node *node = (getRootNode())->nextTraversal() ; node; node = node->nextTraversal()) {
			if (node->isGroupingNode())
				return (GroupingNode *)node;
		}
		return NULL;
	}

	AnchorNode *findAnchorNode() const {
		return (AnchorNode *)findNode(ANCHOR_NODE);
	}

	AppearanceNode *findAppearanceNode() const {
		return (AppearanceNode *)findNode(APPEARANCE_NODE);
	}

	AudioClipNode *findAudioClipNode() const {
		return (AudioClipNode *)findNode(AUDIOCLIP_NODE);
	}

	BackgroundNode *findBackgroundNode() const {
		return (BackgroundNode *)findNode(BACKGROUND_NODE);
	}

	BillboardNode *findBillboardNode() const {
		return (BillboardNode *)findNode(BILLBOARD_NODE);
	}

	BoxNode *findBoxNode() const {
		return (BoxNode *)findNode(BOX_NODE);
	}

	CollisionNode *findCollisionNode() const {
		return (CollisionNode *)findNode(COLLISION_NODE);
	}

	ColorNode *findColorNode() const {
		return (ColorNode *)findNode(COLOR_NODE);
	}

	ColorInterpolatorNode *findColorInterpolatorNode() const {
		return (ColorInterpolatorNode *)findNode(COLORINTERPOLATOR_NODE);
	}

	ConeNode *findConeNode() const {
		return (ConeNode *)findNode(CONE_NODE);
	}

	CoordinateNode *findCoordinateNode() const {
		return (CoordinateNode *)findNode(COORDINATE_NODE);
	}

	CoordinateInterpolatorNode *findCoordinateInterpolatorNode() const {
		return (CoordinateInterpolatorNode *)findNode(COORDINATEINTERPOLATOR_NODE);
	}

	CylinderNode *findCylinderNode() const {
		return (CylinderNode *)findNode(CYLINDER_NODE);
	}

	CylinderSensorNode *findCylinderSensorNode() const {
		return (CylinderSensorNode *)findNode(CYLINDERSENSOR_NODE);
	}

	DirectionalLightNode *findDirectionalLightNode() const {
		return (DirectionalLightNode *)findNode(DIRECTIONALLIGHT_NODE);
	}

	ElevationGridNode *findElevationGridNode() const {
		return (ElevationGridNode *)findNode(ELEVATIONGRID_NODE);
	}

	ExtrusionNode *findExtrusionNode() const {
		return (ExtrusionNode *)findNode(EXTRUSION_NODE);
	}

	FogNode *findFogNode() const {
		return (FogNode *)findNode(FOG_NODE);
	}

	FontStyleNode *findFontStyleNode() const {
		return (FontStyleNode *)findNode(FONTSTYLE_NODE);
	}

	GroupNode *findGroupNode() const {
		return (GroupNode *)findNode(GROUP_NODE);
	}

	ImageTextureNode *findImageTextureNode() const {
		return (ImageTextureNode *)findNode(IMAGETEXTURE_NODE);
	}

	IndexedFaceSetNode *findIndexedFaceSetNode() const {
		return (IndexedFaceSetNode *)findNode(INDEXEDFACESET_NODE);
	}

	IndexedLineSetNode *findIndexedLineSetNode() const {
		return (IndexedLineSetNode *)findNode(INDEXEDLINESET_NODE);
	}

	InlineNode *findInlineNode() const {
		return (InlineNode *)findNode(INLINE_NODE);
	}

	LODNode *findLODNode() const {
		return (LODNode *)findNode(LOD_NODE);
	}

	MaterialNode *findMaterialNode() const {
		return (MaterialNode *)findNode(MATERIAL_NODE);
	}

	MovieTextureNode *findMovieTextureNode() const {
		return (MovieTextureNode *)findNode(MOVIETEXTURE_NODE);
	}

	NavigationInfoNode *findNavigationInfoNode() const {
		return (NavigationInfoNode *)findNode(NAVIGATIONINFO_NODE);
	}

	NormalNode *findNormalNode() const {
		return (NormalNode *)findNode(NORMAL_NODE);
	}

	NormalInterpolatorNode *findNormalInterpolatorNode() const {
		return (NormalInterpolatorNode *)findNode(NORMALINTERPOLATOR_NODE);
	}

	OrientationInterpolatorNode *findOrientationInterpolatorNode() const {
		return (OrientationInterpolatorNode *)findNode(ORIENTATIONINTERPOLATOR_NODE);
	}

	PixelTextureNode *findPixelTextureNode() const {
		return (PixelTextureNode *)findNode(PIXELTEXTURE_NODE);
	}

	PlaneSensorNode *findPlaneSensorNode() const {
		return (PlaneSensorNode *)findNode(PLANESENSOR_NODE);
	}

	PointLightNode *findPointLightNode() const {
		return (PointLightNode *)findNode(POINTLIGHT_NODE);
	}

	PointSetNode *findPointSetNode() const {
		return (PointSetNode *)findNode(POINTSET_NODE);
	}

	PositionInterpolatorNode *findPositionInterpolatorNode() const {
		return (PositionInterpolatorNode *)findNode(POSITIONINTERPOLATOR_NODE);
	}

	ProximitySensorNode *findProximitySensorNode() const {
		return (ProximitySensorNode *)findNode(PROXIMITYSENSOR_NODE);
	}

	ScalarInterpolatorNode *findScalarInterpolatorNode() const {
		return (ScalarInterpolatorNode *)findNode(SCALARINTERPOLATOR_NODE);
	}

	ScriptNode *findScriptNode() const {
		return (ScriptNode *)findNode(SCRIPT_NODE);
	}

	ShapeNode *findShapeNode() const {
		return (ShapeNode *)findNode(SHAPE_NODE);
	}

	SoundNode *findSoundNode() const {
		return (SoundNode *)findNode(SOUND_NODE);
	}

	SphereNode *findSphereNode() const {
		return (SphereNode *)findNode(SPHERE_NODE);
	}

	SphereSensorNode *findSphereSensorNode() const {
		return (SphereSensorNode *)findNode(SPHERESENSOR_NODE);
	}

	SpotLightNode *findSpotLightNode() const {
		return (SpotLightNode *)findNode(SPOTLIGHT_NODE);
	}

	SwitchNode *findSwitchNode() const {
		return (SwitchNode *)findNode(SWITCH_NODE);
	}

	TextNode *findTextNode() const {
		return (TextNode *)findNode(TEXT_NODE);
	}

	TextureCoordinateNode *findTextureCoordinateNode() const {
		return (TextureCoordinateNode *)findNode(TEXTURECOORDINATE_NODE);
	}

	TextureTransformNode *findTextureTransformNode() const {
		return (TextureTransformNode *)findNode(TEXTURETRANSFORM_NODE);
	}

	TimeSensorNode *findTimeSensorNode() const {
		return (TimeSensorNode *)findNode(TIMESENSOR_NODE);
	}

	TouchSensorNode *findTouchSensorNode() const {
		return (TouchSensorNode *)findNode(TOUCHSENSOR_NODE);
	}

	TransformNode *findTransformNode() const {
		return (TransformNode *)findNode(TRANSFORM_NODE);
	}

	ViewpointNode *findViewpointNode() const {
		return (ViewpointNode *)findNode(VIEWPOINT_NODE);
	}

	VisibilitySensorNode *findVisibilitySensorNode() const {
		return (VisibilitySensorNode *)findNode(VISIBILITYSENSOR_NODE);
	}

	WorldInfoNode *findWorldInfoNode() const {
		return (WorldInfoNode *)findNode(WORLDINFO_NODE);
	}

	// Scene (X3D)

	SceneNode *findSceneNode() const {
		return (SceneNode *)findNode(SCENE_NODE);
	}

	// 9. Networking component (X3D)

	LoadSensorNode *findLoadSensorNode() const {
		return (LoadSensorNode *)findNode(LOADSENSOR_NODE);
	}

	// 10. Grouping component (X3D)

	StaticGroupNode *findStaticGroupNode() const {
		return (StaticGroupNode *)findNode(STATICGROUP_NODE);
	}

	// 11. Rendering component (X3D)

	ColorRGBANode *findColorRGBANode() const {
		return (ColorRGBANode *)findNode(COLORRGBA_NODE);
	}

	TriangleSetNode *findTriangleSetNode() const {
		return (TriangleSetNode *)findNode(TRIANGLESET_NODE);
	}

	TriangleFanSetNode *findTriangleFanSetNode()const  {
		return (TriangleFanSetNode *)findNode(TRIANGLEFANSET_NODE);
	}

	TriangleStripSetNode *findTriangleStripSetNode() const {
		return (TriangleStripSetNode *)findNode(TRIANGLESTRIPSET_NODE);
	}
	
	// 12. Shape component (X3D)

	FillPropertiesNode *findFillPropertiesNode() const {
		return (FillPropertiesNode *)findNode(FILLPROPERTIES_NODE);
	}

	LinePropertiesNode *findLinePropertiesNode() const {
		return (LinePropertiesNode *)findNode(LINEPROPERTIES_NODE);
	}

	// 14. Geometry2D component (X3D)

	Arc2DNode *findArc2DNode() const {
		return (Arc2DNode *)findNode(ARC2D_NODE);
	}

	ArcClose2DNode *findArcClose2DNode() const {
		return (ArcClose2DNode *)findNode(ARCCLOSE2D_NODE);
	}

	Circle2DNode *findCircle2DNode() const {
		return (Circle2DNode *)findNode(CIRCLE2D_NODE);
	}

	Disk2DNode *findDisk2DNode() const {
		return (Disk2DNode *)findNode(DISK2D_NODE);
	}

	Polyline2DNode *findPolyline2DNode() const {
		return (Polyline2DNode *)findNode(POLYLINE2D_NODE);
	}

	Polypoint2DNode *findPolypoint2DNode() const {
		return (Polypoint2DNode *)findNode(POLYPOINT2D_NODE);
	}

	Rectangle2DNode *findRectangle2DNode() const {
		return (Rectangle2DNode *)findNode(RECTANGLE2D_NODE);
	}

	TriangleSet2DNode *findTriangleSet2DNode() const {
		return (TriangleSet2DNode *)findNode(TRIANGLESET2D_NODE);
	}
	
	// 18. Texturing component (x3D)

	MultiTextureNode *findMultiTextureNode() const {
		return (MultiTextureNode *)findNode(MULTITEXTURE_NODE);
	}

	MultiTextureCoordinateNode *findMultiTextureCoordinateNode() const {
		return (MultiTextureCoordinateNode *)findNode(MULTITEXTURECOORD_NODE);
	}

	MultiTextureTransformNode *findMultiTextureTransformNode() const {
		return (MultiTextureTransformNode *)findNode(MULTITEXTURETRANSFORM_NODE);
	}
	
	TextureCoordinateGeneratorNode *findTextureCoordinateGeneratorNode() const {
		return (TextureCoordinateGeneratorNode *)findNode(TEXCOORDGEN_NODE);
	}
	
	// 19. Interpolation component (X3D)

	CoordinateInterpolator2DNode *findCoordinateInterpolator2DNode() const {
		return (CoordinateInterpolator2DNode *)findNode(COORDINATEINTERPOLATOR2D_NODE);
	}

	PositionInterpolator2DNode *findPositionInterpolator2DNode() const {
		return (PositionInterpolator2DNode *)findNode(POSITIONINTERPOLATOR2D_NODE);
	}

	// 21. Key device sensor component (X3D)

	KeySensorNode *findKeySensorNode() const {
		return (KeySensorNode *)findNode(KEYSENSOR_NODE);
	}

	StringSensorNode *findStringSensorNode() const {
		return (StringSensorNode *)findNode(STRINGSENSOR_NODE);
	}

	// 30. Event Utilities component (X3D)

	BooleanFilterNode *findBooleanFilterNode() const {
		return (BooleanFilterNode *)findNode(BOOLEANFILTER_NODE);
	}

	BooleanToggleNode *findBooleanToggleNode() const {
		return (BooleanToggleNode *)findNode(BOOLEANTOGGLE_NODE);
	}

	BooleanTriggerNode *findBooleanTriggerNode() const {
		return (BooleanTriggerNode *)findNode(BOOLEANTRIGGER_NODE);
	}

	BooleanSequencerNode *findBooleanSequencerNode() const {
		return (BooleanSequencerNode *)findNode(BOOLEANSEQUENCER_NODE);
	}

	IntegerTriggerNode *findIntegerTriggerNode() const {
		return (IntegerTriggerNode *)findNode(INTEGERTRIGGER_NODE);
	}

	IntegerSequencerNode *findIntegerSequencerNode() const {
		return (IntegerSequencerNode *)findNode(INTEGERSEQUENCER_NODE);
	}

	TimeTriggerNode *findTimeTriggerNode() const {
		return (TimeTriggerNode *)findNode(TIMETRIGGER_NODE);
	}
	
	// Deprecated components (X3D)

	NodeSequencerNode *findNodeSequencerNode() const {
		return (NodeSequencerNode *)findNode(NODESEQUENCER_NODE);
	}

	Shape2DNode *findShape2DNode() const {
		return (Shape2DNode *)findNode(SHAPE2D_NODE);
	}

	BooleanTimeTriggerNode *findBooleanTimeTriggerNode() const {
		return (BooleanTimeTriggerNode *)findNode(BOOLEANTIMETRIGGER_NODE);
	}

	Transform2DNode *findTransform2DNode() const {
		return (Transform2DNode *)findNode(TRANSFORM2D_NODE);
	}

	////////////////////////////////////////////////
	//	find*(const char *name)
	////////////////////////////////////////////////


	AnchorNode *findAnchorNode(const char *name)const  {
		return (AnchorNode *)findNode(ANCHOR_NODE, name);
	}

	AppearanceNode *findAppearanceNode(const char *name) const {
		return (AppearanceNode *)findNode(APPEARANCE_NODE, name);
	}

	AudioClipNode *findAudioClipNode(const char *name) const {
		return (AudioClipNode *)findNode(AUDIOCLIP_NODE, name);
	}

	BackgroundNode *findBackgroundNode(const char *name) const {
		return (BackgroundNode *)findNode(BACKGROUND_NODE, name);
	}

	BillboardNode *findBillboardNode(const char *name) const {
		return (BillboardNode *)findNode(BILLBOARD_NODE, name);
	}

	BoxNode *findBoxNode(const char *name) const {
		return (BoxNode *)findNode(BOX_NODE, name);
	}

	CollisionNode *findCollisionNode(const char *name) const {
		return (CollisionNode *)findNode(COLLISION_NODE, name);
	}

	ColorNode *findColorNode(const char *name) const {
		return (ColorNode *)findNode(COLOR_NODE, name);
	}

	ColorInterpolatorNode *findColorInterpolatorNode(const char *name) const {
		return (ColorInterpolatorNode *)findNode(COLORINTERPOLATOR_NODE, name);
	}

	ConeNode *findConeNode(const char *name) const {
		return (ConeNode *)findNode(CONE_NODE, name);
	}

	CoordinateNode *findCoordinateNode(const char *name) const {
		return (CoordinateNode *)findNode(COORDINATE_NODE, name);
	}

	CoordinateInterpolatorNode *findCoordinateInterpolatorNode(const char *name) const {
		return (CoordinateInterpolatorNode *)findNode(COORDINATEINTERPOLATOR_NODE, name);
	}

	CylinderNode *findCylinderNode(const char *name) const {
		return (CylinderNode *)findNode(CYLINDER_NODE, name);
	}

	CylinderSensorNode *findCylinderSensorNode(const char *name) const {
		return (CylinderSensorNode *)findNode(CYLINDERSENSOR_NODE, name);
	}

	DirectionalLightNode *findDirectionalLightNode(const char *name) const {
		return (DirectionalLightNode *)findNode(DIRECTIONALLIGHT_NODE, name);
	}

	ElevationGridNode *findElevationGridNode(const char *name) const {
		return (ElevationGridNode *)findNode(ELEVATIONGRID_NODE, name);
	}

	ExtrusionNode *findExtrusionNode(const char *name) const {
		return (ExtrusionNode *)findNode(EXTRUSION_NODE, name);
	}

	FogNode *findFogNode(const char *name) const {
		return (FogNode *)findNode(FOG_NODE, name);
	}

	FontStyleNode *findFontStyleNode(const char *name) const {
		return (FontStyleNode *)findNode(FONTSTYLE_NODE, name);
	}

	GroupNode *findGroupNode(const char *name) const {
		return (GroupNode *)findNode(GROUP_NODE, name);
	}

	ImageTextureNode *findImageTextureNode(const char *name) const {
		return (ImageTextureNode *)findNode(IMAGETEXTURE_NODE, name);
	}

	IndexedFaceSetNode *findIndexedFaceSetNode(const char *name) const {
		return (IndexedFaceSetNode *)findNode(INDEXEDFACESET_NODE, name);
	}

	IndexedLineSetNode *findIndexedLineSetNode(const char *name) const {
		return (IndexedLineSetNode *)findNode(INDEXEDLINESET_NODE, name);
	}

	InlineNode *findInlineNode(const char *name) const {
		return (InlineNode *)findNode(INLINE_NODE, name);
	}

	LODNode *findLODNode(const char *name) const {
		return (LODNode *)findNode(LOD_NODE, name);
	}

	MaterialNode *findMaterialNode(const char *name) const {
		return (MaterialNode *)findNode(MATERIAL_NODE, name);
	}

	MovieTextureNode *findMovieTextureNode(const char *name) const {
		return (MovieTextureNode *)findNode(MOVIETEXTURE_NODE, name);
	}

	NavigationInfoNode *findNavigationInfoNode(const char *name) const {
		return (NavigationInfoNode *)findNode(NAVIGATIONINFO_NODE, name);
	}

	NormalNode *findNormalNode(const char *name) const {
		return (NormalNode *)findNode(NORMAL_NODE, name);
	}

	NormalInterpolatorNode *findNormalInterpolatorNode(const char *name) const {
		return (NormalInterpolatorNode *)findNode(NORMALINTERPOLATOR_NODE, name);
	}

	OrientationInterpolatorNode *findOrientationInterpolatorNode(const char *name) const {
		return (OrientationInterpolatorNode *)findNode(ORIENTATIONINTERPOLATOR_NODE, name);
	}

	PixelTextureNode *findPixelTextureNode(const char *name) const {
		return (PixelTextureNode *)findNode(PIXELTEXTURE_NODE, name);
	}

	PlaneSensorNode *findPlaneSensorNode(const char *name) const {
		return (PlaneSensorNode *)findNode(PLANESENSOR_NODE, name);
	}

	PointLightNode *findPointLightNode(const char *name) const {
		return (PointLightNode *)findNode(POINTLIGHT_NODE, name);
	}

	PointSetNode *findPointSetNode(const char *name) const {
		return (PointSetNode *)findNode(POINTSET_NODE, name);
	}

	PositionInterpolatorNode *findPositionInterpolatorNode(const char *name) const {
		return (PositionInterpolatorNode *)findNode(POSITIONINTERPOLATOR_NODE, name);
	}

	ProximitySensorNode *findProximitySensorNode(const char *name) const {
		return (ProximitySensorNode *)findNode(PROXIMITYSENSOR_NODE, name);
	}

	ScalarInterpolatorNode *findScalarInterpolatorNode(const char *name) const {
		return (ScalarInterpolatorNode *)findNode(SCALARINTERPOLATOR_NODE, name);
	}

	ScriptNode *findScriptNode(const char *name) const {
		return (ScriptNode *)findNode(SCRIPT_NODE, name);
	}

	ShapeNode *findShapeNode(const char *name) const {
		return (ShapeNode *)findNode(SHAPE_NODE, name);
	}

	SoundNode *findSoundNode(const char *name) const {
		return (SoundNode *)findNode(SOUND_NODE, name);
	}

	SphereNode *findSphereNode(const char *name) const {
		return (SphereNode *)findNode(SPHERE_NODE, name);
	}

	SphereSensorNode *findSphereSensorNode(const char *name) const {
		return (SphereSensorNode *)findNode(SPHERESENSOR_NODE, name);
	}

	SpotLightNode *findSpotLightNode(const char *name) const {
		return (SpotLightNode *)findNode(SPOTLIGHT_NODE, name);
	}

	SwitchNode *findSwitchNode(const char *name) const {
		return (SwitchNode *)findNode(SWITCH_NODE, name);
	}

	TextNode *findTextNode(const char *name) const {
		return (TextNode *)findNode(TEXT_NODE, name);
	}

	TextureCoordinateNode *findTextureCoordinateNode(const char *name) const {
		return (TextureCoordinateNode *)findNode(TEXTURECOORDINATE_NODE, name);
	}

	TextureTransformNode *findTextureTransformNode(const char *name) const {
		return (TextureTransformNode *)findNode(TEXTURETRANSFORM_NODE, name);
	}

	TimeSensorNode *findTimeSensorNode(const char *name) const {
		return (TimeSensorNode *)findNode(TIMESENSOR_NODE, name);
	}

	TouchSensorNode *findTouchSensorNode(const char *name) const {
		return (TouchSensorNode *)findNode(TOUCHSENSOR_NODE, name);
	}

	TransformNode *findTransformNode(const char *name) const {
		return (TransformNode *)findNode(TRANSFORM_NODE, name);
	}

	ViewpointNode *findViewpointNode(const char *name) const {
		return (ViewpointNode *)findNode(VIEWPOINT_NODE, name);
	}

	VisibilitySensorNode *findVisibilitySensorNode(const char *name) const {
		return (VisibilitySensorNode *)findNode(VISIBILITYSENSOR_NODE, name);
	}

	WorldInfoNode *findWorldInfoNode(const char *name) const {
		return (WorldInfoNode *)findNode(WORLDINFO_NODE, name);
	}

	// 9. Networking component (X3D)

	LoadSensorNode *findLoadSensorNode(const char *name) const {
		return (LoadSensorNode *)findNode(LOADSENSOR_NODE, name);
	}

	// 10. Grouping component (X3D)

	StaticGroupNode *findStaticGroupNode(const char *name) const {
		return (StaticGroupNode *)findNode(STATICGROUP_NODE, name);
	}

	// 11. Rendering component (X3D)

	ColorRGBANode *findColorRGBANode(const char *name) const {
		return (ColorRGBANode *)findNode(COLORRGBA_NODE, name);
	}

	TriangleSetNode *findTriangleSetNode(const char *name) const {
		return (TriangleSetNode *)findNode(TRIANGLESET_NODE, name);
	}

	TriangleFanSetNode *findTriangleFanSetNode(const char *name) const {
		return (TriangleFanSetNode *)findNode(TRIANGLEFANSET_NODE, name);
	}

	TriangleStripSetNode *findTriangleStripSetNode(const char *name) const {
		return (TriangleStripSetNode *)findNode(TRIANGLESTRIPSET_NODE, name);
	}
	
	// 12. Shape component (X3D)

	FillPropertiesNode *findFillPropertiesNode(const char *name) const {
		return (FillPropertiesNode *)findNode(FILLPROPERTIES_NODE, name);
	}

	LinePropertiesNode *findLinePropertiesNode(const char *name) const {
		return (LinePropertiesNode *)findNode(LINEPROPERTIES_NODE, name);
	}

	// 14. Geometry2D component (X3D)

	Arc2DNode *findArc2DNode(const char *name) const {
		return (Arc2DNode *)findNode(ARC2D_NODE, name);
	}

	ArcClose2DNode *findArcClose2DNode(const char *name) const {
		return (ArcClose2DNode *)findNode(ARCCLOSE2D_NODE, name);
	}

	Circle2DNode *findCircle2DNode(const char *name) const {
		return (Circle2DNode *)findNode(CIRCLE2D_NODE, name);
	}

	Disk2DNode *findDisk2DNode(const char *name) const {
		return (Disk2DNode *)findNode(DISK2D_NODE, name);
	}

	Polyline2DNode *findPolyline2DNode(const char *name) const {
		return (Polyline2DNode *)findNode(POLYLINE2D_NODE, name);
	}

	Polypoint2DNode *findPolypoint2DNode(const char *name) const {
		return (Polypoint2DNode *)findNode(POLYPOINT2D_NODE, name);
	}

	Rectangle2DNode *findRectangle2DNode(const char *name) const {
		return (Rectangle2DNode *)findNode(RECTANGLE2D_NODE, name);
	}

	TriangleSet2DNode *findTriangleSet2DNode(const char *name) const {
		return (TriangleSet2DNode *)findNode(TRIANGLESET2D_NODE, name);
	}
	
	// 18. Texturing component (x3D)

	MultiTextureNode *findMultiTextureNode(const char *name) const {
		return (MultiTextureNode *)findNode(MULTITEXTURE_NODE, name);
	}

	MultiTextureCoordinateNode *findMultiTextureCoordinateNode(const char *name) const {
		return (MultiTextureCoordinateNode *)findNode(MULTITEXTURECOORD_NODE, name);
	}

	MultiTextureTransformNode *findMultiTextureTransformNode(const char *name) const {
		return (MultiTextureTransformNode *)findNode(MULTITEXTURETRANSFORM_NODE, name);
	}
	
	TextureCoordinateGeneratorNode *findTextureCoordinateGeneratorNode(const char *name) const {
		return (TextureCoordinateGeneratorNode *)findNode(TEXCOORDGEN_NODE, name);
	}
	
	// 19. Interpolation component (X3D)

	CoordinateInterpolator2DNode *findCoordinateInterpolator2DNode(const char *name) const {
		return (CoordinateInterpolator2DNode *)findNode(COORDINATEINTERPOLATOR2D_NODE, name);
	}

	PositionInterpolator2DNode *findPositionInterpolator2DNode(const char *name) const {
		return (PositionInterpolator2DNode *)findNode(POSITIONINTERPOLATOR2D_NODE, name);
	}

	// 21. Key device sensor component (X3D)

	KeySensorNode *findKeySensorNode(const char *name) const {
		return (KeySensorNode *)findNode(KEYSENSOR_NODE, name);
	}

	StringSensorNode *findStringSensorNode(const char *name) const {
		return (StringSensorNode *)findNode(STRINGSENSOR_NODE, name);
	}

	// 30. Event Utilities component (X3D)

	BooleanFilterNode *findBooleanFilterNode(const char *name) const {
		return (BooleanFilterNode *)findNode(BOOLEANFILTER_NODE, name);
	}

	BooleanToggleNode *findBooleanToggleNode(const char *name) const {
		return (BooleanToggleNode *)findNode(BOOLEANTOGGLE_NODE, name);
	}

	BooleanTriggerNode *findBooleanTriggerNode(const char *name) const {
		return (BooleanTriggerNode *)findNode(BOOLEANTRIGGER_NODE, name);
	}

	BooleanSequencerNode *findBooleanSequencerNode(const char *name) const {
		return (BooleanSequencerNode *)findNode(BOOLEANSEQUENCER_NODE, name);
	}

	IntegerTriggerNode *findIntegerTriggerNode(const char *name) const {
		return (IntegerTriggerNode *)findNode(INTEGERTRIGGER_NODE, name);
	}

	IntegerSequencerNode *findIntegerSequencerNode(const char *name) const {
		return (IntegerSequencerNode *)findNode(INTEGERSEQUENCER_NODE, name);
	}

	TimeTriggerNode *findTimeTriggerNode(const char *name) const {
		return (TimeTriggerNode *)findNode(TIMETRIGGER_NODE, name);
	}
	
	// Deprecated components (X3D)

	NodeSequencerNode *findNodeSequencerNode(const char *name) const {
		return (NodeSequencerNode *)findNode(NODESEQUENCER_NODE, name);
	}

	Shape2DNode *findShape2DNode(const char *name) const {
		return (Shape2DNode *)findNode(SHAPE2D_NODE, name);
	}

	BooleanTimeTriggerNode *findBooleanTimeTriggerNode(const char *name) const {
		return (BooleanTimeTriggerNode *)findNode(BOOLEANTIMETRIGGER_NODE, name);
	}

	Transform2DNode *findTransform2DNode(const char *name) const {
		return (Transform2DNode *)findNode(TRANSFORM2D_NODE, name);
	}

};

}

#endif
