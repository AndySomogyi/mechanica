/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	Node.h
*
******************************************************************/

#ifndef _CX3D_NODE_H_
#define _CX3D_NODE_H_

#include <iostream>
#include <fstream>
#include <assert.h>

#include <x3d/CyberX3DConfig.h>

#include <x3d/Vector.h>
#include <x3d/Field.h>
#include <x3d/StringUtil.h>
#include <x3d/LinkedList.h>
#include <x3d/JNode.h>

#include <x3d/NodeType.h>
#include <x3d/NodeString.h>

namespace CyberX3D {

class	SceneGraph;

class	SFMatrix;
	
class	AnchorNode;
class	AppearanceNode;
class	AudioClipNode;
class	BackgroundNode;
class	BillboardNode;
class	BoxNode;
class	CollisionNode;
class	ColorNode;
class	ColorInterpolatorNode;
class	ConeNode;
class	CoordinateNode;
class	CoordinateInterpolatorNode;
class	CylinderNode;
class	CylinderSensorNode;
class	DirectionalLightNode;
class	ElevationGridNode;
class	ExtrusionNode;
class	FogNode;
class	FontStyleNode;
class	GroupNode;
class	ImageTextureNode;
class	IndexedFaceSetNode;
class	IndexedLineSetNode;
class	InlineNode;
class	LODNode;
class	MaterialNode;
class	MovieTextureNode;
class	NavigationInfoNode;
class	NormalNode;
class	NormalInterpolatorNode;
class	OrientationInterpolatorNode;
class	PixelTextureNode;
class	PlaneSensorNode;
class	PointLightNode;
class	PointSetNode;
class	PositionInterpolatorNode;
class	ProximitySensorNode;
class	ScalarInterpolatorNode;
class	ScriptNode;
class	ShapeNode;
class	SoundNode;
class	SphereNode;
class	SphereSensorNode;
class	SpotLightNode;
class	SwitchNode;
class	TextNode;
class	TextureNode;
class	TextureCoordinateNode;
class	TextureTransformNode;
class	TimeSensorNode;
class	TouchSensorNode;
class	TransformNode;
class	ViewpointNode;
class	VisibilitySensorNode;
class	WorldInfoNode;

// 9. Networking component (X3D)
class LoadSensorNode;
// 10. Grouping component (X3D)
class StaticGroupNode;
// 11. Rendering component (X3D)
class ColorRGBANode;
class TriangleSetNode;
class TriangleFanSetNode;
class TriangleStripSetNode;
// 12. Shape component (X3D)
class FillPropertiesNode;
class LinePropertiesNode;
// 14. Geometry2D component (X3D)
class Arc2DNode;
class ArcClose2DNode;
class Circle2DNode;
class Disk2DNode;
class Polyline2DNode;
class Polypoint2DNode;
class Rectangle2DNode;
class TriangleSet2DNode;
// 18. Texturing component (x3D)
class MultiTextureNode;
class MultiTextureCoordinateNode;
class MultiTextureTransformNode;
class TextureCoordinateGeneratorNode;
// 19. Interpolation component (X3D)
class CoordinateInterpolator2DNode;
class PositionInterpolator2DNode;
// 21. Key device sensor component (X3D)
class KeySensorNode;
class StringSensorNode;
// 30. Event Utilities component (X3D)
class BooleanFilterNode;
class BooleanToggleNode;
class BooleanTriggerNode;
class BooleanSequencerNode;
class IntegerTriggerNode;
class IntegerSequencerNode;
class TimeTriggerNode;
// Deprecated components (X3D)
class NodeSequencerNode;
class Shape2DNode;
class BooleanTimeTriggerNode;
class Transform2DNode;

class	GroupingNode;
class	Geometry3DNode;

class	DEFNode;

class Node : public LinkedListNode<Node> {

public:
	String				*mName;
	int						mType;
	Vector<Field>		*mExposedField;
	Vector<Field>		*mEventInField;
	Vector<Field>		*mEventOutField;
	Vector<Field>		*mField;
	Vector<Field>		*mPrivateField;
	Vector<Node>		*mPrivateNodeVector;
	bool				*mInitialized;

	String				*mOrgName;
	int						mOrgType;
	Vector<Field>		*mOrgExposedField;
	Vector<Field>		*mOrgEventInField;
	Vector<Field>		*mOrgEventOutField;
	Vector<Field>		*mOrgField;
	Vector<Field>		*mOrgPrivateField;

private:
	Node				*mParentNode;
	LinkedList<Node>	*mChildNodes;
	SceneGraph			*mSceneGraph;

#if defined(CX3D_SUPPORT_JSAI)
	JNode				*mJNode;
#endif

	void				*mValue;
	Node				*mReferenceNode;

public:

	Node();
	
	Node(const int nodeType, const char * nodeName);
	
	virtual ~Node();

	void initializeMember();

	void remove();

	void setName(const char * name);
	const char *getName() const;
	bool hasName() const;

	void setType(const int type);
	int getType() const;
	const char *getTypeString() const;

	////////////////////////////////////////////////
	//	Java
	////////////////////////////////////////////////

#if defined(CX3D_SUPPORT_JSAI)

	void createJavaNodeObject() {
		mJNode = new JNode(this);
	}

	void setJavaNodeObject(JNode *jnode) {
		mJNode = jnode;
	}

	JNode *getJavaNodeObject() {
		return mJNode;
	}

#endif

	////////////////////////////////////////////////
	//	Field
	////////////////////////////////////////////////

	static Field *createField(int type);
	int getNAllFields() const;

	Field *findField(const char *name) const;
	bool hasMField() const;

	////////////////////////////////////////////////
	//	EventIn
	////////////////////////////////////////////////

	Field *getEventIn(const char * fieldString) const;
	int getNEventIn() const;
	void addEventIn(Field *field);
	void addEventIn(const char * name, Field *field);
	void addEventIn(const char * name, int fieldType);
	Field *getEventIn(int index) const;
	int getEventInNumber(Field *field) const;

	////////////////////////////////////////////////
	//	EventOut
	////////////////////////////////////////////////

	Field *getEventOut(const char *fieldString) const;
	int getNEventOut() const;
	void addEventOut(Field *field);
	void addEventOut(const char *name, Field *field);
	void addEventOut(const char * name, int fieldType);
	Field *getEventOut(int index) const;
	int getEventOutNumber(Field *field) const;

	////////////////////////////////////////////////
	//	ExposedField
	////////////////////////////////////////////////

	Field *getExposedField(const char * fieldString) const;
	int getNExposedFields() const;
	void addExposedField(Field *field);
	void addExposedField(const char * name, Field *field);
	void addExposedField(const char * name, int fieldType);
	Field *getExposedField(int index) const;
	int getExposedFieldNumber(Field *field) const;

	////////////////////////////////////////////////
	//	Field
	////////////////////////////////////////////////

	Field *getField(const char *fieldString) const;
	int getNFields() const;
	void addField(Field *field);
	void addField(const char * name, Field *field);
	void addField(const char * name, int fieldType);
	Field *getField(int index) const;
	int getFieldNumber(Field *field) const;

	////////////////////////////////////////////////
	//	PrivateField
	////////////////////////////////////////////////

	Field *getPrivateField(const char *fieldString) const;
	int getNPrivateFields() const;
	void addPrivateField(Field *field);
	void addPrivateField(const char * name, Field *field);
	Field *getPrivateField(int index) const;
	int getPrivateFieldNumber(Field *field) const;

	////////////////////////////////////////////////
	//	PrivateField
	////////////////////////////////////////////////

	int getNPrivateNodeElements() const;
	void addPrivateNodeElement(Node *node);
	Node *getPrivateNodeElementAt(int n) const;
	void removeAllNodeElement();

	////////////////////////////////////////////////
	//	Parent node
	////////////////////////////////////////////////

	void setParentNode(Node *parentNode);
	Node *getParentNode() const;
	bool isParentNode(Node *node) const;
	bool isAncestorNode(Node *node) const;

	////////////////////////////////////////////////
	//	Traversal node list
	////////////////////////////////////////////////

	Node *nextTraversal() const;
	Node *nextTraversalByType(const int type) const;
	Node *nextTraversalByName(const char *nameString) const;
	
	Node *nextTraversalSameType() const {
		return nextTraversalByType(getType());
	}

	////////////////////////////////////////////////
	//	next node list
	////////////////////////////////////////////////

	Node *next() const;
	Node *next(const int type) const;

	////////////////////////////////////////////////
	//	child node list
	////////////////////////////////////////////////

	Node *getChildNodes() const;
	Node *getChildNodeByType(int type) const;
	Node *getChildNode(int n) const;
	
	int getNChildNodes() const;
	bool hasChildNodes() const;
	
	virtual bool isChildNodeType(Node *node) const = 0;

	void addChildNode(Node *node, bool initialize = true);
	void addChildNodeAtFirst(Node *node, bool initialize = true);

	void moveChildNode(Node *node);
	void moveChildNodeAtFirst(Node *node);

	void deleteChildNodes(void);

	void removeRoutes();
	void removeSFNodes();
	void removeInstanceNodes();

	////////////////////////////////////////////////
	//	Add / Remove children (for Groupingnode)
	////////////////////////////////////////////////

	bool isChildNode(Node *parentNode, Node *node) const;
	bool isChildNode(Node *node) const;

	////////////////////////////////////////////////
	//	get child node list
	////////////////////////////////////////////////

	GroupingNode *getGroupingNodes() const;
	Geometry3DNode *getGeometry3DNode() const;
	TextureNode *getTextureNode() const;

	AnchorNode *getAnchorNodes() const {
		return (AnchorNode *)getChildNodeByType(ANCHOR_NODE);
	}

	AppearanceNode *getAppearanceNodes() const {
		return (AppearanceNode *)getChildNodeByType(APPEARANCE_NODE);
	}

	AudioClipNode *getAudioClipNodes() const {
		return (AudioClipNode *)getChildNodeByType(AUDIOCLIP_NODE);
	}

	BackgroundNode *getBackgroundNodes() const {
		return (BackgroundNode *)getChildNodeByType(BACKGROUND_NODE);
	}

	BillboardNode *getBillboardNodes() const {
		return (BillboardNode *)getChildNodeByType(BILLBOARD_NODE);
	}

	BoxNode *getBoxeNodes() const {
		return (BoxNode *)getChildNodeByType(BOX_NODE);
	}

	CollisionNode *getCollisionNodes() const {
		return (CollisionNode *)getChildNodeByType(COLLISION_NODE);
	}

	ColorNode *getColorNodes() const {
		return (ColorNode *)getChildNodeByType(COLOR_NODE);
	}

	ColorInterpolatorNode *getColorInterpolatorNodes() const {
		return (ColorInterpolatorNode *)getChildNodeByType(COLORINTERPOLATOR_NODE);
	}

	ConeNode *getConeNodes() const {
		return (ConeNode *)getChildNodeByType(CONE_NODE);
	}

	CoordinateNode *getCoordinateNodes() const {
		return (CoordinateNode *)getChildNodeByType(COORDINATE_NODE);
	}

	CoordinateInterpolatorNode *getCoordinateInterpolatorNodes() const {
		return (CoordinateInterpolatorNode *)getChildNodeByType(COORDINATEINTERPOLATOR_NODE);
	}

	CylinderNode *getCylinderNodes() const {
		return (CylinderNode *)getChildNodeByType(CYLINDER_NODE);
	}

	CylinderSensorNode *getCylinderSensorNodes() const {
		return (CylinderSensorNode *)getChildNodeByType(CYLINDERSENSOR_NODE);
	}

	DirectionalLightNode *getDirectionalLightNodes() const {
		return (DirectionalLightNode *)getChildNodeByType(DIRECTIONALLIGHT_NODE);
	}

	ElevationGridNode *getElevationGridNodes() const {
		return (ElevationGridNode *)getChildNodeByType(ELEVATIONGRID_NODE);
	}

	ExtrusionNode *getExtrusionNodes() const {
		return (ExtrusionNode *)getChildNodeByType(EXTRUSION_NODE);
	}

	FogNode *getFogNodes() const {
		return (FogNode *)getChildNodeByType(FOG_NODE);
	}

	FontStyleNode *getFontStyleNodes() const {
		return (FontStyleNode *)getChildNodeByType(FONTSTYLE_NODE);
	}

	GroupNode *getGroupNodes() const {
		return (GroupNode *)getChildNodeByType(GROUP_NODE);
	}

	ImageTextureNode *getImageTextureNodes() const {
		return (ImageTextureNode *)getChildNodeByType(IMAGETEXTURE_NODE);
	}

	IndexedFaceSetNode *getIndexedFaceSetNodes() const {
		return (IndexedFaceSetNode *)getChildNodeByType(INDEXEDFACESET_NODE);
	}

	IndexedLineSetNode *getIndexedLineSetNodes() const {
		return (IndexedLineSetNode *)getChildNodeByType(INDEXEDLINESET_NODE);
	}

	InlineNode *getInlineNodes() const {
		return (InlineNode *)getChildNodeByType(INLINE_NODE);
	}

	LODNode *getLODNodes() const {
		return (LODNode *)getChildNodeByType(LOD_NODE);
	}

	MaterialNode *getMaterialNodes() const {
		return (MaterialNode *)getChildNodeByType(MATERIAL_NODE);
	}

	MovieTextureNode *getMovieTextureNodes() const {
		return (MovieTextureNode *)getChildNodeByType(MOVIETEXTURE_NODE);
	}

	NavigationInfoNode *getNavigationInfoNodes() const {
		return (NavigationInfoNode *)getChildNodeByType(NAVIGATIONINFO_NODE);
	}

	NormalNode *getNormalNodes() const {
		return (NormalNode *)getChildNodeByType(NORMAL_NODE);
	}

	NormalInterpolatorNode *getNormalInterpolatorNodes() const {
		return (NormalInterpolatorNode *)getChildNodeByType(NORMALINTERPOLATOR_NODE);
	}

	OrientationInterpolatorNode *getOrientationInterpolatorNodes() const {
		return (OrientationInterpolatorNode *)getChildNodeByType(ORIENTATIONINTERPOLATOR_NODE);
	}

	PixelTextureNode *getPixelTextureNodes() const {
		return (PixelTextureNode *)getChildNodeByType(PIXELTEXTURE_NODE);
	}

	PlaneSensorNode *getPlaneSensorNodes() const {
		return (PlaneSensorNode *)getChildNodeByType(PLANESENSOR_NODE);
	}

	PointLightNode *getPointLightNodes() const {
		return (PointLightNode *)getChildNodeByType(POINTLIGHT_NODE);
	}

	PointSetNode *getPointSetNodes() const {
		return (PointSetNode *)getChildNodeByType(POINTSET_NODE);
	}

	PositionInterpolatorNode *getPositionInterpolatorNodes() const {
		return (PositionInterpolatorNode *)getChildNodeByType(POSITIONINTERPOLATOR_NODE);
	}

	ProximitySensorNode *getProximitySensorNodes() const {
		return (ProximitySensorNode *)getChildNodeByType(PROXIMITYSENSOR_NODE);
	}

	ScalarInterpolatorNode *getScalarInterpolatorNodes() const {
		return (ScalarInterpolatorNode *)getChildNodeByType(SCALARINTERPOLATOR_NODE);
	}

	ScriptNode *getScriptNodes() const {
		return (ScriptNode *)getChildNodeByType(SCRIPT_NODE);
	}

	ShapeNode *getShapeNodes() const {
		return (ShapeNode *)getChildNodeByType(SHAPE_NODE);
	}

	SoundNode *getSoundNodes() const {
		return (SoundNode *)getChildNodeByType(SOUND_NODE);
	}

	SphereNode *getSphereNodes() const {
		return (SphereNode *)getChildNodeByType(SPHERE_NODE);
	}

	SphereSensorNode *getSphereSensorNodes() const {
		return (SphereSensorNode *)getChildNodeByType(SPHERESENSOR_NODE);
	}

	SpotLightNode *getSpotLightNodes() const {
		return (SpotLightNode *)getChildNodeByType(SPOTLIGHT_NODE);
	}

	SwitchNode *getSwitchNodes() const {
		return (SwitchNode *)getChildNodeByType(SWITCH_NODE);
	}

	TextNode *getTextNodes() const {
		return (TextNode *)getChildNodeByType(TEXT_NODE);
	}

	TextureCoordinateNode *getTextureCoordinateNodes() const {
		return (TextureCoordinateNode *)getChildNodeByType(TEXTURECOORDINATE_NODE);
	}

	TextureTransformNode *getTextureTransformNodes() const {
		return (TextureTransformNode *)getChildNodeByType(TEXTURETRANSFORM_NODE);
	}

	TimeSensorNode *getTimeSensorNodes() const {
		return (TimeSensorNode *)getChildNodeByType(TIMESENSOR_NODE);
	}

	TouchSensorNode *getTouchSensorNodes() const {
		return (TouchSensorNode *)getChildNodeByType(TOUCHSENSOR_NODE);
	}

	TransformNode *getTransformNodes() const {
		return (TransformNode *)getChildNodeByType(TRANSFORM_NODE);
	}

	ViewpointNode *getViewpointNodes() const {
		return (ViewpointNode *)getChildNodeByType(VIEWPOINT_NODE);
	}

	VisibilitySensorNode *getVisibilitySensorNodes() const {
		return (VisibilitySensorNode *)getChildNodeByType(VISIBILITYSENSOR_NODE);
	}

	WorldInfoNode *getWorldInfoNodes() const {
		return (WorldInfoNode *)getChildNodeByType(WORLDINFO_NODE);
	}

	// 9. Networking component (X3D)

	LoadSensorNode *getLoadSensorNodes() const {
		return (LoadSensorNode *)getChildNodeByType(LOADSENSOR_NODE);
	}

	// 10. Grouping component (X3D)

	StaticGroupNode *getStaticGroupNodes() const {
		return (StaticGroupNode *)getChildNodeByType(STATICGROUP_NODE);
	}

	// 11. Rendering component (X3D)

	ColorRGBANode *getColorRGBANodes() const {
		return (ColorRGBANode *)getChildNodeByType(COLORRGBA_NODE);
	}

	TriangleSetNode *getTriangleSetNodes() const {
		return (TriangleSetNode *)getChildNodeByType(TRIANGLESET_NODE);
	}

	TriangleFanSetNode *getTriangleFanSetNodes() const {
		return (TriangleFanSetNode *)getChildNodeByType(TRIANGLEFANSET_NODE);
	}

	TriangleStripSetNode *getTriangleStripSetNodes() const {
		return (TriangleStripSetNode *)getChildNodeByType(TRIANGLESTRIPSET_NODE);
	}
	
	// 12. Shape component (X3D)

	FillPropertiesNode *getFillPropertiesNodes() const {
		return (FillPropertiesNode *)getChildNodeByType(FILLPROPERTIES_NODE);
	}

	LinePropertiesNode *getLinePropertiesNodes() const {
		return (LinePropertiesNode *)getChildNodeByType(LINEPROPERTIES_NODE);
	}

	// 14. Geometry2D component (X3D)

	Arc2DNode *getArc2DNodes() const {
		return (Arc2DNode *)getChildNodeByType(ARC2D_NODE);
	}

	ArcClose2DNode *getArcClose2DNodes() const {
		return (ArcClose2DNode *)getChildNodeByType(ARCCLOSE2D_NODE);
	}

	Circle2DNode *getCircle2DNodes() const {
		return (Circle2DNode *)getChildNodeByType(CIRCLE2D_NODE);
	}

	Disk2DNode *getDisk2DNodes() const {
		return (Disk2DNode *)getChildNodeByType(DISK2D_NODE);
	}

	Polyline2DNode *getPolyline2DNodes() const {
		return (Polyline2DNode *)getChildNodeByType(POLYLINE2D_NODE);
	}

	Polypoint2DNode *getPolypoint2DNodes() const {
		return (Polypoint2DNode *)getChildNodeByType(POLYPOINT2D_NODE);
	}

	Rectangle2DNode *getRectangle2DNodes() const {
		return (Rectangle2DNode *)getChildNodeByType(RECTANGLE2D_NODE);
	}

	TriangleSet2DNode *getTriangleSet2DNodes() const {
		return (TriangleSet2DNode *)getChildNodeByType(TRIANGLESET2D_NODE);
	}
	
	// 18. Texturing component (x3D)

	MultiTextureNode *getMultiTextureNodes() const {
		return (MultiTextureNode *)getChildNodeByType(MULTITEXTURE_NODE);
	}

	MultiTextureCoordinateNode *getMultiTextureCoordinateNodes() const {
		return (MultiTextureCoordinateNode *)getChildNodeByType(MULTITEXTURECOORD_NODE);
	}

	MultiTextureTransformNode *getMultiTextureTransformNodes() const {
		return (MultiTextureTransformNode *)getChildNodeByType(MULTITEXTURETRANSFORM_NODE);
	}

	TextureCoordinateGeneratorNode *getTextureCoordinateGeneratorNodes() const {
		return (TextureCoordinateGeneratorNode *)getChildNodeByType(TEXCOORDGEN_NODE);
	}
	
	// 19. Interpolation component (X3D)

	CoordinateInterpolator2DNode *getCoordinateInterpolator2DNodes() const {
		return (CoordinateInterpolator2DNode *)getChildNodeByType(COORDINATEINTERPOLATOR2D_NODE);
	}

	PositionInterpolator2DNode *getPositionInterpolator2DNodes() const {
		return (PositionInterpolator2DNode *)getChildNodeByType(POSITIONINTERPOLATOR2D_NODE);
	}

	// 21. Key device sensor component (X3D)

	KeySensorNode *getKeySensorNodes() const {
		return (KeySensorNode *)getChildNodeByType(KEYSENSOR_NODE);
	}

	StringSensorNode *getStringSensorNodes() const {
		return (StringSensorNode *)getChildNodeByType(STRINGSENSOR_NODE);
	}

	// 30. Event Utilities component (X3D)

	BooleanFilterNode *getBooleanFilterNodes() const {
		return (BooleanFilterNode *)getChildNodeByType(BOOLEANFILTER_NODE);
	}

	BooleanToggleNode *getBooleanToggleNodes() const {
		return (BooleanToggleNode *)getChildNodeByType(BOOLEANTOGGLE_NODE);
	}

	BooleanTriggerNode *getBooleanTriggerNodes() const {
		return (BooleanTriggerNode *)getChildNodeByType(BOOLEANTRIGGER_NODE);
	}

	BooleanSequencerNode *getBooleanSequencerNodes() const {
		return (BooleanSequencerNode *)getChildNodeByType(BOOLEANSEQUENCER_NODE);
	}

	IntegerTriggerNode *getIntegerTriggerNodes() const {
		return (IntegerTriggerNode *)getChildNodeByType(INTEGERTRIGGER_NODE);
	}

	IntegerSequencerNode *getIntegerSequencerNodes() const {
		return (IntegerSequencerNode *)getChildNodeByType(INTEGERSEQUENCER_NODE);
	}

	TimeTriggerNode *getTimeTriggerNodes() const {
		return (TimeTriggerNode *)getChildNodeByType(TIMETRIGGER_NODE);
	}
	
	// Deprecated components (X3D)

	NodeSequencerNode *getNodeSequencerNodes() const {
		return (NodeSequencerNode *)getChildNodeByType(NODESEQUENCER_NODE);
	}

	Shape2DNode *getShape2DNodes() const {
		return (Shape2DNode *)getChildNodeByType(SHAPE2D_NODE);
	}

	BooleanTimeTriggerNode *getBooleanTimeTriggerNodes() const {
		return (BooleanTimeTriggerNode *)getChildNodeByType(BOOLEANTIMETRIGGER_NODE);
	}

	Transform2DNode *getTransform2DNodes() const {
		return (Transform2DNode *)getChildNodeByType(TRANSFORM2D_NODE);
	}

	////////////////////////////////////////////////
	//	is*
	////////////////////////////////////////////////

	bool isNode(const int type) const;
	bool isRootNode() const;
	bool isDEFNode() const;
	bool isInlineChildNode() const;

	bool isGroupingNode() const {
		if (isAnchorNode() || isBillboardNode() || isCollisionNode() || isGroupNode() || isLODNode() || isSwitchNode() || isTransformNode())
			return true;
		return false;
	}

	bool isBoundedGroupingNode() const {
		return isGroupingNode();
	}

	bool isSpecialGroupNode() const {
		if (isInlineNode() || isLODNode() || isSwitchNode())
			return true;
		return false;
	}

	bool isCommonNode() const {
		if (isLightNode() || isAudioClipNode() || isScriptNode() || isShapeNode() || isSoundNode() || isWorldInfoNode())
			return true;
		return false;
	}

	bool isLightNode() const {
		if (isDirectionalLightNode() || isSpotLightNode() || isPointLightNode())
			return true;
		return false;
	}

	bool isGeometry3DNode() const {
		if (isBoxNode() || isConeNode() || isCylinderNode() || isElevationGridNode() || isExtrusionNode() || isIndexedFaceSetNode() || isIndexedLineSetNode() || isPointSetNode() || isSphereNode() || isTextNode())
			return true;
		return false;
	}

	bool isGeometry3DPropertyNode() const {
		if (isColorNode() || isCoordinateNode() || isNormalNode() || isTextureCoordinateNode())
			return true;
		return false;
	}

	bool isTextureNode() const {
		if (isMovieTextureNode() || isPixelTextureNode() || isImageTextureNode() )
			return true;
		return false;
	}

	bool isSensorNode() const {
		if (isCylinderSensorNode() || isPlaneSensorNode() || isSphereSensorNode() || isProximitySensorNode() || isTimeSensorNode() || isTouchSensorNode() || isVisibilitySensorNode())
			return true;
		return false;
	}

	bool isInterpolatorNode() const {
		if (isColorInterpolatorNode() || isCoordinateInterpolatorNode() || isNormalInterpolatorNode() || isOrientationInterpolatorNode() || isPositionInterpolatorNode() || isScalarInterpolatorNode())
			return true;
		return false;
	}

	bool isBindableNode() const {
		if (isBackgroundNode() || isFogNode() || isNavigationInfoNode() || isViewpointNode())
			return true;
		return false;
	}

	// VRML97 component

	bool isAnchorNode() const {
		return isNode(ANCHOR_NODE);
	}

	bool isAppearanceNode() const {
		return isNode(APPEARANCE_NODE);
	}

	bool isAudioClipNode() const {
		return isNode(AUDIOCLIP_NODE);
	}

	bool isBackgroundNode() const {
		return isNode(BACKGROUND_NODE);
	}

	bool isBillboardNode() const {
		return isNode(BILLBOARD_NODE);
	}

	bool isBoxNode() const {
		return isNode(BOX_NODE);
	}

	bool isCollisionNode() const {
		return isNode(COLLISION_NODE);
	}

	bool isColorNode() const {
		return isNode(COLOR_NODE);
	}

	bool isColorInterpolatorNode() const {
		return isNode(COLORINTERPOLATOR_NODE);
	}

	bool isConeNode() const {
		return isNode(CONE_NODE);
	}

	bool isCoordinateNode() const {
		return isNode(COORDINATE_NODE);
	}

	bool isCoordinateInterpolatorNode() const {
		return isNode(COORDINATEINTERPOLATOR_NODE);
	}

	bool isCylinderNode() const {
		return isNode(CYLINDER_NODE);
	}

	bool isCylinderSensorNode() const {
		return isNode(CYLINDERSENSOR_NODE);
	}

	bool isDirectionalLightNode() const {
		return isNode(DIRECTIONALLIGHT_NODE);
	}

	bool isElevationGridNode() const {
		return isNode(ELEVATIONGRID_NODE);
	}

	bool isExtrusionNode() const {
		return isNode(EXTRUSION_NODE);
	}

	bool isFogNode() const {
		return isNode(FOG_NODE);
	}

	bool isFontStyleNode() const {
		return isNode(FONTSTYLE_NODE);
	}

	bool isGroupNode() const {
		return isNode(GROUP_NODE);
	}

	bool isImageTextureNode() const {
		return isNode(IMAGETEXTURE_NODE);
	}

	bool isIndexedFaceSetNode() const {
		return isNode(INDEXEDFACESET_NODE);
	}

	bool isIndexedLineSetNode() const {
		return isNode(INDEXEDLINESET_NODE);
	}

	bool isInlineNode() const {
		return isNode(INLINE_NODE);
	}

	bool isLODNode() const {
		return isNode(LOD_NODE);
	}

	bool isMaterialNode() const {
		return isNode(MATERIAL_NODE);
	}

	bool isMovieTextureNode() const {
		return isNode(MOVIETEXTURE_NODE);
	}

	bool isNavigationInfoNode() const {
		return isNode(NAVIGATIONINFO_NODE);
	}

	bool isNormalNode() const {
		return isNode(NORMAL_NODE);
	}

	bool isNormalInterpolatorNode() const {
		return isNode(NORMALINTERPOLATOR_NODE);
	}

	bool isOrientationInterpolatorNode() const {
		return isNode(ORIENTATIONINTERPOLATOR_NODE);
	}

	bool isPixelTextureNode() const {
		return isNode(PIXELTEXTURE_NODE);
	}

	bool isPlaneSensorNode() const {
		return isNode(PLANESENSOR_NODE);
	}

	bool isPointLightNode() const {
		return isNode(POINTLIGHT_NODE);
	}

	bool isPointSetNode() const {
		return isNode(POINTSET_NODE);
	}

	bool isPositionInterpolatorNode() const {
		return isNode(POSITIONINTERPOLATOR_NODE);
	}

	bool isProximitySensorNode() const {
		return isNode(PROXIMITYSENSOR_NODE);
	}

	bool isScalarInterpolatorNode() const {
		return isNode(SCALARINTERPOLATOR_NODE);
	}

	bool isScriptNode() const {
		return isNode(SCRIPT_NODE);
	}

	bool isShapeNode() const {
		return isNode(SHAPE_NODE);
	}

	bool isSoundNode() const {
		return isNode(SOUND_NODE);
	}

	bool isSphereNode() const {
		return isNode(SPHERE_NODE);
	}

	bool isSphereSensorNode() const {
		return isNode(SPHERESENSOR_NODE);
	}

	bool isSpotLightNode() const {
		return isNode(SPOTLIGHT_NODE);
	}

	bool isSwitchNode() const {
		return isNode(SWITCH_NODE);
	}

	bool isTextNode() const {
		return isNode(TEXT_NODE);
	}

	bool isTextureCoordinateNode() const {
		return isNode(TEXTURECOORDINATE_NODE);
	}

	bool isTextureTransformNode() const {
		return isNode(TEXTURETRANSFORM_NODE);
	}

	bool isTimeSensorNode() const {
		return isNode(TIMESENSOR_NODE);
	}

	bool isTouchSensorNode() const {
		return isNode(TOUCHSENSOR_NODE);
	}

	bool isTransformNode() const {
		return isNode(TRANSFORM_NODE);
	}

	bool isViewpointNode() const {
		return isNode(VIEWPOINT_NODE);
	}

	bool isVisibilitySensorNode() const {
		return isNode(VISIBILITYSENSOR_NODE);
	}

	bool isWorldInfoNode() const {
		return isNode(WORLDINFO_NODE);
	}

	// 9. Networking component (X3D)
	
	bool isLoadSensorNode() const {
		return isNode(LOADSENSOR_NODE);
	}
	
	// 10. Grouping component (X3D)
	
	bool isStaticGroupNode() const {
		return isNode(STATICGROUP_NODE);
	}

	// 11. Rendering component (X3D)
	
	bool isColorRGBANode() const {
		return isNode(COLORRGBA_NODE);
	}

	bool isTriangleSetNode() const {
		return isNode(TRIANGLESET_NODE);
	}

	bool isTriangleFanSetNode() const {
		return isNode(TRIANGLEFANSET_NODE);
	}

	bool isTriangleStripSetNode() const {
		return isNode(TRIANGLESTRIPSET_NODE);
	}

	// 12. Shape component (X3D)

	bool isFillPropertiesNode() const {
		return isNode(FILLPROPERTIES_NODE);
	}

	bool isLinePropertiesNode() const {
		return isNode(LINEPROPERTIES_NODE);
	}

	// 14. Geometry2D component (X3D)

	bool isArc2DNode() const {
		return isNode(ARC2D_NODE);
	}

	bool isArcClose2DNode() const {
		return isNode(ARCCLOSE2D_NODE);
	}

	bool isCircle2DNode() const {
		return isNode(CIRCLE2D_NODE);
	}

	bool isDisk2DNode() const {
		return isNode(DISK2D_NODE);
	}

	bool isPolyline2DNode() const {
		return isNode(POLYLINE2D_NODE);
	}

	bool isPolypoint2DNode() const {
		return isNode(POLYPOINT2D_NODE);
	}

	bool isRectangle2DNode() const {
		return isNode(RECTANGLE2D_NODE);
	}

	bool isTriangleSet2DNode() const {
		return isNode(TRIANGLESET2D_NODE);
	}

	// 18. Texturing component (x3D)

	bool isMultiTextureNode() const {
		return isNode(MULTITEXTURE_NODE);
	}

	bool isMultiTextureCoordinateNode() const {
		return isNode(MULTITEXTURECOORD_NODE);
	}

	bool isMultiTextureTransformNode() const {
		return isNode(MULTITEXTURETRANSFORM_NODE);
	}

	bool isTextureCoordinateGeneratorNode() const {
		return isNode(TEXCOORDGEN_NODE);
	}

	// 19. Interpolation component (X3D)

	bool isCoordinateInterpolator2DNode() const {
		return isNode(COORDINATEINTERPOLATOR2D_NODE);
	}

	bool isPositionInterpolator2DNode() const {
		return isNode(POSITIONINTERPOLATOR2D_NODE);
	}

	// 21. Key device sensor component (X3D)

	bool isKeySensorNode() const {
		return isNode(KEYSENSOR_NODE);
	}

	bool isStringSensorNode() const {
		return isNode(STRINGSENSOR_NODE);
	}

	// 30. Event Utilities component (X3D)

	bool isBooleanFilterNode() const {
		return isNode(BOOLEANFILTER_NODE);
	}

	bool isBooleanToggleNode() const {
		return isNode(BOOLEANTOGGLE_NODE);
	}

	bool isBooleanTriggerNode() const {
		return isNode(BOOLEANTRIGGER_NODE);
	}

	bool isBooleanSequencerNode() const {
		return isNode(BOOLEANSEQUENCER_NODE);
	}

	bool isIntegerTriggerNode() const {
		return isNode(INTEGERTRIGGER_NODE);
	}

	bool isIntegerSequencerNode() const {
		return isNode(INTEGERSEQUENCER_NODE);
	}

	bool isTimeTriggerNode() const {
		return isNode(TIMETRIGGER_NODE);
	}

	// Deprecated components (X3D)

	bool isNodeSequencerNode() const {
		return isNode(NODESEQUENCER_NODE);
	}

	bool isShape2DNode() const {
		return isNode(SHAPE2D_NODE);
	}

	bool isBooleanTimeTriggerNode() const {
		return isNode(BOOLEANTIMETRIGGER_NODE);
	}

	bool isTransform2DNode() const {
		return isNode(TRANSFORM2D_NODE);
	}

	////////////////////////////////////////////////
	//	is* (XML)
	////////////////////////////////////////////////

	bool isXMLNode() const {
		return isNode(XML_NODE);
	}

	////////////////////////////////////////////////
	//	isRouteNode (X3D)
	////////////////////////////////////////////////

	bool isRouteNode() const {
		return isNode(ROUTE_NODE);
	}

	////////////////////////////////////////////////
	//	is (VRML97|X3D)Node
	////////////////////////////////////////////////

	bool isX3DNode() const {
		if (isXMLNode())
			return true;
		return false;
	}

	bool isVRML97Node() const {
		return !isX3DNode();
	}

	////////////////////////////////////////////////
	//	output (VRML97)
	////////////////////////////////////////////////

	char *getIndentLevelString(int nIndentLevel) const;
	char *getSpaceString(int nSpaces) const;

	void outputHead(std::ostream& printStream, const char *indentString) const;

	void outputTail(std::ostream& printStream, const char * indentString) const;

	virtual void outputContext(std::ostream &printStream, const char *indentString) const = 0;

	void outputContext(std::ostream& printStream, const char *indentString1, const char *indentString2) const;

	void output(std::ostream& printStream, int indentLevet) const;

	void print(std::ostream& printStream, int indentLevel) const {
		output(printStream, indentLevel);
	}

	void print() const {
		output(std::cout, 0);
	}

	void save(std::ofstream outputStream) const {
		output(outputStream, 0);
	}

	int save(const char * filename) const {
		std::ofstream outputFile(filename, std::ios::out);
		if (outputFile) {
			output(outputFile, 0);
			return 1;
		}
		else
			return 0;
	}

	////////////////////////////////////////////////
	//	output (XML)
	////////////////////////////////////////////////

	void outputXML(std::ostream& printStream, int indentLevet) const;
	void outputXMLField(std::ostream& printStream, Field *field, int indentLevel) const;
	void outputXMLField(std::ostream& printStream, Field *field, int indentLevel, bool isSingleLine) const;

	void printXML(std::ostream& printStream, int indentLevel) const {
		outputXML(printStream, indentLevel);
	}

	void printXML() const {
		outputXML(std::cout, 0);
	}

	int saveXML(const char * filename) const {
		std::ofstream outputFile(filename, std::ios::out);
		if (outputFile) {
			output(outputFile, 0);
			return 1;
		}
		else
			return 0;
	}


	////////////////////////////////////////////////
	//	Virtual functions
	////////////////////////////////////////////////

	virtual void update()		= 0;
	virtual void initialize()	= 0;
	virtual void uninitialize()	= 0;

	////////////////////////////////////////////////
	//	Transform matrix
	////////////////////////////////////////////////

	void	getTransformMatrix(SFMatrix *matrix) const;
	void	getTransformMatrix(float value[4][4]) const;

	////////////////////////////////////////////////
	//	Translation matrix
	////////////////////////////////////////////////

	void	getTranslationMatrix(SFMatrix *matrix) const;
	void	getTranslationMatrix(float value[4][4]) const;

	////////////////////////////////////////////////
	//	SceneGraph
	////////////////////////////////////////////////

	void setSceneGraph(SceneGraph *sceneGraph);
	SceneGraph	*getSceneGraph() const;

	////////////////////////////////////////////////
	//	Route
	////////////////////////////////////////////////

	void		sendEvent(Field *eventOutField);
	void		sendEvent(const char *eventOutFieldString);

	////////////////////////////////////////////////
	//	Value
	////////////////////////////////////////////////

	void		setValue(void *value) { mValue = value; }
	void		*getValue() const     { return mValue; }

	////////////////////////////////////////////////
	//	Initialized
	////////////////////////////////////////////////

	void		setInitialized(bool flag)	{ *mInitialized = flag; }
	bool		isInitialized()	const  	{ return *mInitialized; }

	////////////////////////////////////////////////
	//	BoundingBox
	////////////////////////////////////////////////

	virtual void recomputeBoundingBox() {
	}

	////////////////////////////////////////////////
	//	DisplayList
	////////////////////////////////////////////////

#ifdef CX3D_SUPPORT_OPENGL

	virtual void recomputeDisplayList() {
	}

#endif

	////////////////////////////////////////////////
	//	Instance node
	////////////////////////////////////////////////

	bool isInstanceNode() const {
		return (getReferenceNode() != NULL ? true : false);
	}

	void setReferenceNodeMembers(Node *node);

	void setOriginalMembers();
	
	void setReferenceNode(Node *node) {
		mReferenceNode = node;
	}
	
	Node *getReferenceNode() const {
		return mReferenceNode;
	}

	void setAsInstanceNode(Node *node) {
		setReferenceNode(node);
		setReferenceNodeMembers(node);
	}
	
	Node *createInstanceNode();

	////////////////////////////////////////////////
	//	DEF node
	////////////////////////////////////////////////

	DEFNode *createDEFNode();
};

}

#endif
