/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	SceneGraph.h
*
******************************************************************/

#ifndef _CX3D_SCENEGRAPH_H_
#define _CX3D_SCENEGRAPH_H_

#include <iostream>
#include <fstream>
#include <time.h>
#include <x3d/StringUtil.h>
#include <x3d/VRML97Fields.h>
#include <x3d/VRML97Nodes.h>
#include <x3d/X3DNodes.h>
#include <x3d/Scene.h>
#include <x3d/JavaVM.h>
#include <x3d/UrlFile.h>
#include <x3d/BoundingBox.h>
#include <x3d/RouteList.h>
#include <x3d/MathUtil.h>
#include <x3d/Parser.h>
#include <x3d/ParserResult.h>

namespace CyberX3D {

enum {
SCENEGRAPH_OPTION_NONE			= 0x00,
SCENEGRAPH_NORMAL_GENERATION	= 0x01,
SCENEGRAPH_TEXTURE_GENERATION	= 0x02,
};

#if defined(CX3D_SUPPORT_JSAI)
class SceneGraph : public Scene, public JavaVM {
#else
class SceneGraph : public Scene {
#endif

	int						mOption;

	Vector<BindableNode>	*mBackgroundNodeVector;
	Vector<BindableNode>	*mFogNodeVector;
	Vector<BindableNode>	*mNavigationInfoNodeVector;
	Vector<BindableNode>	*mViewpointNodeVector;	

	ShapeNode				*mSelectedShapeNode;
	Node					*mSelectedNode;
	
	BackgroundNode			*mDefaultBackgroundNode;
	FogNode					*mDefaultFogNode;
	NavigationInfoNode		*mDefaultNavigationInfoNode;
	ViewpointNode			*mDefaultViewpointNode;

	float					mBoundingBoxSize[3];
	float					mBoundingBoxCenter[3];
	float					mFrameRate;

	ParserResult	mParserResult;

#if defined(CX3D_SUPPORT_URL)
	UrlFile					*mUrl;
#endif
	std::string m_base_path;
public:

	SceneGraph();

#if defined(CX3D_SUPPORT_JSAI)
	void setJavaEnv(const char *javaClassPath = NULL, jint (JNICALL *printfn)(FILE *fp, const char *format, va_list args) = NULL);
#endif

	virtual ~SceneGraph();

	////////////////////////////////////////////////
	//	Option
	////////////////////////////////////////////////

	void setOption(int option) {
		mOption = option;
	}

	int getOption() const {
		return mOption;
	}

	////////////////////////////////////////////////
	//	child node list
	////////////////////////////////////////////////

	int getNAllNodes() const;
	int getNNodes() const;
	Node *getNodes(const int type) const;
	Node *getNodes() const;

	////////////////////////////////////////////////
	//	child node list
	////////////////////////////////////////////////

	GroupingNode *getGroupingNodes() const {
		for (Node *node = getNodes(); node; node = node->next()) {
			if (node->isGroupingNode())
				return (GroupingNode *)node;
		}
		return NULL;
	}

	AnchorNode *getAnchorNodes() const {
		return (AnchorNode *)getNodes(ANCHOR_NODE);
	}

	AppearanceNode *getAppearanceNodes() const {
		return (AppearanceNode *)getNodes(APPEARANCE_NODE);
	}

	AudioClipNode *getAudioClipNodes() const {
		return (AudioClipNode *)getNodes(AUDIOCLIP_NODE);
	}

	BackgroundNode *getBackgroundNodes() const {
		return (BackgroundNode *)getNodes(BACKGROUND_NODE);
	}

	BillboardNode *getBillboardNodes() const {
		return (BillboardNode *)getNodes(BILLBOARD_NODE);
	}

	BoxNode *getBoxeNodes() const {
		return (BoxNode *)getNodes(BOX_NODE);
	}

	CollisionNode *getCollisionNodes() const {
		return (CollisionNode *)getNodes(COLLISION_NODE);
	}

	ColorNode *getColorNodes() const {
		return (ColorNode *)getNodes(COLOR_NODE);
	}

	ColorInterpolatorNode *getColorInterpolatorNodes() const {
		return (ColorInterpolatorNode *)getNodes(COLORINTERPOLATOR_NODE);
	}

	ConeNode *getConeNodes() const {
		return (ConeNode *)getNodes(CONE_NODE);
	}

	CoordinateNode *getCoordinateNodes() const {
		return (CoordinateNode *)getNodes(COORDINATE_NODE);
	}

	CoordinateInterpolatorNode *getCoordinateInterpolatorNodes() const {
		return (CoordinateInterpolatorNode *)getNodes(COORDINATEINTERPOLATOR_NODE);
	}

	CylinderNode *getCylinderNodes() const {
		return (CylinderNode *)getNodes(CYLINDER_NODE);
	}

	CylinderSensorNode *getCylinderSensorNodes() const {
		return (CylinderSensorNode *)getNodes(CYLINDERSENSOR_NODE);
	}

	DirectionalLightNode *getDirectionalLightNodes() const {
		return (DirectionalLightNode *)getNodes(DIRECTIONALLIGHT_NODE);
	}

	ElevationGridNode *getElevationGridNodes() const {
		return (ElevationGridNode *)getNodes(ELEVATIONGRID_NODE);
	}

	ExtrusionNode *getExtrusionNodes() const {
		return (ExtrusionNode *)getNodes(EXTRUSION_NODE);
	}

	FogNode *getFogNodes() const {
		return (FogNode *)getNodes(FOG_NODE);
	}

	FontStyleNode *getFontStyleNodes() const {
		return (FontStyleNode *)getNodes(FONTSTYLE_NODE);
	}

	GroupNode *getGroupNodes() const {
		return (GroupNode *)getNodes(GROUP_NODE);
	}

	ImageTextureNode *getImageTextureNodes() const {
		return (ImageTextureNode *)getNodes(IMAGETEXTURE_NODE);
	}

	IndexedFaceSetNode *getIndexedFaceSetNodes() const {
		return (IndexedFaceSetNode *)getNodes(INDEXEDFACESET_NODE);
	}

	IndexedLineSetNode *getIndexedLineSetNodes() const {
		return (IndexedLineSetNode *)getNodes(INDEXEDLINESET_NODE);
	}

	InlineNode *getInlineNodes() const {
		return (InlineNode *)getNodes(INLINE_NODE);
	}

	LODNode *getLODNodes() const {
		return (LODNode *)getNodes(LOD_NODE);
	}

	MaterialNode *getMaterialNodes() const {
		return (MaterialNode *)getNodes(MATERIAL_NODE);
	}

	MovieTextureNode *getMovieTextureNodes() const {
		return (MovieTextureNode *)getNodes(MOVIETEXTURE_NODE);
	}

	NavigationInfoNode *getNavigationInfoNodes() const {
		return (NavigationInfoNode *)getNodes(NAVIGATIONINFO_NODE);
	}

	NormalNode *getNormalNodes() const {
		return (NormalNode *)getNodes(NORMAL_NODE);
	}

	NormalInterpolatorNode *getNormalInterpolatorNodes() const {
		return (NormalInterpolatorNode *)getNodes(NORMALINTERPOLATOR_NODE);
	}

	OrientationInterpolatorNode *getOrientationInterpolatorNodes() const {
		return (OrientationInterpolatorNode *)getNodes(ORIENTATIONINTERPOLATOR_NODE);
	}

	PixelTextureNode *getPixelTextureNodes() const {
		return (PixelTextureNode *)getNodes(PIXELTEXTURE_NODE);
	}

	PlaneSensorNode *getPlaneSensorNodes() const {
		return (PlaneSensorNode *)getNodes(PLANESENSOR_NODE);
	}

	PointLightNode *getPointLightNodes() const {
		return (PointLightNode *)getNodes(POINTLIGHT_NODE);
	}

	PointSetNode *getPointSetNodes() const {
		return (PointSetNode *)getNodes(POINTSET_NODE);
	}

	PositionInterpolatorNode *getPositionInterpolatorNodes() const {
		return (PositionInterpolatorNode *)getNodes(POSITIONINTERPOLATOR_NODE);
	}

	ProximitySensorNode *getProximitySensorNodes() const {
		return (ProximitySensorNode *)getNodes(PROXIMITYSENSOR_NODE);
	}

	ScalarInterpolatorNode *getScalarInterpolatorNodes() const {
		return (ScalarInterpolatorNode *)getNodes(SCALARINTERPOLATOR_NODE);
	}

	ScriptNode *getScriptNodes() const {
		return (ScriptNode *)getNodes(SCRIPT_NODE);
	}

	ShapeNode *getShapeNodes() const {
		return (ShapeNode *)getNodes(SHAPE_NODE);
	}

	SoundNode *getSoundNodes() const {
		return (SoundNode *)getNodes(SOUND_NODE);
	}

	SphereNode *getSphereNodes() const {
		return (SphereNode *)getNodes(SPHERE_NODE);
	}

	SphereSensorNode *getSphereSensorNodes() const {
		return (SphereSensorNode *)getNodes(SPHERESENSOR_NODE);
	}

	SpotLightNode *getSpotLightNodes() const {
		return (SpotLightNode *)getNodes(SPOTLIGHT_NODE);
	}

	SwitchNode *getSwitchNodes() const {
		return (SwitchNode *)getNodes(SWITCH_NODE);
	}

	TextNode *getTextNodes() const {
		return (TextNode *)getNodes(TEXT_NODE);
	}

	TextureCoordinateNode *getTextureCoordinateNodes() const {
		return (TextureCoordinateNode *)getNodes(TEXTURECOORDINATE_NODE);
	}

	TextureTransformNode *getTextureTransformNodes() const {
		return (TextureTransformNode *)getNodes(TEXTURETRANSFORM_NODE);
	}

	TimeSensorNode *getTimeSensorNodes() const {
		return (TimeSensorNode *)getNodes(TIMESENSOR_NODE);
	}

	TouchSensorNode *getTouchSensorNodes() const {
		return (TouchSensorNode *)getNodes(TOUCHSENSOR_NODE);
	}

	TransformNode *getTransformNodes() const {
		return (TransformNode *)getNodes(TRANSFORM_NODE);
	}

	ViewpointNode *getViewpointNodes() const {
		return (ViewpointNode *)getNodes(VIEWPOINT_NODE);
	}

	VisibilitySensorNode *getVisibilitySensorNodes() const {
		return (VisibilitySensorNode *)getNodes(VISIBILITYSENSOR_NODE);
	}

	WorldInfoNode *getWorldInfoNodes() const {
		return (WorldInfoNode *)getNodes(WORLDINFO_NODE);
	}

	// 9. Networking component (X3D)

	LoadSensorNode *getLoadSensorNodes() const {
		return (LoadSensorNode *)getNodes(LOADSENSOR_NODE);
	}

	// 10. Grouping component (X3D)

	StaticGroupNode *getStaticGroupNodes() const {
		return (StaticGroupNode *)getNodes(STATICGROUP_NODE);
	}

	// 11. Rendering component (X3D)

	ColorRGBANode *getColorRGBANodes() const {
		return (ColorRGBANode *)getNodes(COLORRGBA_NODE);
	}

	TriangleSetNode *getTriangleSetNodes() const {
		return (TriangleSetNode *)getNodes(TRIANGLESET_NODE);
	}

	TriangleFanSetNode *getTriangleFanSetNodes() const {
		return (TriangleFanSetNode *)getNodes(TRIANGLEFANSET_NODE);
	}

	TriangleStripSetNode *getTriangleStripSetNodes() const {
		return (TriangleStripSetNode *)getNodes(TRIANGLESTRIPSET_NODE);
	}
	
	// 12. Shape component (X3D)

	FillPropertiesNode *getFillPropertiesNodes() const {
		return (FillPropertiesNode *)getNodes(FILLPROPERTIES_NODE);
	}

	LinePropertiesNode *getLinePropertiesNodes() const {
		return (LinePropertiesNode *)getNodes(LINEPROPERTIES_NODE);
	}

	// 14. Geometry2D component (X3D)

	Arc2DNode *getArc2DNodes() const {
		return (Arc2DNode *)getNodes(ARC2D_NODE);
	}

	ArcClose2DNode *getArcClose2DNodes() const {
		return (ArcClose2DNode *)getNodes(ARCCLOSE2D_NODE);
	}

	Circle2DNode *getCircle2DNodes() const {
		return (Circle2DNode *)getNodes(CIRCLE2D_NODE);
	}

	Disk2DNode *getDisk2DNodes() const {
		return (Disk2DNode *)getNodes(DISK2D_NODE);
	}

	Polyline2DNode *getPolyline2DNodes() const {
		return (Polyline2DNode *)getNodes(POLYLINE2D_NODE);
	}

	Polypoint2DNode *getPolypoint2DNodes() const {
		return (Polypoint2DNode *)getNodes(POLYPOINT2D_NODE);
	}

	Rectangle2DNode *getRectangle2DNodes() const {
		return (Rectangle2DNode *)getNodes(RECTANGLE2D_NODE);
	}

	TriangleSet2DNode *getTriangleSet2DNodes() const {
		return (TriangleSet2DNode *)getNodes(TRIANGLESET2D_NODE);
	}
	
	// 18. Texturing component (x3D)

	MultiTextureNode *getMultiTextureNodes() const {
		return (MultiTextureNode *)getNodes(MULTITEXTURE_NODE);
	}

	MultiTextureCoordinateNode *getMultiTextureCoordinateNodes() const {
		return (MultiTextureCoordinateNode *)getNodes(MULTITEXTURECOORD_NODE);
	}

	MultiTextureTransformNode *getMultiTextureTransformNodes() const {
		return (MultiTextureTransformNode *)getNodes(MULTITEXTURETRANSFORM_NODE);
	}
	
	TextureCoordinateGeneratorNode *getTextureCoordinateGeneratorNodes() const {
		return (TextureCoordinateGeneratorNode *)getNodes(TEXCOORDGEN_NODE);
	}
	
	// 19. Interpolation component (X3D)

	CoordinateInterpolator2DNode *getCoordinateInterpolator2DNodes() const {
		return (CoordinateInterpolator2DNode *)getNodes(COORDINATEINTERPOLATOR2D_NODE);
	}

	PositionInterpolator2DNode *getPositionInterpolator2DNodes() const {
		return (PositionInterpolator2DNode *)getNodes(POSITIONINTERPOLATOR2D_NODE);
	}

	// 21. Key device sensor component (X3D)

	KeySensorNode *getKeySensorNodes() const {
		return (KeySensorNode *)getNodes(KEYSENSOR_NODE);
	}

	StringSensorNode *getStringSensorNodes() const {
		return (StringSensorNode *)getNodes(STRINGSENSOR_NODE);
	}

	// 30. Event Utilities component (X3D)

	BooleanFilterNode *getBooleanFilterNodes() const {
		return (BooleanFilterNode *)getNodes(BOOLEANFILTER_NODE);
	}

	BooleanToggleNode *getBooleanToggleNodes() const {
		return (BooleanToggleNode *)getNodes(BOOLEANTOGGLE_NODE);
	}

	BooleanTriggerNode *getBooleanTriggerNodes() const {
		return (BooleanTriggerNode *)getNodes(BOOLEANTRIGGER_NODE);
	}

	BooleanSequencerNode *getBooleanSequencerNodes() const {
		return (BooleanSequencerNode *)getNodes(BOOLEANSEQUENCER_NODE);
	}

	IntegerTriggerNode *getIntegerTriggerNodes() const {
		return (IntegerTriggerNode *)getNodes(INTEGERTRIGGER_NODE);
	}

	IntegerSequencerNode *getIntegerSequencerNodes() const {
		return (IntegerSequencerNode *)getNodes(INTEGERSEQUENCER_NODE);
	}

	TimeTriggerNode *getTimeTriggerNodes() const {
		return (TimeTriggerNode *)getNodes(TIMETRIGGER_NODE);
	}
	
	// Deprecated components (X3D)

	NodeSequencerNode *getNodeSequencerNodes() const {
		return (NodeSequencerNode *)getNodes(NODESEQUENCER_NODE);
	}

	Shape2DNode *getShape2DNodes() const {
		return (Shape2DNode *)getNodes(SHAPE2D_NODE);
	}

	BooleanTimeTriggerNode *getBooleanTimeTriggerNodes() const {
		return (BooleanTimeTriggerNode *)getNodes(BOOLEANTIMETRIGGER_NODE);
	}

	Transform2DNode *getTransform2DNodes() const {
		return (Transform2DNode *)getNodes(TRANSFORM2D_NODE);
	}

	////////////////////////////////////////////////
	//	Node Number
	////////////////////////////////////////////////

	unsigned int getNodeNumber(Node *node) const ;

	////////////////////////////////////////////////
	//	initialize
	////////////////////////////////////////////////

	void initialize(void (*callbackFn)(int nNode, void *info) = NULL, void *callbackFnInfo = NULL);

	void uninitialize(void (*callbackFn)(int nNode, void *info) = NULL, void *callbackFnInfo = NULL);

	////////////////////////////////////////////////
	//	update
	////////////////////////////////////////////////

	void update();
	void updateRoute(Node *eventOutNode, Field *eventOutField);

	///////////////////////////////////////////////
	//	Output node infomations
	///////////////////////////////////////////////
	
	void print();
	void printXML();

	///////////////////////////////////////////////
	//	Delete/Remove Node
	///////////////////////////////////////////////

	void removeNode(Node *node);
	void deleteNode(Node *node);

	///////////////////////////////////////////////
	//	Bindable Nodes
	///////////////////////////////////////////////

	void setBindableNode(Vector<BindableNode> *nodeVector, BindableNode *node, bool bind);

	void setBindableNode(BindableNode *node, bool bind); 

	void setBackgroundNode(BackgroundNode *bg, bool bind) {
		setBindableNode(mBackgroundNodeVector, bg, bind);
	}

	void setFogNode(FogNode *fog, bool bind) {
		setBindableNode(mFogNodeVector, fog, bind);
	}

	void setNavigationInfoNode(NavigationInfoNode *navInfo, bool bind) {
		setBindableNode(mNavigationInfoNodeVector, navInfo, bind);
	}

	void setViewpointNode(ViewpointNode *view, bool bind) {
		setBindableNode(mViewpointNodeVector, view, bind);
	}

	BackgroundNode *getBackgroundNode() const {
		return (BackgroundNode *)mBackgroundNodeVector->lastElement();
	}

	FogNode *getFogNode() const {
		return (FogNode *)mFogNodeVector->lastElement();
	}

	NavigationInfoNode *getNavigationInfoNode() const {
		return (NavigationInfoNode *)mNavigationInfoNodeVector->lastElement();
	}

	ViewpointNode *getViewpointNode() const {
		return (ViewpointNode *)mViewpointNodeVector->lastElement();
	}

	////////////////////////////////////////////////
	//	BoundingBoxSize
	////////////////////////////////////////////////

	void setBoundingBoxSize(float value[]);
	void setBoundingBoxSize(float x, float y, float z);
	void getBoundingBoxSize(float value[]) const;

	////////////////////////////////////////////////
	//	BoundingBoxCenter
	////////////////////////////////////////////////

	void setBoundingBoxCenter(float value[]);
	void setBoundingBoxCenter(float x, float y, float z);
	void getBoundingBoxCenter(float value[]) const;

	////////////////////////////////////////////////
	//	BoundingBox
	////////////////////////////////////////////////

	void setBoundingBox(BoundingBox *bbox) ;
	void recomputeBoundingBox();

	////////////////////////////////////////////////
	//	Polygons
	////////////////////////////////////////////////

	int getNPolygons() const;

	///////////////////////////////////////////////
	//	Load
	///////////////////////////////////////////////

	void clear();

	bool load(const char *filename, bool bInitialize = true, void (*callbackFn)(int nLine, void *info) = NULL, void *callbackFnInfo = NULL);
	bool add(const char *filename, bool bInitialize = true, void (*callbackFn)(int nLine, void *info) = NULL, void *callbackFnInfo = NULL);

	///////////////////////////////////////////////
	//	Parser
	///////////////////////////////////////////////

	void moveParserNodes(Parser *parser);
	void moveParserRoutes(Parser *parser);

	ParserResult *getParserResult()
	{
		return &mParserResult;
	}

	int	getParserErrorLineNumber(void) {
		return getParserResult()->getErrorLineNumber();
	}

	const char *getParserErrorMessage(void) { 
		return getParserResult()->getErrorMessage();
	}

	const char *getParserErrorToken(void) { 
		return getParserResult()->getErrorToken(); 
	}

	const char *getParserErrorLineString(void) {
		return getParserResult()->getErrorLineString();
	}

	///////////////////////////////////////////////
	//	Save node infomations
	///////////////////////////////////////////////
	
	bool save(const char *filename, void (*callbackFn)(int nNode, void *info) = NULL, void *callbackFnInfo = NULL);
  //bool save(const wchar_t *filename, void (*callbackFn)(int nNode, void *info) = NULL, void *callbackFnInfo = NULL);
	bool saveXML(const char *filename, void (*callbackFn)(int nNode, void *info) = NULL, void *callbackFnInfo = NULL);
  //bool saveXML(const wchar_t    *filename, void (*callbackFn)(int nNode, void *info) = NULL, void *callbackFnInfo = NULL);

	///////////////////////////////////////////////
	//	URL
	///////////////////////////////////////////////

#if defined(CX3D_SUPPORT_URL)

	void	setUrl(const char *url)				{ mUrl->setUrl(url); }
	const char	*getUrl() const						{ return mUrl->getUrl(); }
	bool	getUrlStream(const char *urlString) const	{ return mUrl->getStream(urlString); }
	const char	*getUrlOutputFilename()	const		{ return mUrl->getOutputFilename(); }
	bool	deleteUrlOutputFilename()		{ return mUrl->deleteOutputFilename(); }

#endif

	//////////////////////////////////////////////////
	// Selected Shape/Node
	//////////////////////////////////////////////////

	void			setSelectedShapeNode(ShapeNode *shape)	{ mSelectedShapeNode = shape; }
	ShapeNode		*getSelectedShapeNode()	const				{ return mSelectedShapeNode; }

	void			setSelectedNode(Node *node)				{ mSelectedNode = node; }
	Node			*getSelectedNode()	const					{ return mSelectedNode; }

	//////////////////////////////////////////////////
	// Default Bindable Nodes
	//////////////////////////////////////////////////

	BackgroundNode		*getDefaultBackgroundNode() const		{ return mDefaultBackgroundNode; }
	FogNode				*getDefaultFogNode() const			{ return mDefaultFogNode; }
	NavigationInfoNode	*getDefaultNavigationInfoNode()	const { return mDefaultNavigationInfoNode; }
	ViewpointNode		*getDefaultViewpointNode()	const	{ return mDefaultViewpointNode; }

	//////////////////////////////////////////////////
	// Zoom All
	//////////////////////////////////////////////////

	void			zoomAllViewpoint();

	//////////////////////////////////////////////////
	// FrameRate
	//////////////////////////////////////////////////

	void setFrameRate(float value)
	{
		mFrameRate = value;
	}

	void addFrameRate(float value)
	{
		mFrameRate += value;
		mFrameRate /= 2.0;
	}

	void addFrameRateByTime(time_t value)
	{
		addFrameRate(1.0f / (value / 1000.0f));
	}

	float getFrameRate() const
	{
		return mFrameRate;
	}

	void setBasePath(const std::string &base_path)
	{
		m_base_path = base_path;
	}

	const std::string &getBasePath() const
	{
		return m_base_path;
	}
};

}

#endif
