/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	SceneGraph.cpp
*
*	Revision:
*	
*	07/22/04
*		- Thanks Peter DeSantis <peter@jumbovision.com.au>
*		- Fixed recomputeBoundingBox() to multi the transform matrix.
*
******************************************************************/

#include <x3d/SceneGraph.h>
#include <x3d/VRML97Parser.h>
#include <x3d/X3DParser.h>
#include <x3d/ParserFunc.h>

using namespace CyberX3D;

////////////////////////////////////////////////////////////
//	SceneGraph::SceneGraph
////////////////////////////////////////////////////////////

SceneGraph::SceneGraph() 
{
//	setHeaderFlag(false);
	setOption(SCENEGRAPH_OPTION_NONE);
	setBoundingBoxCenter(0.0f, 0.0f, 0.0f);
	setBoundingBoxSize(-1.0f, -1.0f, -1.0f);
	setSelectedShapeNode(NULL);
	setSelectedNode(NULL);

	mBackgroundNodeVector		= new Vector<BindableNode>;
	mFogNodeVector				= new Vector<BindableNode>;
	mNavigationInfoNodeVector	= new Vector<BindableNode>;
	mViewpointNodeVector		= new Vector<BindableNode>;	

	mDefaultBackgroundNode		= new BackgroundNode();
	mDefaultFogNode				= new FogNode();
	mDefaultNavigationInfoNode	= new NavigationInfoNode();
	mDefaultViewpointNode		= new ViewpointNode();

#if defined(CX3D_SUPPORT_URL)
	mUrl = new UrlFile();
#endif

	setFrameRate(0.0f);
}

////////////////////////////////////////////////
//	Node Number
////////////////////////////////////////////////

unsigned int SceneGraph::getNodeNumber(Node *node) const {
	unsigned int nNode = 1;
	for (Node *n = getNodes(); n; n = n->nextTraversal()) {
		if (n == node)
			return nNode;
		nNode++;
	}
	return 0;
}

////////////////////////////////////////////////////////////
//	SceneGraph::SceneGraph
////////////////////////////////////////////////////////////

#if defined(CX3D_SUPPORT_JSAI)

void SceneGraph::setJavaEnv(const char *javaClassPath, jint (JNICALL *printfn)(FILE *fp, const char *format, va_list args)) 
{
	CreateJavaVM(javaClassPath, printfn);
}

#endif

////////////////////////////////////////////////////////////
//	SceneGraph::~SceneGraph
////////////////////////////////////////////////////////////

SceneGraph::~SceneGraph() 
{
	Node *node=getNodes();
	while (node) {
		delete node;
		node = getNodes();
	}
	Route *route=getRoutes();
	while (route) {
		Route *nextRoute=route->next();
		delete route;
		route = nextRoute;
	}

	delete mBackgroundNodeVector;
	delete mFogNodeVector;
	delete mNavigationInfoNodeVector;
	delete mViewpointNodeVector;	

	delete mDefaultBackgroundNode;
	delete mDefaultFogNode;
	delete mDefaultNavigationInfoNode;
	delete mDefaultViewpointNode;

#if defined(CX3D_SUPPORT_URL)
	delete mUrl;
#endif

#if defined(CX3D_SUPPORT_JSAI)
	DeleteJavaVM();
#endif
}

////////////////////////////////////////////////
//	child node list
////////////////////////////////////////////////

int SceneGraph::getNAllNodes() const
{
	int nNode = 0;
	for (Node *node = getNodes(); node; node = node->nextTraversal())
		nNode++;
	return nNode;
}

int SceneGraph::getNNodes() const
{
	int nNode = 0;
	for (Node *node = getNodes(); node; node = node->next())
		nNode++;
	return nNode;
}

Node *SceneGraph::getNodes(const int type) const
{
	Node *node = getNodes();
	if (node == NULL)
		return NULL;
	if (node->getType() == type)
		return node;
	return node->next(type);
}

Node *SceneGraph::getNodes() const
{
	return Scene::getNodes();
}

////////////////////////////////////////////////////////////
//	SceneGraph::clear
////////////////////////////////////////////////////////////

void SceneGraph::clear() 
{
	clearNodeList();
	clearRouteList();
}

////////////////////////////////////////////////////////////
//	SceneGraph::load
////////////////////////////////////////////////////////////

bool SceneGraph::load(const char *filename, bool bInitialize, void (*callbackFn)(int nLine, void *info), void *callbackFnInfo)
{
	clear();
	return add(filename, bInitialize, callbackFn, callbackFnInfo);
}

bool SceneGraph::add(const char *filename, bool bInitialize, void (*callbackFn)(int nLine, void *info), void *callbackFnInfo)
{
	int fileFormat = GetFileFormat(filename);

	Parser *parser = NULL;
	ParserResult *presult = getParserResult();

	presult->init();
	presult->setParserResult(false);

	switch (fileFormat) {
	case FILE_FORMAT_VRML:
		parser = new VRML97Parser();
		break;
#ifdef CX3D_SUPPORT_X3D
	case FILE_FORMAT_XML:
		parser = new X3DParser();
		break;
#endif
	}

	if (parser == NULL)
		return false;

	SetParserResultObject(presult);

	bool parserRet = parser->load(filename, callbackFn, callbackFnInfo);
	presult->setParserResult(parserRet);
	if (parserRet == false) {
		delete parser;
		return false;
	}

	moveParserNodes(parser);
	moveParserRoutes(parser);

        delete parser;

	if (bInitialize)
		initialize();

	setBackgroundNode(findBackgroundNode(), true);
	setFogNode(findFogNode(), true);
	setNavigationInfoNode(findNavigationInfoNode(), true);
	setViewpointNode(findViewpointNode(), true);

	return true;
}

////////////////////////////////////////////////////////////
//	moveParser*
////////////////////////////////////////////////////////////

void SceneGraph::moveParserNodes(Parser *parser)
{
	Node *node = parser->getNodes();
	if (node->isXMLNode() == true) {
		Node *sceneNode = parser->findSceneNode();
		if (sceneNode != NULL)
			node = sceneNode->getChildNodes();
		else
			node = NULL;
	}

	while (node != NULL) {
		Node *nextNode = node->next();
		moveNode(node);
		node = nextNode;
	}
}

void SceneGraph::moveParserRoutes(Parser *parser)
{
	Route *route = parser->getRoutes();
	while (route != NULL) {
		Route *nextRoute = route->next();
		route->remove();
		addRoute(route);
		route = nextRoute;
	}
}

////////////////////////////////////////////////////////////
//	SceneGraph::save
////////////////////////////////////////////////////////////
	
bool SceneGraph::save(const char *filename, void (*callbackFn)(int nNode, void *info), void *callbackFnInfo) 
{
	
	std::ofstream outputFile(filename);

	if (!outputFile)
		return false;

	uninitialize();

	outputFile << "#VRML V2.0 utf8" << std::endl;

	int nNode = 0;
	for (Node *node = getNodes(); node; node = node->next()) {
		node->output(outputFile, 0);
		nNode++;
		if (callbackFn)
			callbackFn(nNode, callbackFnInfo);
	}
	for (Route *route = getRoutes(); route; route = route->next()) {
		route->output(outputFile);
	}

	initialize();

	return true;
}


/*
///////////////////////////////////////////////////////////////
bool SceneGraph::save(const wchar_t *filename, void (*callbackFn)(int nNode, void *info), void *callbackFnInfo)
{
	std::wofstream outputFile(filename);

	if (!outputFile)
		return false;

	uninitialize();

	outputFile << "#VRML V2.0 utf8" << std::endl;

	int nNode = 0;
	for (Node *node = getNodes(); node; node = node->next()) {
		node->output(outputFile, 0);
		nNode++;
		if (callbackFn)
			callbackFn(nNode, callbackFnInfo);
	}
	for (Route *route = getRoutes(); route; route = route->next()) {
		route->output(outputFile);
	}

	initialize();

	return true;
}
 
*/
////////////////////////////////////////////////////////////////
bool SceneGraph::saveXML(const char *filename, void (*callbackFn)(int nNode, void *info), void *callbackFnInfo) 
{
	
	std::ofstream outputFile(filename);

	if (!outputFile)
		return false;

	uninitialize();

	outputFile << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << std::endl;

	int nNode = 0;
	for (Node *node = getNodes(); node; node = node->next()) {
		node->outputXML(outputFile, 0);
		nNode++;
		if (callbackFn)
			callbackFn(nNode, callbackFnInfo);
	}
	for (Route *route = getRoutes(); route; route = route->next()) {
		route->outputXML(outputFile);
	}

	initialize();

	return true;
}

/*
////////////////////////////////////////////////////////////////
bool SceneGraph::saveXML(const wchar_t *filename, void (*callbackFn)(int nNode, void *info), void *callbackFnInfo)
{
	std::ofstream outputFile(filename);

	if (!outputFile)
		return false;

	uninitialize();

	outputFile << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << std::endl;
  outputFile << "<X3D>" << std::endl;
  outputFile << "<x3d/Scene>" << std::endl;

	int nNode = 0;
	for (Node *node = getNodes(); node; node = node->next()) {
		node->outputXML(outputFile, 0);
		nNode++;
		if (callbackFn)
			callbackFn(nNode, callbackFnInfo);
	}
	for (Route *route = getRoutes(); route; route = route->next()) {
		route->outputXML(outputFile);
	}

	initialize();

  outputFile << "</Scene>" << std::endl;
  outputFile << "</X3D>" << std::endl;

	return true;
}
*/


///////////////////////////////////////////////////////////
//	SceneGraph::initialize
////////////////////////////////////////////////////////////

void SceneGraph::initialize(void (*callbackFn)(int nNode, void *info), void *callbackFnInfo) 
{
	Node *node;

	int nNode = 0;
	for (node = getNodes(); node; node = node->nextTraversal()) {
		node->setSceneGraph(this);
		if (node->isInstanceNode() == false)		
			node->initialize();
		nNode++;
		if (callbackFn)
			callbackFn(nNode, callbackFnInfo);
	}

	// Convert from InstanceNode into DEFNode 
	node = getNodes();
	while(node != NULL) {
		Node *nextNode = node->nextTraversal();
		if (node->isInstanceNode() == true && node->isDEFNode() == false) {
			Node *referenceNode	= node->getReferenceNode();
			Node *parentNode	= node->getParentNode();
			Node *defNode;
			
			defNode = referenceNode->createDEFNode();
			if (parentNode != NULL)
				parentNode->addChildNode(defNode, false);
			else
				addNode(defNode, false);

			node->remove();
			delete node;

			nextNode = defNode->nextTraversal();
		}
		node = nextNode;
	}

	// Convert from DEFNode into InstanceNode 
	node = getNodes();
	while(node != NULL) {
		Node *nextNode = node->nextTraversal();

		if (node->isDEFNode() == true) {
			Node *defNode = findNode(node->getName());
			assert(defNode);
			if (defNode) {	
				Node *instanceNode = defNode->createInstanceNode();
				Node *parentNode = node->getParentNode();
				if (parentNode != NULL)
					parentNode->moveChildNode(instanceNode);
				else
					moveNode(instanceNode);
				node->remove();
				delete node;
			}
		}

		node = nextNode;
	}

	recomputeBoundingBox();

	for (Route *route = getRoutes(); route; route = route->next())
		route->initialize();
}

////////////////////////////////////////////////
//	update
////////////////////////////////////////////////

void SceneGraph::update() 
{
	for (Node *node = getNodes(); node; node = node->nextTraversal()) {
		node->update();
	}
}

void SceneGraph::updateRoute(Node *eventOutNode, Field *eventOutField) 
{
	for (Route *route = getRoutes(); route; route = route->next()) {
		if (route->getEventOutNode() == eventOutNode && route->getEventOutField() == eventOutField) {
			route->update();
			route->getEventInNode()->update();
			updateRoute(route->getEventInNode(), route->getEventInField());
		}
	}
}

///////////////////////////////////////////////
//	Output node infomations
///////////////////////////////////////////////
	
void SceneGraph::print()
{
	uninitialize();

	for (Node *node = getNodes(); node; node = node->next())
		node->print();
	for (Route *route = getRoutes(); route; route = route->next())
		route->print();

	initialize();
}

void SceneGraph::printXML()
{
	uninitialize();

	for (Node *node = getNodes(); node; node = node->next())
		node->printXML();
	for (Route *route = getRoutes(); route; route = route->next()) 
		route->printXML();

	initialize();
}

///////////////////////////////////////////////
//	Delete/Remove Node
///////////////////////////////////////////////

void SceneGraph::removeNode(Node *node) 
{
	deleteRoutes(node);
	node->remove();
}

void SceneGraph::deleteNode(Node *node) 
{
	deleteRoutes(node);
	delete node;
}

////////////////////////////////////////////////////////////
//	SceneGraph::uninitialize
////////////////////////////////////////////////////////////

void SceneGraph::uninitialize(void (*callbackFn)(int nNode, void *info), void *callbackFnInfo) 
{
	int nNode = 0;
	for (Node *node = getNodes(); node; node = node->nextTraversal()) {
		node->uninitialize();
		nNode++;
		if (callbackFn)
			callbackFn(nNode, callbackFnInfo);
	}
}

////////////////////////////////////////////////
//	BoundingBoxSize
////////////////////////////////////////////////

void SceneGraph::setBoundingBoxSize(float value[]) 
{
	mBoundingBoxSize[0] = value[0];
	mBoundingBoxSize[1] = value[1];
	mBoundingBoxSize[2] = value[2];
}

void SceneGraph::setBoundingBoxSize(float x, float y, float z) 
{
	mBoundingBoxSize[0] = x;
	mBoundingBoxSize[1] = y;
	mBoundingBoxSize[2] = z;
}

void SceneGraph::getBoundingBoxSize(float value[]) const
{
	value[0] = mBoundingBoxSize[0];
	value[1] = mBoundingBoxSize[1];
	value[2] = mBoundingBoxSize[2];
}

////////////////////////////////////////////////
//	BoundingBoxCenter
////////////////////////////////////////////////

void SceneGraph::setBoundingBoxCenter(float value[]) 
{
	mBoundingBoxCenter[0] = value[0];
	mBoundingBoxCenter[1] = value[1];
	mBoundingBoxCenter[2] = value[2];
}

void SceneGraph::setBoundingBoxCenter(float x, float y, float z) 
{
	mBoundingBoxCenter[0] = x;
	mBoundingBoxCenter[1] = y;
	mBoundingBoxCenter[2] = z;
}

void SceneGraph::getBoundingBoxCenter(float value[]) const
{
	value[0] = mBoundingBoxCenter[0];
	value[1] = mBoundingBoxCenter[1];
	value[2] = mBoundingBoxCenter[2];
}

////////////////////////////////////////////////
//	BoundingBox
////////////////////////////////////////////////

void SceneGraph::setBoundingBox(BoundingBox *bbox) 
{
	float center[3];
	float size[3];
	bbox->getCenter(center);
	bbox->getSize(size);
	setBoundingBoxCenter(center);
	setBoundingBoxSize(size);
}

void SceneGraph::recomputeBoundingBox() 
{
	Node	*node;
	float	center[3];
	float	size[3];
	float	m4[4][4];
	SFMatrix mx;

	BoundingBox bbox;

	for (node=getNodes(); node; node=node->nextTraversal()) {
		if (node->isBoundedGroupingNode()) {
			BoundedGroupingNode *gnode = (BoundedGroupingNode *)node;
			gnode->getBoundingBoxCenter(center);
			gnode->getBoundingBoxSize(size);
			// Thanks for Peter DeSantis (07/22/04)
			gnode->getTransformMatrix(m4);
			mx.setValue(m4);
			bbox.addBoundingBox(&mx, center, size);
		}
		else if (node->isGeometry3DNode()) { 
			Geometry3DNode *gnode = (Geometry3DNode *)node; 
			gnode->getBoundingBoxCenter(center); 
			gnode->getBoundingBoxSize(size); 
			// Thanks for Peter DeSantis (07/22/04)
			gnode->getTransformMatrix(m4);
			mx.setValue(m4);
			bbox.addBoundingBox(&mx, center, size);
		} 
	}

	setBoundingBox(&bbox);
}

////////////////////////////////////////////////
//	Polygons
////////////////////////////////////////////////

int SceneGraph::getNPolygons() const
{
	int nPolys = 0;

	for (Node *node=getNodes(); node; node=node->nextTraversal()) {
		if (node->isGeometry3DNode()) {
			Geometry3DNode *geo = (Geometry3DNode *)node;
			nPolys += geo->getNPolygons();
		}
	}

	return nPolys;
}

///////////////////////////////////////////////
//	Bindable Nodes
///////////////////////////////////////////////

void SceneGraph::setBindableNode(Vector<BindableNode> *nodeVector, BindableNode *node, bool bind)
{
	if (!node)
		return;

	BindableNode *topNode = nodeVector->lastElement();

	if (bind) {
		if (topNode != node) {
			if (topNode) {
				topNode->setIsBound(false);
				topNode->sendEvent(topNode->getIsBoundField());
			}

			nodeVector->removeElement(node);
			nodeVector->addElement(node, false);

			node->setIsBound(true);
			node->sendEvent(node->getIsBoundField());
		}
	}
	else {
		if (topNode == node) {
			node->setIsBound(false);
			node->sendEvent(node->getIsBoundField());

			nodeVector->removeElement(node);

			BindableNode *newTopNode = nodeVector->lastElement();
			if (newTopNode) {
				newTopNode->setIsBound(true);
				newTopNode->sendEvent(newTopNode->getIsBoundField());
			}
		}
		else {
			nodeVector->removeElement(node);
		}
	}
}

void SceneGraph::setBindableNode(BindableNode *node, bool bind) 
{
	if (node->isBackgroundNode())		setBackgroundNode((BackgroundNode *)node, bind);
	if (node->isFogNode())					setFogNode((FogNode *)node, bind);
	if (node->isNavigationInfoNode())	setNavigationInfoNode((NavigationInfoNode *)node, bind);
	if (node->isViewpointNode())		setViewpointNode((ViewpointNode *)node, bind);
}


///////////////////////////////////////////////
//	Zoom All Viewpoint
///////////////////////////////////////////////

void SceneGraph::zoomAllViewpoint() 
{
	float	bboxCenter[3];
	float	bboxSize[3];

	getBoundingBoxCenter(bboxCenter);
	getBoundingBoxSize(bboxSize);

	ViewpointNode *view = getViewpointNode();
	if (view == NULL)
		view = getDefaultViewpointNode();

	float fov = view->getFieldOfView();
	float zoffset = bboxSize[0] / (float)tan(fov);
	view->setPosition(bboxCenter[0], bboxCenter[1], bboxCenter[2] + zoffset*5.0f);
	view->setOrientation(0.0f, 0.0f, 1.0f, 0.0f);
}
