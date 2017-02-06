/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	Node.cpp
*
*	03/15/04
*	- Thanks for Simon Goodall <sg02r@ecs.soton.ac.uk>
*	- Fixed a memory leak in Node::outputContext().
*	03/17/04
*	- Thanks for Simon Goodall <sg02r@ecs.soton.ac.uk>
*	- Changed getChildNodeByType() to check the reference node type if the child node is the instance node.
*
******************************************************************/

#include <assert.h>
#include <x3d/X3DFields.h>
#include <x3d/Node.h>
#include <x3d/SceneGraph.h>
#include <x3d/StringUtil.h>

using namespace std;
using namespace CyberX3D;

////////////////////////////////////////////////
//	Node::Node
////////////////////////////////////////////////

void Node::initializeMember()
{
	mName				= mOrgName				= new String();
	mType				= mOrgType				= UNKNOWN_NODE;
	mExposedField		= mOrgExposedField		= new Vector<Field>();
	mEventInField		= mOrgEventInField		= new Vector<Field>();
	mEventOutField		= mOrgEventOutField		= new Vector<Field>();
	mField				= mOrgField				= new Vector<Field>();
	mPrivateField		= mOrgPrivateField		= new Vector<Field>();

	mPrivateNodeVector	= new Vector<Node>();
	mInitialized		= new bool;

	mChildNodes			= new LinkedList<Node>();

	setName(NULL);
	setParentNode(NULL);
	setSceneGraph(NULL);
#if defined(CX3D_SUPPORT_JSAI)
	setJavaNodeObject(NULL);
#endif
	setValue(NULL);
	setInitialized(false);
	setName(NULL);
	setReferenceNode(NULL);
}

Node::Node() 
{
	setHeaderFlag(true);
	initializeMember();
}
	
Node::Node(const int nodeType, const char * nodeName) 
{
	setHeaderFlag(false);
	initializeMember();

	setType(nodeType);
	setName(nodeName);
}
	
////////////////////////////////////////////////
//	Node::~Node
////////////////////////////////////////////////

void Node::deleteChildNodes(void)
{
	Node *node=getChildNodes();
	while (node) {
		Node *nextNode = node->next();
		delete node;
		node = nextNode;
	}
}

Node::~Node() 
{
  deleteChildNodes(); //Remove this string for optimise process of release memory

	SceneGraph *sg = getSceneGraph();
	if (sg) {
		if (sg->getSelectedShapeNode() == this)
			sg->setSelectedShapeNode(NULL);
		if (sg->getSelectedNode() == this)
			sg->setSelectedNode(NULL);
	}

	remove();

	if (isInstanceNode() == true)
		setOriginalMembers();

#if defined(CX3D_SUPPORT_JSAI)
	delete mJNode;
#endif

	delete mName;
	delete mExposedField;
	delete mEventInField;
	delete mEventOutField;
	delete mField;
	delete mPrivateField;
	delete mPrivateNodeVector;
	delete mChildNodes;
	delete mInitialized;
}

////////////////////////////////////////////////
//	Name
////////////////////////////////////////////////

void Node::setName(const char *name) 
{
	if (name != 0) {
		char *nameString = strdup(name);
		int l = strlen(nameString);
        
		for (int n=0; n < l; ++n) {
			if (nameString[n] <= 0x20)
				nameString[n] = '_';
		}
		mName->setValue(nameString);
		free(nameString);
	} else {
		mName->setValue(NULL);
	}
}

const char *Node::getName() const
{
	return mName->getValue();
}

bool Node::hasName() const
{
	const char *name = getName();
	if (name == NULL)
		return false;
	if (strlen(name) <= 0)
		return false;
	return true;
}

////////////////////////////////////////////////
//	Type
////////////////////////////////////////////////

void Node::setType(const int type) 
{
	mType = type;
}

int Node::getType() const
{
	return mType;
}

const char *Node::getTypeString() const
{
	return GetNodeTypeString(mType);
}

////////////////////////////////////////////////
//	Node::addChildNode
////////////////////////////////////////////////

void Node::addChildNode(Node *node, bool initialize) {
	moveChildNode(node);
	if (initialize)
		node->initialize();
}

void Node::addChildNodeAtFirst(Node *node, bool initialize) {
	moveChildNodeAtFirst(node);
	if (initialize)
		node->initialize();
}

////////////////////////////////////////////////
//	Node::moveChildNode
////////////////////////////////////////////////

void Node::moveChildNode(Node *node) {
	mChildNodes->addNode(node); 
	node->setParentNode(this);
	node->setSceneGraph(getSceneGraph());
}

void Node::moveChildNodeAtFirst(Node *node) {
	mChildNodes->addNodeAtFirst(node); 
	node->setParentNode(this);
	node->setSceneGraph(getSceneGraph());
}

////////////////////////////////////////////////
//	Node::remove
////////////////////////////////////////////////

void Node::removeRoutes() 
{
	SceneGraph *sg = getSceneGraph();
	if (sg) {
		Route *route=sg->getRoutes();
		while (route) {
			Route *nextRoute = route->next();
			if (route->getEventInNode() == this || route->getEventOutNode() == this)
				delete route;
			route = nextRoute;
		}
	}
}

void Node::removeSFNodes() 
{
	SceneGraph *sg = getSceneGraph();
	if (sg) {
		for (ScriptNode *script = sg->findScriptNode(); script; script=script->nextTraversal()) {
			for (int n=0; n<script->getNFields(); n++) {
				Field *field = script->getField(n);
				if (field->getType() == fieldTypeSFNode) {
					SFNode *sfnode = (SFNode *)field;
					if (sfnode->getValue() == this)
						sfnode->setValue((Node *)NULL);
				}
			}
		}
	}
}

void Node::removeInstanceNodes() 
{
	SceneGraph *sg = getSceneGraph();
	if (sg && isInstanceNode() == false) {
		Node *node = sg->getNodes();
		while (node) {
			Node *nextNode = node->nextTraversal();
			if (node->isInstanceNode() == true) {
				Node *refNode = node->getReferenceNode();
				while (refNode->isInstanceNode() == true)
					refNode = refNode->getReferenceNode();
				if (refNode == this) {
					node->deleteChildNodes();
					nextNode = node->nextTraversal();
					delete node;
				}
			}
			node = nextNode;
		}
	
	}
}

void Node::remove() 
{
	LinkedListNode<Node>::remove();

	if (isInstanceNode() == false) {
		removeRoutes();
		removeSFNodes();
		removeInstanceNodes();

		if (isBindableNode()) {
			SceneGraph *sceneGraph = getSceneGraph();
			if (sceneGraph)
				sceneGraph->setBindableNode((BindableNode *)this, false);			
		}
	}

	setParentNode(NULL);
	setSceneGraph(NULL);
}

////////////////////////////////////////////////
//	Node::createField
////////////////////////////////////////////////

Field *Node::createField(int type)
{
	Field	*field = NULL;

	switch (type) {
	case fieldTypeSFBool:
		field = new SFBool();
		break;
	case fieldTypeSFFloat:
		field = new SFFloat();
		break;
	case fieldTypeSFInt32:
		field = new SFInt32();
		break;
	case fieldTypeSFVec2f:
		field = new SFVec2f();
		break;
	case fieldTypeSFVec3f:
		field = new SFVec3f();
		break;
	case fieldTypeSFString:
		field = new SFString();
		break;
	case fieldTypeSFColor:
		field = new SFColor();
		break;
	case fieldTypeSFTime:
		field = new SFTime();
		break;
	case fieldTypeSFRotation:
		field = new SFRotation();
		break;
	}

	assert(field != NULL);

	return field;
}

int Node::getNAllFields() const
{
	int fieldCount = 0;
	fieldCount += getNFields();
	fieldCount += getNExposedFields();
	fieldCount += getNEventIn();
	fieldCount += getNEventOut();
	return fieldCount;
}

Field *Node::findField(const char *name) const
{
	Field *field = NULL;

	field = getField(name);
	if (field != NULL)
		return field;

	field = getExposedField(name);
	if (field != NULL)
		return field;

	field = getEventIn(name);
	if (field != NULL)
		return field;

	field = getEventOut(name);
	if (field != NULL)
		return field;

	field = getPrivateField(name);
	if (field != NULL)
		return field;

	return NULL;
}

bool Node::hasMField() const
{
	int fieldSize = getNFields();
	int exposedfieldSize = getNExposedFields();
	int eventInSize = getNEventIn();
	int eventOutSize = getNEventOut();
	int privateFieldSize = getNPrivateFields();

	int n;
	for (n=0; n<fieldSize; n++) {
		Field *field = getField(n);
		if (field->isMField() == true)
			return true;
	}

	for (n=0; n<exposedfieldSize; n++) {
		Field *field = getExposedField(n);
		if (field->isMField() == true)
			return true;
	}

	for (n=0; n<eventInSize; n++) {
		Field *field = getEventIn(n);
		if (field->isMField() == true)
			return true;
	}

	for (n=0; n<eventOutSize; n++) {
		Field *field = getEventOut(n);
		if (field->isMField() == true)
			return true;
	}

	for (n=0; n<privateFieldSize; n++) {
		Field *field = getPrivateField(n);
		if (field->isMField() == true)
			return true;
	}

	return false;
}

////////////////////////////////////////////////
//	EventIn
////////////////////////////////////////////////

Field *Node::getEventIn(const char * fieldString) const
{

	String fieldName(fieldString);
		
	int nEventIn = getNEventIn();
	for (int n=0; n<nEventIn; n++) {
		Field *field = getEventIn(n);
		if (fieldName.compareTo(field->getName()) == 0)
			return field;
		if (fieldName.startsWith(eventInStripString) == 0) {
			if (fieldName.endsWith(field->getName()) == 0)
				return field;
		}
	}

	return NULL;
}

int Node::getNEventIn() const
{
	return mEventInField->size();
}

void Node::addEventIn(Field *field) 
{
	assert(field->getName() && strlen(field->getName()));
	assert(!getEventIn(field->getName()));
	mEventInField->addElement(field);
}

void Node::addEventIn(const char * name, Field *field) 
{
	assert(name && strlen(name));
	assert(!getEventIn(name));
	field->setName(name);
	mEventInField->addElement(field);
}

void Node::addEventIn(const char * name, int fieldType) 
{
	addEventIn(name, createField(fieldType));
}

Field *Node::getEventIn(int index) const
{
	return (Field *)mEventInField->elementAt(index);
}

int Node::getEventInNumber(Field *field)  const
{
	int nEventIn = getNEventIn();
	for (int n=0; n<nEventIn; n++) {
		if (getEventIn(n) == field)
			return n;
	}
	return -1;
}

////////////////////////////////////////////////
//	EventOut
////////////////////////////////////////////////

Field *Node::getEventOut(const char *fieldString) const
{

	String fieldName(fieldString);

	int nEventOut = getNEventOut();
	for (int n=0; n<nEventOut; n++) {
		Field *field = getEventOut(n);
		if (fieldName.compareTo(field->getName()) == 0)
			return field;
		if (fieldName.endsWith(eventOutStripString) == 0) {
			if (fieldName.startsWith(field->getName())  == 0)
				return field;
		}
	}
	return NULL;
}

int Node::getNEventOut() const
{
	return mEventOutField->size();
}

void Node::addEventOut(Field *field) 
{
	assert(field->getName() && strlen(field->getName()));
	assert(!getEventOut(field->getName()));
	mEventOutField->addElement(field);
}

void Node::addEventOut(const char *name, Field *field) 
{
	assert(name && strlen(name));
	assert(!getEventOut(name));
	field->setName(name);
	mEventOutField->addElement(field);
}

void Node::addEventOut(const char * name, int fieldType) 
{
	addEventOut(name, createField(fieldType));
}

Field *Node::getEventOut(int index) const
{
	return (Field *)mEventOutField->elementAt(index);
}

int Node::getEventOutNumber(Field *field) const
{
	int nEventOut = getNEventOut();
	for (int n=0; n<nEventOut; n++) {
		if (getEventOut(n) == field)
			return n;
	}
	return -1;
}

////////////////////////////////////////////////
//	ExposedField
////////////////////////////////////////////////

Field *Node::getExposedField(const char * fieldString) const
{
	
	String fieldName(fieldString);

	int nExposedField = getNExposedFields();
	for (int n=0; n<nExposedField; n++) {
		Field *field = getExposedField(n);
		const char *filedName = field->getName();
		if (fieldName.compareTo(filedName) == 0)
			return field;
		if (fieldName.startsWith(eventInStripString) == 0) {
			if (fieldName.endsWith(filedName) == 0)
				return field;
		}
		if (fieldName.endsWith(eventOutStripString) == 0) {
			if (fieldName.startsWith(filedName) == 0)
				return field;
		}
	}
	return NULL;
}

int Node::getNExposedFields() const
{
	return mExposedField->size();
}

void Node::addExposedField(Field *field) 
{
	assert(field->getName() && strlen(field->getName()));
	assert(!getExposedField(field->getName()));
	mExposedField->addElement(field);
}

void Node::addExposedField(const char * name, Field *field) 
{
	assert(name && strlen(name));
	assert(!getExposedField(name));
	field->setName(name);
	mExposedField->addElement(field);
}

void Node::addExposedField(const char * name, int fieldType) 
{
	addExposedField(name, createField(fieldType));
}

Field *Node::getExposedField(int index) const
{
	return (Field *)mExposedField->elementAt(index);
}

int Node::getExposedFieldNumber(Field *field) const
{
	int nExposedField = getNExposedFields();
	for (int n=0; n<nExposedField; n++) {
		if (getExposedField(n) == field)
			return n;
	}
	return -1;
}

////////////////////////////////////////////////
//	Field
////////////////////////////////////////////////

Field *Node::getField(const char *fieldString) const
{
	String fieldName(fieldString);

	int nField = getNFields();
	for (int n=0; n<nField; n++) {
		Field *field = getField(n);
		if (fieldName.compareTo(field->getName()) == 0)
			return field;
	}
	return NULL;
}

int Node::getNFields() const
{
	return mField->size();
}

void Node::addField(Field *field) 
{
	assert(field->getName() && strlen(field->getName()));
	assert(!getField(field->getName()));
	mField->addElement(field);
}

void Node::addField(const char * name, Field *field) 
{
	assert(name && strlen(name));
	assert(!getField(name));
	field->setName(name);
	mField->addElement(field);
}

void Node::addField(const char * name, int fieldType) 
{
	addField(name, createField(fieldType));
}

Field *Node::getField(int index) const
{
	return (Field *)mField->elementAt(index);
}

int Node::getFieldNumber(Field *field) const
{
	int nField = getNFields();
	for (int n=0; n<nField; n++) {
		if (getField(n) == field)
			return n;
	}
	return -1;
}

////////////////////////////////////////////////
//	PrivateField
////////////////////////////////////////////////

Field *Node::getPrivateField(const char *fieldString) const
{
		
	String fieldName(fieldString);

	int nPrivateField = getNPrivateFields();
	for (int n=0; n<nPrivateField; n++) {
		Field *field = getPrivateField(n);
		if (fieldName.compareTo(field->getName()) == 0)
			return field;
	}
	return NULL;
}

int Node::getNPrivateFields() const
{
	return mPrivateField->size();
}

void Node::addPrivateField(Field *field) 
{
	assert(field->getName() && strlen(field->getName()));
	assert(!getPrivateField(field->getName()));
	mPrivateField->addElement(field);
}

void Node::addPrivateField(const char * name, Field *field) 
{
	assert(name && strlen(name));
	assert(!getPrivateField(name));
	field->setName(name);
	mPrivateField->addElement(field);
}

Field *Node::getPrivateField(int index) const
{
	return (Field *)mPrivateField->elementAt(index);
}

int Node::getPrivateFieldNumber(Field *field) const
{
	int nPrivateField = getNPrivateFields();
	for (int n=0; n<nPrivateField; n++) {
		if (getPrivateField(n) == field)
			return n;
	}
	return -1;
}

////////////////////////////////////////////////
//	PrivateField
////////////////////////////////////////////////

int Node::getNPrivateNodeElements() const
{
	return mPrivateNodeVector->size();
}

void Node::addPrivateNodeElement(Node *node) 
{
	mPrivateNodeVector->addElement(node, false);
}

Node *Node::getPrivateNodeElementAt(int n)  const
{
	return mPrivateNodeVector->elementAt(n);
}

void Node::removeAllNodeElement() 
{
	mPrivateNodeVector->removeAllElements();
}

////////////////////////////////////////////////
//	Parent node
////////////////////////////////////////////////

void Node::setParentNode(Node *parentNode) 
{
	mParentNode = parentNode;
}

Node *Node::getParentNode() const
{
	return mParentNode;
}

bool Node::isParentNode(Node *node) const
{
	return (getParentNode() == node) ? true : false;
}

bool Node::isAncestorNode(Node *node) const
{
	for (Node *parentNode = getParentNode(); parentNode; parentNode = parentNode->getParentNode()) {
		if (node == parentNode)
				return true;
	}
	return false;
}

////////////////////////////////////////////////
//	Traversal node list
////////////////////////////////////////////////

Node *Node::nextTraversal() const
{
	Node *nextNode = getChildNodes();
	if (nextNode != NULL)
		return nextNode;
	nextNode = next();
	if (nextNode == NULL) {
		Node *parentNode = getParentNode();
		while (parentNode != NULL) { 
			Node *parentNextNode = parentNode->next();
			if (parentNextNode != NULL)
				return parentNextNode;
			parentNode = parentNode->getParentNode();
		}
	}
	return nextNode;
}

Node *Node::nextTraversalByType(const int type) const
{
	for (Node *node = nextTraversal(); node != NULL; node = node->nextTraversal()) {
		if (node->getType() == type)
				return node;
	}
	return NULL;
}

Node *Node::nextTraversalByName(const char *nameString) const
{
	if (nameString == NULL)
		return NULL;

	String name(nameString);

	for (Node *node = nextTraversal(); node != NULL; node = node->nextTraversal()) {
		if (node->getName() != NULL) {
			if (name.compareTo(node->getName()) == 0)
				return node;
		}
	}
	return NULL;
}

////////////////////////////////////////////////
//	next node list
////////////////////////////////////////////////

Node *Node::next() const
{
	return LinkedListNode<Node>::next(); 
}

Node *Node::next(const int type) const
{
	for (Node *node = next(); node != NULL; node = node->next()) {
		if (node->getType() == type)
			return node;
	}
	return NULL;
}

////////////////////////////////////////////////
//	child node list
////////////////////////////////////////////////

Node *Node::getChildNodes() const
{
	return mChildNodes->getNodes();
}

Node *Node::getChildNodeByType(int type) const
{
	// Thanks for Simon Goodall (03/17/04)
	for (Node *node = getChildNodes(); node != NULL; node = node->next()) {
		int nodeType = node->getType();
		if (node->isInstanceNode() == true) 
			nodeType = node->getReferenceNode()->getType();
		if (nodeType == type)
			return node;
	}
	return NULL;
}

Node *Node::getChildNode(int n) const
{
	return mChildNodes->getNode(n);
}

int Node::getNChildNodes() const
{
	return mChildNodes->getNNodes();
}

bool Node::hasChildNodes() const
{
	if (getChildNodes() == NULL)
		return false;
	return true;
}

////////////////////////////////////////////////
//	Add / Remove children (for Groupingnode)
////////////////////////////////////////////////

bool Node::isChildNode(Node *parentNode, Node *node) const
{
	for (Node *cnode = parentNode->getChildNodes(); cnode != NULL; cnode = cnode->next()) {
		if (cnode == node)
			return true;
		if (isChildNode(cnode, node) == true)
			return true;
	}
	return false;
}

bool Node::isChildNode(Node *node) const
{
	for (Node *cnode = getChildNodes(); cnode != NULL; cnode = cnode->next()) {
		if (isChildNode(cnode, node) == true)
			return true;
	}
	return false;
}

////////////////////////////////////////////////
//	get child node list
////////////////////////////////////////////////

GroupingNode *Node::getGroupingNodes() const
{
	for (Node *node = getChildNodes(); node != NULL; node = node->next()) {
		if (node->isGroupingNode())
			return (GroupingNode *)node;
	}
	return NULL;
}

Geometry3DNode *Node::getGeometry3DNode() const
{
	for (Node *node = getChildNodes(); node != NULL; node = node->next()) {
		if (node->isGeometry3DNode())
			return (Geometry3DNode *)node;
	}
	return NULL;
}

TextureNode *Node::getTextureNode() const
{
	for (Node *node = getChildNodes(); node != NULL; node = node->next()) {
		if (node->isTextureNode())
			return (TextureNode *)node;
	}
	return NULL;
}

////////////////////////////////////////////////
//	Node::getTransformMatrix(SFMatrix *matrix)
////////////////////////////////////////////////

void Node::getTransformMatrix(SFMatrix *mxOut) const
{
	mxOut->init();

	for (const Node *node=this; node; node=node->getParentNode()) {
		if (node->isTransformNode() || node->isBillboardNode()) {
			SFMatrix	mxNode;
			if (node->isTransformNode())
				((TransformNode *)node)->getSFMatrix(&mxNode);
			else
				((BillboardNode *)node)->getSFMatrix(&mxNode);
			mxNode.add(mxOut);
			mxOut->setValue(&mxNode);
		}
	}
}

////////////////////////////////////////////////
// is*
////////////////////////////////////////////////

bool Node::isNode(const int type) const
{
	if (getType() == type)
		return true;
	return false;
}

bool Node::isRootNode() const
{
	return isNode(ROOT_NODE);
}

bool Node::isDEFNode() const
{
	return isNode(DEF_NODE);
}

bool Node::isInlineChildNode() const
{
	Node *parentNode = getParentNode();
	while (parentNode != NULL) {
		if (parentNode->isInlineNode() == true)
			return true;
		parentNode = parentNode->getParentNode();
	}
	return false;
}

////////////////////////////////////////////////
//	SceneGraph
////////////////////////////////////////////////

void Node::setSceneGraph(SceneGraph *sceneGraph)	
{
	mSceneGraph = sceneGraph;
	for (Node *node = getChildNodes(); node; node = node->next()) {
			node->setSceneGraph(sceneGraph);
	}
}

SceneGraph *Node::getSceneGraph() const
{
	return mSceneGraph;
}

////////////////////////////////////////////////
//	Node::getTransformMatrix(float value[4][4])
////////////////////////////////////////////////

void Node::getTransformMatrix(float value[4][4]) const
{
	SFMatrix	mx;
	getTransformMatrix(&mx);
	mx.getValue(value);
}

////////////////////////////////////////////////
//	Node::getTranslationMatrix(SFMatrix *matrix)
////////////////////////////////////////////////

void Node::getTranslationMatrix(SFMatrix *mxOut) const
{
	mxOut->init();

	for (const Node *node=this; node; node=node->getParentNode()) {
		if (node->isTransformNode() || node->isBillboardNode()) {
			SFMatrix	mxNode;
			if (node->isTransformNode()) {
				float	translation[3];
				TransformNode *transNode = (TransformNode *)node;
				transNode->getTranslation(translation);
				mxNode.setTranslation(translation);
			}
			mxNode.add(mxOut);
			mxOut->setValue(&mxNode);
		}
	}
}

////////////////////////////////////////////////
//	Node::getTranslationMatrix(float value[4][4])
////////////////////////////////////////////////

void Node::getTranslationMatrix(float value[4][4]) const
{
	SFMatrix	mx;
	getTranslationMatrix(&mx);
	mx.getValue(value);
}

////////////////////////////////////////////////
//	Node::Route
////////////////////////////////////////////////

void Node::sendEvent(Field *eventOutField) {
	getSceneGraph()->updateRoute(this, eventOutField);
}

void Node::sendEvent(const char *eventOutFieldString) {
	getSceneGraph()->updateRoute(this, getEventOut(eventOutFieldString));
}

////////////////////////////////////////////////
//	Node::output (VRML97)
////////////////////////////////////////////////

char *Node::getIndentLevelString(int nIndentLevel) const
{
	char *indentString = new char[nIndentLevel+1];
	for (int n=0; n<nIndentLevel; n++)
		indentString[n] = '\t';
	indentString[nIndentLevel] = '\0';
	return indentString;
}

char *Node::getSpaceString(int nSpaces) const {
	char *spaceString = new char[nSpaces+1];
	for (int n=0; n<nSpaces; n++)
		spaceString[n] = ' ';
	spaceString[nSpaces] = '\0';
	return spaceString;
}

void Node::outputHead(std::ostream& printStream, const char *indentString) const
{
	if (getName() != NULL && strlen(getName()))
		printStream << indentString << "DEF " << getName() << " " << getTypeString() << " {" << std::endl;
	else
		printStream << indentString << getTypeString() << " {" << std::endl;
}

void Node::outputTail(std::ostream& printStream, const char * indentString) const
{
	printStream << indentString << "}" << std::endl;
}

void Node::outputContext(std::ostream& printStream, const char *indentString1, const char *indentString2) const
{
	char *indentString = new char[strlen(indentString1)+strlen(indentString2)+1];
	strcpy(indentString, indentString1);
	strcat(indentString, indentString2);
	outputContext(printStream, indentString);
	// Thanks for Simon Goodall (03/15/04)
	delete []indentString;
}

void Node::output(std::ostream& printStream, int indentLevet) const
{
	char *indentString = getIndentLevelString(indentLevet);

	if (isXMLNode() == true) {
		delete [] indentString;
		return;
	}

	if (isInstanceNode() == true) {
		printStream << indentString << "USE " << getName() << std::endl;
		delete [] indentString;
		return;
	}

	outputHead(printStream, indentString);
	outputContext(printStream, indentString);
	
	if (!isElevationGridNode() && !isShapeNode() && !isSoundNode() && !isPointSetNode() && !isIndexedFaceSetNode() && 
		!isIndexedLineSetNode() && !isTextNode() && !isAppearanceNode()) {
		if (getChildNodes() != NULL) {
			if (isLODNode()) 
				printStream << indentString << "\tlevel [" << std::endl;
			else if (isSwitchNode()) 
				printStream << indentString << "\tchoice [" << std::endl;
			else
				printStream << indentString <<"\tchildren [" << std::endl;
		
			for (Node *node = getChildNodes(); node; node = node->next()) {
				if (node->isInstanceNode() == false) 
					node->output(printStream, indentLevet+2);
				else
					node->output(printStream, indentLevet+2);
			}
			
			printStream << indentString << "\t]" << std::endl;
		}
	}

	outputTail(printStream, indentString);

	delete []indentString;
}

////////////////////////////////////////////////
//	Node::output (X3D)
////////////////////////////////////////////////

static bool hasOutputXMLField(const Field *field) 
{
	if (field->isSFNode() || field->isMFNode())
		return false;
		
	if (field->isMField()) {
		const MField *mfield = (MField *)field;
		int fieldSize = mfield->getSize();
		if (fieldSize == 0)
			return false;
	}

	return true;
}

static bool isOutputXMLFieldInSingleLine(const Node *node) 
{
	if (node->getType() == XML_NODE)
		return true;
	if (node->hasMField() == false)
		return true;
	return false;
}

void Node::outputXMLField(std::ostream& ps, Field *field, int indentLevel, bool isSingleLine)  const
{
	char *indentString = getIndentLevelString(indentLevel+1);
	const char *fieldName = field->getName();
	char *spaceString = getSpaceString(StringLength(fieldName)+2);

	if (hasOutputXMLField(field) == false) {
		delete [] indentString;
		delete [] spaceString;
		return;
	}

	if (isSingleLine == true)
		ps << " ";
	else
		ps << std::endl;
	
	if (field->isSField() == true) {
		if (isSingleLine == false)
			ps << indentString;
    if (field->toXMLString ())
      ps <<  fieldName << "=\"" << field->toXMLString() << "\"";
		delete [] indentString;
		delete [] spaceString;
		return;
	}
		
	if (field->isMField() == true) {
		MField *mfield = (MField *)field;
		int fieldSize = mfield->getSize();

		if (fieldSize == 0) {
			ps << indentString << fieldName << "=\"" << "\"";
			delete [] indentString;
			delete [] spaceString;
			return;
		}
			
		if (fieldSize == 1) {
			Field *eleField	= (Field*)mfield->getObject(0);
			string eleString = eleField->toXMLString();
      ps << indentString << fieldName << "=\"" << eleString.c_str() << "\"";
			delete [] indentString;
			delete [] spaceString;
			return;
		}

		for (int n=0; n<fieldSize; n++) {
			if (n==0)
				ps << indentString << fieldName << "=\"";
			ps << indentString << spaceString;
					
			Field *eleField	= (Field*)mfield->getObject(n);
			string eleString = eleField->toXMLString();
				
			ps << eleString;
				
			if (n < (fieldSize-1)) {
				ps << std::endl;
				/*
				if (mfield->isSingleValueMField() == true)
					ps << std::endl;
				else
					ps << "," << std::endl;
				*/
			}
			else
				ps << "\"";
		}
		delete [] indentString;
		delete [] spaceString;
		return;
	}

	delete []indentString;
	delete []spaceString;
}

void Node::outputXML(std::ostream& ps, int indentLevel) const
{
	char *indentString = getIndentLevelString(indentLevel);

	if (isInstanceNode() == true) {
		const char *typeName = getTypeString();
		const char *nodeName = getName(); 			
		ps << indentString << "<" << typeName << " USE=\"" << nodeName << "\"/>" << std::endl;
		delete [] indentString;
		return;
	}

	const char *tagName;

	if (isVRML97Node() == true) {
		tagName = getTypeString();
		const char *nodeName = getName(); 			
		if (HasString(nodeName) == true)
			ps << indentString << "<" << tagName << " DEF=\"" << nodeName << "\"";
		else
			ps << indentString << "<" << tagName;
	}
	else {
		tagName = getName();
		ps << indentString << "<" << tagName;
	}

	bool isSingleLine = isOutputXMLFieldInSingleLine(this);

	int allFieldSize = getNAllFields();
	int fieldSize = getNFields();
	int exposedfieldSize = getNExposedFields();
	int eventInSize = getNEventIn();
	int eventOutSize = getNEventOut();

	int n;
	for (n=0; n<fieldSize; n++) 
		outputXMLField(ps, getField(n), indentLevel, isSingleLine);

	for (n=0; n<exposedfieldSize; n++)
		outputXMLField(ps, getExposedField(n), indentLevel, isSingleLine);

	for (n=0; n<eventInSize; n++) 
		outputXMLField(ps, getEventIn(n), indentLevel, isSingleLine);

	for (n=0; n<eventOutSize; n++)
		outputXMLField(ps, getEventOut(n), indentLevel, isSingleLine);
		
	if (hasChildNodes() == false) {
		ps << ("/>") << std::endl;
		delete [] indentString;
		return;
	}

	ps << ">" << std::endl;
		
	for (Node *cnode = getChildNodes(); cnode; cnode = cnode->next())
		cnode->outputXML(ps, indentLevel+1);

	ps << indentString << "</" << tagName << ">" << std::endl;

	delete []indentString;
}

////////////////////////////////////////////////
//	InstanceNode
////////////////////////////////////////////////

void Node::setReferenceNodeMembers(Node *node)
{
	if (!node)
		return;

	mName				= node->mName;
	//mType				= node->mType;
	mExposedField		= node->mExposedField;
	mEventInField		= node->mEventInField;
	mEventOutField		= node->mEventOutField;
	mField				= node->mField;
	mPrivateField		= node->mPrivateField;
}

void Node::setOriginalMembers() 
{
	mName				= mOrgName;
	//mType				= mOrgType;
	mExposedField		= mOrgExposedField;
	mEventInField		= mOrgEventInField;
	mEventOutField		= mOrgEventOutField;
	mField				= mOrgField;
	mPrivateField		= mOrgPrivateField;
}
	
Node *Node::createInstanceNode()
{
	Node *instanceNode = NULL;
		
	if (isAnchorNode())
		instanceNode = new AnchorNode();
	else if (isAppearanceNode()) 
		instanceNode = new AppearanceNode();
	else if (isAudioClipNode())
		instanceNode = new AudioClipNode();
	else if (isBackgroundNode())
		instanceNode = new BackgroundNode();
	else if (isBillboardNode())
		instanceNode = new BillboardNode();
	else if (isBoxNode())
		instanceNode = new BoxNode();
	else if (isCollisionNode())
		instanceNode = new CollisionNode();
	else if (isColorNode())
		instanceNode = new ColorNode();
	else if (isColorInterpolatorNode())
		instanceNode = new ColorInterpolatorNode();
	else if (isConeNode())
		instanceNode = new ConeNode();
	else if (isCoordinateNode())
		instanceNode = new CoordinateNode();
	else if (isCoordinateInterpolatorNode())
		instanceNode = new CoordinateInterpolatorNode();
	else if (isCylinderNode())
		instanceNode = new CylinderNode();
	else if (isCylinderSensorNode())
		instanceNode = new CylinderSensorNode();
	else if (isDirectionalLightNode())
		instanceNode = new DirectionalLightNode();
	else if (isElevationGridNode())
		instanceNode = new ElevationGridNode();
	else if (isExtrusionNode())
		instanceNode = new ExtrusionNode();
	else if (isFogNode())
		instanceNode = new FogNode();
	else if (isFontStyleNode())
		instanceNode = new FontStyleNode();
	else if (isGroupNode())
		instanceNode = new GroupNode();
	else if (isImageTextureNode())
		instanceNode = new ImageTextureNode();
	else if (isIndexedFaceSetNode())
		instanceNode = new IndexedFaceSetNode();
	else if (isIndexedLineSetNode()) 
		instanceNode = new IndexedLineSetNode();
	else if (isInlineNode()) 
		instanceNode = new InlineNode();
	else if (isLODNode())
		instanceNode = new LODNode();
	else if (isMaterialNode())
		instanceNode = new MaterialNode();
	else if (isMovieTextureNode())
		instanceNode = new MovieTextureNode();
	else if (isNavigationInfoNode())
		instanceNode = new NavigationInfoNode();
	else if (isNormalNode())
		instanceNode = new NormalNode();
	else if (isNormalInterpolatorNode())
		instanceNode = new NormalInterpolatorNode();
	else if (isOrientationInterpolatorNode())
		instanceNode = new OrientationInterpolatorNode();
	else if (isPixelTextureNode())
		instanceNode = new PixelTextureNode();
	else if (isPlaneSensorNode())
		instanceNode = new PlaneSensorNode();
	else if (isPointLightNode())
		instanceNode = new PointLightNode();
	else if (isPointSetNode())
		instanceNode = new PointSetNode();
	else if (isPositionInterpolatorNode())
		instanceNode = new PositionInterpolatorNode();
	else if (isProximitySensorNode())
		instanceNode = new ProximitySensorNode();
	else if (isScalarInterpolatorNode())
		instanceNode = new ScalarInterpolatorNode();
	else if (isScriptNode())
		instanceNode = new ScriptNode();
	else if (isShapeNode())
		instanceNode = new ShapeNode();
	else if (isSoundNode())
		instanceNode = new SoundNode();
	else if (isSphereNode())
		instanceNode = new SphereNode();
	else if (isSphereSensorNode())
		instanceNode = new SphereSensorNode();
	else if (isSpotLightNode())
		instanceNode = new SpotLightNode();
	else if (isSwitchNode())
		instanceNode = new SwitchNode();
	else if (isTextNode())
		instanceNode = new TextNode();
	else if (isTextureCoordinateNode())
		instanceNode = new TextureCoordinateNode();
	else if (isTextureTransformNode())
		instanceNode = new TextureTransformNode();
	else if (isTimeSensorNode())
		instanceNode = new TimeSensorNode();
	else if (isTouchSensorNode())
		instanceNode = new TouchSensorNode();
	else if (isTransformNode())
		instanceNode = new TransformNode();
	else if (isViewpointNode())
		instanceNode = new ViewpointNode();
	else if (isVisibilitySensorNode())
		instanceNode = new VisibilitySensorNode();
	else if (isWorldInfoNode())
		instanceNode = new WorldInfoNode();

	assert(instanceNode);

	if (instanceNode) {
		Node *refNode = this;
		while (refNode->isInstanceNode() == true) 
			refNode = refNode->getReferenceNode();
		instanceNode->setAsInstanceNode(refNode);
		for (Node *cnode=getChildNodes(); cnode; cnode = cnode->next()) {
			Node *childInstanceNode = cnode->createInstanceNode();
			instanceNode->addChildNode(childInstanceNode);
		}
	}		
		
	return instanceNode;
}

////////////////////////////////////////////////
//	DEF node
////////////////////////////////////////////////

DEFNode *Node::createDEFNode()
{
	DEFNode *defNode = new DEFNode();

	Node *refNode = this;
	while (refNode->isInstanceNode() == true) 
		refNode = refNode->getReferenceNode();
	defNode->setAsInstanceNode(refNode);

	return defNode;
}
