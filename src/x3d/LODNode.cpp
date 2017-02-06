/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	Node.cpp
*
*	11/19/02
*		- Changed the super class from Node to BoundedGroupingNode.
*
******************************************************************/

#include <x3d/SceneGraph.h>

using namespace CyberX3D;

LODNode::LODNode() 
{
	setHeaderFlag(false);
	setType(LOD_NODE);

	// center field
	centerField = new SFVec3f(0.0f, 0.0f, 0.0f);
	addField(centerFieldString, centerField);

	// range field
	rangeField = new MFFloat();
	addField(rangeFieldString, rangeField);
}

LODNode::~LODNode() 
{
}
	
////////////////////////////////////////////////
//	center
////////////////////////////////////////////////

SFVec3f *LODNode::getCenterField() const
{
	if (isInstanceNode() == false)
		return centerField;
	return (SFVec3f *)getField(centerFieldString);
}

void LODNode::setCenter(float value[]) 
{
	getCenterField()->setValue(value);
}

void LODNode::setCenter(float x, float y, float z) 
{
	getCenterField()->setValue(x, y, z);
}

void LODNode::getCenter(float value[]) const
{
	getCenterField()->getValue(value);
}

////////////////////////////////////////////////
//	range 
////////////////////////////////////////////////

MFFloat *LODNode::getRangeField() const
{
	if (isInstanceNode() == false)
		return rangeField;
	return (MFFloat *)getField(rangeFieldString);
}

void LODNode::addRange(float value) 
{
	getRangeField()->addValue(value);
}

int LODNode::getNRanges() const
{
	return getRangeField()->getSize();
}

float LODNode::getRange(int index) const
{
	return getRangeField()->get1Value(index);
}


////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

LODNode *LODNode::next() const
{
	return (LODNode *)Node::next(getType());
}

LODNode *LODNode::nextTraversal() const
{
	return (LODNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool LODNode::isChildNodeType(Node *node) const
{
	if (node->isCommonNode() || node->isBindableNode() ||node->isInterpolatorNode() || node->isSensorNode() || node->isGroupingNode() || node->isSpecialGroupNode())
		return true;
	else
		return false;
}

////////////////////////////////////////////////
//	Infomation
////////////////////////////////////////////////

void LODNode::outputContext(std::ostream &printStream, const char *indentString) const
{
	SFVec3f *center = getCenterField();
	printStream << indentString << "\t" << "center " << center << std::endl;

	if (0 < getNRanges()) {
		MFFloat *range = getRangeField();
		printStream << indentString << "\t" << "range [" << std::endl;
		range->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t" << "]" << std::endl;
	}
}

////////////////////////////////////////////////
//	LODNode::update
////////////////////////////////////////////////

void UpdateLOD(LODNode *lod)
{
	int nNodes = lod->getNPrivateNodeElements();
	for (int n=0; n<nNodes; n++) {
		Node *node = lod->getPrivateNodeElementAt(n);
		node->remove();
	}

	SceneGraph *sg = lod->getSceneGraph();

	ViewpointNode *vpoint = sg->getViewpointNode();
	if (vpoint == NULL)
		vpoint = sg->getDefaultViewpointNode();

	if (vpoint) {
		SFMatrix	viewMatrix;
		float		viewPosition[3];
		vpoint->getTransformMatrix(&viewMatrix);
		vpoint->getPosition(viewPosition);
		viewMatrix.multi(viewPosition);

		SFMatrix	lodMatrix;
		float		lodCenter[3];
		lod->getTransformMatrix(&lodMatrix);
		lod->getCenter(lodCenter);
		lodMatrix.multi(lodCenter);

		float lx = lodCenter[0] - viewPosition[0];
		float ly = lodCenter[1] - viewPosition[1];
		float lz = lodCenter[2] - viewPosition[2];
		float distance = (float)sqrt(lx*lx + ly*ly + lz*lz);

	
		int numRange = lod->getNRanges();
		int nRange = 0;
		for (nRange=0; nRange<numRange; nRange++) {
			if (distance < lod->getRange(nRange))
				break;
		}

		Node *node = lod->getPrivateNodeElementAt(nRange);
		if (!node)
			node = lod->getPrivateNodeElementAt(lod->getNPrivateNodeElements() - 1);
		assert(node);
		lod->addChildNode(node);
	}
}

void LODNode::update() 
{
	UpdateLOD(this);
}

////////////////////////////////////////////////
//	LODNode::initialize
////////////////////////////////////////////////

void InitializeLOD(LODNode *lod)
{
	lod->uninitialize();

	Node *node = lod->getChildNodes();
	while (node) {
		Node *nextNode = node->next();
//		node->remove();
		lod->addPrivateNodeElement(node);
		node = nextNode;
	}
/*
	Node *firstNode = lod->getPrivateNodeElementAt(0);
	if (firstNode)
		lod->addChildNode(firstNode);
*/
}

void LODNode::initialize() 
{
	if (isInitialized() == false) {
		InitializeLOD(this);
		setInitialized(true);
	}
}

////////////////////////////////////////////////
//	LODNode::uninitialize
////////////////////////////////////////////////

void UninitializeLOD(LODNode *lod) 
{
	int nNodes = lod->getNPrivateNodeElements();
	for (int n=0; n<nNodes; n++) {
		Node *node = lod->getPrivateNodeElementAt(n);
		node->remove();
		lod->addChildNode(node);
	}
	lod->removeAllNodeElement();
}

void LODNode::uninitialize() 
{
	if (isInitialized() == true) {
		UninitializeLOD(this);
		setInitialized(false);
	}
}

