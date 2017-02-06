/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	PointSetNode.cpp
*
******************************************************************/

#include <x3d/PointSetNode.h>
#include <x3d/Graphic3D.h>

using namespace CyberX3D;

PointSetNode::PointSetNode() 
{
	setHeaderFlag(false);
	setType(POINTSET_NODE);
}

PointSetNode::~PointSetNode() {
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

PointSetNode *PointSetNode::next() const
{
	return (PointSetNode *)Node::next(getType());
}

PointSetNode *PointSetNode::nextTraversal() const
{
	return (PointSetNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool PointSetNode::isChildNodeType(Node *node) const
{
	if (node->isCoordinateNode() || node->isColorNode())
		return true;
	else
		return false;
}

void PointSetNode::initialize() 
{
	if (!isInitialized()) {
#ifdef CX3D_SUPPORT_OPENGL
		recomputeDisplayList();
#endif
		recomputeBoundingBox();
		setInitialized(1);
	}
}

void PointSetNode::uninitialize() 
{
}

void PointSetNode::update() 
{
}

////////////////////////////////////////////////
//	Infomation
////////////////////////////////////////////////

void PointSetNode::outputContext(std::ostream &printStream, const char *indentString) const 
{
	ColorNode *color = getColorNodes();
	if (color != NULL) {
		if (color->isInstanceNode() == false) {
			if (color->getName() != NULL && strlen(color->getName()))
				printStream << indentString << "\t" << "color " << "DEF " << color->getName() << " Color {" << std::endl;
			else
				printStream << indentString << "\t" << "color Color {" << std::endl;
			color->Node::outputContext(printStream, indentString, "\t");
			printStream << indentString << "\t" << "}" << std::endl;
		}
		else 
			printStream << indentString << "\t" << "color USE " << color->getName() << std::endl;
	}

	CoordinateNode *coord = getCoordinateNodes();
	if (coord != NULL) {
		if (coord->isInstanceNode() == false) {
			if (coord->getName() != NULL && strlen(coord->getName()))
				printStream << indentString << "\t" << "coord " << "DEF " << coord->getName() << " Coordinate {" << std::endl;
			else
				printStream << indentString << "\t" << "coord Coordinate {" << std::endl;
			coord->Node::outputContext(printStream, indentString, "\t");
			printStream << indentString << "\t" << "}" << std::endl;
		}
		else 
			printStream << indentString << "\t" << "coord USE " << coord->getName() << std::endl;
	}
}

////////////////////////////////////////////////////////////
//	PointSetNode::recomputeBoundingBox
////////////////////////////////////////////////////////////

void PointSetNode::recomputeBoundingBox() 
{
	CoordinateNode *coordinate = getCoordinateNodes();
	if (!coordinate) {
		setBoundingBoxCenter(0.0f, 0.0f, 0.0f);
		setBoundingBoxSize(-1.0f, -1.0f, -1.0f);
		return;
	}

	BoundingBox bbox;
	float		point[3];

	int nCoordinatePoints = coordinate->getNPoints();
	for (int n=0; n<nCoordinatePoints; n++) {
		coordinate->getPoint(n, point);
		bbox.addPoint(point);
	}

	setBoundingBox(&bbox);
}

////////////////////////////////////////////////
//	PointSetNode::recomputeDisplayList
////////////////////////////////////////////////

#ifdef CX3D_SUPPORT_OPENGL

static void DrawPointSet(PointSetNode *pointSet)
{
	CoordinateNode *coordinate = pointSet->getCoordinateNodes();
	if (!coordinate)
		return;

	NormalNode	*normal = pointSet->getNormalNodes();
	ColorNode	*color = pointSet->getColorNodes();

	float	vpoint[3];
	float	pcolor[3];

	glColor3f(1.0f, 1.0f, 1.0f);

	glBegin(GL_POINTS);

	int nCoordinatePoint = coordinate->getNPoints();
	for (int n=0; n<nCoordinatePoint; n++) {

		if (color) {
			color->getColor(n, pcolor);
			glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, pcolor);
//			glColor3fv(pcolor);
		}

		coordinate->getPoint(n, vpoint);
		glVertex3fv(vpoint);
	}

	glEnd();
}

void PointSetNode::recomputeDisplayList() 
{
	CoordinateNode *coordinate = getCoordinateNodes();
	if (!coordinate)
		return;

	unsigned int nCurrentDisplayList = getDisplayList();
	if (0 < nCurrentDisplayList)
		glDeleteLists(nCurrentDisplayList, 1);

	unsigned int nNewDisplayList = glGenLists(1);
	glNewList(nNewDisplayList, GL_COMPILE);
		DrawPointSet(this);
	glEndList();

	setDisplayList(nNewDisplayList);
};

#endif
