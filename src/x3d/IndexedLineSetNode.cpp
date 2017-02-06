/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	IndexedLineSetNode.cpp
*
*	Revisions:
*
*	11/25/02
*		- Added the following new X3Df fields.
*			lineWidth
*
******************************************************************/

#include <x3d/IndexedLineSetNode.h>
#include <x3d/Graphic3D.h>

using namespace CyberX3D;

static const char lineWidthFieldString[] = "lineWidth";
static const char colorExposedFieldString[] = "color";
static const char coordExposedFieldString[] = "coord";
static const char coordIndexEventInString[] = "coordIndex";
static const char colorIndexEventInString[] = "colorIndex";

IndexedLineSetNode::IndexedLineSetNode() 
{
	setHeaderFlag(false);
	setType(INDEXEDLINESET_NODE);

	///////////////////////////
	// ExposedField 
	///////////////////////////

	// color field
	colorField = new SFNode();
	addExposedField(colorExposedFieldString, colorField);

	// coord field
	coordField = new SFNode();
	addExposedField(coordExposedFieldString, coordField);

	///////////////////////////
	// Field 
	///////////////////////////

	// colorPerVertex  field
	colorPerVertexField = new SFBool(true);
	colorPerVertexField->setName(colorPerVertexFieldString);
	addField(colorPerVertexField);

	// coordIndex  field
	coordIndexField = new MFInt32();
	coordIndexField->setName(coordIndexFieldString);
	addField(coordIndexField);

	// colorIndex  field
	colorIndexField = new MFInt32();
	colorIndexField->setName(colorIndexFieldString);
	addField(colorIndexField);

	// lineWidth  field
	lineWidthField = new SFFloat(1.0f);
	lineWidthField->setName(lineWidthFieldString);
	addField(lineWidthField);

	///////////////////////////
	// EventIn
	///////////////////////////

	// coordIndex  EventIn
	set_coordIndexField = new MFInt32();
	set_coordIndexField->setName(coordIndexFieldString);
	addEventIn(set_coordIndexField);

	// colorIndex  EventIn
	set_colorIndexField = new MFInt32();
	set_colorIndexField->setName(colorIndexFieldString);
	addEventIn(set_colorIndexField);
}

IndexedLineSetNode::~IndexedLineSetNode() 
{
}
	
////////////////////////////////////////////////
//	Color
////////////////////////////////////////////////

SFNode *IndexedLineSetNode::getColorField() const
{
	if (isInstanceNode() == false)
		return colorField;
	return (SFNode *)getExposedField(colorExposedFieldString);
}
	
////////////////////////////////////////////////
//	Coord
////////////////////////////////////////////////

SFNode *IndexedLineSetNode::getCoordField() const
{
	if (isInstanceNode() == false)
		return coordField;
	return (SFNode *)getExposedField(coordExposedFieldString);
}

////////////////////////////////////////////////
//	ColorPerVertex
////////////////////////////////////////////////

SFBool *IndexedLineSetNode::getColorPerVertexField() const
{
	if (isInstanceNode() == false)
		return colorPerVertexField;
	return (SFBool *)getField(colorPerVertexFieldString);
}
	
void IndexedLineSetNode::setColorPerVertex(bool value) 
{
	getColorPerVertexField()->setValue(value);
}

void IndexedLineSetNode::setColorPerVertex(int value) 
{
	setColorPerVertex(value ? true : false);
}

bool IndexedLineSetNode::getColorPerVertex() const
{
	return getColorPerVertexField()->getValue();
}

////////////////////////////////////////////////
// CoordIndex
////////////////////////////////////////////////

MFInt32 *IndexedLineSetNode::getCoordIndexField() const
{
	if (isInstanceNode() == false)
		return coordIndexField;
	return (MFInt32 *)getField(coordIndexFieldString);
}

void IndexedLineSetNode::addCoordIndex(int value) 
{
	getCoordIndexField()->addValue(value);
}

int IndexedLineSetNode::getNCoordIndexes() const
{
	return getCoordIndexField()->getSize();
}

int IndexedLineSetNode::getCoordIndex(int index) const
{
	return getCoordIndexField()->get1Value(index);
}

void IndexedLineSetNode::clearCoordIndex() 
{
	getCoordIndexField()->clear();
}
	
////////////////////////////////////////////////
// ColorIndex
////////////////////////////////////////////////

MFInt32 *IndexedLineSetNode::getColorIndexField() const
{
	if (isInstanceNode() == false)
		return colorIndexField;
	return (MFInt32 *)getField(colorIndexFieldString);
}

void IndexedLineSetNode::addColorIndex(int value) 
{
	getColorIndexField()->addValue(value);
}

int IndexedLineSetNode::getNColorIndexes() const
{
	return getColorIndexField()->getSize();
}

int IndexedLineSetNode::getColorIndex(int index) const
{
	return getColorIndexField()->get1Value(index);
}

void IndexedLineSetNode::clearColorIndex() 
{
	getColorIndexField()->clear();
}

////////////////////////////////////////////////
//	LineWidth
////////////////////////////////////////////////

SFFloat *IndexedLineSetNode::getLineWidthField() const
{
	if (isInstanceNode() == false)
		return lineWidthField;
	return (SFFloat *)getField(lineWidthFieldString);
}
	
void IndexedLineSetNode::setLineWidth(float value) 
{
	getLineWidthField()->setValue(value);
}

float IndexedLineSetNode::getLineWidth() const
{
	return getLineWidthField()->getValue();
}

////////////////////////////////////////////////
// set_coordIndex
////////////////////////////////////////////////

MFInt32 *IndexedLineSetNode::getSetCoordIndexField() const
{
	if (isInstanceNode() == false)
		return set_coordIndexField;
	return (MFInt32 *)getEventIn(coordIndexEventInString);
}

////////////////////////////////////////////////
// set_colorIndex
////////////////////////////////////////////////

MFInt32 *IndexedLineSetNode::getSetColorIndexField() const
{
	if (isInstanceNode() == false)
		return set_colorIndexField;
	return (MFInt32 *)getEventIn(colorIndexEventInString);
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool IndexedLineSetNode::isChildNodeType(Node *node) const
{
	if (node->isColorNode() || node->isCoordinateNode())
		return true;
	else
		return false;
}

void IndexedLineSetNode::initialize() 
{
	if (!isInitialized()) {
#ifdef CX3D_SUPPORT_OPENGL
		recomputeDisplayList();
#endif
		recomputeBoundingBox();
		setInitialized(true);
	}
}

void IndexedLineSetNode::uninitialize() 
{
}

void IndexedLineSetNode::update() 
{
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

IndexedLineSetNode *IndexedLineSetNode::next() const
{
	return (IndexedLineSetNode *)Node::next(getType());
}

IndexedLineSetNode *IndexedLineSetNode::nextTraversal() const
{
	return (IndexedLineSetNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	Infomation
////////////////////////////////////////////////

void IndexedLineSetNode::outputContext(std::ostream &printStream, const char *indentString) const
{
	const SFBool *colorPerVertex = getColorPerVertexField();

	printStream << indentString << "\t" << "colorPerVertex " << colorPerVertex << std::endl;

	const ColorNode *color = getColorNodes();
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

	const CoordinateNode *coord = getCoordinateNodes();
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

	if (0 < getNCoordIndexes()) {
		const MFInt32 *coordIndex = getCoordIndexField();
		printStream << indentString << "\t" << "coordIndex [" << std::endl;
		coordIndex->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t" << "]" << std::endl;
	}

	if (0 < getNColorIndexes()) {
		const MFInt32 *colorIndex = getColorIndexField();
		printStream << indentString << "\t" << "colorIndex [" << std::endl;
		colorIndex->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t" << "]" << std::endl;
	}
}

////////////////////////////////////////////////////////////
//	IndexedLineSetNode::recomputeBoundingBox
////////////////////////////////////////////////////////////

/*
void IndexedLineSetNode::recomputeBoundingBox() 
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
*/

////////////////////////////////////////////////
//	IndexedLineSetNode::recomputeDisplayList
////////////////////////////////////////////////

#ifdef CX3D_SUPPORT_OPENGL

static void DrawIdxLineSet(IndexedLineSetNode *idxLineSet)
{
	CoordinateNode *coordinate = idxLineSet->getCoordinateNodes();
	if (!coordinate)
		return;

	NormalNode	*normal = idxLineSet->getNormalNodes();
	ColorNode	*color = idxLineSet->getColorNodes();
	int		bColorPerVertex =idxLineSet->getColorPerVertex();

	bool	bLineBegin = true;
	bool	bLineClose = true;
	int		nLine = 0;

	float	vpoint[3];
	float	pcolor[3];

	glColor3f(1.0f, 1.0f, 1.0f);

	int nCoordIndexes = idxLineSet->getNCoordIndexes();
	for (int nCoordIndex=0; nCoordIndex<nCoordIndexes; nCoordIndex++) {

		int coordIndex = idxLineSet->getCoordIndex(nCoordIndex);

		if (bLineBegin) {
			glBegin(GL_LINE_STRIP);
			bLineBegin = false;
			bLineClose = false;

			if (color && !bColorPerVertex) {
				color->getColor(nLine, pcolor);
				glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, pcolor);
//				glColor3fv(pcolor);
			}

			nLine++;
		}

		if (coordIndex != -1) {
			coordinate->getPoint(coordIndex, vpoint);
			glVertex3fv(vpoint);

			if (color && bColorPerVertex) {
				color->getColor(coordIndex, pcolor);
				glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, pcolor);
//				glColor3fv(pcolor);
			}
		}
		else {
			glEnd();
			bLineBegin = true;
			bLineClose = true;
		}
	}

	if (bLineClose == false)
		glEnd();
}

void IndexedLineSetNode::recomputeDisplayList() 
{
	CoordinateNode *coordinate = getCoordinateNodes();
	if (!coordinate)
		return;

	unsigned int nCurrentDisplayList = getDisplayList();
	if (0 < nCurrentDisplayList)
		glDeleteLists(nCurrentDisplayList, 1);

	unsigned int nNewDisplayList = glGenLists(1);
	glNewList(nNewDisplayList, GL_COMPILE);
		DrawIdxLineSet(this);
	glEndList();

	setDisplayList(nNewDisplayList);
};

#endif
