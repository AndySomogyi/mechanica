/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	ExtrusionNode.cpp
*
*	Revisions:
*
*	12/03/02
*		- Changed the super class from Geometry3DNode to ExtrusionNode.
*	03/30/04
*		- Added DrawExtrusionVercities() to set each normals.
*		- Changed to inverse the vertex sequence of the side polygons.
*
******************************************************************/

#include <x3d/ExtrusionNode.h>
#include <x3d/Graphic3D.h>
#include <x3d/MathUtil.h>

using namespace CyberX3D;

void AddDefaultParameters(ExtrusionNode *ex);

ExtrusionNode::ExtrusionNode() 
{

	setHeaderFlag(false);
	setType(EXTRUSION_NODE);

	///////////////////////////
	// Field 
	///////////////////////////
		
	// beginCap field
	beginCapField = new SFBool(true);
	addField(beginCapFieldString, beginCapField);

	// endCap field
	endCapField = new SFBool(true);
	addField(endCapFieldString, endCapField);

	// convex field
	convexField = new SFBool(true);
	convexField->setName(convexFieldString);
	addField(convexField);

	// creaseAngle field
	creaseAngleField = new SFFloat(0.0f);
	creaseAngleField->setName(creaseAngleFieldString);
	addField(creaseAngleField);

	// orientation field
	orientationField = new MFRotation();
	orientationField->setName(orientationFieldString);
	addField(orientationField);

	// scale field
	scaleField = new MFVec2f();
	scaleField->setName(scaleFieldString);
	addField(scaleField);

	// crossSection field
	crossSectionField = new MFVec2f();
	addField(crossSectionFieldString, crossSectionField);

	// spine field
	spineField = new MFVec3f();
	addField(spineFieldString, spineField);

	// ccw  field
	ccwField = new SFBool(true);
	ccwField->setName(ccwFieldString);
	addField(ccwField);

	// solid  field
	solidField = new SFBool(true);
	solidField->setName(solidFieldString);
	addField(solidField);

	///////////////////////////
	// EventIn
	///////////////////////////

	// orientation EventIn
	MFRotation *setOrientationField = new MFRotation();
	setOrientationField->setName(orientationFieldString);
	addEventIn(setOrientationField);

	// scale EventIn
	MFVec2f *setScaleField = new MFVec2f();
	setScaleField->setName(scaleFieldString);
	addEventIn(setScaleField);

	// crossSection EventIn
	MFVec2f *setCrossSectionField = new MFVec2f();
	addEventIn(crossSectionFieldString, setCrossSectionField);

	// spine EventIn
	MFVec3f *setSpineField = new MFVec3f();
	addEventIn(spineFieldString, setSpineField);
}

ExtrusionNode::~ExtrusionNode() 
{
}

////////////////////////////////////////////////
//	BeginCap
////////////////////////////////////////////////

SFBool *ExtrusionNode::getBeginCapField() const
{
	if (isInstanceNode() == false)
		return beginCapField;
	return (SFBool *)getField(beginCapFieldString);
}
	
void ExtrusionNode::setBeginCap(bool value) 
{
	getBeginCapField()->setValue(value);
}

void ExtrusionNode::setBeginCap(int value) 
{
	setBeginCap(value ? true : false);
}

bool ExtrusionNode::getBeginCap() const
{
	return getBeginCapField()->getValue();
}

////////////////////////////////////////////////
//	EndCap
////////////////////////////////////////////////

SFBool *ExtrusionNode::getEndCapField() const
{
	if (isInstanceNode() == false)
		return endCapField;
	return (SFBool *)getField(endCapFieldString);
}
	
void ExtrusionNode::setEndCap(bool value) 
{
	getEndCapField()->setValue(value);
}

void ExtrusionNode::setEndCap(int value) 
{
	setEndCap(value ? true : false);
}

bool ExtrusionNode::getEndCap() const
{
	return getEndCapField()->getValue();
}

////////////////////////////////////////////////
//	Convex
////////////////////////////////////////////////

SFBool *ExtrusionNode::getConvexField() const
{
	if (isInstanceNode() == false)
		return convexField;
	return (SFBool *)getField(convexFieldString);
}
	
void ExtrusionNode::setConvex(bool value) 
{
	getConvexField()->setValue(value);
}

void ExtrusionNode::setConvex(int value) 
{
	setConvex(value ? true : false);
}

bool ExtrusionNode::getConvex() const
{
	return getConvexField()->getValue();
}

////////////////////////////////////////////////
//	CCW
////////////////////////////////////////////////

SFBool *ExtrusionNode::getCCWField() const
{
	if (isInstanceNode() == false)
		return ccwField;
	return (SFBool *)getField(ccwFieldString);
}
	
void ExtrusionNode::setCCW(bool value) 
{
	getCCWField()->setValue(value);
}

void ExtrusionNode::setCCW(int value) 
{
	setCCW(value ? true : false);
}

bool ExtrusionNode::getCCW() const
{
	return getCCWField()->getValue();
}


////////////////////////////////////////////////
//	Solid
////////////////////////////////////////////////

SFBool *ExtrusionNode::getSolidField() const
{
	if (isInstanceNode() == false)
		return solidField;
	return (SFBool *)getField(solidFieldString);
}

void ExtrusionNode::setSolid(bool value) 
{
	getSolidField()->setValue(value);
}
	
void ExtrusionNode::setSolid(int value) 
{
	setSolid(value ? true : false);
}

bool ExtrusionNode::getSolid() const
{
	return getSolidField()->getValue();
}

////////////////////////////////////////////////
//	CreaseAngle
////////////////////////////////////////////////

SFFloat *ExtrusionNode::getCreaseAngleField() const
{
	if (isInstanceNode() == false)
		return creaseAngleField;
	return (SFFloat *)getField(creaseAngleFieldString);
}
	
void ExtrusionNode::setCreaseAngle(float value) 
{
	getCreaseAngleField()->setValue(value);
}

float ExtrusionNode::getCreaseAngle() const
{
	return getCreaseAngleField()->getValue();
}

////////////////////////////////////////////////
// orientation
////////////////////////////////////////////////

MFRotation *ExtrusionNode::getOrientationField() const
{
	if (isInstanceNode() == false)
		return orientationField;
	return (MFRotation *)getField(orientationFieldString);
}

void ExtrusionNode::addOrientation(float value[]) 
{
	getOrientationField()->addValue(value);
}

void ExtrusionNode::addOrientation(float x, float y, float z, float angle) 
{
	getOrientationField()->addValue(x, y, z, angle);
}

int ExtrusionNode::getNOrientations() const
{
	return getOrientationField()->getSize();
}

void ExtrusionNode::getOrientation(int index, float value[]) const
{
	getOrientationField()->get1Value(index, value);
}

////////////////////////////////////////////////
// scale
////////////////////////////////////////////////

MFVec2f *ExtrusionNode::getScaleField() const
{
	if (isInstanceNode() == false)
		return scaleField;
	return (MFVec2f *)getField(scaleFieldString);
}

void ExtrusionNode::addScale(float value[]) 
{
	getScaleField()->addValue(value);
}

void ExtrusionNode::addScale(float x, float z) 
{
	getScaleField()->addValue(x, z);
}

int ExtrusionNode::getNScales() const
{
	return getScaleField()->getSize();
}

void ExtrusionNode::getScale(int index, float value[]) const
{
	getScaleField()->get1Value(index, value);
}

////////////////////////////////////////////////
// crossSection
////////////////////////////////////////////////

MFVec2f *ExtrusionNode::getCrossSectionField() const
{
	if (isInstanceNode() == false)
		return crossSectionField;
	return (MFVec2f *)getField(crossSectionFieldString);
}

void ExtrusionNode::addCrossSection(float value[]) 
{
	getCrossSectionField()->addValue(value);
}

void ExtrusionNode::addCrossSection(float x, float z) 
{
	getCrossSectionField()->addValue(x, z);
}

int ExtrusionNode::getNCrossSections() const
{
	return getCrossSectionField()->getSize();
}

void ExtrusionNode::getCrossSection(int index, float value[]) const
{
	getCrossSectionField()->get1Value(index, value);
}

////////////////////////////////////////////////
// spine
////////////////////////////////////////////////

MFVec3f *ExtrusionNode::getSpineField() const
{
	if (isInstanceNode() == false)
		return spineField;
	return (MFVec3f *)getField(spineFieldString);
}

void ExtrusionNode::addSpine(float value[]) 
{
	getSpineField()->addValue(value);
}

void ExtrusionNode::addSpine(float x, float y, float z) 
{
	getSpineField()->addValue(x, y, z);
}

int ExtrusionNode::getNSpines() const
{
	return getSpineField()->getSize();
}

void ExtrusionNode::getSpine(int index, float value[]) const
{
	getSpineField()->get1Value(index, value);
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

ExtrusionNode *ExtrusionNode::next() const
{
	return (ExtrusionNode *)Node::next(getType());
}

ExtrusionNode *ExtrusionNode::nextTraversal() const
{
	return (ExtrusionNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool ExtrusionNode::isChildNodeType(Node *node) const
{
	return false;
}

void ExtrusionNode::initialize() 
{
	if (!isInitialized()) {
		AddDefaultParameters(this);
#ifdef CX3D_SUPPORT_OPENGL
		recomputeDisplayList();
#endif
		recomputeBoundingBox();
		setInitialized(true);
	}
}

void ExtrusionNode::uninitialize() 
{
}

void ExtrusionNode::update() 
{
}

////////////////////////////////////////////////
//	Infomation
////////////////////////////////////////////////

void ExtrusionNode::outputContext(std::ostream &printStream, const char *indentString) const
{
	const SFBool *beginCap = getBeginCapField();
	const SFBool *endCap = getEndCapField();
	const SFBool *ccw = getCCWField();
	const SFBool *convex = getConvexField();
	const SFBool *solid = getSolidField();

	printStream << indentString << "\t" << "beginCap " << beginCap << std::endl;
	printStream << indentString << "\t" << "endCap " << endCap << std::endl;
	printStream << indentString << "\t" << "solid " << solid << std::endl;
	printStream << indentString << "\t" << "ccw " << ccw << std::endl;
	printStream << indentString << "\t" << "convex " << convex << std::endl;
	printStream << indentString << "\t" << "creaseAngle " << getCreaseAngle() << std::endl;

	if (0 < getNCrossSections()) {
		const MFVec2f *crossSection = getCrossSectionField();
		printStream << indentString << "\t" << "crossSection [" << std::endl;
		crossSection->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t" << "]" << std::endl;
	}

	if (0 < getNOrientations()) {
		const MFRotation *orientation = getOrientationField();
		printStream << indentString << "\t" << "orientation [" << std::endl;
		orientation->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t" << "]" << std::endl;
	}

	if (0 < getNScales()) {
		const MFVec2f *scale = getScaleField();
		printStream << indentString << "\t" << "scale [" << std::endl;
		scale->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t" << "]" << std::endl;
	}

	if (0 < getNSpines()) {
		const MFVec3f *spine = getSpineField();
		printStream << indentString << "\t" << "spine [" << std::endl;
		spine->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t" << "]" << std::endl;
	}
}

////////////////////////////////////////////////
//	GroupingNode::recomputeBoundingBox
////////////////////////////////////////////////

void AddDefaultParameters(ExtrusionNode *ex)
{
	if (ex->getNCrossSections() == 0) {
		ex->addCrossSection(1.0f, 1.0);
		ex->addCrossSection(1.0f, -1.0);
		ex->addCrossSection(-1.0f, -1.0);
		ex->addCrossSection(-1.0f, 1.0);
		ex->addCrossSection(1.0f, 1.0);
	}
	if (ex->getNSpines() == 0) {
		ex->addSpine(0.0f, 0.0f, 0.0f);
		ex->addSpine(0.0f, 1.0f, 0.0f);
	}
}

////////////////////////////////////////////////
//	GroupingNode::recomputeBoundingBox
////////////////////////////////////////////////

void ExtrusionNode::recomputeBoundingBox()
{
}

////////////////////////////////////////////////
//	Polygons
////////////////////////////////////////////////

int ExtrusionNode::getNPolygons() const
{
	int nCrossSections = getNCrossSections();
	int nSpines = getNSpines();

	return nCrossSections + (nCrossSections * (nSpines-1)) + nCrossSections;
}

////////////////////////////////////////////////
//	GroupingNode::recomputeBoundingBox
////////////////////////////////////////////////

#ifdef CX3D_SUPPORT_OPENGL

static void initializePoint(ExtrusionNode *ex, SFVec3f *point)
{
	int nCrossSections = ex->getNCrossSections();
	for (int n=0; n<nCrossSections; n++) {
		float cs[2];
		ex->getCrossSection(n, cs);
		point[n].setValue(cs[0], 0.0f, cs[1]);
	}
}

static void transformPoint(SFVec3f *point, float scale[2], float scp[3][3], float orientation[4], float spine[3])
{
	point->scale(scale[0], 1.0f, scale[1]);
	
	float value[3];
	point->getValue(value);

	if (0.0f < VectorGetLength(scp[0]) && 0.0f < VectorGetLength(scp[1]) && 0.0f < VectorGetLength(scp[2])) {
		float x = value[0]*scp[0][0]+value[1]*scp[1][0]+value[2]*scp[2][0];
		float y = value[0]*scp[0][1]+value[1]*scp[1][1]+value[2]*scp[2][1];
		float z = value[0]*scp[0][2]+value[1]*scp[1][2]+value[2]*scp[2][2];
		value[0] = x;
		value[1] = y;
		value[2] = z;
	}

	point->setValue(value);

	point->translate(spine);
	point->rotate(orientation);
}

static void DrawExtrusionVercities(SFVec3f	*vertex[3])
{
	int n;
	float point[3][3];
	float normal[3];
	for (n=0; n<3; n++)
		vertex[n]->getValue(point[n]);
	GetNormalFromVertices(point, normal);
	glNormal3fv(normal);
	glBegin(GL_POLYGON);
	for (n=0; n<3; n++)
		glVertex3f(point[n][0], point[n][1], point[n][2]);
	glEnd();
}

static void DrawExtrusion(ExtrusionNode *ex)
{
	bool ccw = ex->getCCW();
	if (ccw == true)
		glFrontFace(GL_CCW);
	else
		glFrontFace(GL_CW);

	bool solid = ex->getSolid();
//	if (solid == false)
		glDisable(GL_CULL_FACE);
//	else
//		glEnable(GL_CULL_FACE);

	int nCrossSections = ex->getNCrossSections();

	SFVec3f *point[2];
	point[0] = new SFVec3f[nCrossSections];
	point[1] = new SFVec3f[nCrossSections];

	int		nOrientations	= ex->getNOrientations();
	int		nScales			= ex->getNScales();
	int		nSpines			= ex->getNSpines();

	float	spineStart[3];
	float	spineEnd[3];
	bool	bClosed;

	ex->getSpine(0,			spineStart);
	ex->getSpine(nSpines-1, spineEnd);
	bClosed = VectorEquals(spineStart, spineEnd);
	
	float	scale[2];
	float	orientation[4];
	float	spine[3];
	float	scp[3][3];

	for (int n=0; n<(nSpines-1); n++) {
		initializePoint(ex, point[0]);
		initializePoint(ex, point[1]);

		for (int i=0; i<2; i++) {
			
			if (nScales == 1)
				ex->getScale(0, scale);
			else  if ((n+i) < nScales) 
				ex->getScale(n+i, scale);
			else {
				scale[0] = 1.0f; 
				scale[1] = 1.0f;
			}

			if (nOrientations == 1)
				ex->getOrientation(0, orientation);
			else if ((n+i) < nOrientations)
				ex->getOrientation(n+i, orientation);
			else {
				orientation[0] = 0.0f; 
				orientation[1] = 0.0f; 
				orientation[2] = 1.0f; 
				orientation[3] = 0.0f;
			}

			ex->getSpine(n+i, spine);

			// SCP Y
			float spine0[3], spine1[3], spine2[3];
			if (nSpines <= 2) {
				ex->getSpine(1, spine1);
				ex->getSpine(0, spine2);
			}
			else if (bClosed && (n+i == 0 || n+i == (nSpines-1))) {
				ex->getSpine(1,			spine1);
				ex->getSpine(nSpines-2,	spine2);
			}
			else if (n+i == 0) {
				ex->getSpine(1, spine1);
				ex->getSpine(0, spine2);
			}
			else if (n+i == (nSpines-1)) {
				ex->getSpine(nSpines-1, spine1);
				ex->getSpine(nSpines-2, spine2);
			}
			else {
				ex->getSpine(n+i+1, spine1);
				ex->getSpine(n+i-1, spine2);
			}
			VectorGetDirection(spine1, spine2, scp[1]);
			VectorNormalize(scp[1]);

			// SCP Z
			float v1[3], v2[3];
			if (nSpines <= 2) {
				ex->getSpine(0, spine0);
				ex->getSpine(1, spine1);
				ex->getSpine(1, spine2);
			}
			else if (bClosed && (n+i == 0 || n+i == (nSpines-1))) {
				ex->getSpine(0,			spine0);
				ex->getSpine(1,			spine1);
				ex->getSpine(nSpines-2,	spine2);
			}
			else if (n+i == 0) {
				ex->getSpine(1,	spine0);
				ex->getSpine(2,	spine1);
				ex->getSpine(0,	spine2);
			}
			else if (n+i == (nSpines-1)) {
				ex->getSpine(nSpines-2, spine1);
				ex->getSpine(nSpines-1, spine1);
				ex->getSpine(nSpines-3, spine2);
			}
			else {
				ex->getSpine(n+i,	spine0);
				ex->getSpine(n+i+1,	spine1);
				ex->getSpine(n+i-1,	spine2);
			}
			VectorGetDirection(spine1, spine0, v1);
			VectorGetDirection(spine2, spine0, v2);
			VectorGetCross(v1, v2, scp[2]);

			// SCP X
			VectorGetCross(scp[1], scp[2], scp[0]);

			for (int j=0; j<nCrossSections; j++)  
				transformPoint(&point[i][j], scale, scp, orientation, spine);
		}

		for (int k=0; k<nCrossSections-1; k++) {

			float	vpoint[3][3];
			float	normal[3];
			
			point[1][k].getValue(vpoint[0]); 
			point[0][k].getValue(vpoint[1]);
			point[1][(k+1)%nCrossSections].getValue(vpoint[2]);
			GetNormalFromVertices(vpoint, normal);
			glNormal3fv(normal);
			
			SFVec3f	*vertex[3];

			vertex[2] = &point[1][(k+1)%nCrossSections];
			vertex[1] = &point[1][k];
			vertex[0] = &point[0][k];
			DrawExtrusionVercities(vertex);

			vertex[2] = &point[0][(k+1)%nCrossSections];
			vertex[1] = &point[1][(k+1)%nCrossSections];
			vertex[0] = &point[0][k];
			DrawExtrusionVercities(vertex);
		}

		if (n==0 && ex->getBeginCap() == true) {
			glBegin(GL_POLYGON);
			for (int k=0; k<nCrossSections; k++)
				glVertex3f(point[0][k].getX(), point[0][k].getY(), point[0][k].getZ());
			glEnd();
		}

		if (n==(nSpines-1)-1 && ex->getEndCap() == true) {
			glBegin(GL_POLYGON);
			for (int k=0; k<nCrossSections; k++)
				glVertex3f(point[1][k].getX(), point[1][k].getY(), point[1][k].getZ());
			glEnd();
		}
	}

	if (ccw == false)
		glFrontFace(GL_CCW);

//	if (solid == false)
		glEnable(GL_CULL_FACE);

	delete []point[0];
	delete []point[1];
}		

void ExtrusionNode::recomputeDisplayList()
{
	unsigned int nCurrentDisplayList = getDisplayList();
	if (0 < nCurrentDisplayList)
		glDeleteLists(nCurrentDisplayList, 1);

	unsigned int nNewDisplayList = glGenLists(1);
	glNewList(nNewDisplayList, GL_COMPILE);
		DrawExtrusion(this);
	glEndList();

	setDisplayList(nNewDisplayList);
}

#endif

