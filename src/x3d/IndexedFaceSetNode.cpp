/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	IndexedFaceSetNode.cpp
*
*	Revisions:
*
*	12/03/02
*		- Changed the super class from Geometry3DNode to ComposedGeometryNode.
*
******************************************************************/

#include <x3d/SceneGraph.h>
#include <x3d/BoundingBox.h>
#include <x3d/Graphic3D.h>
#include <x3d/MathUtil.h>

using namespace CyberX3D;

IndexedFaceSetNode::IndexedFaceSetNode() 
{
	setHeaderFlag(false);
	setType(INDEXEDFACESET_NODE);

	///////////////////////////
	// Field 
	///////////////////////////

	// convex  field
	convexField = new SFBool(true);
	convexField->setName(convexFieldString);
	addField(convexField);

	// creaseAngle  field
	creaseAngleField = new SFFloat(0.0f);
	creaseAngleField->setName(creaseAngleFieldString);
	addField(creaseAngleField);

	// coordIndex  field
	coordIdxField = new MFInt32();
	coordIdxField->setName(coordIndexFieldString);
	addField(coordIdxField);

	// texCoordIndex  field
	texCoordIndexField = new MFInt32();
	texCoordIndexField->setName(texCoordIndexFieldString);
	addField(texCoordIndexField);

	// colorIndex  field
	colorIndexField = new MFInt32();
	colorIndexField->setName(colorIndexFieldString);
	addField(colorIndexField);

	// normalIndex  field
	normalIndexField = new MFInt32();
	normalIndexField->setName(normalIndexFieldString);
	addField(normalIndexField);

	///////////////////////////
	// EventIn
	///////////////////////////

	// coordIndex  EventIn
	MFInt32 *setCoordIdxField = new MFInt32();
	setCoordIdxField->setName(coordIndexFieldString);
	addEventIn(setCoordIdxField);

	// texCoordIndex  EventIn
	MFInt32 *setTexCoordIndex = new MFInt32();
	setTexCoordIndex->setName(texCoordIndexFieldString);
	addEventIn(setTexCoordIndex);

	// colorIndex  EventIn
	MFInt32 *setColorIndex = new MFInt32();
	setColorIndex->setName(colorIndexFieldString);
	addEventIn(setColorIndex);

	// normalIndex  EventIn
	MFInt32 *setNormalIndex = new MFInt32();
	setNormalIndex->setName(normalIndexFieldString);
	addEventIn(setNormalIndex);
}

IndexedFaceSetNode::~IndexedFaceSetNode() 
{
}
	
////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

IndexedFaceSetNode *IndexedFaceSetNode::next() const
{
	return (IndexedFaceSetNode *)Node::next(getType());
}

IndexedFaceSetNode *IndexedFaceSetNode::nextTraversal() const
{
	return (IndexedFaceSetNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	Convex
////////////////////////////////////////////////

SFBool *IndexedFaceSetNode::getConvexField() const
{
	if (isInstanceNode() == false)
		return convexField;
	return (SFBool *)getField(convexFieldString);
}
	
void IndexedFaceSetNode::setConvex(bool value) 
{
	getConvexField()->setValue(value);
}

void IndexedFaceSetNode::setConvex(int value) 
{
	setConvex(value ? true : false);
}

bool IndexedFaceSetNode::getConvex() const
{
	return getConvexField()->getValue();
}

////////////////////////////////////////////////
//	CreaseAngle
////////////////////////////////////////////////

SFFloat *IndexedFaceSetNode::getCreaseAngleField() const
{
	if (isInstanceNode() == false)
		return creaseAngleField;
	return (SFFloat *)getField(creaseAngleFieldString);
}

void IndexedFaceSetNode::setCreaseAngle(float value) 
{
	getCreaseAngleField()->setValue(value);
}

float IndexedFaceSetNode::getCreaseAngle() const
{
	return getCreaseAngleField()->getValue();
}

////////////////////////////////////////////////
// CoordIndex
////////////////////////////////////////////////

MFInt32 *IndexedFaceSetNode::getCoordIndexField() const
{
	if (isInstanceNode() == false)
		return coordIdxField;
	return (MFInt32 *)getField(coordIndexFieldString);
}

void IndexedFaceSetNode::addCoordIndex(int value) 
{
	getCoordIndexField()->addValue(value);
}

int IndexedFaceSetNode::getNCoordIndexes() const
{
	return getCoordIndexField()->getSize();
}

int IndexedFaceSetNode::getCoordIndex(int index) const 
{
	return getCoordIndexField()->get1Value(index);
}
	
////////////////////////////////////////////////
// TexCoordIndex
////////////////////////////////////////////////

MFInt32 *IndexedFaceSetNode::getTexCoordIndexField() const
{
	if (isInstanceNode() == false)
		return texCoordIndexField;
	return (MFInt32 *)getField(texCoordIndexFieldString);
}

void IndexedFaceSetNode::addTexCoordIndex(int value) 
{
	getTexCoordIndexField()->addValue(value);
}

int IndexedFaceSetNode::getNTexCoordIndexes() const
{
	return getTexCoordIndexField()->getSize();
}

int IndexedFaceSetNode::getTexCoordIndex(int index) const
{
	return getTexCoordIndexField()->get1Value(index);
}
	
////////////////////////////////////////////////
// ColorIndex
////////////////////////////////////////////////

MFInt32 *IndexedFaceSetNode::getColorIndexField() const
{
	if (isInstanceNode() == false)
		return colorIndexField;
	return (MFInt32 *)getField(colorIndexFieldString);
}

void IndexedFaceSetNode::addColorIndex(int value) 
{
	getColorIndexField()->addValue(value);
}

int IndexedFaceSetNode::getNColorIndexes() const
{
	return getColorIndexField()->getSize();
}

int IndexedFaceSetNode::getColorIndex(int index) const
{
	return getColorIndexField()->get1Value(index);
}

////////////////////////////////////////////////
// NormalIndex
////////////////////////////////////////////////

MFInt32 *IndexedFaceSetNode::getNormalIndexField() const
{
	if (isInstanceNode() == false)
		return normalIndexField;
	return (MFInt32 *)getField(normalIndexFieldString);
}

void IndexedFaceSetNode::addNormalIndex(int value) 
{
	getNormalIndexField()->addValue(value);
}

int IndexedFaceSetNode::getNNormalIndexes() const
{
	return getNormalIndexField()->getSize();
}

int IndexedFaceSetNode::getNormalIndex(int index) const 
{
	return getNormalIndexField()->get1Value(index);
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool IndexedFaceSetNode::isChildNodeType(Node *node) const
{
	if (node->isColorNode() || node->isCoordinateNode() || node->isNormalNode() || node->isTextureCoordinateNode())
		return true;
	else
		return false;
}

void IndexedFaceSetNode::uninitialize() 
{
}

void IndexedFaceSetNode::update() 
{
}

////////////////////////////////////////////////
//	Infomation
////////////////////////////////////////////////

void IndexedFaceSetNode::outputContext(std::ostream &printStream, const char *indentString) const
{
	const SFBool *convex = getConvexField();
	const SFBool *solid = getSolidField();
	const SFBool *normalPerVertex = getNormalPerVertexField();
	const SFBool *colorPerVertex = getColorPerVertexField();
	const SFBool *ccw = getCCWField();

	printStream << indentString << "\t" << "ccw " << ccw << std::endl;
	printStream << indentString << "\t" << "colorPerVertex " << colorPerVertex << std::endl;
	printStream << indentString << "\t" << "normalPerVertex " << normalPerVertex << std::endl;
	printStream << indentString << "\t" << "convex " << convex << std::endl;
	printStream << indentString << "\t" << "creaseAngle " << getCreaseAngle() << std::endl;
	printStream << indentString << "\t" << "solid " << solid << std::endl;

	const NormalNode *normal = getNormalNodes();
	if (normal != NULL) {
		if (normal->isInstanceNode() == false) {
			if (normal->getName() != NULL && strlen(normal->getName()))
				printStream << indentString << "\t" << "normal " << "DEF " << normal->getName() << " Normal {" << std::endl;
			else
				printStream << indentString << "\t" << "normal Normal {" << std::endl;
			normal->Node::outputContext(printStream, indentString , "\t");
			printStream << indentString << "\t" << "}" << std::endl;
		}
		else 
			printStream << indentString << "\t" << "normal USE " << normal->getName() << std::endl;
	}

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

	const TextureCoordinateNode *texCoord = getTextureCoordinateNodes();
	if (texCoord != NULL) {
		if (texCoord->isInstanceNode() == false) {
			if (texCoord->getName() != NULL && strlen(texCoord->getName()))
				printStream << indentString << "\t" << "texCoord " << "DEF " << texCoord->getName() << " TextureCoordinate {" << std::endl;
			else
				printStream << indentString << "\t" << "texCoord TextureCoordinate {" << std::endl;
			texCoord->Node::outputContext(printStream, indentString, "\t");
			printStream << indentString << "\t" << "}" << std::endl;
		}
		else 
			printStream << indentString << "\t" << "texCoord USE " << texCoord->getName() << std::endl;
	}

	if (0 < getNCoordIndexes()) {
		const MFInt32 *coordIndex = getCoordIndexField();
		printStream << indentString << "\t" << "coordIndex [" << std::endl;
		coordIndex->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t" << "]" << std::endl;
	}
		
	if (0 < getNTexCoordIndexes()) {
		const MFInt32 *texCoordIndex = getTexCoordIndexField();
		printStream << indentString << "\t" << "texCoordIndex [" << std::endl;
		texCoordIndex->MField::outputContext(printStream, indentString,"\t\t");
		printStream << indentString << "\t" << "]" << std::endl;
	}
		
	if (0 < getNColorIndexes()) {
		const MFInt32 *colorIndex = getColorIndexField();
		printStream << indentString << "\t" << "colorIndex [" << std::endl;
		colorIndex->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t" << "]" << std::endl;
	}
		
	if (0 < getNNormalIndexes()) {
		const MFInt32 *normalIndex = getNormalIndexField();
		printStream << indentString << "\t" << "normalIndex [" << std::endl;
		normalIndex->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t" << "]" << std::endl;
	}
}

////////////////////////////////////////////////////////////
//	IndexedFaceSetNode::generateNormals
////////////////////////////////////////////////////////////

bool IndexedFaceSetNode::generateNormals() 
{
	NormalNode *normal = getNormalNodes();
	if (normal)
		return false;

	CoordinateNode *coordinateNode = getCoordinateNodes();
	if (!coordinateNode)
		return false;

	normal = new NormalNode();

	int		nPolygon = 0;
	int		nVertex = 0;
	float	point[3][3];
	float	vector[3];

	int		nCoordIndexes = getNCoordIndexes();

	for (int nCoordIndex=0; nCoordIndex<nCoordIndexes; nCoordIndex++) {

		int coordIndex = getCoordIndex(nCoordIndex);

		if (coordIndex != -1) {
			if (nVertex < 3)
				coordinateNode->getPoint(coordIndex, point[nVertex]);
			nVertex++;
		}
		else {
			GetNormalFromVertices(point, vector);
			normal->addVector(vector);

			nVertex = 0;
			nPolygon++;
		}
	}

	addChildNode(normal);

	setNormalPerVertex(false);

	return true;
}

////////////////////////////////////////////////////////////
//	IndexedFaceSetNode::recomputeBoundingBox
////////////////////////////////////////////////////////////

void IndexedFaceSetNode::recomputeBoundingBox() 
{
	CoordinateNode *coordinateNode = getCoordinateNodes();
	if (!coordinateNode) {
		setBoundingBoxCenter(0.0f, 0.0f, 0.0f);
		setBoundingBoxSize(-1.0f, -1.0f, -1.0f);
		return;
	}

	BoundingBox bbox;
	float		point[3];

	int nPoints = coordinateNode->getNPoints();
	for (int n=0; n<nPoints; n++) {
		coordinateNode->getPoint(n, point);
		bbox.addPoint(point);
	}

	setBoundingBox(&bbox);
}

////////////////////////////////////////////////
//	IndexedFaceSetNode::initialize
////////////////////////////////////////////////

void IndexedFaceSetNode::initialize() 
{
	if (!getSceneGraph())
		return;

	if (!isInitialized()) {
		if (getSceneGraph()->getOption() & SCENEGRAPH_NORMAL_GENERATION)
			generateNormals();

		if (getSceneGraph()->getOption() & SCENEGRAPH_TEXTURE_GENERATION) {
			Node *parentNode = getParentNode();
			if (parentNode) {
				AppearanceNode *appearance = parentNode->getAppearanceNodes();
				if (appearance) {
					if (appearance->getTextureNode())
						generateTextureCoordinate();
				}
			}
		}

#ifdef CX3D_SUPPORT_OPENGL
		recomputeDisplayList();
#endif
		recomputeBoundingBox();
		setInitialized(true);
	}
}

////////////////////////////////////////////////
//	IndexedFaceSetNode::recomputeDisplayList
////////////////////////////////////////////////

#ifdef CX3D_SUPPORT_OPENGL

static void DrawIdxFaceSet(IndexedFaceSetNode *idxFaceSet)
{
	CoordinateNode *coordinateNode = idxFaceSet->getCoordinateNodes();
	if (!coordinateNode)
		return;

	TextureCoordinateNode	*texCoordNode	= idxFaceSet->getTextureCoordinateNodes();
	NormalNode				*normalNode		= idxFaceSet->getNormalNodes();
	ColorNode				*colorNode		= idxFaceSet->getColorNodes();

	bool colorPerVertex =idxFaceSet->getColorPerVertex();
	bool normalPerVertex =idxFaceSet->getNormalPerVertex();

	bool ccw = idxFaceSet->getCCW();
	if (ccw == true)
		glFrontFace(GL_CCW);
	else
		glFrontFace(GL_CW);

	bool solid = idxFaceSet->getSolid();
	if (solid == false)
		glDisable(GL_CULL_FACE);
	else
		glEnable(GL_CULL_FACE);

	bool convex = idxFaceSet->getConvex();
	GLUtriangulatorObj *tessObj = NULL;

	if (convex == false) {
		tessObj = gluNewTess();
		gluTessCallback(tessObj, GLU_BEGIN,		(GLUtessCallBackFunc)glBegin);
		gluTessCallback(tessObj, GLU_VERTEX,	(GLUtessCallBackFunc)glVertex3dv);
		gluTessCallback(tessObj, GLU_END,		(GLUtessCallBackFunc)glEnd);
	}

	bool	bPolygonBegin = true;
	bool	bPolygonClose = true;
	
	int		nPolygon	= 0;
	int		nVertex		= 0;

	float	point[3];
	float	vector[3];
	float	color[4]; color[3] = 1.0f;
	float	coord[2];
	double	(*tessPoint)[3];

	if ((idxFaceSet->getColorPerVertex() && idxFaceSet->getColorNodes()) || (idxFaceSet->getNormalPerVertex() && idxFaceSet->getNormalNodes()))
		glShadeModel (GL_SMOOTH);
	else
		glShadeModel (GL_FLAT);


	int nColorIndexes = idxFaceSet->getNColorIndexes();
	int nNormalIndexes = idxFaceSet->getNNormalIndexes();
	int nTexCoordIndexes = idxFaceSet->getNTexCoordIndexes();
	int nCoordIndexes = idxFaceSet->getNCoordIndexes();

	for (int nCoordIndex=0; nCoordIndex<nCoordIndexes; nCoordIndex++) {

		int coordIndex = idxFaceSet->getCoordIndex(nCoordIndex);

		if (bPolygonBegin) {

			if (convex == false) 
				gluBeginPolygon(tessObj);
			else
				glBegin(GL_POLYGON);

			bPolygonBegin = false;
			bPolygonClose = false;

			int nVertices = 0;
			int index = coordIndex;
			while (index != -1) {
				nVertices++;
				int nIndex = nCoordIndex + nVertices;
				if (nIndex < nCoordIndexes)
					index = idxFaceSet->getCoordIndex(nIndex);
				else
					break;
			}

			if (convex == false)
				tessPoint = new double[nVertices][3];

			// dafault color
			//glColor3f(1.0f, 1.0f, 1.0f);

			// default normal
			if ((nCoordIndex + 2) < nCoordIndexes) {
				float point[3][3];
				float normal[3];
				for (int n=0; n<3; n++) {
					int index = idxFaceSet->getCoordIndex(nCoordIndex+n);
					coordinateNode->getPoint(index, point[n]);
				}
				GetNormalFromVertices(point, normal);
				glNormal3fv(normal);
			}
			else
				glNormal3f(0.0f, 0.0f, 1.0f);

			if (colorNode && !colorPerVertex) {
				if (0 < nColorIndexes)
					colorNode->getColor(idxFaceSet->getColorIndex(nPolygon), color);
				else
					colorNode->getColor(nPolygon, color);
				glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
//				glColor3fv(color);
			}

			if (normalNode && !normalPerVertex) {
				if (0 < nNormalIndexes)
					normalNode->getVector(idxFaceSet->getNormalIndex(nPolygon), vector);
				else
					normalNode->getVector(nPolygon, vector);
				glNormal3fv(vector);
			}

			nPolygon++;
			nVertex = 0;
		}

		if (coordIndex != -1) {

			if (colorNode && colorPerVertex) {
				if (0 < nColorIndexes)
					colorNode->getColor(idxFaceSet->getColorIndex(nCoordIndex), color);
				else
					colorNode->getColor(coordIndex, color);
				glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
//				glColor3fv(color);
			}

			if (normalNode && normalPerVertex) {
				if (0 < nNormalIndexes)
					normalNode->getVector(idxFaceSet->getNormalIndex(nCoordIndex), vector);
				else
					normalNode->getVector(coordIndex, vector);
				glNormal3fv(vector);
			}

			if (texCoordNode) {
				if (0 < nTexCoordIndexes)
					texCoordNode->getPoint(idxFaceSet->getTexCoordIndex(nCoordIndex), coord);
				else
					texCoordNode->getPoint(coordIndex, coord);
				coord[1] = 1.0f - coord[1];
				glTexCoord2fv(coord);
			}

			coordinateNode->getPoint(coordIndex, point);
			if (convex == false) { 
				tessPoint[nVertex][0] = point[0];
				tessPoint[nVertex][1] = point[1];
				tessPoint[nVertex][2] = point[2];
				gluTessVertex(tessObj, tessPoint[nVertex], tessPoint[nVertex]);
			}
			else
				glVertex3fv(point);

			nVertex++;
		}
		else {
			if (convex == false)  {
				gluEndPolygon(tessObj);
				delete[] tessPoint;
			}
			else
				glEnd();
			bPolygonBegin = true;
			bPolygonClose = true;
		}
	}

	if (bPolygonClose == false) {
		if (convex == false) { 
			gluEndPolygon(tessObj);
			delete[] tessPoint;
		}	
		else
			glEnd();
	}

	if (ccw == false)
		glFrontFace(GL_CCW);

	if (solid == false)
		glEnable(GL_CULL_FACE);

	if (convex == false)
		gluDeleteTess(tessObj);

	glShadeModel(GL_SMOOTH);
}

void IndexedFaceSetNode::recomputeDisplayList() 
{
	CoordinateNode *coordinateNode = getCoordinateNodes();
	if (!coordinateNode)
		return;

	unsigned int nCurrentDisplayList = getDisplayList();
	if (0 < nCurrentDisplayList)
		glDeleteLists(nCurrentDisplayList, 1);

	unsigned int nNewDisplayList = glGenLists(1);
	glNewList(nNewDisplayList, GL_COMPILE);
		DrawIdxFaceSet(this);
	glEndList();

	setDisplayList(nNewDisplayList);
}

#endif

////////////////////////////////////////////////////////////
//	IndexedFaceSetNode::getNPolygons
////////////////////////////////////////////////////////////

int IndexedFaceSetNode::getNPolygons() const
{
	CoordinateNode *coordinateNode = getCoordinateNodes();
	if (!coordinateNode)
		return 0;

	int nCoordIndexes = getNCoordIndexes();
	int nCoordIndex = 0;
	for (int n=0; n<nCoordIndexes; n++) {
		if (getCoordIndex(n) == -1)
			nCoordIndex++;
		else if (n == (nCoordIndexes-1))
			nCoordIndex++;
	}
	return nCoordIndex;
}

////////////////////////////////////////////////////////////
//	IndexedFaceSetNode::generateTextureCoordinate
////////////////////////////////////////////////////////////

static void GetRotateMatrixFromNormal(
float		normal[3],
SFMatrix	&matrix)
{
	SFMatrix	mx;
	SFMatrix	my;
	float		mxValue[4][4];
	float		myValue[4][4];

	mx.getValue(mxValue);
	my.getValue(myValue);

	float d = (float)sqrt(normal[1]*normal[1] + normal[2]*normal[2]);

	if (d) {
		float cosa = normal[2] / d;
		float sina = normal[1] / d;
		mxValue[0][0] = 1.0;
		mxValue[0][1] = 0.0;
		mxValue[0][2] = 0.0;
		mxValue[1][0] = 0.0;
		mxValue[1][1] = cosa;
		mxValue[1][2] = sina;
		mxValue[2][0] = 0.0;
		mxValue[2][1] = -sina;
		mxValue[2][2] = cosa;
	}
	
	float cosb = d;
	float sinb = normal[0];
	
	myValue[0][0] = cosb;
	myValue[0][1] = 0.0;
	myValue[0][2] = sinb;
	myValue[1][0] = 0.0;
	myValue[1][1] = 1.0;
	myValue[1][2] = 0.0;
	myValue[2][0] = -sinb;
	myValue[2][1] = 0.0;
	myValue[2][2] = cosb;

	mx.setValue(mxValue);
	my.setValue(myValue);

	matrix.init();
	matrix.add(&my);
	matrix.add(&mx);
}

static void SetExtents(
SFVec3f	&maxExtents,
SFVec3f	&minExtents,
float	point[3])
{
	if (maxExtents.getX() < point[0])
		maxExtents.setX(point[0]);
	if (maxExtents.getY() < point[1])
		maxExtents.setY(point[1]);
	if (maxExtents.getZ() < point[2])
		maxExtents.setZ(point[2]);
	if (minExtents.getX() > point[0])
		minExtents.setX(point[0]);
	if (minExtents.getY() > point[1])
		minExtents.setY(point[1]);
	if (minExtents.getZ() > point[2])
		minExtents.setZ(point[2]);
}

bool IndexedFaceSetNode::generateTextureCoordinate() 
{
	TextureCoordinateNode *texCoord = getTextureCoordinateNodes();
	if (texCoord)
		return false;

	CoordinateNode *coordinateNode = getCoordinateNodes();
	if (!coordinateNode)
		return false;

	texCoord = new TextureCoordinateNode();

	int nPolygon = getNPolygons();

	if (nPolygon <= 0)
		return false;

	float	(*normal)[3] = new float[nPolygon][3];
	SFVec3f	*center = new SFVec3f[nPolygon];
	SFVec3f	*maxExtents = new SFVec3f[nPolygon];
	SFVec3f	*minExtents = new SFVec3f[nPolygon];

	bool	bPolygonBegin;
	int		polyn;

	float	point[3][3];
	float	coord[3];

	int		vertexn = 0;
	int		n;


	bPolygonBegin = true;
	polyn = 0;
/*
	int nColorIndexes = idxFaceSet->getNColorIndexes();
	int nNormalIndexes = idxFaceSet->getNNormalIndexes();
	int nTexCoordIndexes = idxFaceSet->getNTexCoordIndexes();
*/
	int nCoordIndexes = getNCoordIndexes();


	for (n=0; n<nCoordIndexes; n++) {
		int coordIndex = getCoordIndex(n);
		if (coordIndex != -1) {

			if (vertexn < 3)
				coordinateNode->getPoint(coordIndex, point[vertexn]);

			float point[3];
			coordinateNode->getPoint(coordIndex, point);
			if (bPolygonBegin) {
				maxExtents[polyn].setValue(point);
				minExtents[polyn].setValue(point);
				center[polyn].setValue(point);
				bPolygonBegin = false;
			}
			else {
				SetExtents(maxExtents[polyn], minExtents[polyn], point);
				center[polyn].add(point);
			}

			vertexn++;
		}
		else {
			GetNormalFromVertices(point, normal[polyn]);
			center[polyn].scale(1.0f / (float)vertexn);
			maxExtents[polyn].sub(center[polyn]);
			minExtents[polyn].sub(center[polyn]);
			vertexn = 0;
			bPolygonBegin = true;
			polyn++;
		}
	}

	float		minx, miny, maxx, maxy, xlength, ylength;
	SFMatrix	matrix;

	bPolygonBegin = true;
	polyn = 0;

	for (n=0; n<nCoordIndexes; n++) {
		int coordIndex = getCoordIndex(n);
		if (coordIndex != -1) {

			if (bPolygonBegin) {
				GetRotateMatrixFromNormal(normal[polyn], matrix);
				matrix.multi(&minExtents[polyn]);
				matrix.multi(&maxExtents[polyn]);
				minx = minExtents[polyn].getX();
				miny = minExtents[polyn].getY();
				maxx = maxExtents[polyn].getX();
				maxy = maxExtents[polyn].getY();
				xlength = (float)fabs(maxExtents[polyn].getX() - minExtents[polyn].getX());
				ylength = (float)fabs(maxExtents[polyn].getY() - minExtents[polyn].getY());

				if (xlength == 0.0f || ylength == 0.0f) {
					delete texCoord;
					delete []minExtents;
					delete []maxExtents;
					delete []center;
					delete []normal;
					return false;
				}

				bPolygonBegin = false;
			}

			coordinateNode->getPoint(coordIndex, coord);

			coord[0] -= center[polyn].getX();
			coord[1] -= center[polyn].getY();
			coord[2] -= center[polyn].getZ();

			matrix.multi(coord);

			coord[0] = (float)fabs(coord[0] - minx) / xlength;
			coord[1] = (float)fabs(coord[1] - miny) / ylength;

			texCoord->addPoint(coord);
		}
		else {
//			coord[0] = coord[1] = 0.0f;
//			texCoord->addPoint(coord);
			bPolygonBegin = true;
			polyn++;
		}
	}

	addChildNode(texCoord);

	delete []minExtents;
	delete []maxExtents;
	delete []center;
	delete []normal;

	return true;
}
