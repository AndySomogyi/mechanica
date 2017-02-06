/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	NodeType.cpp
*
******************************************************************/

#include <string>
#include <x3d/NodeType.h>
#include <x3d/NodeString.h>
#include <x3d/StringUtil.h>

using namespace CyberX3D;

static const char *nodeTypeString[] = {
anchorNodeString, //"Anchor";
appearanceNodeString, //"Appearance";
audioClipNodeString, //"AudioClip";
backgroundNodeString, //"Background";
billboardNodeString, //"Billboard";
boxNodeString, //"Box";
collisionNodeString, //"Collision";
colorInterpolatorNodeString, //"ColorInterpolator";
colorNodeString, //"Color";
coneNodeString, //"Cone";
coordinateInterpolatorNodeString, //"CoordinateInterpolator";
coordinateNodeString, //"Coordinate";
cylinderNodeString, //"Cylinder";
cylinderSensorNodeString, //"CylinderSensor";
defNodeString, //"DEF";
directionalLightNodeString, //"DirectionalLight";
elevationGridNodeString, //"ElevationGrid";
extrusionNodeString, //"Extrusion";
fogNodeString, //"Fog";
fontStyleNodeString, //"FontStyle";
groupNodeString, //"Group";
imageTextureNodeString, //"ImageTexture";
indexedFaceSetNodeString, //"IndexedFaceSet";
indexedLineSetNodeString, //"IndexedLineSet";
inlineNodeString, //"Inline";
lodNodeString, //"LOD";
materialNodeString, //"Material";
movieTextureNodeString, //"MovieTexture";
navigationInfoNodeString, //"NavigationInfo";
normalInterpolatorNodeString, //"NormalInterpolator";
normalNodeString, //"Normal";
orientationInterpolatorNodeString, //"OrientationInterpolator";
pixelTextureNodeString, //"PixelTexture";
planeSensorNodeString, //"PlaneSensor";
pointLightNodeString, //"PointLight";
pointSetNodeString, //"PointSet";
positionInterpolatorNodeString, //"PositionInterpolator";
proximitySensorNodeString, //"ProximitySensor";
rootNodeString, //"Root";
scalarInterpolatorNodeString, //"ScalarInterpolator";
scriptNodeString, //"Script";
shapeNodeString, //"Shape";
soundNodeString, //"Sound";
sphereNodeString, //"Sphere";
sphereSensorNodeString, //"SphereSensor";
spotLightNodeString, //"SpotLight";
switchNodeString, //"Switch";
textNodeString, //"Text";
textureCoordinateNodeString, //"TextureCoordinate";
textureTransformNodeString, //"TextureTransform";
timeSensorNodeString, //"TimeSensor";
touchSensorNodeString, //"TouchSensor";
transformNodeString, //"Transform";
viewpointNodeString, //"Viewpoint";
visibilitySensorNodeString, //"VisibilitySensor";
worldInfoNodeString, //"WorldInfo";
"XML",
// 9. Networking component (X3D)
"LoadSensor",
// 10. Grouping component (X3D)
"StaticGroup",
// 11. Rendering component (X3D)
"ColorRGBA",
"TriangleSet",
"TriangleFanSet",
"TriangleStripSet",
// 12. Shape component (X3D)
"FillProperties",
"LineProperties",
// 14. Geometry2D component (X3D)
"Arc2D",
"ArcClose2D",
"Circle2D",
"Disk2D",
"Polyline2D",
"Polypoint2D",
"Rectangle2D",
"TriangleSet2D",
// 18. Texturing component (x3D)
"MultiTexture",
"MultiTextureCoordinate",
"MultiTextureTransformNode",
"TextureCoordinateGenerator",
// 19. Interpolation component (X3D)
"CoordinateInterpolator2D",
"PositionInterpolator2D", 
// 21. Key device sensor component (X3D)
"KeySensor",
"StringSensor",
// 30. Event Utilities component (X3D)
"BooleanFilter",
"BooleanToggle",
"BooleanTrigger",
"BooleanSequencer",
"IntegerTrigger",
"IntegerSequencer",
"TimeTrigger",
// Deprecated components (X3D)
"BooleanTimeTrigger",
"Shape2D",
"Transform2D",
"NodeSequencer",
// Scene (X3D)
"Scene",
// ROUTE (X3D)
"ROUTE",
};

static const char *fieldTypeString[] = {
ambientIntensityFieldString, //"ambientIntensity";
attenuationFieldString, //"attenuation";
autoOffsetFieldString, //"autoOffset";
avatarSizeFieldString, //"avatarSize";
axisOfRotationFieldString, //"axisOfRotation";
backUrlFieldString, //"backUrl";
beamWidthFieldString, //"beamWidth";
beginCapFieldString, //"beginCap";
bindTimeFieldString, //"bindTime";
bottomFieldString, //"bottom";
bottomRadiusFieldString, //"bottomRadius";
bottomUrlFieldString, //"bottomUrl";
ccwFieldString, //"ccw";
centerFieldString, //"center";
collideFieldString, //"collide";
collideTimeFieldString, //"collideTime";
colorFieldString, //"color";
colorIndexFieldString, //"colorIndex";
colorPerVertexFieldString, //"colorPerVertex";
convexFieldString, //"convex";
coordIndexFieldString, //"coordIndex";
creaseAngleFieldString, //"creaseAngle";
crossSectionFieldString, //"crossSection";
cutOffAngleFieldString, //"cutOffAngle";
cycleIntervalFieldString, //"cycleInterval";
cycleTimeFieldString, //"cycleTime";
descriptionFieldString, //"description";
diffuseColorFieldString, //"diffuseColor";
directOutputFieldString, //"directOutput";
directionFieldString, //"direction";
diskAngleFieldString, //"diskAngle";
durationFieldString, //"duration";
emissiveColorFieldString, //"emissiveColor";
enabledFieldString, //"enabled";
endCapFieldString, //"endCap";
enterTimeFieldString, //"enterTime";
exitTimeFieldString, //"exitTime";
familyFieldString, //"family";
fieldOfViewFieldString, //"fieldOfView";
fogTypeFieldString, //"fogType";
fractionFieldString, //"fraction";
frontUrlFieldString, //"frontUrl";
groundAngleFieldString, //"groundAngle";
groundColorFieldString, //"groundColor";
headlightFieldString, //"headlight";
heightFieldString, //"height";
hitNormalFieldString, //"hitNormal";
hitPointFieldString, //"hitPoint";
hitTexCoordFieldString, //"hitTexCoord";
horizontalFieldString, //"horizontal";
imageFieldString, //"image";
inRegionPrivateFieldString, //"inRegion";
infoFieldString, //"info";
intensityFieldString, //"intensity";
isActiveFieldString, //"isActive";
isBoundFieldString, //"isBound";
isOverFieldString, //"isOver";
jumpFieldString, //"jump";
justifyFieldString, //"justify";
keyFieldString, //"key";
keyValueFieldString, //"keyValue";
languageFieldString, //"language";
leftToRightFieldString, //"leftToRight";
leftUrlFieldString, //"leftUrl";
lengthFieldString, //"length";
locationFieldString, //"location";
loopFieldString, //"loop";
maxAngleFieldString, //"maxAngle";
maxBackFieldString, //"maxBack";
maxExtentFieldString, //"maxExtent";
maxFrontFieldString, //"maxFront";
maxPositionFieldString, //"maxPosition";
minAngleFieldString, //"minAngle";
minBackFieldString, //"minBack";
minFrontFieldString, //"minFront";
minPositionFieldString, //"minPosition";
mustEvaluateFieldString, //"mustEvaluate";
normalIndexFieldString, //"normalIndex";
normalPerVertexFieldString, //"normalPerVertex";
offsetFieldString, //"offset";
onFieldString, //"on";
orientationFieldString, //"orientation";
parameterFieldString, //"parameter";
pitchFieldString, //"pitch";
pointFieldString, //"point";
positionFieldString, //"position";
priorityFieldString, //"priority";
radiusFieldString, //"radius";
rangeFieldString, //"range";
repeatSFieldString, //"repeatS";
repeatTFieldString, //"repeatT";
rightUrlFieldString, //"rightUrl";
rotationFieldString, //"rotation";
scaleFieldString, //"scale";
scaleOrientationFieldString, //"scaleOrientation";
setBindFieldString, //"set_bind";
shininessFieldString, //"shininess";
sideFieldString, //"side";
sizeFieldString, //"size";
skyAngleFieldString, //"skyAngle";
skyColorFieldString, //"skyColor";
solidFieldString, //"solid";
spacingFieldString, //"spacing";
spatializeFieldString, //"spatialize";
specularColorFieldString, //"specularColor";
speedFieldString, //"speed";
speedTimeFieldString, //"speedTime";
spineFieldString, //"spine";
startTimeFieldString, //"startTime";
stopTimeFieldString, //"stopTime";
stringFieldString, //"string";
styleFieldString, //"style";
texCoordIndexFieldString, //"texCoordIndex";
timeFieldString, //"time";
titleFieldString, //"title";
topFieldString, //"top";
topToBottomFieldString, //"topToBottom";
topUrlFieldString, //"topUrl";
touchTimeFieldString, //"touchTime";
trackPointFieldString, //"trackPoint";
translationFieldString, //"translation";
transparencyFieldString, //"transparency";
typeFieldString, //"type";
urlFieldString, //"url";
valueFieldString, //"value";
vectorFieldString, //"vector";
visibilityLimitFieldString, //"visibilityLimit";
visibilityRangeFieldString, //"visibilityRange";
whichChoiceFieldString, //"whichChoice";
xDimensionFieldString, //"xDimension";
xSpacingFieldString, //"xSpacing";
zDimensionFieldString, //"zDimension";
zSpacingFieldString, //"zSpacing";
};

int CyberX3D::GetNodeType(const char *nodeString)
{
	std::string nodeStr = nodeString;
	int nNodes = sizeof(nodeTypeString) / sizeof(nodeTypeString[0]);
	for (int n=0; n<nNodes; n++) {
		if (nodeStr.compare(nodeTypeString[n]) == 0)
			return n;
	}
	return -1;
}

const char *CyberX3D::GetNodeTypeString(int nodeType)
{
	return nodeTypeString[nodeType];
}

int CyberX3D::GetFieldType(const char *fieldString)
{
	std::string fieldStr = fieldString;
	int nFields = sizeof(fieldTypeString) / sizeof(fieldTypeString[0]);
	for (int n=0; n<nFields; n++) {
		if (fieldStr.compare(fieldTypeString[n]) == 0)
			return n;
	}
	return -1;
}
