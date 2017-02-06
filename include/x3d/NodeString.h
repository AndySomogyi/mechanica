/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	NodeString.h
*
******************************************************************/

#ifndef _CX3D_NODESTRING_H_
#define _CX3D_NODESTRING_H_

namespace CyberX3D {

/******************************************************************
* Node Constants
******************************************************************/

const char anchorNodeString[] = "Anchor";
const char appearanceNodeString[] = "Appearance";
const char audioClipNodeString[] = "AudioClip";
const char backgroundNodeString[] = "Background";
const char billboardNodeString[] = "Billboard";
const char boxNodeString[] = "Box";
const char collisionNodeString[] = "Collision";
const char colorInterpolatorNodeString[] = "ColorInterpolator";
const char colorNodeString[] = "Color";
const char coneNodeString[] = "Cone";
const char coordinateInterpolatorNodeString[] = "CoordinateInterpolator";
const char coordinateNodeString[] = "Coordinate";
const char cylinderNodeString[] = "Cylinder";
const char cylinderSensorNodeString[] = "CylinderSensor";
const char defNodeString[] = "DEF";
const char directionalLightNodeString[] = "DirectionalLight";
const char elevationGridNodeString[] = "ElevationGrid";
const char extrusionNodeString[] = "Extrusion";
const char fogNodeString[] = "Fog";
const char fontStyleNodeString[] = "FontStyle";
const char groupNodeString[] = "Group";
const char imageTextureNodeString[] = "ImageTexture";
const char indexedFaceSetNodeString[] = "IndexedFaceSet";
const char indexedLineSetNodeString[] = "IndexedLineSet";
const char inlineNodeString[] = "Inline";
const char lodNodeString[] = "LOD";
const char materialNodeString[] = "Material";
const char movieTextureNodeString[] = "MovieTexture";
const char navigationInfoNodeString[] = "NavigationInfo";
const char normalInterpolatorNodeString[] = "NormalInterpolator";
const char normalNodeString[] = "Normal";
const char orientationInterpolatorNodeString[] = "OrientationInterpolator";
const char pixelTextureNodeString[] = "PixelTexture";
const char planeSensorNodeString[] = "PlaneSensor";
const char pointLightNodeString[] = "PointLight";
const char pointSetNodeString[] = "PointSet";
const char positionInterpolatorNodeString[] = "PositionInterpolator";
const char proximitySensorNodeString[] = "ProximitySensor";
const char rootNodeString[] = "Root";
const char scalarInterpolatorNodeString[] = "ScalarInterpolator";
const char scriptNodeString[] = "Script";
const char shapeNodeString[] = "Shape";
const char soundNodeString[] = "Sound";
const char sphereNodeString[] = "Sphere";
const char sphereSensorNodeString[] = "SphereSensor";
const char spotLightNodeString[] = "SpotLight";
const char switchNodeString[] = "Switch";
const char textNodeString[] = "Text";
const char textureCoordinateNodeString[] = "TextureCoordinate";
const char textureTransformNodeString[] = "TextureTransform";
const char timeSensorNodeString[] = "TimeSensor";
const char touchSensorNodeString[] = "TouchSensor";
const char transformNodeString[] = "Transform";
const char viewpointNodeString[] = "Viewpoint";
const char visibilitySensorNodeString[] = "VisibilitySensor";
const char worldInfoNodeString[] = "WorldInfo";

/******************************************************************
* Field Constants
******************************************************************/

const char ambientIntensityFieldString[] = "ambientIntensity";
const char attenuationFieldString[] = "attenuation";
const char autoOffsetFieldString[] = "autoOffset";
const char avatarSizeFieldString[] = "avatarSize";
const char axisOfRotationFieldString[] = "axisOfRotation";
const char backUrlFieldString[] = "backUrl";
const char beamWidthFieldString[] = "beamWidth";
const char beginCapFieldString[] = "beginCap";
const char bindTimeFieldString[] = "bindTime";
const char bottomFieldString[] = "bottom";
const char bottomRadiusFieldString[] = "bottomRadius";
const char bottomUrlFieldString[] = "bottomUrl";
const char ccwFieldString[] = "ccw";
const char centerFieldString[] = "center";
const char collideFieldString[] = "collide";
const char collideTimeFieldString[] = "collideTime";
const char colorFieldString[] = "color";
const char colorIndexFieldString[] = "colorIndex";
const char colorPerVertexFieldString[] = "colorPerVertex";
const char convexFieldString[] = "convex";
const char coordIndexFieldString[] = "coordIndex";
const char creaseAngleFieldString[] = "creaseAngle";
const char crossSectionFieldString[] = "crossSection";
const char cutOffAngleFieldString[] = "cutOffAngle";
const char cycleIntervalFieldString[] = "cycleInterval";
const char cycleTimeFieldString[] = "cycleTime";
const char descriptionFieldString[] = "description";
const char diffuseColorFieldString[] = "diffuseColor";
const char directOutputFieldString[] = "directOutput";
const char directionFieldString[] = "direction";
const char diskAngleFieldString[] = "diskAngle";
const char durationFieldString[] = "duration";
const char emissiveColorFieldString[] = "emissiveColor";
const char enabledFieldString[] = "enabled";
const char endCapFieldString[] = "endCap";
const char enterTimeFieldString[] = "enterTime";
const char exitTimeFieldString[] = "exitTime";
const char familyFieldString[] = "family";
const char fieldOfViewFieldString[] = "fieldOfView";
const char fogTypeFieldString[] = "fogType";
const char fractionFieldString[] = "fraction";
const char frontUrlFieldString[] = "frontUrl";
const char groundAngleFieldString[] = "groundAngle";
const char groundColorFieldString[] = "groundColor";
const char headlightFieldString[] = "headlight";
const char heightFieldString[] = "height";
const char hitNormalFieldString[] = "hitNormal";
const char hitPointFieldString[] = "hitPoint";
const char hitTexCoordFieldString[] = "hitTexCoord";
const char horizontalFieldString[] = "horizontal";
const char imageFieldString[] = "image";
const char inRegionPrivateFieldString[] = "inRegion";
const char infoFieldString[] = "info";
const char intensityFieldString[] = "intensity";
const char isActiveFieldString[] = "isActive";
const char isBoundFieldString[] = "isBound";
const char isOverFieldString[] = "isOver";
const char jumpFieldString[] = "jump";
const char justifyFieldString[] = "justify";
const char keyFieldString[] = "key";
const char keyValueFieldString[] = "keyValue";
const char languageFieldString[] = "language";
const char leftToRightFieldString[] = "leftToRight";
const char leftUrlFieldString[] = "leftUrl";
const char lengthFieldString[] = "length";
const char locationFieldString[] = "location";
const char loopFieldString[] = "loop";
const char maxAngleFieldString[] = "maxAngle";
const char maxBackFieldString[] = "maxBack";
const char maxExtentFieldString[] = "maxExtent";
const char maxFrontFieldString[] = "maxFront";
const char maxPositionFieldString[] = "maxPosition";
const char minAngleFieldString[] = "minAngle";
const char minBackFieldString[] = "minBack";
const char minFrontFieldString[] = "minFront";
const char minPositionFieldString[] = "minPosition";
const char mustEvaluateFieldString[] = "mustEvaluate";
const char normalIndexFieldString[] = "normalIndex";
const char normalPerVertexFieldString[] = "normalPerVertex";
const char offsetFieldString[] = "offset";
const char onFieldString[] = "on";
const char orientationFieldString[] = "orientation";
const char parameterFieldString[] = "parameter";
const char pitchFieldString[] = "pitch";
const char pointFieldString[] = "point";
const char positionFieldString[] = "position";
const char priorityFieldString[] = "priority";
const char radiusFieldString[] = "radius";
const char rangeFieldString[] = "range";
const char repeatSFieldString[] = "repeatS";
const char repeatTFieldString[] = "repeatT";
const char rightUrlFieldString[] = "rightUrl";
const char rotationFieldString[] = "rotation";
const char scaleFieldString[] = "scale";
const char scaleOrientationFieldString[] = "scaleOrientation";
const char setBindFieldString[] = "set_bind";
const char shininessFieldString[] = "shininess";
const char sideFieldString[] = "side";
const char sizeFieldString[] = "size";
const char skyAngleFieldString[] = "skyAngle";
const char skyColorFieldString[] = "skyColor";
const char solidFieldString[] = "solid";
const char spacingFieldString[] = "spacing";
const char spatializeFieldString[] = "spatialize";
const char specularColorFieldString[] = "specularColor";
const char speedFieldString[] = "speed";
const char speedTimeFieldString[] = "speedTime";
const char spineFieldString[] = "spine";
const char startTimeFieldString[] = "startTime";
const char stopTimeFieldString[] = "stopTime";
const char stringFieldString[] = "string";
const char styleFieldString[] = "style";
const char texCoordIndexFieldString[] = "texCoordIndex";
const char timeFieldString[] = "time";
const char titleFieldString[] = "title";
const char topFieldString[] = "top";
const char topToBottomFieldString[] = "topToBottom";
const char topUrlFieldString[] = "topUrl";
const char touchTimeFieldString[] = "touchTime";
const char trackPointFieldString[] = "trackPoint";
const char translationFieldString[] = "translation";
const char transparencyFieldString[] = "transparency";
const char typeFieldString[] = "type";
const char urlFieldString[] = "url";
const char valueFieldString[] = "value";
const char vectorFieldString[] = "vector";
const char visibilityLimitFieldString[] = "visibilityLimit";
const char visibilityRangeFieldString[] = "visibilityRange";
const char whichChoiceFieldString[] = "whichChoice";
const char xDimensionFieldString[] = "xDimension";
const char xSpacingFieldString[] = "xSpacing";
const char zDimensionFieldString[] = "zDimension";
const char zSpacingFieldString[] = "zSpacing";

}

#endif
