/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	X3DNodes.h
*
******************************************************************/

#ifndef _CX3D_X3DNODES_H_
#define _CX3D_X3DNODES_H_

#include <x3d/VRML97Nodes.h>
#include <x3d/XMLNode.h>

#include <x3d/SceneNode.h>

// 9. Networking component (X3D)
#include "LoadSensorNode.h"

// 10. Grouping component (X3D)
#include "StaticGroupNode.h"

// 11. Rendering component (X3D)
#include "ColorRGBANode.h"
#include "TriangleSetNode.h"
#include "TriangleFanSetNode.h"
#include "TriangleStripSetNode.h"
	
// 12. Shape component (X3D)
#include "FillPropertiesNode.h"
#include "LinePropertiesNode.h"

// 14. Geometry2D component (X3D)
#include "Arc2DNode.h"
#include "ArcClose2DNode.h"
#include "Circle2DNode.h"
#include "Disk2DNode.h"
#include "Polyline2DNode.h"
#include "Polypoint2DNode.h"
#include "Rectangle2DNode.h"
#include "TriangleSet2DNode.h"

// 18. Texturing component (x3D)
#include "MultiTextureNode.h"
#include "MultiTextureCoordinateNode.h"
#include "MultiTextureTransformNode.h"
#include "TextureCoordinateGeneratorNode.h"
	
// 19. Interpolation component (X3D)
#include "CoordinateInterpolator2DNode.h"
#include "PositionInterpolator2DNode.h"

// 21. Key device sensor component (X3D)
#include "KeySensorNode.h"
#include "StringSensorNode.h"

// 30. Event Utilities component (X3D)
#include "BooleanFilterNode.h"
#include "BooleanToggleNode.h"
#include "BooleanTriggerNode.h"
#include "BooleanSequencerNode.h"
#include "IntegerTriggerNode.h"
#include "IntegerSequencerNode.h"
#include "TimeTriggerNode.h"

// Deprecated components (X3D)
#include "NodeSequencerNode.h"
#include "Shape2DNode.h"
#include "BooleanTimeTriggerNode.h"
#include "Transform2DNode.h"

// RouteNode (X3D)
#include "RouteNode.h"

#endif
