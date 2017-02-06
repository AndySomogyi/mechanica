/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2003
*
*	File:	X3DBrowserGLUT.cpp
*
******************************************************************/

#include <iostream>
#include <cybergarage/x3d/CyberX3D.h>
#include "X3DBrowserFunc.h"

using namespace std;
using namespace CyberX3D;

static SceneGraph sceneGraph;

static int mouseButton = MOUSE_BUTTON_NONE;
static int mousePos[2];

////////////////////////////////////////////////////////// 
//  GLUT Callback Functions
////////////////////////////////////////////////////////// 

#ifdef CX3D_SUPPORT_GLUT

void SetSize(int width, int height) 
{
	UpdateViewport(&sceneGraph, width, height);
}

void MouseMotion(int x, int y)
{
	mousePos[0] = x;
	mousePos[1] = y;
}

void MouseButton(int button, int state, int x, int y)
{
	mouseButton = MOUSE_BUTTON_NONE;
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
		mouseButton = MOUSE_BUTTON_LEFT;
	mousePos[0] = x;
	mousePos[1] = y;
}

void MoveViewpoint(SceneGraph *sg, int mosx, int mosy)
{
	int width = glutGet(GLUT_WINDOW_WIDTH);
	int height = glutGet(GLUT_WINDOW_HEIGHT);
	MoveViewpoint(sg, width, height, mosx, mosy);
}

void UpdateSceneGraph()
{
	if (mouseButton == MOUSE_BUTTON_LEFT)
		MoveViewpoint(&sceneGraph, mousePos[0], mousePos[1]);
	sceneGraph.update();
	glutPostRedisplay();
}

void DrawSceneGraph(void) 
{
	DrawSceneGraph(&sceneGraph, OGL_RENDERING_TEXTURE);
	glutSwapBuffers();
}

#endif
	
////////////////////////////////////////////////////////// 
//  main
////////////////////////////////////////////////////////// 

int main(int argc, char **argv)
{
#ifdef CX3D_SUPPORT_GLUT
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutCreateWindow("X3D Browser for GLUT");

	glutReshapeFunc(SetSize);
	glutMotionFunc(MouseMotion);
	glutPassiveMotionFunc(MouseMotion);
	glutMouseFunc(MouseButton);
	glutDisplayFunc(DrawSceneGraph);
	glutIdleFunc(UpdateSceneGraph);

	if (2 <= argc) {
		char *filename = argv[1];
		if (sceneGraph.load(filename) == false) {
			std::cout << "Line Number: " <<  sceneGraph.getParserErrorLineNumber() << std::endl;
			std::cout << "Error Message: " <<  sceneGraph.getParserErrorMessage() << std::endl;
			std::cout << "Error Token: " <<   sceneGraph.getParserErrorToken() << std::endl;
			std::cout << "Error Line: " <<   sceneGraph.getParserErrorLineString() << std::endl;
			std::cout << std::flush;
		}
		sceneGraph.initialize();
		if (sceneGraph.getViewpointNode() == NULL)
			sceneGraph.zoomAllViewpoint();
	}

	glutMainLoop();
#else
	cout << "GLUT not found - please install GLUT" << endl;	
#endif	
	return 0;
}
