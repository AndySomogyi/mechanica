//http://www.youtube.com/user/thecplusplusguy
#ifndef CAMERA_H
#define CAMERA_H
#include <cmath>
#include <iostream>
#include <GLFW/glfw3.h>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include "vector3d.h"

class camera{
	vector3d loc;
	float camPitch,camYaw;
	float movevel;
	float mousevel;
	bool mi,ismoved;
	void lockCamera();
	void moveCamera(float dir);
	void moveCameraUp(float dir);
    GLFWwindow *window;
	public:
    camera(GLFWwindow *w);
		camera(vector3d loc);
		camera(vector3d loc,float yaw,float pitch);
		camera(vector3d loc,float yaw,float pitch,float mv,float mov);
		void Control();
		void UpdateCamera();
		vector3d getVector();
		vector3d getLocation();
		float getPitch();
		float getYaw();
		float getMovevel();
		float getMousevel();
		bool isMouseIn();
		
		void setLocation(vector3d vec);
		void lookAt(float pitch,float yaw);
		void mouseIn(bool b);
		void setSpeed(float mv,float mov);
		
		bool isMoved();
};

#endif
