/**********************************************************************

  arcball.cpp


          --------------------------------------------------

  GLUI User Interface Toolkit
  Copyright (c) 1998 Paul Rademacher
     Feb 1998, Paul Rademacher (rademach@cs.unc.edu)
     Oct 2003, Nigel Stewart - GLUI Code Cleaning

  WWW:    https://github.com/libglui/glui
  Issues: https://github.com/libglui/glui/issues

  This software is provided 'as-is', without any express or implied
  warranty. In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
  claim that you wrote the original software. If you use this software
  in a product, an acknowledgment in the product documentation would be
  appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
  misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

**********************************************************************/

#include "ArcBallInteractor.h"

#include <cstdio>
#include <iostream>

enum {VX, VY, VZ, VW};           // axes

static inline Magnum::Matrix4 q2m(const Magnum::Quaternion q)
{
    /*
    float xs, ys, zs, wx, wy, wz, xx, xy, xz, yy, yz, zz;

    Magnum::Vector3 v = q.vector();
    float s = q.scalar();

    float t  = 2.0f / (Magnum::Math::dot(v, v) + s*s);

    xs = v[VX]*t;   ys = v[VY]*t;   zs = v[VZ]*t;
    wx = s*xs;      wy = s*ys;      wz = s*zs;
    xx = v[VX]*xs;  xy = v[VX]*ys;  xz = v[VX]*zs;
    yy = v[VY]*ys;  yz = v[VY]*zs;  zz = v[VZ]*zs;

    Magnum::Matrix4 matrix{
        {1.0f-(yy+zz), xy+wz,        xz-wy,        0.0f},
        {xy-wz,        1.0f-(xx+zz), yz+wx,        0.0f},
        {xz+wy,        yz-wx,        1.0f-(xx+yy), 0.0f},
        {0.0f,         0.0f,         0.0f,         1.0f }};

    return matrix;
     */
    return Magnum::Matrix4::from(q.toMatrix(), {});
}


/**************************************** ArcBall::ArcBall() ****/
/* Default () constructor for ArcBall                         */

ArcBallInteractor::ArcBallInteractor()
{
    reset();
}

/**************************************** ArcBall::ArcBall() ****/
/* Takes as argument a Magnum::Matrix4 to use instead of the internal rot  */

ArcBallInteractor::ArcBallInteractor(const Magnum::Matrix4 &mtx)
{
    rot = mtx;
}


/**************************************** ArcBall::ArcBall() ****/
/* A constructor that accepts the screen center and arcball radius*/

ArcBallInteractor::ArcBallInteractor(const Magnum::Vector2 &_center, float _radius)
{
    reset();
    setParams(_center, _radius);
}


/************************************** ArcBall::set_params() ****/

void ArcBallInteractor::setParams(const Magnum::Vector2 &_center, float _radius)
{
    center      = _center;
    radius      = _radius;
}

/*************************************** ArcBall::init() **********/

void ArcBallInteractor::reset()
{
    center = Magnum::Vector2{{ 0.0f, 0.0f }};
    radius         = 1.0;
    q_now          = Magnum::Quaternion(Magnum::Math::IdentityInit);
    rot            = Magnum::Matrix4{Magnum::Math::IdentityInit};
    q_increment    = Magnum::Quaternion(Magnum::Math::IdentityInit);
    rot_increment  = Magnum::Matrix4{Magnum::Math::IdentityInit};
    is_mouse_down  = false;
    is_spinning    = false;
    damp_factor    = 0.0;
    zero_increment = true;
}

Magnum::Matrix4 ArcBallInteractor::rotation() const
{
    return rot;
}

void ArcBallInteractor::setWindowSize(int width, int height)
{
    center = {{width / 2.f, height / 2.f}};
    radius = center.length() / 2.;
}

/*********************************** ArcBall::mouse_to_sphere() ****/

Magnum::Vector3 ArcBallInteractor::mouseToSphere(const Magnum::Vector2 &p)
{
    float mag;
    Magnum::Vector2  v2 = (p - center) / radius;
    Magnum::Vector3  v3( v2[0], v2[1], 0.0 );

    mag = Magnum::Math::dot(v2, v2);

    if ( mag > 1.0 )
        v3 = v3.normalized();
    else
        v3[VZ] = (float) sqrt( 1.0 - mag );

    /* Now we add constraints - X takes precedence over Y */
    if ( constraint_x )
    {
        v3 = constrainVector( v3, Magnum::Vector3( 1.0, 0.0, 0.0 ));
    }
    else if ( constraint_y )
        {
            v3 = constrainVector( v3, Magnum::Vector3( 0.0, 1.0, 0.0 ));
        }

    return v3;
}


/************************************ ArcBall::constrain_vector() ****/

Magnum::Vector3 ArcBallInteractor::constrainVector(const Magnum::Vector3 &vector, const Magnum::Vector3 &axis)
{
    return (vector-(vector*axis)*axis).normalized();
}

/************************************ ArcBall::mouse_down() **********/

void ArcBallInteractor::mouseDown(int x, int y)
{
    std::cout << "mouse_down(" << x << ", " << y << ")" << std::endl;

    // move the mouse to the center, change from screen coordinates.
    y = (int) floor(2.0 * center[1] - y);

    down_pt = {{ (float)x, (float) y }};
    is_mouse_down = true;

    q_increment   = Magnum::Quaternion(Magnum::Math::IdentityInit);
    rot_increment = Magnum::Matrix4(Magnum::Math::IdentityInit);
    zero_increment = true;
}


/************************************ ArcBall::mouse_up() **********/

void ArcBallInteractor::mouseUp()
{
    q_now = q_drag * q_now;
    is_mouse_down = false;
}


/********************************** ArcBall::mouse_motion() **********/

void ArcBallInteractor::mouseMotion(int x, int y, int shift, int ctrl, int alt)
{
    std::cout << "mouse_motion(x:" << x << ", y:" << y << ", shift: " << shift << ", ctrl: " << ctrl << ", alt: " << alt << ")" << std::endl;


    // move the mouse to the center, change from screen coordinates.
    y = (int) floor(2.0 * center[1] - y);
    
    /* Set the X constraint if CONTROL key is pressed, Y if ALT key */
    setConstraints( ctrl != 0, alt != 0 );

    Magnum::Vector2 new_pt( (float)x, (float) y );
    Magnum::Vector3 v0 = mouseToSphere( down_pt );
    Magnum::Vector3 v1 = mouseToSphere( new_pt );

    Magnum::Vector3 cross = Magnum::Math::cross(v0,v1);

    q_drag = Magnum::Quaternion{{cross}, Magnum::Math::dot(v0,v1) };

    //    *rot_ptr = (q_drag * q_now).to_Magnum::Matrix4();
    Magnum::Matrix4 temp = q2m(q_drag);
    
    std::cout << "rot: " << std::endl;
    std::cout << rot << std::endl;
    std::cout << "temp:" << std::endl;
    std::cout << temp << std::endl;
    
    rot =  temp * rot;
    
    std::cout << "new rot:" << std::endl;
    std::cout << rot << std::endl;
    
    

    down_pt = new_pt;

    /* We keep a copy of the current incremental rotation (= q_drag) */
    q_increment   = q_drag;
    rot_increment = q2m(q_increment);

    setConstraints(false, false);

    if ( q_increment.scalar() < .999999 )
    {
        is_spinning = true;
        zero_increment = false;
    }
    else
    {
        is_spinning = false;
        zero_increment = true;
    }
}


/********************************** ArcBall::mouse_motion() **********/

void ArcBallInteractor::mouseMotion(int x, int y)
{
    mouseMotion(x, y, 0, 0, 0);
}


/***************************** ArcBall::set_constraints() **********/

void ArcBallInteractor::setConstraints(bool _constraint_x, bool _constraint_y)
{
    constraint_x = _constraint_x;
    constraint_y = _constraint_y;
}

/***************************** ArcBall::idle() *********************/

void ArcBallInteractor::idle()
{
    if (is_mouse_down)
    {
        is_spinning = false;
        zero_increment = true;
    }


    if (damp_factor < 1.0f) {
        q_increment = Magnum::Quaternion::rotation(Magnum::Rad(1.0f - damp_factor), q_increment.axis());
    }

    rot_increment = q2m(q_increment);

    if (q_increment.scalar() >= .999999f)
    {
        is_spinning = false;
        zero_increment = true;
    }
}


/************************ ArcBall::set_damping() *********************/

void ArcBallInteractor::setDampening(float d)
{
    damp_factor = d;
}





