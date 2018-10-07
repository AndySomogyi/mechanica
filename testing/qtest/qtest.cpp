/*
 * test.cpp
 *
 *  Created on: Oct 4, 2018
 *      Author: andy
 */

#include <string.h>
#include <iostream>

using namespace std;

#include "ArcBallInteractor.h"


using namespace Magnum;

ostream& operator<<(ostream& os, const Quaternion& q)
{
    Vector3 v = q.vector();
    os << "quat {{" << v[0] << ", " << v[1] << ", " << v[2] << "}, " << q.scalar() << "}";
    return os;
}

ostream& operator<<(ostream& os, const ArcBallInteractor& ball)
{
    os << "q_down: " << ball.q_down << std::endl;
    os << "q_now: " << ball.q_now << std::endl;
    os << "q_drag: " << ball.q_drag << std::endl;
    os << "q_increment: " << ball.q_increment << std::endl;
    os << "rot: " << std::endl << ball.rot << std::endl;
    return os;
}



void mouseDown(ArcBallInteractor& ball, int x, int y) {
    
    cout << "mouseDown(" << x << ", " << y << ")" << std::endl;
    
    ball.mouseDown( x, y );
}


void mouseMotion(ArcBallInteractor &ball, int local_x, int local_y, bool inside)
{
    cout << "mouseMotion(" << local_x << ", " << local_y << ")" << std::endl;

    ball.mouseMotion( local_x, local_y);
}

void  initBall(ArcBallInteractor &ball, float w, float h)
{
    cout << "initBall(" << w << ", " << h << ")" << std::endl;



  /*ball->set_damping( .05 );              */
  /*float( MIN(w/2,h/2))*2.0  );              */
  /*    ball->reset_mouse();              */
}


void testMat() {
    
    std::cout << "matrix test..." << std::endl;
    
    Matrix4 m1{{ 0.997062,   0.0761639, -0.00816716, 0},
               {-0.0728401,  0.975706,   0.206619,   0},
               { 0.0237057, -0.205417,   0.978387,   0},
               { 0,          0,          0,          1}};
    
    Matrix4 m2{{0.972691,   -0.124996,  -0.195572,   0},
               {0.0838129,   0.974905,  -0.206241,   0},
               {0.216444,    0.184217,   0.958758,   0},
               {0,           0,          0,          1}};
    
    std::cout << "m1: " << std::endl;
    std::cout << m1 << std::endl;
    std::cout << "m2:" << std::endl;
    std::cout << m2 << std::endl;
    
    Matrix4 result = m1 * m2;
    
    std::cout << "m1 * m2:" << std::endl;
    std::cout << result << std::endl;
    
    Matrix4 r2 = m2 * m1;
    
    std::cout << "m2 * m1:" << std::endl;
    std::cout << r2 << std::endl;
}

int main(int argc, const char** argv) {

    testMat();

    ArcBallInteractor ab;

    initBall(ab, 640, 480);

    cout << ab;

    mouseDown(ab, 0, 0);

    cout << ab;

    mouseMotion(ab, 0, 100, false);

    cout << ab;

    mouseMotion(ab, 100, 0, false);

    cout << ab;

    return 0;
}




