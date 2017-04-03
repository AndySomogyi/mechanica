//http://www.youtube.com/user/thecplusplusguy
#include "vector3d.h"


vector3d::vector3d()
{
	x=y=z=0;
}
vector3d::vector3d(float a,float b)
{
	x=a;
	y=b;
	z=0;
}

vector3d::vector3d(float a,float b,float c)
{
	x=a;
	y=b;
	z=c;
}
		
float vector3d::dotProduct(const vector3d& vec2)
{
	return (x*vec2.x+y*vec2.y+z*vec2.z);
}

vector3d vector3d::crossProduct(const vector3d& vec2)
{
	return vector3d();
}

float vector3d::length()
{
	return sqrt(x*x+y*y+z*z);
}

void vector3d::normalize()
{
	float len=length();
	if(len!=0)
	{
		x/=len;
		y/=len;
		z/=len;
	}
}
		
void vector3d::change(float a,float b,float c)
{
	x=a;
	y=b;
	z=c;
}

void vector3d::change(vector3d vec2)
{
	x=vec2.x;
	y=vec2.y;
	z=vec2.z;
}

void vector3d::changeX(float a)
{
	x=a;
}
void vector3d::changeY(float a)
{
	y=a;
}
void vector3d::changeZ(float a)
{
	z=a;
}
		
vector3d vector3d::operator+(const vector3d& vec2)
{
	return vector3d(x+vec2.x,y+vec2.y,z+vec2.z);
}

vector3d vector3d::operator-(const vector3d& vec2)
{
	return vector3d(x-vec2.x,y-vec2.y,z-vec2.z);
}

vector3d vector3d::operator*(float num)
{
	return vector3d(x*num,y*num,z*num);
}

vector3d vector3d::operator/(float num)
{
	if(num!=0)
		return vector3d(x/num,y/num,z/num);
	else
		return vector3d();
}

vector3d& vector3d::operator+=(const vector3d& vec2)
{
	x+=vec2.x;
	y+=vec2.y;
	z+=vec2.z;
	return *this;
}

vector3d& vector3d::operator-=(const vector3d& vec2)
{
	x-=vec2.x;
	y-=vec2.y;
	z-=vec2.z;
	return *this;
}

vector3d& vector3d::operator*=(float num)
{
	x*=num;
	y*=num;
	z*=num;
	return *this;
}

vector3d& vector3d::operator/=(float num)	
{
	if(num!=0)
	{
		x/=num;
		y/=num;
		z/=num;
	}
	return *this;
}

bool vector3d::operator==(const vector3d vec2)
{
	return (x==vec2.x && y==vec2.y && z==vec2.z);
}

bool vector3d::operator!=(const vector3d vec2)
{
	!(*this==vec2);
}
		
std::ostream& operator<<(std::ostream& out,const vector3d& vec)
{
	out << vec.x << " " << vec.y << " " << vec.z << std::endl;
	return out;
}
