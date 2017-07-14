/*
 * MxDebug.h
 *
 *  Created on: Jul 13, 2017
 *      Author: andy
 */

#ifndef SRC_MXDEBUG_H_
#define SRC_MXDEBUG_H_

#include <iostream>
#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector3.h>

inline std::ostream& operator<<(std::ostream& os, const Magnum::Vector3& vec)
{
    os << "{" << vec[0] << "," << vec[1] << "," << vec[2] << "}";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const Magnum::Vector3ui& vec)
{
    os << "{" << vec[0] << "," << vec[1] << "," << vec[2] << "}";
    return os;
}

//inline std::ostream& operator<<(std::ostream& os, const Magnum::Vector3us& vec)
//{
//   os << "{" << vec[0] << "," << vec[1] << "," << vec[2] << "}";
//    return os;
//}





#endif /* SRC_MXDEBUG_H_ */
