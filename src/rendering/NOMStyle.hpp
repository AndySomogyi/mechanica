/*
 * NOMStyle.hpp
 *
 *  Created on: Jul 29, 2020
 *      Author: andy
 */

#ifndef SRC_RENDERING_NOMSTYLE_HPP_
#define SRC_RENDERING_NOMSTYLE_HPP_

#include <NOMStyle.h>

#include <Magnum/Magnum.h>
#include <Magnum/Math/Color.h>

struct NOMStyle : public PyObject
{
    Magnum::Color3 color;
};


HRESULT _NOMStyle_init(PyObject *m);

#endif /* SRC_RENDERING_NOMSTYLE_HPP_ */
