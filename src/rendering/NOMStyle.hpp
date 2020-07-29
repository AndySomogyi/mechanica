/*
 * NOMStyle.hpp
 *
 *  Created on: Jul 29, 2020
 *      Author: andy
 */

#ifndef SRC_RENDERING_NOMSTYLE_HPP_
#define SRC_RENDERING_NOMSTYLE_HPP_

#include <carbon.h>

struct NOMStyle : public PyObject
{
};


HRESULT NOMStyle_init(PyObject *m);

#endif /* SRC_RENDERING_NOMSTYLE_HPP_ */
