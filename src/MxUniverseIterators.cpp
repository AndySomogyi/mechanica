/*
 * MxUniverseIterators.cpp
 *
 *  Created on: May 12, 2020
 *      Author: andy
 */

#include <MxUniverseIterators.h>
#include <MxUniverse.h>

/*
    tests/test_sequences_and_iterators.cpp -- supporting Pythons' sequence protocol, iterators,
    etc.

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
 */


#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <constructor_stats.h>

namespace py = pybind11;


// Obsolete: special data structure for exposing custom iterator types to python
// kept here for illustrative purposes because there might be some use cases which
// are not covered by the much simpler py::make_iterator

struct PyParticlesIterator {
    PyParticlesIterator(py::object ref) :  ref(ref) { }

    py::handle next() {
        if (index == _Engine.s.nr_parts)
            throw py::stop_iteration();

        MxParticle *part = _Engine.s.partlist[index++];
        MxPyParticle *py = MxPyParticle_New(part);
        return py;
    }

    py::object ref; // keep a reference
    size_t index = 0;
};



struct PyBondsIterator {
    PyBondsIterator(py::object ref) :  ref(ref) { }

    py::handle next() {
        if (index == _Engine.nr_bonds)
            throw py::stop_iteration();

        PyObject *bond = &_Engine.bonds[index++];

        return bond;
    }

    py::object ref; // keep a reference
    size_t index = 0;
};




/**
 * Init and add to python module
 */
HRESULT _MxUniverseIterators_init(PyObject *_m) {

    py::module m = pybind11::reinterpret_borrow<py::module>(_m);



    py::class_<PyParticlesIterator>(m, "_PyParticlesIterator")
                .def("__iter__", [](PyParticlesIterator &it) -> PyParticlesIterator& { return it; })
                .def("__next__", &PyParticlesIterator::next);

    py::class_<PyParticles>(m, "PyParticles")
                .def(py::init<>())

                /// Bare bones interface
                .def("__getitem__", [](const PyParticles &s, size_t i) -> py::handle {
        if (i >= _Engine.s.nr_parts) throw py::index_error();
        return MxPyParticle_New(_Engine.s.partlist[i]);
    })
    .def("__setitem__", [](PyParticles &s, size_t i, py::handle) {
        throw py::index_error();
    })
    .def("__len__", [](const PyParticles &s) -> int {
        return _Engine.s.nr_parts;

    });


    py::class_<PyBondsIterator>(m, "_PyBondsIterator")
                .def("__iter__", [](PyBondsIterator &it) -> PyBondsIterator& { return it; })
                .def("__next__", &PyBondsIterator::next);

    py::class_<PyBonds>(m, "PyBonds")
                .def(py::init<>())

                /// Bare bones interface
                .def("__getitem__", [](const PyBonds &s, size_t i) -> py::handle {
        if (i >= _Engine.nr_bonds) throw py::index_error();
        return (PyObject*)&(_Engine.bonds[i]);
    })
    .def("__setitem__", [](PyParticles &s, size_t i, py::handle) {
        throw py::index_error();
    })
    .def("__len__", [](const PyBonds &s) -> int {
        return _Engine.nr_bonds;

    });

    /*

        .def("__iter__", [](py::object s) { return PyParticlesIterator(s); })

        .def("__contains__", [](const PyParticles &s, float v) { return s.contains(v); })
        .def("__reversed__", [](const PyParticles &s) -> PyParticles { return s.reversed(); })
        /// Slicing protocol (optional)
        .def("__getitem__", [](const PyParticles &s, py::slice slice) -> PyParticles* {
            size_t start, stop, step, slicelength;
            if (!slice.compute(s.size(), &start, &stop, &step, &slicelength))
                throw py::error_already_set();
            PyParticles *seq = new PyParticles(slicelength);
            for (size_t i = 0; i < slicelength; ++i) {
                (*seq)[i] = s[start]; start += step;
            }
            return seq;
        })
        .def("__setitem__", [](PyParticles &s, py::slice slice, const PyParticles &value) {
            size_t start, stop, step, slicelength;
            if (!slice.compute(s.size(), &start, &stop, &step, &slicelength))
                throw py::error_already_set();
            if (slicelength != value.size())
                throw std::runtime_error("Left and right hand size of slice assignment have different sizes!");
            for (size_t i = 0; i < slicelength; ++i) {
                s[start] = value[i]; start += step;
            }
        })
     */
    /// Comparisons
    //.def(py::self == py::self)
    //.def(py::self != py::self)
    // Could also define py::self + py::self for concatenation, etc.
    ;

    return S_OK;
}


