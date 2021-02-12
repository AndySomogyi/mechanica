/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2010 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 ******************************************************************************/

/* Include configuration header */
#include "mdcore_config.h"

/* Include some standard header files */
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <limits.h>
#include <iostream>
#include <sstream>


/* Include some conditional headers. */
#include "mdcore_config.h"
#ifdef __SSE__
    #include <xmmintrin.h>
#endif
#ifdef WITH_MPI
    #include <mpi.h>
#endif

/* Include local headers */
#include "cycle.h"
#include "errs.h"
#include "fptype.h"
#include "lock.h"
#include <MxParticle.h>
#include <MxPotential.h>
#include "potential_eval.hpp"
#include <space_cell.h>
#include "space.h"
#include "engine.h"
#include <bond.h>

#include <MxConvert.hpp>
#include <../../MxUtil.h>
#include <../../rendering/NOMStyle.hpp>

NOMStyle *MxBond_StylePtr = NULL;

/* Global variables. */
/** The ID of the last error. */
int bond_err = bond_err_ok;
unsigned int bond_rcount = 0;

/* the error macro. */
#define error(id)				( bond_err = errs_register( id , bond_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )

/* list of error messages. */
const char *bond_err_msg[2] = {
	"Nothing bad happened.",
    "An unexpected NULL pointer was encountered."
};

static PyObject* bond_destroy(MxBondHandle *_self, PyObject *args, PyObject *kwargs);
static PyObject* bond_energy(MxBondHandle *_self, PyObject *args, PyObject *kwargs);
static PyObject *bond_bonds();


/**
 * check if a type pair is in a list of pairs
 * pairs has to be a python list of tuples of types
 */
static bool pair_check(PyObject *pairs, short a_typeid, short b_typeid);

/**
 * @brief Evaluate a list of bonded interactoins
 *
 * @param b Pointer to an array of #bond.
 * @param N Nr of bonds in @c b.
 * @param e Pointer to the #engine in which these bonds are evaluated.
 * @param epot_out Pointer to a double in which to aggregate the potential energy.
 * 
 * @return #bond_err_ok or <0 on error (see #bond_err)
 */
 
int bond_eval ( struct MxBond *bonds , int N , struct engine *e , double *epot_out ) {

    int bid, pid, pjd, k, *loci, *locj, shift[3], ld_pots;
    double h[3], epot = 0.0;
    struct space *s;
    struct MxParticle *pi, *pj, **partlist;
    struct space_cell **celllist;
    struct MxPotential *pot;
    struct MxBond *b;
    FPTYPE r2, w;
#if defined(VECTORIZE)
    struct MxPotential *potq[VEC_SIZE];
    int icount = 0, l;
    FPTYPE dx[4] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE pix[4] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE *effi[VEC_SIZE], *effj[VEC_SIZE];
    FPTYPE r2q[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE ee[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE eff[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE dxq[VEC_SIZE*3];
#else
    FPTYPE ee, eff, dx[4], pix[4];
#endif
    
    /* Check inputs. */
    if ( bonds == NULL || e == NULL )
        return error(bond_err_null);
        
    /* Get local copies of some variables. */
    s = &e->s;
    partlist = s->partlist;
    celllist = s->celllist;
    ld_pots = e->max_type;
    for ( k = 0 ; k < 3 ; k++ )
        h[k] = s->h[k];
    pix[3] = FPTYPE_ZERO;
        
    /* Loop over the bonds. */
    for ( bid = 0 ; bid < N ; bid++ ) {

        b = &bonds[bid];

        if(!(b->flags & BOND_ACTIVE))
            continue;
    
        /* Get the particles involved. */
        pid = bonds[bid].i; pjd = bonds[bid].j;
        if ( ( pi = partlist[ pid ] ) == NULL )
            continue;
        if ( ( pj = partlist[ pjd ] ) == NULL )
            continue;
        
        /* Skip if both ghosts. */
        if ( ( pi->flags & PARTICLE_GHOST ) && 
             ( pj->flags & PARTICLE_GHOST ) )
            continue;
            
        /* Get the potential. */
        pot = b->potential;
        if (!pot) {
            continue;
        }
    
        /* get the distance between both particles */
        loci = celllist[ pid ]->loc; locj = celllist[ pjd ]->loc;
        for ( k = 0 ; k < 3 ; k++ ) {
            shift[k] = loci[k] - locj[k];
            if ( shift[k] > 1 )
                shift[k] = -1;
            else if ( shift[k] < -1 )
                shift[k] = 1;
            pix[k] = pi->x[k] + h[k]*shift[k];
        }
        r2 = fptype_r2( pix , pj->x , dx );
        
        if ( r2 < pot->a*pot->a || r2 > pot->b*pot->b ) {
            //printf( "bond_eval: bond %i (%s-%s) out of range [%e,%e], r=%e.\n" ,
            //    bid , e->types[pi->typeId].name , e->types[pj->typeId].name , pot->a , pot->b , sqrt(r2) );
            r2 = fmax( pot->a*pot->a , fmin( pot->b*pot->b , r2 ) );
        }

        #ifdef VECTORIZE
            /* add this bond to the interaction queue. */
            r2q[icount] = r2;
            dxq[icount*3] = dx[0];
            dxq[icount*3+1] = dx[1];
            dxq[icount*3+2] = dx[2];
            effi[icount] = pi->f;
            effj[icount] = pj->f;
            potq[icount] = pot;
            icount += 1;

            /* evaluate the interactions if the queue is full. */
            if ( icount == VEC_SIZE ) {

                #if defined(FPTYPE_SINGLE)
                    #if VEC_SIZE==8
                    potential_eval_vec_8single( potq , r2q , ee , eff );
                    #else
                    potential_eval_vec_4single( potq , r2q , ee , eff );
                    #endif
                #elif defined(FPTYPE_DOUBLE)
                    #if VEC_SIZE==4
                    potential_eval_vec_4double( potq , r2q , ee , eff );
                    #else
                    potential_eval_vec_2double( potq , r2q , ee , eff );
                    #endif
                #endif

                /* update the forces and the energy */
                for ( l = 0 ; l < VEC_SIZE ; l++ ) {
                    epot += ee[l];
                    for ( k = 0 ; k < 3 ; k++ ) {
                        w = eff[l] * dxq[l*3+k];
                        effi[l][k] -= w;
                        effj[l][k] += w;
                    }
                }

                /* re-set the counter. */
                icount = 0;

            }
        #else // NOT VECTORIZE
            /* evaluate the bond */
            #ifdef EXPLICIT_POTENTIALS
                potential_eval_expl( pot , r2 , &ee , &eff );
            #else
                potential_eval( pot , r2 , &ee , &eff );
            #endif
        
            if(eff >= b->dissociation_energy) {
                MxBond_Destroy(b);
            }
            else {

                /* update the forces */
                for ( k = 0 ; k < 3 ; k++ ) {
                    w = eff * dx[k];
                    pi->f[k] -= w;
                    pj->f[k] += w;
                }
                /* tabulate the energy */
                epot += ee;
            }


        #endif

        } /* loop over bonds. */
        
    #if defined(VECTORIZE)
        /* are there any leftovers? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < VEC_SIZE ; k++ ) {
                potq[k] = potq[0];
                r2q[k] = r2q[0];
                }

            /* evaluate the potentials */
            #if defined(VEC_SINGLE)
                #if VEC_SIZE==8
                potential_eval_vec_8single( potq , r2q , ee , eff );
                #else
                potential_eval_vec_4single( potq , r2q , ee , eff );
                #endif
            #elif defined(VEC_DOUBLE)
                #if VEC_SIZE==4
                potential_eval_vec_4double( potq , r2q , ee , eff );
                #else
                potential_eval_vec_2double( potq , r2q , ee , eff );
                #endif
            #endif

            /* for each entry, update the forces and energy */
            for ( l = 0 ; l < icount ; l++ ) {
                epot += ee[l];
                for ( k = 0 ; k < 3 ; k++ ) {
                    w = eff[l] * dxq[l*3+k];
                    effi[l][k] -= w;
                    effj[l][k] += w;
                    }
                }

            }
    #endif
    
    /* Store the potential energy. */
    if ( epot_out != NULL )
        *epot_out += epot;
    
    /* We're done here. */
    return bond_err_ok;
    
}



/**
 * @brief Evaluate a list of bonded interactoins
 *
 * @param b Pointer to an array of #bond.
 * @param N Nr of bonds in @c b.
 * @param e Pointer to the #engine in which these bonds are evaluated.
 * @param f An array of @c FPTYPE in which to aggregate the resulting forces.
 * @param epot_out Pointer to a double in which to aggregate the potential energy.
 * 
 * This function differs from #bond_eval in that the forces are added to
 * the array @c f instead of directly in the particle data.
 * 
 * @return #bond_err_ok or <0 on error (see #bond_err)
 */
 
int bond_evalf ( struct MxBond *b , int N , struct engine *e , FPTYPE *f , double *epot_out ) {

    int bid, pid, pjd, k, *loci, *locj, shift[3], ld_pots;
    double h[3], epot = 0.0;
    struct space *s;
    struct MxParticle *pi, *pj, **partlist;
    struct space_cell **celllist;
    struct MxPotential *pot;
    FPTYPE r2, w;
#if defined(VECTORIZE)
    struct MxPotential *potq[VEC_SIZE];
    int icount = 0, l;
    FPTYPE dx[4] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE pix[4] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE *effi[VEC_SIZE], *effj[VEC_SIZE];
    FPTYPE r2q[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE ee[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE eff[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE dxq[VEC_SIZE*3];
#else
    FPTYPE ee, eff, dx[4], pix[4];
#endif
    
    /* Check inputs. */
    if ( b == NULL || e == NULL || f == NULL )
        return error(bond_err_null);
        
    /* Get local copies of some variables. */
    s = &e->s;
    partlist = s->partlist;
    celllist = s->celllist;
    ld_pots = e->max_type;
    for ( k = 0 ; k < 3 ; k++ )
        h[k] = s->h[k];
    pix[3] = FPTYPE_ZERO;
        
    /* Loop over the bonds. */
    for ( bid = 0 ; bid < N ; bid++ ) {
    
        /* Get the particles involved. */
        pid = b[bid].i; pjd = b[bid].j;
        if ( ( pi = partlist[ pid ] ) == NULL )
            continue;
        if ( ( pj = partlist[ pjd ] ) == NULL )
            continue;
        
        /* Skip if both ghosts. */
        if ( pi->flags & PARTICLE_GHOST && pj->flags & PARTICLE_GHOST )
            continue;
            
        /* Get the potential. */
        if ( ( pot = b[bid].potential ) == NULL )
            continue;
    
        /* get the distance between both particles */
        loci = celllist[ pid ]->loc; locj = celllist[ pjd ]->loc;
        for ( k = 0 ; k < 3 ; k++ ) {
            shift[k] = loci[k] - locj[k];
            if ( shift[k] > 1 )
                shift[k] = -1;
            else if ( shift[k] < -1 )
                shift[k] = 1;
            pix[k] = pi->x[k] + h[k]*shift[k];
            }
        r2 = fptype_r2( pix , pj->x , dx );

        if ( r2 < pot->a*pot->a || r2 > pot->b*pot->b ) {
            printf( "bond_evalf: bond %i (%s-%s) out of range [%e,%e], r=%e.\n" ,
                bid , e->types[pi->typeId].name , e->types[pj->typeId].name , pot->a , pot->b , sqrt(r2) );
            r2 = fmax( pot->a*pot->a , fmin( pot->b*pot->b , r2 ) );
            }

        #ifdef VECTORIZE
            /* add this bond to the interaction queue. */
            r2q[icount] = r2;
            dxq[icount*3] = dx[0];
            dxq[icount*3+1] = dx[1];
            dxq[icount*3+2] = dx[2];
            effi[icount] = &( f[ 4*pid ] );
            effj[icount] = &( f[ 4*pjd ] );
            potq[icount] = pot;
            icount += 1;

            /* evaluate the interactions if the queue is full. */
            if ( icount == VEC_SIZE ) {

                #if defined(FPTYPE_SINGLE)
                    #if VEC_SIZE==8
                    potential_eval_vec_8single( potq , r2q , ee , eff );
                    #else
                    potential_eval_vec_4single( potq , r2q , ee , eff );
                    #endif
                #elif defined(FPTYPE_DOUBLE)
                    #if VEC_SIZE==4
                    potential_eval_vec_4double( potq , r2q , ee , eff );
                    #else
                    potential_eval_vec_2double( potq , r2q , ee , eff );
                    #endif
                #endif

                /* update the forces and the energy */
                for ( l = 0 ; l < VEC_SIZE ; l++ ) {
                    epot += ee[l];
                    for ( k = 0 ; k < 3 ; k++ ) {
                        w = eff[l] * dxq[l*3+k];
                        effi[l][k] -= w;
                        effj[l][k] += w;
                        }
                    }

                /* re-set the counter. */
                icount = 0;

                }
        #else
            /* evaluate the bond */
            #ifdef EXPLICIT_POTENTIALS
                potential_eval_expl( pot , r2 , &ee , &eff );
            #else
                potential_eval( pot , r2 , &ee , &eff );
            #endif

            /* update the forces */
            for ( k = 0 ; k < 3 ; k++ ) {
                w = eff * dx[k];
                f[ 4*pid + k ] -= w;
                f[ 4*pjd + k ] += w;
                }

            /* tabulate the energy */
            epot += ee;
        #endif

        } /* loop over bonds. */
        
    #if defined(VECTORIZE)
        /* are there any leftovers? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < VEC_SIZE ; k++ ) {
                potq[k] = potq[0];
                r2q[k] = r2q[0];
                }

            /* evaluate the potentials */
            #if defined(VEC_SINGLE)
                #if VEC_SIZE==8
                potential_eval_vec_8single( potq , r2q , ee , eff );
                #else
                potential_eval_vec_4single( potq , r2q , ee , eff );
                #endif
            #elif defined(VEC_DOUBLE)
                #if VEC_SIZE==4
                potential_eval_vec_4double( potq , r2q , ee , eff );
                #else
                potential_eval_vec_2double( potq , r2q , ee , eff );
                #endif
            #endif

            /* for each entry, update the forces and energy */
            for ( l = 0 ; l < icount ; l++ ) {
                epot += ee[l];
                for ( k = 0 ; k < 3 ; k++ ) {
                    w = eff[l] * dxq[l*3+k];
                    effi[l][k] -= w;
                    effj[l][k] += w;
                    }
                }

            }
    #endif
    
    /* Store the potential energy. */
    if ( epot_out != NULL )
        *epot_out += epot;
    
    /* We're done here. */
    return bond_err_ok;
    
    }


static int _bond_init(MxBondHandle *self, uint32_t flags, int32_t i, int32_t j,
        double half_life, double bond_energy, struct MxPotential *potential) {

    MxBond *bond = NULL;
    
    int result = engine_bond_alloc (&_Engine, &bond );
    
    if(result < 0) {
        return c_error(E_FAIL, "could not allocate bond");
    }
    
    bond->flags = flags;
    bond->i = i;
    bond->j = j;
    bond->half_life = half_life;
    bond->dissociation_energy = bond_energy;
    bond->style = MxBond_StylePtr;
    Py_IncRef(bond->style);
    
    if(bond->i >= 0 && bond->j >= 0) {
        bond->flags = bond->flags | BOND_ACTIVE;
        _Engine.nr_active_bonds++;
    }

    if(potential) {
        Py_INCREF(potential);
        bond->potential = potential;
    }
    
    self->id = result;

    return 0;
}

static int bond_init(MxBondHandle *self, PyObject *args, PyObject *kwargs) {

    std::cout << MX_FUNCTION << std::endl;

    try {
        PyObject *pot  = mx::arg<PyObject*>("potential", 0, args, kwargs);
        PyObject *p1  = mx::arg<PyObject*>("p1", 1, args, kwargs);
        PyObject *p2  = mx::arg<PyObject*>("p2", 2, args, kwargs);
        
        double half_life = mx::arg<double>("half_life", 3, args, kwargs, std::numeric_limits<double>::max());
        double bond_energy = mx::arg<double>("dissociation_energy", 4, args, kwargs, std::numeric_limits<double>::max());
        uint32_t flags = mx::arg<uint32_t>("flags", 5, args, kwargs, 0);
        
        if(PyObject_IsInstance(pot, (PyObject*)&MxPotential_Type) <= 0) {
            PyErr_SetString(PyExc_TypeError, "potential is not a instance of Potential");
            return -1;
        }
        
        if(MxParticle_Check(p1) <= 0) {
            PyErr_SetString(PyExc_TypeError, "p1 is not a instance of Particle");
            return -1;
        }
        
        if(MxParticle_Check(p2) <= 0) {
            PyErr_SetString(PyExc_TypeError, "p2 is not a instance Particle");
            return -1;
        }
        
        return _bond_init(self, flags, ((MxParticleHandle*)p1)->id, ((MxParticleHandle*)p2)->id,
                   half_life, bond_energy, (MxPotential*)pot);

    }
    catch (const std::exception &e) {
        return C_EXP(e);
    }
}

static PyObject* bond_str(MxBondHandle *bh) {
    std::stringstream  ss;
    MxBond *bond = &_Engine.bonds[bh->id];
    
    
    ss << "Bond(i="
       << bond->i
       << ", j="
       << bond->j
       << ")";
    
    return PyUnicode_FromString(ss.str().c_str());
}

static PyGetSetDef bond_getset[] = {
    {
        .name = "parts",
        .get = [](PyObject *_obj, void *p) -> PyObject* {
            MxBond *bond = ((MxBondHandle*)_obj)->get();
            if(bond->flags & BOND_ACTIVE) {
                return MxParticleList_Pack(2, bond->i, bond->j);
            }
            Py_RETURN_NONE;
        },
        .set = [](PyObject *_obj, PyObject *val, void *p) -> int {
            PyErr_SetString(PyExc_PermissionError, "read only");
            return -1;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "potential",
        .get = [](PyObject *_obj, void *p) -> PyObject* {
            MxBond *bond = ((MxBondHandle*)_obj)->get();
            if(bond->flags & BOND_ACTIVE) {
                PyObject *pot = bond->potential;
                Py_INCREF(pot);
                return pot;
            }
            Py_RETURN_NONE;
        },
        .set = [](PyObject *_obj, PyObject *val, void *p) -> int {
            PyErr_SetString(PyExc_PermissionError, "read only");
            return -1;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "id",
        .get = [](PyObject *_obj, void *p) -> PyObject* {
            MxBond *bond = ((MxBondHandle*)_obj)->get();
            if(bond->flags & BOND_ACTIVE) {
                // WARNING: need the (int) cast here to pick up the
                // correct mx::cast template specialization, won't build wiht
                // an int32_specializations, claims it's duplicate for int.
                return mx::cast((int)bond->id);
            }
            Py_RETURN_NONE;
        },
        .set = [](PyObject *_obj, PyObject *val, void *p) -> int {
            PyErr_SetString(PyExc_PermissionError, "read only");
            return -1;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "dissociation_energy",
        .get = [](PyObject *_obj, void *p) -> PyObject* {
            MxBond *bond = ((MxBondHandle*)_obj)->get();
            return mx::cast(bond->dissociation_energy);
        },
        .set = [](PyObject *_obj, PyObject *val, void *p) -> int {
            try {
                MxBond *bond = ((MxBondHandle*)_obj)->get();
                bond->dissociation_energy = mx::cast<float>(val);
                return 0;
            }
            catch (const std::exception &e) {
                return C_EXP(e);
            }
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "active",
        .get = [](PyObject *_obj, void *p) -> PyObject* {
            MxBond *bond = ((MxBondHandle*)_obj)->get();
            return mx::cast((bool)(bond->flags & BOND_ACTIVE));
        },
        .set = [](PyObject *_obj, PyObject *val, void *p) -> int {
            PyErr_SetString(PyExc_PermissionError, "read only");
            return -1;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "style",
        .get = [](PyObject *_obj, void *p) -> PyObject* {
            MxBond *bond = ((MxBondHandle*)_obj)->get();
            Py_INCREF(bond->style);
            return bond->style;
        },
        .set = [](PyObject *_obj, PyObject *val, void *p) -> int {
            if(!NOMStyle_Check(val)) {
                PyErr_SetString(PyExc_TypeError, "style must be a Style object");
                return -1;
            }
            MxBond *bond = ((MxBondHandle*)_obj)->get();
            Py_DECREF(bond->style);
            bond->style = (NOMStyle*)val;
            Py_INCREF(bond->style);
            return 0;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {NULL}
};

PyObject *bond_bonds() {
    PyObject *list = PyList_New(_Engine.nr_bonds);
    
    for(int i = 0; i < _Engine.nr_bonds; ++i) {
        PyList_SET_ITEM(list, i, MxBondHandle_FromId(i));
    }
    
    return list;
}



static PyMethodDef bond_methods[] = {
    { "destroy", (PyCFunction)bond_destroy, METH_VARARGS | METH_KEYWORDS, NULL },
    { "energy", (PyCFunction)bond_energy, METH_NOARGS, NULL },
    { "bonds", (PyCFunction)bond_bonds, METH_STATIC | METH_NOARGS, NULL },
    { "items", (PyCFunction)bond_bonds, METH_STATIC | METH_NOARGS, NULL },
    { NULL, NULL, 0, NULL }
};


PyTypeObject MxBondHandle_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "Bond",
    .tp_basicsize = sizeof(MxBondHandle),
    .tp_itemsize =       0,
    .tp_dealloc =        0,
                         0, // changed to tp_vectorcall_offset in python 3.8
    .tp_getattr =        0,
    .tp_setattr =        0,
    .tp_as_async =       0,
    .tp_repr =           (reprfunc)bond_str,
    .tp_as_number =      0,
    .tp_as_sequence =    0,
    .tp_as_mapping =     0,
    .tp_hash =           0,
    .tp_call =           0,
    .tp_str =            (reprfunc)bond_str,
    .tp_getattro =       0,
    .tp_setattro =       0,
    .tp_as_buffer =      0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Custom objects",
    .tp_traverse =       0,
    .tp_clear =          0,
    .tp_richcompare =    0,
    .tp_weaklistoffset = 0,
    .tp_iter =           0,
    .tp_iternext =       0,
    .tp_methods =        bond_methods,
    .tp_members =        0,
    .tp_getset =         bond_getset,
    .tp_base =           0,
    .tp_dict =           0,
    .tp_descr_get =      0,
    .tp_descr_set =      0,
    .tp_dictoffset =     0,
    .tp_init =           (initproc)bond_init,
    .tp_alloc =          0,
    .tp_new =            PyType_GenericNew,
    .tp_free =           0,
    .tp_is_gc =          0,
    .tp_bases =          0,
    .tp_mro =            0,
    .tp_cache =          0,
    .tp_subclasses =     0,
    .tp_weaklist =       0,
    .tp_del =            0,
    .tp_version_tag =    0,
    .tp_finalize =       0,
};


static PyMethodDef methods[] = {
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "bonds",   /* name of module */
        NULL, /* module documentation, may be NULL */
        -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
        methods
};

static PyObject *bonds_module = NULL;

HRESULT _MxBond_init(PyObject *m)
{
    if (PyType_Ready((PyTypeObject*)&MxBondHandle_Type) < 0) {
        std::cout << "could not initialize MxBondHandle_Type " << std::endl;
        return E_FAIL;
    }

    bonds_module = PyModule_Create(&moduledef);

    Py_INCREF(&MxBondHandle_Type);
    if (PyModule_AddObject(m, "Bond", (PyObject *)&MxBondHandle_Type) < 0) {
        Py_DECREF(&MxBondHandle_Type);
        return E_FAIL;
    }

    if (PyModule_AddObject(m, "bonds", (PyObject *)bonds_module) < 0) {
        Py_DECREF(&MxBondHandle_Type);
        Py_DECREF(&bonds_module);
        return E_FAIL;
    }
    
    MxBond_StylePtr = NOMStyle_NewEx(Color3_Parse("lime"));
    
    if(MxBondHandle_Type.tp_dict) {
        PyDict_SetItemString(MxBondHandle_Type.tp_dict, "style", MxBond_StylePtr);
    }

    return S_OK;
}

MxBondHandle* MxBondHandle_New(uint32_t flags, int32_t i, int32_t j,
        double half_life, double bond_energy, struct MxPotential *potential)
{
    MxBondHandle *bond = (MxBondHandle*)PyType_GenericAlloc(&MxBondHandle_Type, 0);

    _bond_init(bond, flags, i, j, half_life, bond_energy, potential);

    return bond;
}

// list of pairs...
struct Pair {
    int32_t i;
    int32_t j;
};

typedef std::vector<Pair> PairList;

static void make_pairlist(const MxParticleList *parts,
                          float cutoff, PyObject *paircheck_list,
                          PairList& pairs) {
    int i, j;
    struct MxParticle *part_i, *part_j;
    Magnum::Vector4 dx;
    Magnum::Vector4 pix, pjx;
 
    /* get the space and cutoff */
    pix[3] = FPTYPE_ZERO;
    
    float r2;
    
    float c2 = cutoff * cutoff;
    
    // TODO: more effecient to caclulate everythign in reference frame
    // of outer particle.
    
    /* loop over all particles */
    for ( i = 1 ; i < parts->nr_parts ; i++ ) {
        
        /* get the particle */
        part_i = _Engine.s.partlist[parts->parts[i]];
        
        // global position
        double *oi = _Engine.s.celllist[part_i->id]->origin;
        pix[0] = part_i->x[0] + oi[0];
        pix[1] = part_i->x[1] + oi[1];
        pix[2] = part_i->x[2] + oi[2];
        
        /* loop over all other particles */
        for ( j = 0 ; j < i ; j++ ) {
            
            /* get the other particle */
            part_j = _Engine.s.partlist[parts->parts[j]];
            
            // global position
            double *oj = _Engine.s.celllist[part_j->id]->origin;
            pjx[0] = part_j->x[0] + oj[0];
            pjx[1] = part_j->x[1] + oj[1];
            pjx[2] = part_j->x[2] + oj[2];
            
            /* get the distance between both particles */
            r2 = fptype_r2(pix.data(), pjx.data() , dx.data());
            
            if(r2 <= c2 && pair_check(paircheck_list, part_i->typeId, part_j->typeId)) {
                pairs.push_back(Pair{part_i->id,part_j->id});
            }
        } /* loop over all other particles */
    } /* loop over all particles */
}

PyObject* MxBond_PairwiseNew(
    struct MxPotential* pot,
    struct MxParticleList *parts,
    float cutoff,
    PyObject *ppairs,
    PyObject *args,
    PyObject *kwds) {
    
    PairList pairs;
    PyObject *bonds = NULL;
    MxBondHandle *bond = NULL;
    
    try {
        make_pairlist(parts, cutoff, ppairs, pairs);
        
        bonds = PyList_New(pairs.size());
        std::cout << "list size: " << PyList_Size(bonds) << std::endl;
        
        double half_life = mx::arg<double>("half_life", 3, args, kwds, std::numeric_limits<double>::max());
        double bond_energy = mx::arg<double>("bond_energy", 4, args, kwds, std::numeric_limits<double>::max());
        uint32_t flags = mx::arg<uint32_t>("flags", 5, args, kwds, 0);
        
        for(int i = 0; i < pairs.size(); ++i) {
            bond = (MxBondHandle*)PyType_GenericAlloc(&MxBondHandle_Type, 0);
            if(!bond) {
                throw std::logic_error("failed to allocated bond");
            }
            
            if(_bond_init(bond, flags, pairs[i].i, pairs[i].j, half_life, bond_energy, (MxPotential*)pot) != 0) {
                throw std::logic_error("failed to init bond");
            }
                
            // each new bond has a refcount of 1
            PyList_SET_ITEM(bonds, i, bond);
        }
        
        return bonds;
    }
    catch (const std::exception &e) {
        if(bonds) {
            Py_DecRef(bonds);
        }
        C_EXP(e);
    }
    return NULL;
}

MxBondHandle* MxBondHandle_FromId(int id) {
    if(id >= 0 && id < _Engine.nr_bonds) {
        MxBondHandle *h = (MxBondHandle*)PyType_GenericAlloc(&MxBondHandle_Type, 0);
        h->id = id;
        return h;
    }
    PyErr_SetString(PyExc_ValueError, "invalid id");
    return NULL;
}

CAPI_FUNC(HRESULT) MxBond_Destroy(struct MxBond *b) {
    
    std::unique_lock<std::mutex> lock(_Engine.bonds_mutex);
    
    if(b->flags & BOND_ACTIVE) {
        Py_DecRef(b->potential);
        // this clears the BOND_ACTIVE flag
        bzero(b, sizeof(MxBond));
        _Engine.nr_active_bonds -= 1;
    }
    return S_OK;
}


PyObject* bond_destroy(MxBondHandle *self, PyObject *args,
                                 PyObject *kwargs)
{
    std::cout << MX_FUNCTION << std::endl;
    
    MxBond_Destroy(self->get());
    Py_RETURN_NONE;
}

PyObject* bond_energy(MxBondHandle *self, PyObject *args, PyObject *kwargs)
{
    std::cout << MX_FUNCTION << std::endl;
    
    MxBond *bond = self->get();
    double energy = 0;
    
    MxBond_Energy (bond, &energy);
    
    return mx::cast(energy);
    
    Py_RETURN_NONE;
}


HRESULT MxBond_Energy (MxBond *b, double *epot_out) {
    
    int pid, pjd, k, *loci, *locj, shift[3], ld_pots;
    double h[3], epot = 0.0;
    struct space *s;
    struct MxParticle *pi, *pj, **partlist;
    struct space_cell **celllist;
    struct MxPotential *pot;
    FPTYPE r2, w;
    FPTYPE ee, eff, dx[4], pix[4];
    
    
    /* Get local copies of some variables. */
    s = &_Engine.s;
    partlist = s->partlist;
    celllist = s->celllist;
    ld_pots = _Engine.max_type;
    for ( k = 0 ; k < 3 ; k++ )
        h[k] = s->h[k];
    
    pix[3] = FPTYPE_ZERO;
    
    if(!(b->flags & BOND_ACTIVE))
        return -1;
    
    /* Get the particles involved. */
    pid = b->i; pjd = b->j;
    if ( ( pi = partlist[ pid ] ) == NULL )
        return -1;
    if ( ( pj = partlist[ pjd ] ) == NULL )
        return -1;
    
    /* Skip if both ghosts. */
    if ( ( pi->flags & PARTICLE_GHOST ) &&
        ( pj->flags & PARTICLE_GHOST ) )
        return 0;
    
    /* Get the potential. */
    pot = b->potential;
    if (!pot) {
        return 0;
    }
    
    /* get the distance between both particles */
    loci = celllist[ pid ]->loc; locj = celllist[ pjd ]->loc;
    for ( k = 0 ; k < 3 ; k++ ) {
        shift[k] = loci[k] - locj[k];
        if ( shift[k] > 1 )
            shift[k] = -1;
        else if ( shift[k] < -1 )
            shift[k] = 1;
        pix[k] = pi->x[k] + h[k]*shift[k];
    }
    r2 = fptype_r2( pix , pj->x , dx );
    
    if ( r2 < pot->a*pot->a || r2 > pot->b*pot->b ) {
        //printf( "bond_eval: bond %i (%s-%s) out of range [%e,%e], r=%e.\n" ,
        //    bid , e->types[pi->typeId].name , e->types[pj->typeId].name , pot->a , pot->b , sqrt(r2) );
        r2 = fmax( pot->a*pot->a , fmin( pot->b*pot->b , r2 ) );
    }
    
    /* evaluate the bond */
    potential_eval( pot , r2 , &ee , &eff );
    
    /* update the forces */
    //for ( k = 0 ; k < 3 ; k++ ) {
    //    w = eff * dx[k];
    //    pi->f[k] -= w;
    //    pj->f[k] += w;
    //}
    
    /* tabulate the energy */
    epot += ee;
    

    /* Store the potential energy. */
    if ( epot_out != NULL )
        *epot_out += epot;
    
    /* We're done here. */
    return bond_err_ok;
}

std::vector<int32_t> MxBond_IdsForParticle(int32_t pid) {
    std::vector<int32_t> bonds;
    for (int i = 0; i < _Engine.nr_bonds; ++i) {
        MxBond *b = &_Engine.bonds[i];
        if((b->flags & BOND_ACTIVE) && (b->i == pid || b->j == pid)) {
            assert(i == b->id);
            bonds.push_back(b->id);
        }
    }
    return bonds;
}


bool pair_check(PyObject *pairs, short a_typeid, short b_typeid) {
    if(!pairs) {
        return true;
    }
    
    PyObject *a = (PyObject*)&_Engine.types[a_typeid];
    PyObject *b = (PyObject*)&_Engine.types[b_typeid];
    
    for (int i = 0; i < PyList_Size(pairs); ++i) {
        PyObject *o = PyList_GetItem(pairs, i);
        if(PyTuple_Check(o) && PyTuple_Size(o) == 2) {
            if((a == PyTuple_GET_ITEM(o, 0) && b == PyTuple_GET_ITEM(o, 1)) ||
               (b == PyTuple_GET_ITEM(o, 0) && a == PyTuple_GET_ITEM(o, 1))) {
                return true;
            }
        }
    }
    return false;
}
