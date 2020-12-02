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
#include "potential_eval.h"
#include <space_cell.h>
#include "space.h"
#include "engine.h"
#include <bond.h>

#include <MxPy.h>

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
                pi->f[k] -= w;
                pj->f[k] += w;
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
    bond->bond_energy = bond_energy;
    
    if(bond->i >= 0 && bond->j >= 0) {
        bond->flags = bond->flags | BOND_ACTIVE;
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
        PyObject *pot  = arg<PyObject*>("potential", 0, args, kwargs);
        PyObject *p1  = arg<PyObject*>("p1", 1, args, kwargs);
        PyObject *p2  = arg<PyObject*>("p2", 2, args, kwargs);
        
        double half_life = arg<double>("half_life", 3, args, kwargs, std::numeric_limits<double>::max());
        double bond_energy = arg<double>("bond_energy", 4, args, kwargs, std::numeric_limits<double>::max());
        uint32_t flags = arg<uint32_t>("flags", 5, args, kwargs, 0);
        
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
        
        return _bond_init(self, flags, ((MxPyParticle*)p1)->id, ((MxPyParticle*)p2)->id,
                   half_life, bond_energy, (MxPotential*)pot);

    }
    catch (const std::exception &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return -1;
    }
    catch(pybind11::error_already_set &e){
        e.restore();
        return -1;
    }
    return 0;
}

static void bond_dealloc(PyObject *obj) {
    std::cout << MX_FUNCTION << std::endl;
}

PyTypeObject MxBondHandle_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "Bond",
    .tp_basicsize = sizeof(MxBondHandle),
    .tp_itemsize =       0,
    .tp_dealloc =        0,
    .tp_print =          0,
    .tp_getattr =        0,
    .tp_setattr =        0,
    .tp_as_async =       0,
    .tp_repr =           0,
    .tp_as_number =      0,
    .tp_as_sequence =    0,
    .tp_as_mapping =     0,
    .tp_hash =           0,
    .tp_call =           0,
    .tp_str =            0,
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
    .tp_methods =        0,
    .tp_members =        0,
    .tp_getset =         0,
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

static PyObject *module;

HRESULT _MxBond_init(PyObject *m)
{
    if (PyType_Ready((PyTypeObject*)&MxBondHandle_Type) < 0) {
        std::cout << "could not initialize MxBondHandle_Type " << std::endl;
        return E_FAIL;
    }

    module = PyModule_Create(&moduledef);

    Py_INCREF(&MxBondHandle_Type);
    if (PyModule_AddObject(m, "Bond", (PyObject *)&MxBondHandle_Type) < 0) {
        Py_DECREF(&MxBondHandle_Type);
        return E_FAIL;
    }

    if (PyModule_AddObject(m, "bonds", (PyObject *)module) < 0) {
        Py_DECREF(&MxBondHandle_Type);
        Py_DECREF(&module);
        return E_FAIL;
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

static void make_pairlist(const MxParticleList *parts, float cutoff, PairList& pairs) {
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
            
            if(r2 <= c2) {
                pairs.push_back(Pair{i,j});
            }
        } /* loop over all other particles */
    } /* loop over all particles */
}

PyObject* MxBond_PairwiseNew(
    struct MxPotential* pot,
    struct MxParticleList *parts,
    float cutoff,
    PyObject *args,
    PyObject *kwds) {
    
    PairList pairs;
    PyObject *bonds = NULL;
    MxBondHandle *bond = NULL;
    make_pairlist(parts, cutoff, pairs);
    
    bonds = PyList_New(pairs.size());
    std::cout << "list size: " << PyList_Size(bonds) << std::endl;
    
    try {
        
        double half_life = arg<double>("half_life", 3, args, kwds, std::numeric_limits<double>::max());
        double bond_energy = arg<double>("bond_energy", 4, args, kwds, std::numeric_limits<double>::max());
        uint32_t flags = arg<uint32_t>("flags", 5, args, kwds, 0);
        
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
        PyErr_SetString(PyExc_ValueError, e.what());
    }
    catch(pybind11::error_already_set &e){
        if(bonds) {
            Py_DecRef(bonds);
        }
        e.restore();
    }
    return NULL;
}

MxBondHandle* MxBondHandle_FromId(int id) {
    if(id >= 0 && id < _Engine.nr_bonds) {
        MxBondHandle *h = (MxBondHandle*)PyType_GenericAlloc(&MxBondHandle_Type, 0);
        h->id = id;
    }
    PyErr_SetString(PyExc_ValueError, "invalid id");
    return NULL;
}
