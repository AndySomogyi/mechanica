/*
 * MxColorMapper.cpp
 *
 *  Created on: Dec 27, 2020
 *      Author: andy
 */

#include <rendering/MxColorMapper.hpp>
#include "MxParticle.h"
#include <CSpecies.hpp>
#include <CSpeciesList.hpp>
#include <CStateVector.hpp>
#include <CConvert.hpp>
#include <MxConvert.hpp>
#include <MxSimulator.h>

#include "colormaps/colormaps.h"

struct ColormapItem {
    const char* name;
    ColorMapperFunc func;
};


#define COLORMAP_FUNCTION(CMAP) \
static Magnum::Color4 CMAP (MxColorMapper *cm, struct MxParticle *part) { \
    float s = part->state_vector->fvec[cm->species_index];                \
    return Magnum::Color4{colormaps::all:: CMAP (s), 1};                  \
}\


COLORMAP_FUNCTION(grey_0_100_c0);
COLORMAP_FUNCTION(grey_10_95_c0);
COLORMAP_FUNCTION(kryw_0_100_c71);
COLORMAP_FUNCTION(kryw_0_97_c73);
COLORMAP_FUNCTION(green_5_95_c69);
COLORMAP_FUNCTION(blue_5_95_c73);
COLORMAP_FUNCTION(bmw_5_95_c86);
COLORMAP_FUNCTION(bmy_10_95_c71);
COLORMAP_FUNCTION(bgyw_15_100_c67);
COLORMAP_FUNCTION(gow_60_85_c27);
COLORMAP_FUNCTION(gow_65_90_c35);
COLORMAP_FUNCTION(blue_95_50_c20);
COLORMAP_FUNCTION(red_0_50_c52);
COLORMAP_FUNCTION(green_0_46_c42);
COLORMAP_FUNCTION(blue_0_44_c57);
COLORMAP_FUNCTION(bwr_40_95_c42);
COLORMAP_FUNCTION(gwv_55_95_c39);
COLORMAP_FUNCTION(gwr_55_95_c38);
COLORMAP_FUNCTION(bkr_55_10_c35);
COLORMAP_FUNCTION(bky_60_10_c30);
COLORMAP_FUNCTION(bjy_30_90_c45);
COLORMAP_FUNCTION(bjr_30_55_c53);
COLORMAP_FUNCTION(bwr_55_98_c37);
COLORMAP_FUNCTION(cwm_80_100_c22);
COLORMAP_FUNCTION(bgymr_45_85_c67);
COLORMAP_FUNCTION(bgyrm_35_85_c69);
COLORMAP_FUNCTION(bgyr_35_85_c72);
COLORMAP_FUNCTION(mrybm_35_75_c68);
COLORMAP_FUNCTION(mygbm_30_95_c78);
COLORMAP_FUNCTION(wrwbw_40_90_c42);
COLORMAP_FUNCTION(grey_15_85_c0);
COLORMAP_FUNCTION(cgo_70_c39);
COLORMAP_FUNCTION(cgo_80_c38);
COLORMAP_FUNCTION(cm_70_c39);
COLORMAP_FUNCTION(cjo_70_c25);
COLORMAP_FUNCTION(cjm_75_c23);
COLORMAP_FUNCTION(kbjyw_5_95_c25);
COLORMAP_FUNCTION(kbw_5_98_c40);
COLORMAP_FUNCTION(bwy_60_95_c32);
COLORMAP_FUNCTION(bwyk_16_96_c31);
COLORMAP_FUNCTION(wywb_55_96_c33);
COLORMAP_FUNCTION(krjcw_5_98_c46);
COLORMAP_FUNCTION(krjcw_5_95_c24);
COLORMAP_FUNCTION(cwr_75_98_c20);
COLORMAP_FUNCTION(cwrk_40_100_c20);
COLORMAP_FUNCTION(wrwc_70_100_c20);

ColormapItem colormap_items[] = {
    {"Gray", grey_0_100_c0},
    {"DarkGray", grey_10_95_c0},
    {"Heat", kryw_0_100_c71},
    {"DarkHeat", kryw_0_97_c73},
    {"Green", green_5_95_c69},
    {"Blue", blue_5_95_c73},
    {"BlueMagentaWhite", bmw_5_95_c86},
    {"BlueMagentaYellow", bmy_10_95_c71},
    {"BGYW", bgyw_15_100_c67},
    {"GreenOrangeWhite", gow_60_85_c27},
    {"DarkGreenOrangeWhite", gow_65_90_c35},
    {"LightBlue", blue_95_50_c20},
    {"Red", red_0_50_c52},
    {"DarkGreen", green_0_46_c42},
    {"DarkBlue", blue_0_44_c57},
    {"BlueWhiteRed", bwr_40_95_c42},
    {"GreenWhiteViolet", gwv_55_95_c39},
    {"GreenWhiteRed", gwr_55_95_c38},
    {"BlueBlackRed", bkr_55_10_c35},
    {"BlueBlackYellow", bky_60_10_c30},
    {"BlueGrayYellow", bjy_30_90_c45},
    {"BlueGrayRed", bjr_30_55_c53},
    {"BluwWhiteRed", bwr_55_98_c37},
    {"CyanWhiteMagenta", cwm_80_100_c22},
    {"BGYMR", bgymr_45_85_c67},
    {"DarkBGYMR", bgyrm_35_85_c69},
    {"Rainbow", bgyr_35_85_c72},
    {"CyclicMRYBM", mrybm_35_75_c68},
    {"CyclicMYGBM", mygbm_30_95_c78},
    {"CyclicWRWBW", wrwbw_40_90_c42},
    {"CyclicGray", grey_15_85_c0},
    {"DarkCyanGreenOrange", cgo_70_c39},
    {"CyanGreenOrange", cgo_80_c38},
    {"CyanMagenta", cm_70_c39},
    {"CyanGrayOrange", cjo_70_c25},
    {"CyanGrayMagenta", cjm_75_c23},
    {"KBJW", kbjyw_5_95_c25},
    {"BlackBlueWhite", kbw_5_98_c40},
    {"BlueWhiteYellow", bwy_60_95_c32},
    {"CyclicBWYK", bwyk_16_96_c31},
    {"CyclicWYWB", wywb_55_96_c33},
    {"KRJCW", krjcw_5_98_c46},
    {"DarkKRJCW", krjcw_5_95_c24},
    {"CyclicCyanWhiteRed", cwr_75_98_c20},
    {"CyclicCWRK", cwrk_40_100_c20},
    {"CyclicWRWC", wrwc_70_100_c20}
};

static bool iequals(const std::string& a, const std::string& b)
{
    unsigned int sz = a.size();
    if (b.size() != sz)
        return false;
    for (unsigned int i = 0; i < sz; ++i)
        if (tolower(a[i]) != tolower(b[i]))
            return false;
    return true;
}

static int colormap_index_of_name(const char* s) {
    const int size = sizeof(colormap_items) / sizeof(ColormapItem);
    
    for(int i = 0; i < size; ++i) {
        if (iequals(s, colormap_items[i].name)) {
            return i;
        }
    }
    return -1;
}

bool MxColorMapper::set_colormap(const std::string& s) {
    int index = colormap_index_of_name(s.c_str());
    
    if(index >= 0) {
        this->map = colormap_items[index].func;
        
        MxSimulator_Redraw();
        
        return true;
    }
    return false;
}

MxColorMapper *MxColorMapper_New(struct MxParticleType *partType,
                                 const char* speciesName,
                                 const char* name, float min, float max) {
    
    if(partType->species == NULL) {
        std::string msg = "can not create color map for particle type \"";
        msg += partType->name;
        msg += "\" without any species defined";
        PyErr_WarnEx(PyExc_Warning, msg.c_str(), 2);
        return NULL;
    }
    
    int index = partType->species->index_of(speciesName);
    
    if(index < 0) {
        std::string msg = "can not create color map for particle type \"";
        msg += partType->name;
        msg += "\", does not contain species \"";
        msg += speciesName;
        msg += "\"";
        PyErr_WarnEx(PyExc_Warning, msg.c_str(), 2);
        return NULL;
    }
    
    MxColorMapper *obj = (MxColorMapper*)PyType_GenericAlloc(&MxColormap_Type, 0);
    
    int cmap_index = colormap_index_of_name(name);
    
    if(cmap_index >= 0) {
        obj->map = colormap_items[cmap_index].func;
    }
    else {
        obj->map = bgyr_35_85_c72;
    }
    
    obj->species_index = index;
    obj->min_val = min;
    obj->max_val = max;
    
    return obj;
}

MxColorMapper *MxColorMapper_New(PyObject *args, PyObject *kwargs) {
    if(args == nullptr) {
        PyErr_WarnEx(PyExc_Warning, "args to MxColorMapper_New is NULL", 2);
        return NULL;
    }
    
    MxParticleType *type = MxParticleType_Get(args);
    if(type == nullptr) {
        PyErr_WarnEx(PyExc_Warning, "args to MxColorMapper_New is not a ParticleType", 2);
        return NULL;
    }
    
    if(type->species == NULL) {
        PyErr_WarnEx(PyExc_Warning, "can't create color map on a type without any species", 2);
        return NULL;
    }
    
    MxColorMapper *obj = NULL;
    
    
    
    try {
        // always needs a species
        std::string species = carbon::cast<std::string>(PyDict_GetItemString(kwargs, "species"));
        
        PyObject *pmap = PyDict_GetItemString(kwargs, "map");
        
        std::string map = pmap ? carbon::cast<std::string>(pmap) : "rainbow";
        
        return MxColorMapper_New(type, species.c_str(), map.c_str(), 0, 1);
    }
    catch(const std::exception &ex) {
        delete obj;
        PyErr_WarnEx(PyExc_Warning, ex.what(), 2);
        return NULL;
    }
    
    return obj;
}

static PyObject *colormap_names(PyObject *, PyObject *) {
    int size = sizeof(colormap_items) / sizeof(ColormapItem);
    PyObject *items = PyList_New(size);
    
    for(int i = 0; i < size; ++i) {
        PyObject *s = PyUnicode_FromString(colormap_items[i].name);
        PyList_SET_ITEM(items, i, s);
    }
    
    return items;
}

static PyMethodDef colormap_methods[] = {
    { "names", (PyCFunction)colormap_names, METH_STATIC | METH_NOARGS, NULL },
    { NULL, NULL, 0, NULL }
};



PyTypeObject MxColormap_Type = {
    CVarObject_HEAD_INIT(NULL, 0)
    "Colormap"                        , // .tp_name
    sizeof(MxColorMapper)                 , // .tp_basicsize
    0                                     , // .tp_itemsize
    0    , // .tp_dealloc
    0                                     , // .tp_print
    0                                     , // .tp_getattr
    0                                     , // .tp_setattr
    0                                     , // .tp_as_async
    0                                     , // .tp_repr
    0                                     , // .tp_as_number
    0                                     , // .tp_as_sequence
    0                                     , // .tp_as_mapping
    0                                     , // .tp_hash
    0                                     , // .tp_call
    0                                     , // .tp_str
    0                                     , // .tp_getattro
    0                                     , // .tp_setattro
    0                                     , // .tp_as_buffer
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_TYPE_SUBCLASS, // .tp_flags
    0                                     , // .tp_doc
    0                                     , // .tp_traverse
    0                                     , // .tp_clear
    0                                     , // .tp_richcompare
    0                                     , // .tp_weaklistoffset
    0                                     , // .tp_iter
    0                                     , // .tp_iternext
    colormap_methods                     , // .tp_methods
    0                                     , // .tp_members
    0                 , // .tp_getset
    0                                     , // .tp_base
    0                                     , // .tp_dict
    0                                     , // .tp_descr_get
    0                                     , // .tp_descr_set
    0                                     , // .tp_dictoffset
    0          , // .tp_init
    0                                     , // .tp_alloc
    PyType_GenericNew                     , // .tp_new
    0                                     , // .tp_free
    0                                     , // .tp_is_gc
    0                                     , // .tp_bases
    0                                     , // .tp_mro
    0                                     , // .tp_cache
    0                                     , // .tp_subclasses
    0                                     , // .tp_weaklist
    0                                     , // .tp_del
    0                                     , // .tp_version_tag
    0                                     , // .tp_finalize
#ifdef COUNT_ALLOCS
    0                                     , // .tp_allocs
    0                                     , // .tp_frees
    0                                     , // .tp_maxalloc
    0                                     , // .tp_prev
    0                                     , // .tp_next
#endif
};


MX_BASIC_PYTHON_TYPE_INIT(Colormap)


