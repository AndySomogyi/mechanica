    /* StgDict is derived from PyDict_Type */
    PyCStgDict_Type.tp_base = &PyDict_Type;
    if (PyType_Ready(&PyCStgDict_Type) < 0)
        return NULL;
        
        
        
            PyCStructType_Type.tp_base = &PyType_Type;
    if (PyType_Ready(&PyCStructType_Type) < 0)
        return NULL;
        

```
 (PyTypeObject) PyCStructType_Type = {
  ob_base = {
    ob_base = {
      ob_refcnt = 1
      ob_type = 0x0000000000000000
    }
    ob_size = 0
  }
  tp_name = 0x0000000104c74a30 "_ctypes.PyCStructType"
  tp_basicsize = 0
  tp_itemsize = 0
  tp_dealloc = 0x0000000000000000
  tp_print = 0x0000000000000000
  tp_getattr = 0x0000000000000000
  tp_setattr = 0x0000000000000000
  tp_as_async = 0x0000000000000000
  tp_repr = 0x0000000000000000
  tp_as_number = 0x0000000000000000
  tp_as_sequence = 0x0000000104c77760
  tp_as_mapping = 0x0000000000000000
  tp_hash = 0x0000000000000000
  tp_call = 0x0000000000000000
  tp_str = 0x0000000000000000
  tp_getattro = 0x0000000000000000
  tp_setattro = 0x0000000104c659a0
  tp_as_buffer = 0x0000000000000000
  tp_flags = 279552
  tp_doc = 0x0000000104c74a46 "metatype for the CData Objects"
  tp_traverse = 0x0000000104c65a20
  tp_clear = 0x0000000104c65a80
  tp_richcompare = 0x0000000000000000
  tp_weaklistoffset = 0
  tp_iter = 0x0000000000000000
  tp_iternext = 0x0000000000000000
  tp_methods = 0x0000000104c777b0
  tp_members = 0x0000000000000000
  tp_getset = 0x0000000000000000
  tp_base = 0x0000000000000000
  tp_dict = 0x0000000000000000
  tp_descr_get = 0x0000000000000000
  tp_descr_set = 0x0000000000000000
  tp_dictoffset = 0
  tp_init = 0x0000000000000000
  tp_alloc = 0x0000000000000000
  tp_new = 0x0000000104c65ad0
  tp_free = 0x0000000000000000
  tp_is_gc = 0x0000000000000000
  tp_bases = 0x0000000000000000
  tp_mro = 0x0000000000000000
  tp_cache = 0x0000000000000000
  tp_subclasses = 0x0000000000000000
  tp_weaklist = 0x0000000000000000
  tp_del = 0x0000000000000000
  tp_version_tag = 0
  tp_finalize = 0x0000000000000000
}
```



```
int
PyType_Ready(PyTypeObject *type)
{
   ... stuff ...
   
    /* Initialize tp_base (defaults to BaseObject unless that's us) */
    base = type->tp_base;
    if (base == NULL && type != &PyBaseObject_Type) {
        base = type->tp_base = &PyBaseObject_Type;
        Py_INCREF(base);
    }
    
    // base is "type", otherwise initialize to "object" if null
    
    
    // base class is already initialized. 
    /* Initialize the base class */
    if (base != NULL && base->tp_dict == NULL) {
        if (PyType_Ready(base) < 0)
            goto error;
    }
    
    
```


After PyType_Ready
``` 
Printing description of PyCStructType_Type:
(PyTypeObject) PyCStructType_Type = {
  ob_base = {
    ob_base = {
      ob_refcnt = 12
      ob_type = 0x0000000100935c30
    }
    ob_size = 0
  }
  tp_name = 0x0000000104c74a30 "_ctypes.PyCStructType"
  tp_basicsize = 864
  tp_itemsize = 40
  tp_dealloc = 0x0000000100791480
  tp_print = 0x0000000000000000
  tp_getattr = 0x0000000000000000
  tp_setattr = 0x0000000000000000
  tp_as_async = 0x0000000000000000
  tp_repr = 0x0000000100791700
  tp_as_number = 0x0000000000000000
  tp_as_sequence = 0x0000000104c77760
  tp_as_mapping = 0x0000000000000000
  tp_hash = 0x0000000100841ca0
  tp_call = 0x0000000100791830
  tp_str = 0x0000000100793780
  tp_getattro = 0x0000000100791990
  tp_setattro = 0x0000000104c659a0
  tp_as_buffer = 0x0000000000000000
  tp_flags = 2147767296
  tp_doc = 0x0000000104c74a46 "metatype for the CData Objects"
  tp_traverse = 0x0000000104c65a20
  tp_clear = 0x0000000104c65a80
  tp_richcompare = 0x00000001007937a0
  tp_weaklistoffset = 368
  tp_iter = 0x0000000000000000
  tp_iternext = 0x0000000000000000
  tp_methods = 0x0000000104c777b0
  tp_members = 0x0000000000000000
  tp_getset = 0x0000000000000000
  tp_base = 0x0000000100935c30
  tp_dict = 0x0000000104c21e10
  tp_descr_get = 0x0000000000000000
  tp_descr_set = 0x0000000000000000
  tp_dictoffset = 264
  tp_init = 0x0000000100791f70
  tp_alloc = 0x000000010078e580
  tp_new = 0x0000000104c65ad0
  tp_free = 0x00000001008721f0
  tp_is_gc = 0x0000000100793600
  tp_bases = 0x0000000104c22d50
  tp_mro = 0x0000000104c230f0
  tp_cache = 0x0000000000000000
  tp_subclasses = 0x0000000000000000
  tp_weaklist = 0x0000000104c1c8f0
  tp_del = 0x0000000000000000
  tp_version_tag = 0
  tp_finalize = 0x0000000000000000
}
```
