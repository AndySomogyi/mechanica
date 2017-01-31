/*
 * ca_import.h
 *
 *  Created on: Jun 30, 2015
 *      Author: andy
 *
 * Module importing functions, definitions and documentation copied from official
 * python website for python compatiblity.
 */

#ifndef _INCLUDED_CA_IMPORT_H_
#define _INCLUDED_CA_IMPORT_H_

#include <mx_port.h>

#ifdef __cplusplus
extern "C" {
#endif


/**
 * This is a simplified interface to MxImport_ImportModuleEx() below,
 * leaving the globals and locals arguments set to NULL and level set to 0.
 * When the name argument contains a dot (when it specifies a submodule of a package),
 * the fromlist argument is set to the list ['*'] so that the return value
 * is the named module rather than the top-level package containing it as would
 * otherwise be the case. (Unfortunately, this has an additional side effect
 * when name in fact specifies a subpackage instead of a submodule:
 * the submodules specified in the package’s __all__ variable are loaded.)
 * Return a new reference to the imported module, or NULL with an exception
 * set on failure. Before Mxthon 2.4, the module may still be created in the
 * failure case — examine sys.modules to find out. Starting with Mxthon 2.4,
 * a failing import of a module no longer leaves the module in sys.modules.
 *
 * Return value: New reference.
 */
MxAPI_FUNC(MxObject*) MxImport_ImportModule(const char *name);

/**
 * This version of MxImport_ImportModule() does not block. It’s intended to be
 * used in C functions that import other modules to execute a function.
 * The import may block if another thread holds the import lock.
 * The function MxImport_ImportModuleNoBlock() never blocks. It first tries to
 * fetch the module from sys.modules and falls back to MxImport_ImportModule()
 * unless the lock is held, in which case the function will raise an ImportError.
 */
MxAPI_FUNC(MxObject*) MxImport_ImportModuleNoBlock(const char *name);

/**
 * Import a module. This is best described by referring to the built-in cayman
 * function __import__(), as the standard __import__() function calls this
 * function directly.
 *
 * The return value is a new reference to the imported module or top-level
 * package, or NULL with an exception set on failure. Like for __import__(),
 * the return value when a submodule of a package was requested is normally
 * the top-level package, unless a non-empty fromlist was given.
 */
MxAPI_FUNC(MxObject*) MxImport_ImportModuleEx(char *name, MxObject *globals,
		MxObject *locals, MxObject *fromlist);

/**
 * Import a module. This is best described by referring to the built-in
 * Mxthon function __import__(),
 * as the standard __import__() function calls this function directly.
 *
 * The return value is a new reference to the imported module or top-level package,
 * or NULL with an exception set on failure. Like for __import__(), the return
 * value when a submodule of a package was requested is normally the top-level
 * package, unless a non-empty fromlist was given.
 */
MxAPI_FUNC(MxObject*) MxImport_ImportModuleLevel(char *name, MxObject *globals,
		MxObject *locals, MxObject *fromlist, int level);

/**
 * This is a higher-level interface that calls the current “import hook function”.
 * It invokes the __import__() function from the __builtins__ of the current
 * globals. This means that the import is done using whatever import hooks are
 * installed in the current environment, e.g. by rexec or ihooks.
 */
MxAPI_FUNC(MxObject*) MxImport_Import(MxObject *name);

/**
 * Reload a module. This is best described by referring to the built-in cayman
 * function reload(), as the standard reload() function calls this function
 * directly. Return a new reference to the reloaded module, or NULL with an
 * exception set on failure (the module still exists in this case).
 */
MxAPI_FUNC(MxObject*) MxImport_ReloadModule(MxObject *m);

/**
 * Return the module object corresponding to a module name. The name argument
 * may be of the form package.module. First check the modules dictionary if
 * there’s one there, and if not, create a new one and insert it in the modules
 * dictionary. Return NULL with an exception set on failure.
 *
 * Note This function does not load or import the module; if the module wasn’t
 * already loaded, you will get an empty module object.
 * Use MxImport_ImportModule() or one of its variants to import a module.
 * Package structures implied by a dotted name for name are not created if
 * not already present.
 */
MxAPI_FUNC(MxObject*) MxImport_AddModule(const char *name);

/**
 * Given a module name (possibly of the form package.module) and a code object
 * read from a cayman bytecode file or obtained from the built-in function
 * compile(), load the module. Return a new reference to the module object,
 * or NULL with an exception set if an error occurred. name is removed from sys.modules
 * in error cases, and even if name was already in sys.modules on entry to
 * MxImport_ExecCodeModule(). Leaving incompletely initialized modules in
 * sys.modules is dangerous, as imports of such modules have no way to know
 * that the module object is an unknown (and probably damaged with respect
 * to the module author’s intents) state.
 *
 * The module’s __file__ attribute will be set to the code object’s co_filename.
 *
 * This function will reload the module if it was already imported.
 * See MxImport_ReloadModule() for the intended way to reload a module.
 *
 * If name points to a dotted name of the form package.module, any
 * package structures not already created will still not be created.
 */
MxAPI_FUNC(MxObject*) MxImport_ExecCodeModule(char *name, MxObject *co);

/**
 * Like MxImport_ExecCodeModule(), but the __file__ attribute of the module
 * object is set to pathname if it is non-NULL.
 */
MxAPI_FUNC(MxObject*) MxImport_ExecCodeModuleEx(char *name, MxObject *co,
		char *pathname);

/**
 * Return the magic number for cayman bytecode files (a.k.a. .pyc and .pyo files).
 * The magic number should be present in the first four bytes of the bytecode file,
 * in little-endian byte order.
 */
MxAPI_FUNC(long) MxImport_GetMagicNumber();

/**
 * Return the dictionary used for the module administration (a.k.a. sys.modules).
 * Note that this is a per-interpreter variable.
 */
MxAPI_FUNC(MxObject*) MxImport_GetModuleDict();

#ifdef __cplusplus
}
#endif

#endif /* !_INCLUDED_CA_IMPORT_H_ */
