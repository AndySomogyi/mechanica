/*
 * ca_runtime.h
 *
 *  Created on: Jun 30, 2015
 *      Author: andy
 */

#ifndef _INCLUDE_CA_RUNTIME_H_
#define _INCLUDE_CA_RUNTIME_H_

#include <mx_port.h>
#include <mx_object.h>
#include <mx_module.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C"
{
#endif

MxAPI_DATA(MxModule*) module;

MxAPI_FUNC(wchar_t *) Mx_GetProgramName(void);

MxAPI_FUNC(void) Mx_SetMechanicaHome(wchar_t *);
MxAPI_FUNC(wchar_t *) Mx_GetMechanicaHome(void);

/**
 * Initialize the entire runtime.
 */
MxAPI_FUNC(int) Mx_Initialize(int);


MxAPI_FUNC(void) Mx_Finalize(void);

/**
 * Is the runtime initialized?
 */
MxAPI_FUNC(int) Mx_IsInitialized(void);

/* Mx_CaAtExit is for the atexit module, Mx_AtExit is for low-level
 * exit functions.
 */

MxAPI_FUNC(int) Mx_AtExit(void (*func)(void));

MxAPI_FUNC(void) Mx_Exit(int);

/* Restore signals that the interpreter has called SIG_IGN on to SIG_DFL. */



/* In getpath.c */
MxAPI_FUNC(wchar_t *) Mx_GetProgramFullPath(void);
MxAPI_FUNC(wchar_t *) Mx_GetPrefix(void);
MxAPI_FUNC(wchar_t *) Mx_GetExecPrefix(void);
MxAPI_FUNC(wchar_t *) Mx_GetPath(void);
MxAPI_FUNC(void) Mx_SetPath(const wchar_t *);

/* In their own files */
MxAPI_FUNC(const char *) Mx_GetVersion(void);
MxAPI_FUNC(const char *) Mx_GetPlatform(void);
MxAPI_FUNC(const char *) Mx_GetCopyright(void);
MxAPI_FUNC(const char *) Mx_GetCompiler(void);
MxAPI_FUNC(const char *) Mx_GetBuildInfo(void);

/* Signals */
typedef void (*MxOS_sighandler_t)(int);
MxAPI_FUNC(MxOS_sighandler_t) MxOS_getsig(int);
MxAPI_FUNC(MxOS_sighandler_t) MxOS_setsig(int, MxOS_sighandler_t);

/* Random */
MxAPI_FUNC(int) _CaOS_URandom (void *buffer, Mx_ssize_t size);

MxAPI_FUNC(void) MxErr_Print(void);
MxAPI_FUNC(void) MxErr_PrintEx(int);
MxAPI_FUNC(void) MxErr_Display(MxObject *, MxObject *, MxObject *);

MxAPI_FUNC(void) MxErr_SetString(
		MxObject *exception,
		const char *string /* decoded from utf-8 */
);

/**
 * Test whether the error indicator is set. If set, return the exception type
 * (the first argument to the last call to one of the MxErr_Set*() functions or to MxErr_Restore()).
 * If not set, return NULL. You do not own a reference to the return value,
 * so you do not need to Mx_DECREF() it.
 */
MxAPI_FUNC(MxObject *) MxErr_Occurred(void);


MxAPI_FUNC(void) MxErr_Clear(void);
MxAPI_FUNC(void) MxErr_Fetch(MxObject **, MxObject **, MxObject **);
MxAPI_FUNC(void) MxErr_Restore(MxObject *, MxObject *, MxObject *);
MxAPI_FUNC(void) MxErr_GetExcInfo(MxObject **, MxObject **, MxObject **);
MxAPI_FUNC(void) MxErr_SetExcInfo(MxObject *, MxObject *, MxObject *);

/**

 *************************
 The Very High Level Layer
 *************************

 The functions in this chapter will let you execute Mechanica source code given in a
 file or a buffer, but they will not let you interact in a more detailed way with
 the interpreter.

 Several of these functions accept a start symbol from the grammar as a
 parameter.  The available start symbols are :const:`Mx_eval_input`,
 :const:`Mx_file_input`, and :const:`Mx_single_input`.  These are described
 following the functions which accept them as parameters.

 Note also that several of these functions take :c:type:`FILE\*` parameters.  One
 particular issue which needs to be handled carefully is that the :c:type:`FILE`
 structure for different C libraries can be different and incompatible.  Under
 Windows (at least), it is possible for dynamically linked extensions to actually
 use different libraries, so care should be taken that :c:type:`FILE\*` parameters
 are only passed to these functions if it is certain that they were created by
 the same library that the Mxthon runtime is using.
 */


/**
 * Return true (nonzero) if the standard I/O file fp with name filename is deemed interactive.
 * This is the case for files for which "isatty(fileno(fp))" is true. If the global flag
 * Mx_InteractiveFlag is true, this function also returns true if the filename pointer is
 * NULL or if the name is equal to one of the strings '<stdin>' or '???'.
 */
MxAPI_FUNC(int) Mx_FdIsInteractive(FILE *, const char *);

/**
 * The main program for the standard interpreter.  This is made available for
 * programs which embed Mxthon.  The *argc* and *argv* parameters should be
 * prepared exactly as those which are passed to a C program's :c:func:`main`
 * function.  It is important to note that the argument list may be modified (but
 * the contents of the strings pointed to by the argument list are not). The return
 * value will be ``0`` if the interpreter exits normally (ie, without an
 * exception), ``1`` if the interpreter exits due to an exception, or ``2``
 * if the parameter list does not represent a valid Mxthon command line.
 *
 * Note that if an otherwise unhandled :exc:`SystemExit` is raised, this
 * function will not return ``1``, but exit the process, as long as
 * ``Mx_InspectFlag`` is not set.
 */
MxAPI_FUNC(int) Mx_Main(int argc, const char **argv);

/**
 * This is a simplified interface to :c:func:`CaRun_AnyFileExFlags` below, leaving
 * closeit* set to ``0`` and *flags* set to *NULL*.
 */
MxAPI_FUNC(int) MxRun_AnyFile(FILE *fp, const char *filename);

typedef struct
{
	int cf_flags; /* bitmask of CO_xxx flags relevant to future */
} MxCompilerFlags;

/**
 * This is a simplified interface to :c:func:`CaRun_AnyFileExFlags` below, leaving
 * the *closeit* argument set to ``0``.
 */

MxAPI_FUNC(int) MxRun_AnyFileFlags(FILE *fp, const char *filename,
		MxCompilerFlags *flags);

/**
 * This is a simplified interface to :c:func:`CaRun_AnyFileExFlags` below, leaving
 * the *flags* argument set to *NULL*.
 */
MxAPI_FUNC(int) MxRun_AnyFileEx(FILE *fp, const char *filename, int closeit);

/**
 * If *fp* refers to a file associated with an interactive device (console or
 * terminal input or Unix pseudo-terminal), return the value of
 * :c:func:`CaRun_InteractiveLoop`, otherwise return the result of
 * :c:func:`CaRun_SimpleFile`.  If *filename* is *NULL*, this function uses
 * ``"???"`` as the filename.
 */
MxAPI_FUNC(int) MxRun_AnyFileExFlags(FILE *fp, const char *filename,
		int closeit, MxCompilerFlags *flags);

/**
 * This is a simplified interface to :c:func:`CaRun_SimpleStringFlags` below,
 * leaving the *MxCompilerFlags\** argument set to NULL.
 */

MxAPI_FUNC(int) MxRun_SimpleString(const char *command);

/**
 Executes the cayman source code from *command* in the :mod:`__main__` module
 according to the *flags* argument. If :mod:`__main__` does not already exist, it
 is created.  Returns ``0`` on success or ``-1`` if an exception was raised.  If
 there was an error, there is no way to get the exception information. For the
 meaning of *flags*, see below.

 Note that if an otherwise unhandled :exc:`SystemExit` is raised, this
 function will not return ``-1``, but exit the process, as long as
 ``Mx_InspectFlag`` is not set.
 */
MxAPI_FUNC(int) MxRun_SimpleStringFlags(const char *command,
		MxCompilerFlags *flags);

/**
 This is a simplified interface to :c:func:`CaRun_SimpleFileExFlags` below,
 leaving *closeit* set to ``0`` and *flags* set to *NULL*.
 */
MxAPI_FUNC(int) MxRun_SimpleFile(FILE *fp, const char *filename);

/**
 This is a simplified interface to :c:func:`CaRun_SimpleFileExFlags` below,
 leaving *flags* set to *NULL*.
 */
MxAPI_FUNC(int) MxRun_SimpleFileEx(FILE *fp, const char *filename, int closeit);

/**
 Similar to :c:func:`CaRun_SimpleStringFlags`, but the Mxthon source code is read
 from *fp* instead of an in-memory string. *filename* should be the name of the
 file.  If *closeit* is true, the file is closed before MxRun_SimpleFileExFlags
 returns.

 */
MxAPI_FUNC(int) MxRun_SimpleFileExFlags(FILE *fp, const char *filename,
		int closeit, MxCompilerFlags *flags);

/**
 * This is a simplified interface to :c:func:`CaRun_InteractiveOneFlags` below,
 * leaving *flags* set to *NULL*.
 */
MxAPI_FUNC(int) MxRun_InteractiveOne(FILE *fp, const char *filename);

/**
 * Read and execute a single statement from a file associated with an
 * interactive device according to the *flags* argument.  The user will be
 * prompted using ``sys.ps1`` and ``sys.ps2``.  Returns ``0`` when the input was
 * executed successfully, ``-1`` if there was an exception, or an error code
 * from the :file:`errcode.h` include file distributed as part of Mxthon if
 * there was a parse error.  (Note that :file:`errcode.h` is not included by
 * :file:`cayman.h`, so must be included specifically if needed.)
 */
MxAPI_FUNC(int) MxRun_InteractiveOneFlags(FILE *fp, const char *filename,
		MxCompilerFlags *flags);

/**
 * This is a simplified interface to :c:func:`CaRun_InteractiveLoopFlags` below,
 * leaving *flags* set to *NULL*.
 */
MxAPI_FUNC(int) MxRun_InteractiveLoop(FILE *fp, const char *filename);

/**
 * Read and execute statements from a file associated with an interactive device
 * until EOF is reached.  The user will be prompted using ``sys.ps1`` and
 * ``sys.ps2``.  Returns ``0`` at EOF.
 */
MxAPI_FUNC(int) MxRun_InteractiveLoopFlags(FILE *fp, const char *filename,
		MxCompilerFlags *flags);

/*

 MxAPI_FUNC(struct) _node* MxParser_SimpleParseString(const char *str,
 int start);

 This is a simplified interface to
 :c:func:`CaParser_SimpleParseStringFlagsFilename` below, leaving  *filename* set
 to *NULL* and *flags* set to ``0``.


 MxAPI_FUNC(struct) _node* MxParser_SimpleParseStringFlags( const char *str,
 int start, int flags);

 This is a simplified interface to
 :c:func:`CaParser_SimpleParseStringFlagsFilename` below, leaving  *filename* set
 to *NULL*.


 MxAPI_FUNC(struct) _node* MxParser_SimpleParseStringFlagsFilename( const char *str,
 const char *filename, int start, int flags);

 Parse Mxthon source code from *str* using the start token *start* according to
 the *flags* argument.  The result can be used to create a code object which can
 be evaluated efficiently. This is useful if a code fragment must be evaluated
 many times.


 MxAPI_FUNC(struct) _node* MxParser_SimpleParseFile(FILE *fp, const char
 *filename, int start);

 This is a simplified interface to :c:func:`CaParser_SimpleParseFileFlags` below,
 leaving *flags* set to ``0``


 MxAPI_FUNC(struct) _node* MxParser_SimpleParseFileFlags(FILE *fp, const char
 *filename, int start, int flags);

 Similar to :c:func:`CaParser_SimpleParseStringFlagsFilename`, but the Mxthon
 source code is read from *fp* instead of an in-memory string.
 */

/**
 * This is a simplified interface to :c:func:`CaRun_StringFlags` below, leaving
 * *flags* set to *NULL*.
 */
MxAPI_FUNC(MxObject*) MxRun_String(const char *str, int start, MxObject *globals,
		MxObject *locals);

/*
 Execute Mxthon source code from *str* in the context specified by the
 dictionaries *globals* and *locals* with the compiler flags specified by
 *flags*.  The parameter *start* specifies the start token that should be used to
 parse the source code.

 Returns the result of executing the code as a Mxthon object, or *NULL* if an
 exception was raised.
 */
MxAPI_FUNC(MxObject*) MxRun_StringFlags(const char *str, int start,
		MxObject *globals, MxObject *locals, MxCompilerFlags *flags);

/*
 This is a simplified interface to :c:func:`CaRun_FileExFlags` below, leaving
 *closeit* set to ``0`` and *flags* set to *NULL*.

 */

MxAPI_FUNC(MxObject*) MxRun_File(FILE *fp, const char *filename, int start,
		MxObject *globals, MxObject *locals);

/*
 This is a simplified interface to :c:func:`CaRun_FileExFlags` below, leaving
 *flags* set to *NULL*.
 */
MxAPI_FUNC(MxObject*) MxRun_FileEx(FILE *fp, const char *filename, int start,
		MxObject *globals, MxObject *locals, int closeit);

/*
 This is a simplified interface to :c:func:`CaRun_FileExFlags` below, leaving
 *closeit* set to ``0``.

 */
MxAPI_FUNC(MxObject*) MxRun_FileFlags(FILE *fp, const char *filename, int start,
		MxObject *globals, MxObject *locals, MxCompilerFlags *flags);

/*

 Similar to :c:func:`CaRun_StringFlags`, but the Mxthon source code is read from
 *fp* instead of an in-memory string. *filename* should be the name of the file.
 If *closeit* is true, the file is closed before :c:func:`CaRun_FileExFlags`
 returns.

 */

MxAPI_FUNC(MxObject*) MxRun_FileExFlags(FILE *fp, const char *filename, int start,
		MxObject *globals, MxObject *locals, int closeit, MxCompilerFlags *flags);

/*
 This is a simplified interface to :c:func:`Mx_CompileStringFlags` below, leaving
 *flags* set to *NULL*.
 */

MxAPI_FUNC(MxObject*) Mx_CompileString(const char *str, const char *filename,
		int start);

/*
 Parse and compile the Mxthon source code in *str*, returning the resulting code
 object.  The start token is given by *start*; this can be used to constrain the
 code which can be compiled and should be :const:`Mx_eval_input`,
 :const:`Mx_file_input`, or :const:`Mx_single_input`.  The filename specified by
 *filename* is used to construct the code object and may appear in tracebacks or
 :exc:`SyntaxError` exception messages.  This returns *NULL* if the code cannot
 be parsed or compiled.

 */

MxAPI_FUNC(MxObject*) Mx_CompileStringFlags(const char *str, const char *filename,
		int start, MxCompilerFlags *flags);

/*
 This is a simplified interface to :c:func:`MxEval_EvalCodeEx`, with just
 the code object, and the dictionaries of global and local variables.
 The other arguments are set to *NULL*.


 MxAPI_FUNC(MxObject*) MxEval_EvalCode(CaCodeObject *co, MxObject *globals,
 MxObject *locals);



 Evaluate a precompiled code object, given a particular environment for its
 evaluation.  This environment consists of dictionaries of global and local
 variables, arrays of arguments, keywords and defaults, and a closure tuple of
 cells.

 MxAPI_FUNC(MxObject*) MxEval_EvalCodeEx(CaCodeObject *co, MxObject *globals, C
 aObject *locals,  MxObject **args, int argcount, MxObject **kws, int kwcount,
 MxObject **defs, int defcount, MxObject *closure);



 Evaluate an execution frame.  This is a simplified interface to
 MxEval_EvalFrameEx, for backward compatibility.


 MxAPI_FUNC(MxObject*) MxEval_EvalFrame(CaFrameObject *f);


 MxAPI_FUNC(MxObject*) MxEval_EvalFrameEx(CaFrameObject *f, int throwflag);

 This is the main, unvarnished function of Mxthon interpretation.  It is
 literally 2000 lines long.  The code object associated with the execution
 frame *f* is executed, interpreting bytecode and executing calls as needed.
 The additional *throwflag* parameter can mostly be ignored - if true, then
 it causes an exception to immediately be thrown; this is used for the
 :meth:`~generator.throw` methods of generator objects.


 MxAPI_FUNC(int) MxEval_MergeCompilerFlags(MxCompilerFlags *cf);

 This function changes the flags of the current evaluation frame, and returns
 true on success, false on failure.


 */


/**
 * System Functions
 * These are utility functions that make functionality from the sys module accessible to C code.
 * They all work with the current interpreter thread’s sys module’s dict, which is contained in
 * the internal thread state structure.
 */

/**
 * Return value: Borrowed reference.
Return the object name from the sys module or NULL if it does not exist, without setting an exception.
 */
MxAPI_FUNC(MxObject*) MxSys_GetObject(const char *name);

/**
 * Return the FILE* associated with the object name in the sys module, or def if name is
 * not in the module or is not associated with a FILE*.
 */
MxAPI_FUNC(FILE*) MxSys_GetFile(const char *name, FILE *def);

/**
 * Set name in the sys module to v unless v is NULL, in which case name is deleted from
 * the sys module. Returns 0 on success, -1 on error.
 */
MxAPI_FUNC(int) MxSys_SetObject(const char *name, MxObject *v);



#ifdef __cplusplus
}
#endif

#endif /* _INCLUDE_CA_RUNTIME_H_ */
