/*
 * ca_parser.h
 *
 *  Created on: Jul 2, 2015
 *      Author: andy
 */

#ifndef PYCALC_INCLUDE_CA_PARSER_H_
#define PYCALC_INCLUDE_CA_PARSER_H_

#include <mx_ast.h>
#include <stdio.h>


struct MxCompilerFlags {

};

struct MxMemory {

};

typedef struct MxParserFlags {

};


/**
 * Parse and compile the Mechanica source code in *str*, returning the resulting AST
 * object.  The start token is given by *start*; this can be used to constrain the
 * code which can be compiled and should be :const:`Mx_eval_input`,
 * :const:`Mx_file_input`, or :const:`Mx_single_input`.  The filename specified by
 * *filename* is used to construct the code object and may appear in tracebacks or
 * :exc:`SyntaxError` exception messages.  This returns *NULL* if the code cannot
 * be parsed or compiled.
 */

MxAPI_FUNC(MxObject*) Mx_ParseStringFlags(const char *str, const char *filename,
		int start, MxParserFlags *flags);






#endif /* PYCALC_INCLUDE_CA_PARSER_H_ */
