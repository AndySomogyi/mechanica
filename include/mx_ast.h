/*
 * ca_ast.h
 *
 *  Created on: Aug 4, 2015
 *      Author: andy
 *
 * Reflection
 */

#ifndef _INCLUDE_MX_AST_H_
#define _INCLUDE_MX_AST_H_

/**
 * Mechanica AST
 *
 * Creates a set of type definitions in the cayman.ast namespace when used under
 * Python. or a in the 'ast' namespace when stand-alone.
 *
 * Many language elements, such as strings, lists, dictionaries, primitives parse
 * directly into thier native type, i.e. as in scheme, there is no difference
 * between a quasi-quoted string and a string itself, i.e.
 *
 * "foo" is the same as '"foo".
 *
 * Other, more complex language elments parse into a 'defintion' object, such as
 * a function defintion "FuncDef", or a class definition, "ClassDef". Here, there is a
 * difference betwen the compiled object and its definition. This is the same as in
 * scheme:
 *
 * @code{.py}
 * >>> (define foodef '(define (foo x) x))
 * >>> foodef
 * '(define (foo x) x)
 * >>> (eval foodef)
 * >>> foo
 * #<procedure:foo>
 * @endcode
 */

#include "carbon.h"

#include "mx_list.h"


#ifdef __cplusplus
extern "C" {
#endif






#ifdef __cplusplus
}
#endif



#endif /* _INCLUDE_MX_AST_H_ */
