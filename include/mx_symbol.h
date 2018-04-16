/*
 * ca_symbol.h
 *
 *  Created on: Apr 26, 2016
 *      Author: andy
 */

#ifndef INCLUDE_CA_SYMBOL_H_
#define INCLUDE_CA_SYMBOL_H_

#include "mx_string.h"

#ifdef __cplusplus
extern "C" {
#endif


/**
 * A symbol is a unique and immutable string which is used to identify named language
 * elements.
 *
 * Symbols are shared and immutable, which means that they can be compared directly
 * using thier address. The MxSymbol creation functions will always return the same
 * symbol object for the same string (but with the reference count increases).
 *
 *
 * Symbols may be either a single character string, or a dotted name, i.e. foo.bar.fred.
 *
 * Dotted symbols are automatically parsed and generate a MxDottedSymbol object.
 */
struct MxSymbol;
struct MxType;

/**
 * The type object for a MxSymbol.
 */
MxAPI_DATA(MxType) *MxSymbol_Type;

/**
 * Determines if this object is a symbol type.
 * @returns TRUE if a symbol, FALSE otherwise.
 */
MxAPI_FUNC(int) MxSymbolCheck(MxObject *o);

/**
 * Creates a new symbol from a given character string.
 *
 * Interns the given character string -- searches if a symbol already exists
 * for this string, if so, bumps the reference count and returns it, otherwise
 * a new symbol is allocated, interned and returned.
 *
 * The parameter v must not be NULL; it will not be checked.
 *
 * @param str: a NULL terminated C string.
 * @returns New reference, NULL on failure.
 */
MxAPI_FUNC(MxSymbol*) MxSymbol_FromCString(const char *str);

/**
 * Creates a new symbol from a given character string and length
 *
 * Interns the given character string -- searches if a symbol already exists
 * for this string, if so, bumps the reference count and returns it, otherwise
 * a new symbol is allocated, interned and returned.
 *
 * The parameter v must not be NULL; it will not be checked.
 *
 * @param str: a NULL terminated C string.
 * @param len: how much of str to use.
 * @returns New reference, NULL on failure.
 */
MxAPI_FUNC(MxSymbol*) MxSymbol_FromCStringAndSize(const char *str, Mx_ssize_t len);

/**
 * Steals a reference from an existing string, and creates a new symbol.
 *
 * Same semantics as MxSymbol_FromCString, it creates an interned string, and returns
 * a reference to it.
 *
 * @param str: Stolen reference to an existing string.
 * @returns New reference
 */
MxAPI_FUNC(MxSymbol*) MxSymbol_FromString(MxString *str);

/**
 * Get the underlying string for this symbol.
 * @returns Borrowed reference
 */
MxAPI_FUNC(MxString*) MxSymbol_GetString(MxSymbol *sym);


/**
 * Compares two symbols. No error checking is performed. The result is the
 * same as if the symbol string values were compared.
 *
 * result = cmp(o1, o2).
 */
int MxSymbol_Cmp(const MxSymbol *o1, const MxSymbol *o2);


/**
 * Dotted symbols are a sequence of symbols
 *
 * Dotted symbols are also interned and immutable, like symbols.
 */
struct MxDottedSymbol;

/**
 * The type object for a MxDottedSymbol.
 */
MxAPI_DATA(MxType) *MxDottedSymbol_Type;

/**
 * Determines if this object is a dotted symbol type.
 * @returns TRUE if a symbol, FALSE otherwise.
 */
MxAPI_FUNC(int) MxDottedSymbolCheck(MxObject *o);

#ifdef __cplusplus
}
#endif


#endif /* INCLUDE_CA_SYMBOL_H_ */
