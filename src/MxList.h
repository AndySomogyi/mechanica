/*
 * MxList.h
 *
 *  Created on: Apr 9, 2017
 *      Author: andy
 */

#ifndef SRC_MXLIST_H_
#define SRC_MXLIST_H_


struct MxList : MxObject {

    /* Vector of pointers to list elements.  list[0] is ob_item[0], etc. */
	MxObject **ob_item;

    /* ob_item contains space for 'allocated' elements.  The number
     * currently in use is ob_size.
     * Invariants:
     *     0 <= ob_size <= allocated
     *     len(list) == ob_size
     *     ob_item == NULL implies ob_size == allocated == 0
     * list.sort() temporarily sets allocated to -1 to detect mutations.
     *
     * Items must normally not be NULL, except during construction when
     * the list is not yet visible outside the function that builds it.
     */
    Mx_ssize_t allocated;
} ;



#endif /* SRC_MXLIST_H_ */
