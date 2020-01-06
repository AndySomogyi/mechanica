/*
 * ca_list.cpp
 *
 *  Created on: Apr 22, 2016
 *      Author: andy
 */

#include "mx_list.h"
#include "mechanica_private.h"

extern "C" {

MxList* MxList_New(Mx_ssize_t sze)
{
    MX_NOTIMPLEMENTED;
}

Mx_ssize_t MxList_Size(MxList* list)
{
    MX_NOTIMPLEMENTED;
}

CObject * MxList_GetItem(MxList* lst, Mx_ssize_t size)
{
    MX_NOTIMPLEMENTED;
}

int MxList_SetItem(MxList* list, Mx_ssize_t index, CObject* value)
{
    MX_NOTIMPLEMENTED;
}

int MxList_Insert(MxList*, Mx_ssize_t unsignedLongInt, CObject*)
{
    MX_NOTIMPLEMENTED;
}

int MxList_Append(MxList*, CObject*)
{
    MX_NOTIMPLEMENTED;
}

CObject * MxList_GetSlice(MxList*, Mx_ssize_t unsignedLongInt, Mx_ssize_t unsignedLongInt1)
{
    MX_NOTIMPLEMENTED;
}

int MxList_SetSlice(MxList*, Mx_ssize_t unsignedLongInt,
		Mx_ssize_t unsignedLongInt1, CObject*)
{
    MX_NOTIMPLEMENTED;
}

int MxList_Sort(MxList*)
{
    MX_NOTIMPLEMENTED
}

int MxList_Reverse(MxList*)
{
    MX_NOTIMPLEMENTED
}

CObject * MxList_AsTuple(MxList*)
{
    MX_NOTIMPLEMENTED
}

CObject * _MxList_Extend(MxList*, CObject*)
{
    MX_NOTIMPLEMENTED
}

int MxList_ClearFreeList(void)
{
    MX_NOTIMPLEMENTED
}

void _MxList_DebugMallocStats(FILE* out)
{
}

}
