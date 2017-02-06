/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	LinkedListNode.h
*
******************************************************************/

#ifndef _CX3D_LINKEDLISTNODE_H_
#define _CX3D_LINKEDLISTNODE_H_

#include <stdio.h>

#ifdef CX3D_SUPPORT_OLDCPP
#include <OldCpp.h>
#endif

namespace CyberX3D {

template <class T>
class LinkedListNode {

public:

	bool			mbHeader;
	LinkedListNode	*mPrevNode;
	LinkedListNode	*mNextNode;

	LinkedListNode () {
		setHeaderFlag(false);
		mPrevNode = mNextNode = this;
	}

	LinkedListNode (bool header) {
		setHeaderFlag(header);
		mPrevNode = mNextNode = this;
	}

	LinkedListNode (LinkedListNode *prevNode) {
		setHeaderFlag(false);
		insert(prevNode);
	}

	virtual ~LinkedListNode(void)
	{
		remove();
	}

	bool isHeaderNode() const {
		return mbHeader;
	}

	void setHeaderFlag(bool bHeader) {
		mbHeader = bHeader;
	}

	T *next() const {
		if (mNextNode->isHeaderNode())
			return NULL;
		else
			return (T *)mNextNode;
	}

	T* prev() const {
		return (T *)mPrevNode;
	}

	T* nextCircular() const {
		if (mNextNode->isHeaderNode())
			return (T *)mNextNode->mNextNode;
		else
			return (T *)mNextNode;
	}

	T *prevCircular() const {
		if (mPrevNode->isHeaderNode())
			return (T *)mPrevNode->mPrevNode;
		else
			return (T *)mPrevNode;
	}

	void insert(LinkedListNode *prevNode)
	{
		remove();

		mPrevNode = prevNode;
		mNextNode = prevNode->mNextNode;
		prevNode->mNextNode->mPrevNode = this;
		prevNode->mNextNode = this;
	}

	void remove() {
		mNextNode->mPrevNode = mPrevNode;
		mPrevNode->mNextNode = mNextNode;
		mPrevNode = mNextNode = this;
	}
};

}

#endif
