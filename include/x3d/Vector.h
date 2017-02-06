/******************************************************************
*
*       CyberX3D for C++
*
*       Copyright (C) Satoshi Konno 1996-2007
*
*       File:   Vector.h
*
*       03/17/04
*       - Fixed a memory leak in ~Vector() using STL.
*       03/23/04
*       - Thanks for Joerg Scheurich aka MUFTI <rusmufti@helpdesk.rus.uni-stuttgart.de>
*       - Added to include CyberX3DConfig.h for IRIX platform
*
******************************************************************/

#ifndef _CX3D_VECTOR_H_
#define _CX3D_VECTOR_H_

#include <x3d/CyberX3DConfig.h>
#include <x3d/LinkedList.h>

#ifndef NO_USE_STL
#include <vector>
#endif

namespace CyberX3D {

#ifndef NO_USE_STL

template <class T>
class VectorElement {
	bool	mbDelObj;
	T		*mObj;
public:
	VectorElement() {
		setObject(NULL);
		setObjectDeleteFlag(false);
	}
	VectorElement(T *obj, bool delObjFlag = true) {
		setObject(obj);
		setObjectDeleteFlag(delObjFlag);
	}
	virtual ~VectorElement() { 
		if (mbDelObj)
			delete mObj;
	}
	void setObject(T *obj)	{
		mObj = obj;
	}
	T *getObject() const	{
		return mObj;
	}
	void setObjectDeleteFlag(bool flag)
	{
		mbDelObj = flag;
	}
};

template<class T >
class Vector : public std::vector< VectorElement<T> *> {

public:

	Vector() {
	}

	virtual ~Vector() {
		removeAllElements();
		std::vector<VectorElement<T>*>::clear();
	}

	void addElement(T *obj, bool delObjFlag = true) 
	{
		VectorElement<T> *elem = new VectorElement<T>(obj, delObjFlag);
		std::vector<VectorElement<T>*>::push_back(elem);
	}

	void insertElementAt(T *obj, int index, bool delObjFlag = true)
	{
		VectorElement<T> *elem = new VectorElement<T>(obj, delObjFlag);
		std::vector<VectorElement<T>*>::insert(std::vector<VectorElement<T>*>::begin() + index, elem);
	}

	VectorElement<T> *getElement(int index) const
	{
		if (index < 0)
			return (VectorElement<T> *)NULL;
		if ((int)std::vector<VectorElement<T>*>::size() < (index+1))
			return (VectorElement<T> *)NULL;
		return (*this)[index];
	}

	T *elementAt(int index) const
	{
		VectorElement<T> *elem = getElement(index);
		if (elem == NULL)
			return (T *)NULL;
		return elem->getObject();
	}

	void setElementAt(T *obj, int index) {
		VectorElement<T> *elem = getElement(index);
		if (elem == NULL)
			return;
		elem->setObject(obj);
	}

	int indexOf(T *elem) const{
		return indexOf(elem, 0);
	}

	int indexOf(T *elem, int index) const{
		int cnt = std::vector<VectorElement<T>*>::size();
		for (int n=index; n<cnt; n++) {
			if (elem == elementAt(n))
				return n;
		}
		return -1;
	}

	void removeElement(T *obj) {
		removeElementAt(indexOf(obj));
	}

	void removeElementAt(int index) 
	{
		VectorElement<T> *elem = getElement(index);
		if (elem == NULL)
			return;
		delete elem;
		std::vector<VectorElement<T>*>::erase(std::vector<VectorElement<T>*>::begin() + index);
	}

	void removeAllElements() 
	{
		int cnt = std::vector<VectorElement<T>*>::size();
		for (int n=0; n<cnt; n++) {
			VectorElement<T> *elem = (*this)[n];
			delete elem;
		}
		std::vector<VectorElement<T>*>::clear();
	}

	bool isEmpty() const
	{
		return (std::vector<VectorElement<T>*>::size() == 0) ? false : true;
	}

	T *firstElement() const
	{
		return elementAt(0);
	}

	T *lastElement()const
	{
		return elementAt(std::vector<VectorElement<T>*>::size()-1);
	}
};

#else // NO_USE_STL

template <class T>
class VectorElement : public LinkedListNode<T> {
	bool mbDelObj;
	T	*mObj;
public:
	VectorElement() : LinkedListNode<T>(true) {
		setObject(NULL);
		setObjectDeleteFlag(false);
	}
	VectorElement(T *obj, bool delObjFlag = true) : LinkedListNode<T>((bool)false) {
		setObject(obj);
		setObjectDeleteFlag(delObjFlag);
	}
	virtual ~VectorElement() { 
		remove();
		if (mbDelObj)
			delete mObj;
	}
	void setObject(T *obj)	{
		mObj = obj;
	}
	T *getObject()	const{
		return mObj;
	}
	void setObjectDeleteFlag(bool flag)
	{
		mbDelObj = flag;
	}
};

template <class T>
class Vector {
	LinkedList<T>	 mElementList;
public:
	
	Vector() {
	}

	virtual ~Vector() {
		removeAllElements();
		clear();
	}

	void addElement(T *obj, bool delObjFlag = true) {
		VectorElement<T> *element = new VectorElement<T>(obj, delObjFlag);
		mElementList.addNode(element);
	}

	void insertElementAt(T *obj, int index, bool delObjFlag = true) {
		VectorElement<T> *element = (VectorElement<T> *)mElementList.getNode(index);
		if (element) {
			VectorElement<T> *newElement = new VectorElement<T>(obj, delObjFlag);
			newElement->insert((VectorElement<T> *)element->prev());
		}
	}

	int contains(void *elem) const{
		int cnt = size();
		for (int n=0; n<cnt; n++) {
			if (elem == elementAt(n))
				return 1;
		}
		return 0;
	}

	T *elementAt(int index) const{
		VectorElement<T> *element = (VectorElement<T> *)mElementList.getNode(index);
		return element ? element->getObject() : NULL;
	}

	T *firstElement() const{
		VectorElement<T> *element = (VectorElement<T> *)mElementList.getNodes();
		return element ? element->getObject() : NULL;
	}

	int	indexOf(T *elem) const{
		return indexOf(elem, 0);
	}

	int indexOf(T *elem, int index) const{
		int cnt = size();
		for (int n=index; n<cnt; n++) {
			if (elem == elementAt(n))
				return n;
		}
		return -1;
	}

	bool isEmpty() const{
		return mElementList.getNodes() ? false : true;
	}

	T *lastElement() const{
		VectorElement<T> *element = (VectorElement<T> *)mElementList.getNode(size()-1);
		return element ? element->getObject() : NULL;
	}

	int	lastIndexOf(T *elem) const;
	int	lastIndexOf(T *elem, int index) const;

	void removeAllElements() {
		mElementList.deleteNodes();
	}

	void removeElement(T *obj) {
		removeElementAt(indexOf(obj));
	}

	void removeElementAt(int index) {
		VectorElement<T> *element = (VectorElement<T> *)mElementList.getNode(index);
		if (element)
			delete element;
	}

	void setElementAt(T *obj, int index) {
		VectorElement<T> *element = (VectorElement<T> *)mElementList.getNode(index);
		if (element)  
			element->setObject(obj);
	}

	int	size() const {
		return mElementList.getNNodes();
	}
};

#endif // NO_USE_STL

} 

#endif 
