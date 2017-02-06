/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	ScriptNode.h
*
******************************************************************/

#ifndef _CX3D_SCRIPT_H_
#define _CX3D_SCRIPT_H_

#include <x3d/VRML97Fields.h>
#include <x3d/Node.h>
#include <x3d/JavaVM.h>
#include <x3d/JScript.h>

namespace CyberX3D {

#if defined(CX3D_SUPPORT_JSAI)
class ScriptNode : public Node, public CJavaVM { 
#else
class ScriptNode : public Node { 
#endif

	SFBool *directOutputField;
	SFBool *mustEvaluateField;
	MFString *urlField;

#if defined(CX3D_SUPPORT_JSAI)
	JScript			*mpJScriptNode;
#endif

public:

	ScriptNode();
	virtual ~ScriptNode();

	////////////////////////////////////////////////
	// Initialization
	////////////////////////////////////////////////

	void initialize();
	void uninitialize();

	////////////////////////////////////////////////
	// DirectOutput
	////////////////////////////////////////////////

	SFBool *getDirectOutputField() const;

	void setDirectOutput(bool  value);
	void setDirectOutput(int value);
	bool  getDirectOutput() const;

	////////////////////////////////////////////////
	// MustEvaluate
	////////////////////////////////////////////////

	SFBool *getMustEvaluateField() const;

	void setMustEvaluate(bool  value);
	void setMustEvaluate(int value);
	bool  getMustEvaluate() const;

	////////////////////////////////////////////////
	// Url
	////////////////////////////////////////////////

	MFString *getUrlField() const;

	void addUrl(const char * value);
	int getNUrls() const;
	const char *getUrl(int index) const;
	void setUrl(int index, const char *urlString);

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	ScriptNode *next() const;
	ScriptNode *nextTraversal() const;

	////////////////////////////////////////////////
	//	virtual function
	////////////////////////////////////////////////

	bool isChildNodeType(Node *node) const;

	////////////////////////////////////////////////
	//	update
	////////////////////////////////////////////////

	void update();

	////////////////////////////////////////////////
	//	output
	////////////////////////////////////////////////

	void outputContext(std::ostream &printStream, const char *indentString) const;

	////////////////////////////////////////////////
	// JSAI
	////////////////////////////////////////////////

#if defined(CX3D_SUPPORT_JSAI)

	int hasScript() {
		return getJavaNode() ? 1 : 0;
	}

	JScript	*getJavaNode()	{return mpJScriptNode;}

#endif

	////////////////////////////////////////////////
	// Update Java Fields
	////////////////////////////////////////////////

	void	update(Field *eventInField);
	void	updateFields();
};

}

#endif

