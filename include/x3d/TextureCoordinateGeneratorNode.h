/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	TextureCoordinateGeneratorNode.h
*
******************************************************************/

#ifndef _CX3D_TEXCOORDGENNODE_H_
#define _CX3D_TEXCOORDGENNODE_H_

#include <x3d/Node.h>

namespace CyberX3D {

class TextureCoordinateGeneratorNode : public Node 
{
	MFFloat *parameterField;
	SFString *modeField;

public:

	TextureCoordinateGeneratorNode();
	virtual ~TextureCoordinateGeneratorNode();

	////////////////////////////////////////////////
	//	parameter
	////////////////////////////////////////////////

	MFFloat *getParameterField() const;
	
	void addParameter(float value);
	int getNParameters() const;
	float getParameter(int index) const;

	////////////////////////////////////////////////
	//	mode
	////////////////////////////////////////////////
	
	SFString *getModeField() const;

	void setMode(const char *value);
	const char *getMode() const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	TextureCoordinateGeneratorNode	*next() const;
	TextureCoordinateGeneratorNode	*nextTraversal() const;

	////////////////////////////////////////////////
	//	virtual functions
	////////////////////////////////////////////////

	bool	isChildNodeType(Node *node) const;
	void	initialize();
	void	uninitialize();
	void	update();
	void	outputContext(std::ostream &printStream, const char *indentString) const;
};

}

#endif
