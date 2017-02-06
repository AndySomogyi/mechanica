/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	AnchorNode.h
*
******************************************************************/

#ifndef _CX3D_ANCHORNODE_H_
#define _CX3D_ANCHORNODE_H_

#include <x3d/VRML97Fields.h>
#include <x3d/BoundedGroupingNode.h>

namespace CyberX3D {

class AnchorNode : public BoundedGroupingNode {

	SFString *descriptionField;
	MFString *parameterField;
	MFString *urlField;

	SFBool *enabledField;
	SFVec3f *centerField;

public:

	AnchorNode();
	virtual ~AnchorNode(); 

	////////////////////////////////////////////////
	//	Description
	////////////////////////////////////////////////

	SFString *getDescriptionField() const;

	void	setDescription(const char *value);
	const char *getDescription() const;

	////////////////////////////////////////////////
	// Parameter
	////////////////////////////////////////////////

	MFString *getParameterField() const;

	void	addParameter(const char *value);
	int		getNParameters() const;
	const char *getParameter(int index) const;

	////////////////////////////////////////////////
	// Url
	////////////////////////////////////////////////

	MFString *getUrlField() const;

	void	addUrl(const char *value);
	int		getNUrls() const;
	const char *getUrl(int index) const;
	void	setUrl(int index, const char *urlString);

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	AnchorNode	*next() const;
	AnchorNode	*nextTraversal() const;

	////////////////////////////////////////////////
	//	Center (X3D)
	////////////////////////////////////////////////

	SFVec3f *getCenterField() const ;

	void setCenter(float value[]);
	void setCenter(float x, float y, float z);
	void setCenter(String value);
	void getCenter(float value[]) const;
	
	////////////////////////////////////////////////
	//	Enabled (X3D)
	////////////////////////////////////////////////

	SFBool *getEnabledField() const;

	void setEnabled(bool value);
	bool getEnabled() const;
	bool isEnabled() const;

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
