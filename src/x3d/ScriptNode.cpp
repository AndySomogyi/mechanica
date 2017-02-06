/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	ScriptNode.cpp
*
******************************************************************/

#include <x3d/ScriptNode.h>
#include <x3d/Event.h>

using namespace CyberX3D;

////////////////////////////////////////////////
// ScriptNode::ScriptNode
////////////////////////////////////////////////

ScriptNode::ScriptNode() 
{
	setHeaderFlag(false);
	setType(SCRIPT_NODE);

	// directOutput exposed field
	directOutputField = new SFBool(false);
	Node::addField(directOutputFieldString, directOutputField);

	// directOutput exposed field
	mustEvaluateField = new SFBool(false);
	Node::addField(mustEvaluateFieldString, mustEvaluateField);

	// url exposed field
	urlField = new MFString();
	addExposedField(urlFieldString, urlField);

	// Clear Java object
#if defined(CX3D_SUPPORT_JSAI)
	mpJScriptNode = NULL;
#endif
}


////////////////////////////////////////////////
// ScriptNode::~ScriptNode
////////////////////////////////////////////////

ScriptNode::~ScriptNode() 
{
#if defined(CX3D_SUPPORT_JSAI)
	if (mpJScriptNode)
		delete mpJScriptNode;
#endif
}

////////////////////////////////////////////////
// DirectOutput
////////////////////////////////////////////////

SFBool *ScriptNode::getDirectOutputField() const
{
	if (isInstanceNode() == false)
		return directOutputField;
	return (SFBool *)getField(directOutputFieldString);
}

void ScriptNode::setDirectOutput(bool  value) 
{
	getDirectOutputField()->setValue(value);
}

void ScriptNode::setDirectOutput(int value) 
{
	setDirectOutput(value ? true : false);
}

bool ScriptNode::getDirectOutput() const
{
	return getDirectOutputField()->getValue();
}

////////////////////////////////////////////////
// MustEvaluate
////////////////////////////////////////////////

SFBool *ScriptNode::getMustEvaluateField() const
{
	if (isInstanceNode() == false)
		return mustEvaluateField;
	return (SFBool *)getField(mustEvaluateFieldString);
}

void ScriptNode::setMustEvaluate(bool  value) 
{
	getMustEvaluateField()->setValue(value);
}

void ScriptNode::setMustEvaluate(int value) 
{
	setMustEvaluate(value ? true : false);
}

bool ScriptNode::getMustEvaluate() const
{
	return getMustEvaluateField()->getValue();
}

////////////////////////////////////////////////
// Url
////////////////////////////////////////////////

MFString *ScriptNode::getUrlField() const
{
	if (isInstanceNode() == false)
		return urlField;
	return (MFString *)getExposedField(urlFieldString);
}

void ScriptNode::addUrl(const char * value) 
{
	getUrlField()->addValue(value);
}

int ScriptNode::getNUrls() const
{
	return getUrlField()->getSize();
}

const char *ScriptNode::getUrl(int index) const
{
	return getUrlField()->get1Value(index);
}

void ScriptNode::setUrl(int index, const char *urlString) 
{
	getUrlField()->set1Value(index, urlString);
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

ScriptNode *ScriptNode::next() const
{
	return (ScriptNode *)Node::next(getType());
}

ScriptNode *ScriptNode::nextTraversal() const
{
	return (ScriptNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	virtual function
////////////////////////////////////////////////

bool ScriptNode::isChildNodeType(Node *node) const
{
	return false;
}

void ScriptNode::update() 
{
}

////////////////////////////////////////////////
//	output
////////////////////////////////////////////////

void ScriptNode::outputContext(std::ostream &printStream, const char *indentString) const
{
	SFBool *directOutput = getDirectOutputField();
	SFBool *mustEvaluate = getMustEvaluateField();

	printStream << indentString << "\t" << "directOutput " << directOutput << std::endl;
	printStream << indentString << "\t" << "mustEvaluate " << mustEvaluate << std::endl;

	if (0 < getNUrls()) {
		MFString *url = getUrlField();
		printStream << indentString << "\t" << "url [" << std::endl;
		url->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t" << "]" << std::endl;
	}
		
	int	n;

	for (n=0; n<getNEventIn(); n++) {
		Field *field = getEventIn(n);
		printStream << indentString << "\t" << "eventIn " << field->getTypeName() << " " << ((field->getName() && strlen(field->getName())) ? field->getName() : "NONE") << std::endl;
	}

	for (n=0; n<getNFields(); n++) {
		Field *field = getField(n);
		String fieldName(field->getName());
		if (fieldName.compareTo(directOutputFieldString) != 0 && fieldName.compareTo(mustEvaluateFieldString) != 0) {
			if (field->getType() == fieldTypeSFNode) {
				Node	*node = ((SFNode *)field)->getValue();
				const char	*nodeName = NULL;
				if (node)
					nodeName = node->getName();
				if (nodeName && strlen(nodeName))
					printStream << indentString << "\t" << "field " << "SFNode" << " " << ((field->getName() && strlen(field->getName())) ? field->getName() : "NONE") << " USE " << nodeName << std::endl;
				else
					printStream << indentString << "\t" << "field " << "SFNode" << " " << ((field->getName() && strlen(field->getName())) ? field->getName() : "NONE") << " NULL" << std::endl;
			}
			else
				printStream << indentString << "\t" << "field " << field->getTypeName() << " " << ((field->getName() && strlen(field->getName())) ? field->getName() : "NONE") << " " << field << std::endl;
		}
	}

	for (n=0; n<getNEventOut(); n++) {
		Field *field = getEventOut(n);
		printStream << indentString << "\t" << "eventOut " << field->getTypeName() << " " << ((field->getName() && strlen(field->getName())) ? field->getName() : "NONE") << std::endl;
	}
}

////////////////////////////////////////////////
// ScriptNode::initialize
////////////////////////////////////////////////

void ScriptNode::initialize() 
{
#if defined(CX3D_SUPPORT_JSAI)
	if (!isInitialized()) {

		if (mpJScriptNode) {
			delete mpJScriptNode;
			mpJScriptNode = NULL;
		}

		JScript *sjnode = new JScript(this);
	
		assert(sjnode);

		if (sjnode->isOK()) {
			mpJScriptNode = sjnode;
		}
		else
			delete sjnode;

		setInitialized(true);
	}

	if (mpJScriptNode) {
		mpJScriptNode->setValue(this);
		mpJScriptNode->initialize();
		mpJScriptNode->getValue(this);
	}

#endif
}

////////////////////////////////////////////////
// ScriptNode::initialize
////////////////////////////////////////////////

void ScriptNode::uninitialize() 
{
	setInitialized(false);

#if defined(CX3D_SUPPORT_JSAI)

	if (hasScript()) {
		JScript *jscript = getJavaNode();
		jscript->setValue(this);
		jscript->shutdown();
		jscript->getValue(this);
	}

#endif
}

////////////////////////////////////////////////
// ScriptNode::update
////////////////////////////////////////////////

void ScriptNode::update(Field *eventInField) {

#if defined(CX3D_SUPPORT_JSAI)

	if (hasScript()) {

		JScript *jscript = getJavaNode();

		jscript->setValue(this);

		Event event(eventInField);
		jscript->processEvent(&event);

		jscript->getValue(this);

		int nEventOut = getNEventOut();
		for (int n=0; n<nEventOut; n++) {
			Field *field = getEventOut(n);
			sendEvent(field);
		}
	}

#endif

}

////////////////////////////////////////////////
// ScriptNode::updateFields
////////////////////////////////////////////////

void ScriptNode::updateFields() {

#if defined(CX3D_SUPPORT_JSAI)
	if (hasScript()) {
		JScript *jscript = getJavaNode();
		jscript->setValue(this);
		jscript->processEvent(NULL);
		jscript->getValue(this);
	}
#endif

}









