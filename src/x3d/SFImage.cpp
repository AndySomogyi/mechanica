/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	SFImage.cpp
*
******************************************************************/

#include <x3d/SFImage.h>

using namespace CyberX3D;

SFImage::SFImage() 
{
	setType(fieldTypeSFImage);
	InitializeJavaIDs();
}

void SFImage::InitializeJavaIDs() 
{
#if defined(CX3D_SUPPORT_JSAI)
	setJavaIDs();
#endif
}

void SFImage::addValue(int value) 
{
	SFInt32 *sfvalue = new SFInt32(value);
	add(sfvalue);
}

void SFImage::addValue(SFInt32 *sfvalue) 
{
	add(sfvalue);
}

void SFImage::addValue(const char *value) 
{
	SFInt32 *field = new SFInt32();
	field->setValue(value);
	add(field);
}

void SFImage::insertValue(int index, int value) 
{
	SFInt32 *sfvalue = new SFInt32(value);
	insert(sfvalue, index);
}

int SFImage::get1Value(int index) const
{
	SFInt32 *sfvalue = (SFInt32 *)getObject(index);
	return sfvalue->getValue();
}

void SFImage::set1Value(int index, int value) 
{
	SFInt32 *sfvalue = (SFInt32 *)getObject(index);
	sfvalue->setValue(value);
}

void SFImage::setValue(MField *mfield) 
{
}

////////////////////////////////////////////////
//	Output
////////////////////////////////////////////////

void SFImage::outputContext(std::ostream& printStream, const char *indentString) const
{
	int	nOutput = 0;
	printStream << indentString;
  /// write the head of image in decimal system.
  int nStart = getSize ();
  if (nStart > 3) {
    for (int n=0; n<3; n++)
      printStream << (unsigned int)get1Value(n) << " ";
    nStart = 3;
    }
  else
    nStart = 0;
  /// write body of image in hexadecimal system.
  printStream.setf (std::ios_base::hex, std::ios_base::basefield);
  printStream.setf (std::ios_base::showbase);
	for (int n=nStart; n<getSize(); n++) {
		printStream << (unsigned int)get1Value(n) << " ";
		nOutput++;
		if (32 < nOutput) {
			printStream << std::endl;
			printStream << indentString;
			nOutput = 0;
		}
	}
	printStream << std::endl;
  printStream.setf (std::ios_base::dec, std::ios_base::hex);
}

////////////////////////////////////////////////
//	toString
////////////////////////////////////////////////

void SFImage::setValue(const char *value) 
{
}

const char *SFImage::getValue(char *buffer, int bufferLen) const
{
	buffer[0] = '\0';
	return buffer;
}

////////////////////////////////////////////////
//	Java
////////////////////////////////////////////////

#if defined(CX3D_SUPPORT_JSAI)

int			SFImage::mInit = 0;

jclass		SFImage::mFieldClassID = 0;
jclass		SFImage::mConstFieldClassID = 0;

jmethodID	SFImage::mInitMethodID = 0;
jmethodID	SFImage::mSetValueMethodID = 0;
jmethodID	SFImage::mGetValueMethodID = 0;
jmethodID	SFImage::mSetNameMethodID = 0;

jmethodID	SFImage::mConstInitMethodID = 0;
jmethodID	SFImage::mConstSetValueMethodID = 0;
jmethodID	SFImage::mConstGetValueMethodID = 0;
jmethodID	SFImage::mConstSetNameMethodID = 0;

////////////////////////////////////////////////
//	SFImage::setJavaIDs
////////////////////////////////////////////////

void SFImage::setJavaIDs() {

	if (!mInit) {
		JNIEnv *jniEnv = getJniEnv();

		if (jniEnv == NULL)
			return;

		// Class IDs
		mFieldClassID		= jniEnv->FindClass("vrml/field/SFImage");
		mConstFieldClassID	= jniEnv->FindClass("vrml/field/ConstSFImage");

		assert(mFieldClassID && mConstFieldClassID);

		// MethodIDs
		jclass classid		= getFieldID();
		mInitMethodID		= jniEnv->GetMethodID(classid, "<init>", "()V");
		mSetNameMethodID	= jniEnv->GetMethodID(classid, "setName", "(Ljava/lang/String;)V");

		assert(mInitMethodID && mSetNameMethodID);

		// Const MethodIDs
		classid = getConstFieldID();
		mConstInitMethodID		= jniEnv->GetMethodID(classid, "<init>", "()V");
		mConstSetNameMethodID	= jniEnv->GetMethodID(classid, "setName", "(Ljava/lang/String;)V");

		assert(mConstInitMethodID && mConstSetNameMethodID);

		mInit = 1;
	}
}

////////////////////////////////////////////////
//	SFImage::toJavaObject
////////////////////////////////////////////////

jobject SFImage::toJavaObject(int bConstField) {
	JNIEnv		*jniEnv			= getJniEnv();
	jclass		classid			= bConstField ? getConstFieldID() : getFieldID();
	jmethodID	initMethod		= bConstField ? getConstInitMethodID() : getInitMethodID();
	jobject		eventField		= jniEnv->NewObject(classid, initMethod);
	jmethodID	setNameMethod	= bConstField ? getConstSetNameMethodID() : getSetNameMethodID();

	char		*fieldName		= getName();
	jstring		jfieldName		= NULL;
	if (fieldName && strlen(fieldName))
		jfieldName = jniEnv->NewStringUTF(getName());
	jniEnv->CallVoidMethod(eventField, setNameMethod, jfieldName);
	if (jfieldName)
		jniEnv->DeleteLocalRef(jfieldName);
		
	return eventField;
}

////////////////////////////////////////////////
//	SFImage::setValue
////////////////////////////////////////////////

void SFImage::setValue(jobject field, int bConstField) {
}

////////////////////////////////////////////////
//	SFImage::getValue
////////////////////////////////////////////////

void SFImage::getValue(jobject field, int bConstField) {
}

#endif
