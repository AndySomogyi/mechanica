/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	PROTO.cpp
*
******************************************************************/

#include <x3d/PROTO.h>
#include <x3d/VRML97Fields.h>
#include <cassert>

#define FIELD_SEPARATOR_TOKENS	" \t\n"
#define MFIELD_SEPARATOR_TOKENS	" \t\n"
#define PROTO_IGONRE_TOKENS			" \t"
#define PROTO_SEPARATOR_TOKENS	"{}[]\n"

using namespace CyberX3D;

////////////////////////////////////////////////
//	Parse Functions
////////////////////////////////////////////////


static int tokeniser(const char * str, int start_pos, int str_len, int &next_pos, int &len, bool brace_chomp) {

  next_pos = -1;
  len = 0;

  if (start_pos < 0) return 1;
  if (start_pos >= str_len) return 1;

  int pos = start_pos;
  char c = str[pos];

  int square_brackets = 0;
  int curly_brackets = 0;

  while (c != '\0') {
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == ',') {
        // Handle white space
        if (next_pos != -1 && curly_brackets == 0 && square_brackets == 0)  {
          len = pos - next_pos;
          return 0;
        }
    } else if (c == '\'') { // Loop until next single quote is found
      if (next_pos == -1) next_pos = pos;
        c = str[++pos];
      while (c != '\0' && c != '\'') {
        c = str[++pos];
      }
      len = pos - next_pos;
      if (c == '\0') {
        return 1;
      }
    } else if (c == '\"')  { // Loop until next double quote is found
      if (next_pos == -1) next_pos = pos;
        c = str[++pos];
      while (c != '\0' && c != '\"') {
        c = str[++pos];
      }
      len = pos - next_pos;
      if (c == '\0') {
        return 1;
      }
    } else if (c == '{' || c == '}' || c == '[' || c == ']') {
     if (brace_chomp) {
        if (next_pos == -1 || square_brackets || curly_brackets) {
          if (next_pos == -1) next_pos = pos;
          if (c == '{') ++curly_brackets;
          else if (c == '[') ++square_brackets;
          else if (c == '}') --curly_brackets;
          else if (c == ']') --square_brackets;
      } else {
        len = pos - next_pos;
        return 0;
        }
      } else {
        if (next_pos == -1) {
          next_pos = pos;
          len = 1;
          return 0;
      } else {
        len = pos - next_pos;
        return 0;
        }
      }
    } else {
      if (next_pos == -1) {
        next_pos = pos;
      }
    }
    c = str[++pos];
  }
  if (next_pos != -1) {
    len = pos - next_pos;
  }
  return 0;
}

// Small datastructure to store tokeniser state info
typedef struct {
  int str_len;
  int len;
  int buf_len;
  int next_pos;
  int pos;
  const char * str;
  char * buf;
  bool brace_chomp;
} TokenData;

// Convience function to use above function and make a copy of the string
const char * nextToken(TokenData *tk) {
  assert(tk != NULL);

  // Ensure we have a token to return
  if (tk->pos == -1) return NULL;
  if (tk->str_len == 0) return NULL;
  if (tokeniser(tk->str, tk->pos, tk->str_len, tk->next_pos, tk->len, tk->brace_chomp)) return NULL;
  if (tk->len == 0) return NULL;
 
  // Ensure buffer has enough memory
  if (tk->buf_len < tk->len + 1) {
    tk->buf_len = tk->len + 1;
    tk->buf = (char*)realloc(tk->buf, tk->buf_len * sizeof(char));
  }

  // Copy token into buffer
  strncpy(tk->buf, &tk->str[tk->next_pos], tk->len);
  tk->buf[tk->len] = '\0';

  // Check state
  if (tk->next_pos != -1) tk->pos = tk->next_pos + tk->len;
  else tk->pos = -1;

  return tk->buf;
}


static bool IsFieldAccessTypeString(const char *token)
{
	if (token == NULL)
		return false;

	if (strcmp(token, "field") == 0)
		return true;
	if (strcmp(token, "exposedField") == 0)
		return true;
	if (strcmp(token, "eventIn") == 0)
		return true;
	if (strcmp(token, "eventOut") == 0)
		return true;

	return false;
}

static int GetFieldTypeFromString(char *typeString)
{
	SFBool	field;
	field.setType(typeString);
	return field.getType();
}	

static bool IsTokenChar(char c, char *tokenChars) 
{
	if (tokenChars == NULL)
		return false;

	int tokenLen = strlen(tokenChars);
	for (int n=0; n<tokenLen; n++) {
		if  (c == tokenChars[n])
			return true;
	}

	return false;
}

static bool IsTokenChar(char c) 
{
	if ('a' <= c && c <= 'z')
		return true;
	if ('A' <= c && c <= 'Z')
		return true;
	if ('0' <= c && c <= '9')
		return true;
	if ('_' == c)
		return true;
	return false;
}

static char *GetFirstFieldToken(char *fieldString)
{
	return strtok(fieldString, FIELD_SEPARATOR_TOKENS);
}

static char *GetNextFieldToken()
{
	return strtok(NULL, FIELD_SEPARATOR_TOKENS);
}

static char *GetNextMFieldToken()
{
	return strtok(NULL, MFIELD_SEPARATOR_TOKENS);
}

static bool isTokenEOL(char *value) 
{
	if (value == NULL)
		return false;
	if (strlen(value) != 1)
		return false;
	if (value[0] == '\n')
		return true;
	return false;
}

static char *currentStringPos = NULL;

static char *GetStringToken(
char	*string, 
char	*ignoreToken,
char	*separatorToken,
char	*buffer)
{
	if (string == NULL)
		string = currentStringPos;

	int stringLen = (int)strlen(string);
	int tokenLen = 0;

	int pos = 0;
	while (pos < stringLen && IsTokenChar(string[pos], ignoreToken)) 
		pos++;

	int startPos = pos;

	for (;pos < stringLen; pos++) {
		if (IsTokenChar(string[pos], ignoreToken) == true)
			break;
		if (IsTokenChar(string[pos], separatorToken) == true) {
			if (tokenLen == 0)
				tokenLen = 1;
			break;
		}
		tokenLen++;
	}
	
	if (tokenLen == 0)
		return NULL;

	strncpy(buffer, string + startPos, tokenLen);
	buffer[tokenLen] = '\0';
	currentStringPos = string + (pos + 1);

	return buffer;
}

////////////////////////////////////////////////
//	PROTO
////////////////////////////////////////////////

PROTO::PROTO(const char *name, const char *string, const char *fieldString)
{
	setName(name);
	setString(string);
	addDefaultFields(fieldString);
}

PROTO::~PROTO(void)
{
}

void PROTO::setName(const char *name)
{
	mName.setValue(name);
}

const char *PROTO::getName(void) const
{
	return mName.getValue();
}

void PROTO::setString(const char *string) 
{
	mString.setValue(string);
}

const char *PROTO::getString() const
{
	return mString.getValue();
}

void PROTO::addDefaultField(Field *field)
{
	mDefaultFieldVector.addElement(field);
}

void PROTO::addField(Field *field) 
{
	mFieldVector.addElement(field);
}

int PROTO::getNDefaultFields() const
{
	return mDefaultFieldVector.size();
}

int PROTO::getNFields() const
{
	return mFieldVector.size();
}

Field *PROTO::getDefaultField(int n) const
{
	return (Field *)mDefaultFieldVector.elementAt(n);
}

Field *PROTO::getField(int n) const
{
	return (Field *)mFieldVector.elementAt(n);
}

void PROTO::addDefaultFields(const char *fieldString)
{
	deleteDefaultFields();
	addFieldValues(fieldString, 1);
}

void PROTO::addFields(const char *fieldString) 
{
	deleteFields(); 
	addFieldValues(fieldString, 0);
}

void PROTO::deleteDefaultFields(void) 
{
	mDefaultFieldVector.removeAllElements();
}

void PROTO::deleteFields(void) 
{
	mFieldVector.removeAllElements();
}

void PROTO::addFieldValues(
const char		*fieldString, 
int			bDefaultField)
{
	// TODO: Possible problem with limited width strings. Perhaps use std::string?
	char	fieldAccessType[256];
	char	fieldTypeName[256];
	char	fieldName[256];

	bool	mustReadNextToken;

	const char *token;
	String fieldToken;
	String tokenStr;

	// Initialise tokeniser data
	TokenData td;
	td.str = fieldString;
	td.str_len = strlen(fieldString);
	td.pos = 0;
	// Create an initial buffer of 256 chars
	td.buf = (char*)malloc(256);
	td.buf_len = 256;
	td.brace_chomp = true;

	token = nextToken(&td);

	while( token != NULL ) {


		mustReadNextToken = true;

		if (bDefaultField) {
			sscanf(token, "%255s", fieldAccessType);
			if (IsFieldAccessTypeString(fieldAccessType) == false) { 
 				// This leads to an infinite loop. Lets just ignore this....
				// continue;
			} else {
				/* Get field type */
				token = nextToken(&td);
				assert(token != NULL);
				sscanf(token, "%255s", fieldTypeName);
				token = nextToken(&td);
			}

		}
		else {
			fieldTypeName[0] = '\0';
		}

		/* Get field name */
		sscanf(token, "%255s", fieldName);

		int fieldType;
		if (bDefaultField) {
			fieldType = GetFieldTypeFromString(fieldTypeName);
		} else {
			fieldType = getFieldType(fieldName);
		}

		Field *field = NULL;


		{
			int	bigBracket = 0;
			int	smallBracket = 0;
			bool emptyField = false;

			fieldToken = "";
				 
			token = nextToken(&td);
			while (token != NULL) {

				if (bDefaultField) {
					if (IsFieldAccessTypeString(token) == true) {
						mustReadNextToken = false;
						break;
					}
				} else {
					if (getFieldType(token) != fieldTypeNone) {
						mustReadNextToken = false;
						break;
					}
				}

				fieldToken.append(token);
				fieldToken.append(" ");

				token = nextToken(&td);
			}

			if (fieldToken.length() == 0)
				fieldToken.append("[]");

			const char *fieldValue = fieldToken.getValue();
			field = new MFString();
			((MFString *)field)->addValue(fieldValue);

			// std::cout << "Result: " << fieldTypeName << " " << fieldName << " " << fieldValue << std::endl;
		}

		//assert(field);

		if (field) {
			field->setName(fieldName);
			if (bDefaultField)
				addDefaultField(field);
			else
				addField(field);
		}

		if (mustReadNextToken == true) {
			token = nextToken(&td);

		}
	}

	free(td.buf);
}

void PROTO::getString(String  &returnBuffer) const
{
	char tokenBuffer[512];

	returnBuffer.clear();

	const char *string = getString();
	if (!string || !strlen(string))
		return;

	char *defaultString = new char[strlen(string)+1];
	strcpy(defaultString, string);

	const char *token;
	String fieldToken;
	String tokenStr;

	// Initialise tokeniser data
	TokenData td;
	td.str = defaultString;
	td.str_len = strlen(defaultString);
	td.pos = 0;
	// Create an initial buffer of 256 chars
	td.buf = (char*)malloc(256);
	td.buf_len = 256;
	td.brace_chomp = false;

	token = nextToken(&td);

	while( token != NULL ) {
		if (!strcmp(token, "IS")) {
			token = nextToken(&td);

			Field *field = getField(token);
			if (field) {
				String fieldValue;
				field->getValue(fieldValue);
				returnBuffer.append(fieldValue.getValue());
				returnBuffer.append("\n");
			}
			else {
				int bufferLen = returnBuffer.length();
				for (int n=(bufferLen-1-1); 0 <= n; n--) {
					int c = returnBuffer.charAt(n);
					if (IsTokenChar(c) == false)
						break;
				}
			}
		}
		else {
			returnBuffer.append(token);
			returnBuffer.append(" ");
		}
		
		token = nextToken(&td);
	}

	// std::cout << "==== DEFAULT STRING ====" << std::endl;
	// std::cout << defaultString << std::endl;
	// std::cout << "==== REPLACE STRING ====" << std::endl;
	// std::cout << returnBuffer.getValue() << std::endl;

	delete[] defaultString;
}

Field *PROTO::getField(const char *name) const
{
	Field	*field;
	int		n;

	int nField = getNFields();
	for (n = 0; n<nField; n++) {
		field = getField(n);
		if (!strcmp(field->getName(), name))
			return field;
	}

	int nDefaultField = getNDefaultFields();
	for (n = 0; n < nDefaultField; ++n) {
		field = getDefaultField(n);
		if (!strcmp(field->getName(), name))
			return field;
	}

	return NULL;
}

int PROTO::getFieldType(const char *name) const
{
	int nDefaultField = getNDefaultFields();
	for (int n = 0; n < nDefaultField; ++n) {
		Field *field = getDefaultField(n);
		if (!strcmp(field->getName(), name))
			return field->getType();
	}
	return fieldTypeNone;
}
