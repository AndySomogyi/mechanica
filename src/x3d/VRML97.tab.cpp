/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton implementation for Bison's Yacc-like parsers in C

   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.3"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Using locations.  */
#define YYLSP_NEEDED 0



/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     NUMBER = 258,
     FLOAT = 259,
     STRING = 260,
     NAME = 261,
     ANCHOR = 262,
     APPEARANCE = 263,
     AUDIOCLIP = 264,
     BACKGROUND = 265,
     BILLBOARD = 266,
     BOX = 267,
     COLLISION = 268,
     COLOR = 269,
     COLOR_INTERP = 270,
     COORDINATE = 271,
     COORDINATE_INTERP = 272,
     CYLINDER_SENSOR = 273,
     NULL_STRING = 274,
     CONE = 275,
     CUBE = 276,
     CYLINDER = 277,
     DIRECTIONALLIGHT = 278,
     FONTSTYLE = 279,
     ERROR = 280,
     EXTRUSION = 281,
     ELEVATION_GRID = 282,
     FOG = 283,
     INLINE = 284,
     MOVIE_TEXTURE = 285,
     NAVIGATION_INFO = 286,
     PIXEL_TEXTURE = 287,
     GROUP = 288,
     INDEXEDFACESET = 289,
     INDEXEDLINESET = 290,
     S_INFO = 291,
     LOD = 292,
     MATERIAL = 293,
     NORMAL = 294,
     POSITION_INTERP = 295,
     PROXIMITY_SENSOR = 296,
     SCALAR_INTERP = 297,
     SCRIPT = 298,
     SHAPE = 299,
     SOUND = 300,
     SPOTLIGHT = 301,
     SPHERE_SENSOR = 302,
     TEXT = 303,
     TEXTURE_COORDINATE = 304,
     TEXTURE_TRANSFORM = 305,
     TIME_SENSOR = 306,
     SWITCH = 307,
     TOUCH_SENSOR = 308,
     VIEWPOINT = 309,
     VISIBILITY_SENSOR = 310,
     WORLD_INFO = 311,
     NORMAL_INTERP = 312,
     ORIENTATION_INTERP = 313,
     POINTLIGHT = 314,
     POINTSET = 315,
     SPHERE = 316,
     PLANE_SENSOR = 317,
     TRANSFORM = 318,
     S_CHILDREN = 319,
     S_PARAMETER = 320,
     S_URL = 321,
     S_MATERIAL = 322,
     S_TEXTURETRANSFORM = 323,
     S_TEXTURE = 324,
     S_LOOP = 325,
     S_STARTTIME = 326,
     S_STOPTIME = 327,
     S_GROUNDANGLE = 328,
     S_GROUNDCOLOR = 329,
     S_SPEED = 330,
     S_AVATAR_SIZE = 331,
     S_BACKURL = 332,
     S_BOTTOMURL = 333,
     S_FRONTURL = 334,
     S_LEFTURL = 335,
     S_RIGHTURL = 336,
     S_TOPURL = 337,
     S_SKYANGLE = 338,
     S_SKYCOLOR = 339,
     S_AXIS_OF_ROTATION = 340,
     S_COLLIDE = 341,
     S_COLLIDETIME = 342,
     S_PROXY = 343,
     S_SIDE = 344,
     S_AUTO_OFFSET = 345,
     S_DISK_ANGLE = 346,
     S_ENABLED = 347,
     S_MAX_ANGLE = 348,
     S_MIN_ANGLE = 349,
     S_OFFSET = 350,
     S_BBOXSIZE = 351,
     S_BBOXCENTER = 352,
     S_VISIBILITY_LIMIT = 353,
     S_AMBIENT_INTENSITY = 354,
     S_NORMAL = 355,
     S_TEXCOORD = 356,
     S_CCW = 357,
     S_COLOR_PER_VERTEX = 358,
     S_CREASE_ANGLE = 359,
     S_NORMAL_PER_VERTEX = 360,
     S_XDIMENSION = 361,
     S_XSPACING = 362,
     S_ZDIMENSION = 363,
     S_ZSPACING = 364,
     S_BEGIN_CAP = 365,
     S_CROSS_SECTION = 366,
     S_END_CAP = 367,
     S_SPINE = 368,
     S_FOG_TYPE = 369,
     S_VISIBILITY_RANGE = 370,
     S_HORIZONTAL = 371,
     S_JUSTIFY = 372,
     S_LANGUAGE = 373,
     S_LEFT2RIGHT = 374,
     S_TOP2BOTTOM = 375,
     IMAGE_TEXTURE = 376,
     S_SOLID = 377,
     S_KEY = 378,
     S_KEYVALUE = 379,
     S_REPEAT_S = 380,
     S_REPEAT_T = 381,
     S_CONVEX = 382,
     S_BOTTOM = 383,
     S_PICTH = 384,
     S_COORD = 385,
     S_COLOR_INDEX = 386,
     S_COORD_INDEX = 387,
     S_NORMAL_INDEX = 388,
     S_MAX_POSITION = 389,
     S_MIN_POSITION = 390,
     S_ATTENUATION = 391,
     S_APPEARANCE = 392,
     S_GEOMETRY = 393,
     S_DIRECT_OUTPUT = 394,
     S_MUST_EVALUATE = 395,
     S_MAX_BACK = 396,
     S_MIN_BACK = 397,
     S_MAX_FRONT = 398,
     S_MIN_FRONT = 399,
     S_PRIORITY = 400,
     S_SOURCE = 401,
     S_SPATIALIZE = 402,
     S_BERM_WIDTH = 403,
     S_CHOICE = 404,
     S_WHICHCHOICE = 405,
     S_FONTSTYLE = 406,
     S_LENGTH = 407,
     S_MAX_EXTENT = 408,
     S_ROTATION = 409,
     S_SCALE = 410,
     S_CYCLE_INTERVAL = 411,
     S_FIELD_OF_VIEW = 412,
     S_JUMP = 413,
     S_TITLE = 414,
     S_TEXCOORD_INDEX = 415,
     S_HEADLIGHT = 416,
     S_TOP = 417,
     S_BOTTOMRADIUS = 418,
     S_HEIGHT = 419,
     S_POINT = 420,
     S_STRING = 421,
     S_SPACING = 422,
     S_TYPE = 423,
     S_RADIUS = 424,
     S_ON = 425,
     S_INTENSITY = 426,
     S_COLOR = 427,
     S_DIRECTION = 428,
     S_SIZE = 429,
     S_FAMILY = 430,
     S_STYLE = 431,
     S_RANGE = 432,
     S_CENTER = 433,
     S_TRANSLATION = 434,
     S_LEVEL = 435,
     S_DIFFUSECOLOR = 436,
     S_SPECULARCOLOR = 437,
     S_EMISSIVECOLOR = 438,
     S_SHININESS = 439,
     S_TRANSPARENCY = 440,
     S_VECTOR = 441,
     S_POSITION = 442,
     S_ORIENTATION = 443,
     S_LOCATION = 444,
     S_CUTOFFANGLE = 445,
     S_WHICHCHILD = 446,
     S_IMAGE = 447,
     S_SCALEORIENTATION = 448,
     S_DESCRIPTION = 449,
     SFBOOL = 450,
     SFFLOAT = 451,
     SFINT32 = 452,
     SFTIME = 453,
     SFROTATION = 454,
     SFNODE = 455,
     SFCOLOR = 456,
     SFIMAGE = 457,
     SFSTRING = 458,
     SFVEC2F = 459,
     SFVEC3F = 460,
     MFBOOL = 461,
     MFFLOAT = 462,
     MFINT32 = 463,
     MFTIME = 464,
     MFROTATION = 465,
     MFNODE = 466,
     MFCOLOR = 467,
     MFIMAGE = 468,
     MFSTRING = 469,
     MFVEC2F = 470,
     MFVEC3F = 471,
     FIELD = 472,
     EVENTIN = 473,
     EVENTOUT = 474,
     USE = 475,
     S_VALUE_CHANGED = 476
   };
#endif
/* Tokens.  */
#define NUMBER 258
#define FLOAT 259
#define STRING 260
#define NAME 261
#define ANCHOR 262
#define APPEARANCE 263
#define AUDIOCLIP 264
#define BACKGROUND 265
#define BILLBOARD 266
#define BOX 267
#define COLLISION 268
#define COLOR 269
#define COLOR_INTERP 270
#define COORDINATE 271
#define COORDINATE_INTERP 272
#define CYLINDER_SENSOR 273
#define NULL_STRING 274
#define CONE 275
#define CUBE 276
#define CYLINDER 277
#define DIRECTIONALLIGHT 278
#define FONTSTYLE 279
#define ERROR 280
#define EXTRUSION 281
#define ELEVATION_GRID 282
#define FOG 283
#define INLINE 284
#define MOVIE_TEXTURE 285
#define NAVIGATION_INFO 286
#define PIXEL_TEXTURE 287
#define GROUP 288
#define INDEXEDFACESET 289
#define INDEXEDLINESET 290
#define S_INFO 291
#define LOD 292
#define MATERIAL 293
#define NORMAL 294
#define POSITION_INTERP 295
#define PROXIMITY_SENSOR 296
#define SCALAR_INTERP 297
#define SCRIPT 298
#define SHAPE 299
#define SOUND 300
#define SPOTLIGHT 301
#define SPHERE_SENSOR 302
#define TEXT 303
#define TEXTURE_COORDINATE 304
#define TEXTURE_TRANSFORM 305
#define TIME_SENSOR 306
#define SWITCH 307
#define TOUCH_SENSOR 308
#define VIEWPOINT 309
#define VISIBILITY_SENSOR 310
#define WORLD_INFO 311
#define NORMAL_INTERP 312
#define ORIENTATION_INTERP 313
#define POINTLIGHT 314
#define POINTSET 315
#define SPHERE 316
#define PLANE_SENSOR 317
#define TRANSFORM 318
#define S_CHILDREN 319
#define S_PARAMETER 320
#define S_URL 321
#define S_MATERIAL 322
#define S_TEXTURETRANSFORM 323
#define S_TEXTURE 324
#define S_LOOP 325
#define S_STARTTIME 326
#define S_STOPTIME 327
#define S_GROUNDANGLE 328
#define S_GROUNDCOLOR 329
#define S_SPEED 330
#define S_AVATAR_SIZE 331
#define S_BACKURL 332
#define S_BOTTOMURL 333
#define S_FRONTURL 334
#define S_LEFTURL 335
#define S_RIGHTURL 336
#define S_TOPURL 337
#define S_SKYANGLE 338
#define S_SKYCOLOR 339
#define S_AXIS_OF_ROTATION 340
#define S_COLLIDE 341
#define S_COLLIDETIME 342
#define S_PROXY 343
#define S_SIDE 344
#define S_AUTO_OFFSET 345
#define S_DISK_ANGLE 346
#define S_ENABLED 347
#define S_MAX_ANGLE 348
#define S_MIN_ANGLE 349
#define S_OFFSET 350
#define S_BBOXSIZE 351
#define S_BBOXCENTER 352
#define S_VISIBILITY_LIMIT 353
#define S_AMBIENT_INTENSITY 354
#define S_NORMAL 355
#define S_TEXCOORD 356
#define S_CCW 357
#define S_COLOR_PER_VERTEX 358
#define S_CREASE_ANGLE 359
#define S_NORMAL_PER_VERTEX 360
#define S_XDIMENSION 361
#define S_XSPACING 362
#define S_ZDIMENSION 363
#define S_ZSPACING 364
#define S_BEGIN_CAP 365
#define S_CROSS_SECTION 366
#define S_END_CAP 367
#define S_SPINE 368
#define S_FOG_TYPE 369
#define S_VISIBILITY_RANGE 370
#define S_HORIZONTAL 371
#define S_JUSTIFY 372
#define S_LANGUAGE 373
#define S_LEFT2RIGHT 374
#define S_TOP2BOTTOM 375
#define IMAGE_TEXTURE 376
#define S_SOLID 377
#define S_KEY 378
#define S_KEYVALUE 379
#define S_REPEAT_S 380
#define S_REPEAT_T 381
#define S_CONVEX 382
#define S_BOTTOM 383
#define S_PICTH 384
#define S_COORD 385
#define S_COLOR_INDEX 386
#define S_COORD_INDEX 387
#define S_NORMAL_INDEX 388
#define S_MAX_POSITION 389
#define S_MIN_POSITION 390
#define S_ATTENUATION 391
#define S_APPEARANCE 392
#define S_GEOMETRY 393
#define S_DIRECT_OUTPUT 394
#define S_MUST_EVALUATE 395
#define S_MAX_BACK 396
#define S_MIN_BACK 397
#define S_MAX_FRONT 398
#define S_MIN_FRONT 399
#define S_PRIORITY 400
#define S_SOURCE 401
#define S_SPATIALIZE 402
#define S_BERM_WIDTH 403
#define S_CHOICE 404
#define S_WHICHCHOICE 405
#define S_FONTSTYLE 406
#define S_LENGTH 407
#define S_MAX_EXTENT 408
#define S_ROTATION 409
#define S_SCALE 410
#define S_CYCLE_INTERVAL 411
#define S_FIELD_OF_VIEW 412
#define S_JUMP 413
#define S_TITLE 414
#define S_TEXCOORD_INDEX 415
#define S_HEADLIGHT 416
#define S_TOP 417
#define S_BOTTOMRADIUS 418
#define S_HEIGHT 419
#define S_POINT 420
#define S_STRING 421
#define S_SPACING 422
#define S_TYPE 423
#define S_RADIUS 424
#define S_ON 425
#define S_INTENSITY 426
#define S_COLOR 427
#define S_DIRECTION 428
#define S_SIZE 429
#define S_FAMILY 430
#define S_STYLE 431
#define S_RANGE 432
#define S_CENTER 433
#define S_TRANSLATION 434
#define S_LEVEL 435
#define S_DIFFUSECOLOR 436
#define S_SPECULARCOLOR 437
#define S_EMISSIVECOLOR 438
#define S_SHININESS 439
#define S_TRANSPARENCY 440
#define S_VECTOR 441
#define S_POSITION 442
#define S_ORIENTATION 443
#define S_LOCATION 444
#define S_CUTOFFANGLE 445
#define S_WHICHCHILD 446
#define S_IMAGE 447
#define S_SCALEORIENTATION 448
#define S_DESCRIPTION 449
#define SFBOOL 450
#define SFFLOAT 451
#define SFINT32 452
#define SFTIME 453
#define SFROTATION 454
#define SFNODE 455
#define SFCOLOR 456
#define SFIMAGE 457
#define SFSTRING 458
#define SFVEC2F 459
#define SFVEC3F 460
#define MFBOOL 461
#define MFFLOAT 462
#define MFINT32 463
#define MFTIME 464
#define MFROTATION 465
#define MFNODE 466
#define MFCOLOR 467
#define MFIMAGE 468
#define MFSTRING 469
#define MFVEC2F 470
#define MFVEC3F 471
#define FIELD 472
#define EVENTIN 473
#define EVENTOUT 474
#define USE 475
#define S_VALUE_CHANGED 476




/* Copy the first part of user declarations.  */


/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif

#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
#line 20 "VRML97.y"
{
int		ival;
float	fval;
char	*sval;
}
/* Line 187 of yacc.c.  */
#line 545 "VRML97.tab.cpp"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */
#line 72 "VRML97.y"


#include <stdio.h>
#include <stdlib.h>

#ifndef __GNUC__
#define alloca	malloc
#endif

#include <x3d/SceneGraph.h>
#include <x3d/NodeType.h>
#include <x3d/VRML97Parser.h>
#include <x3d/VRML97ParserFunc.h>

using namespace CyberX3D;

static float gColor[3];
static float gVec2f[2];
static float gVec3f[3];
static float gRotation[4];
static int gWidth;
static int gHeight;
static int gComponents;

#define YYINITDEPTH  (1024 * 64)
#define	YYMAXDEPTH	(YYINITDEPTH * 128)

int yyerror(char *s);
int yyparse(void);
int yylex(void);



/* Line 216 of yacc.c.  */
#line 590 "VRML97.tab.cpp"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(e) ((void) (e))
#else
# define YYUSE(e) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(n) (n)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID (int i)
#else
static int
YYID (i)
    int i;
#endif
{
  return i;
}
#endif

#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef _STDLIB_H
#      define _STDLIB_H 1
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (YYID (0))
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined _STDLIB_H \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef _STDLIB_H
#    define _STDLIB_H 1
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss;
  YYSTYPE yyvs;
  };

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  YYSIZE_T yyi;				\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (YYID (0))
#  endif
# endif

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack)					\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack, Stack, yysize);				\
	Stack = &yyptr->Stack;						\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  146
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   1854

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  226
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  295
/* YYNRULES -- Number of rules.  */
#define YYNRULES  706
/* YYNRULES -- Number of states.  */
#define YYNSTATES  1187

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   476

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,   222,     2,   223,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,   224,     2,   225,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     115,   116,   117,   118,   119,   120,   121,   122,   123,   124,
     125,   126,   127,   128,   129,   130,   131,   132,   133,   134,
     135,   136,   137,   138,   139,   140,   141,   142,   143,   144,
     145,   146,   147,   148,   149,   150,   151,   152,   153,   154,
     155,   156,   157,   158,   159,   160,   161,   162,   163,   164,
     165,   166,   167,   168,   169,   170,   171,   172,   173,   174,
     175,   176,   177,   178,   179,   180,   181,   182,   183,   184,
     185,   186,   187,   188,   189,   190,   191,   192,   193,   194,
     195,   196,   197,   198,   199,   200,   201,   202,   203,   204,
     205,   206,   207,   208,   209,   210,   211,   212,   213,   214,
     215,   216,   217,   218,   219,   220,   221
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     5,     7,     9,    12,    13,    15,    17,
      19,    21,    23,    25,    27,    29,    31,    33,    35,    37,
      39,    41,    43,    45,    47,    49,    51,    53,    55,    57,
      59,    61,    63,    65,    67,    69,    71,    73,    75,    77,
      79,    81,    83,    85,    87,    89,    91,    93,    95,    97,
      99,   101,   103,   105,   107,   109,   111,   113,   115,   117,
     119,   121,   123,   125,   127,   129,   131,   133,   135,   137,
     139,   141,   143,   147,   152,   155,   156,   159,   163,   165,
     168,   170,   173,   177,   179,   182,   184,   187,   191,   193,
     196,   198,   201,   205,   207,   210,   212,   215,   219,   221,
     224,   226,   229,   233,   235,   238,   240,   243,   247,   249,
     252,   254,   257,   261,   263,   265,   268,   269,   271,   273,
     276,   279,   281,   284,   287,   290,   292,   294,   296,   301,
     304,   305,   308,   311,   314,   317,   320,   323,   326,   329,
     332,   335,   338,   340,   345,   348,   349,   351,   354,   357,
     360,   363,   366,   369,   371,   376,   379,   380,   382,   384,
     386,   388,   390,   392,   394,   396,   398,   400,   403,   406,
     409,   412,   415,   418,   421,   424,   427,   430,   432,   437,
     440,   441,   443,   446,   448,   450,   452,   457,   460,   461,
     464,   466,   471,   474,   475,   480,   483,   486,   487,   489,
     491,   494,   496,   498,   501,   504,   507,   509,   514,   517,
     518,   521,   523,   528,   531,   532,   534,   536,   539,   542,
     544,   549,   552,   553,   556,   559,   562,   565,   567,   572,
     575,   576,   578,   583,   586,   587,   590,   593,   595,   600,
     603,   604,   607,   610,   613,   616,   619,   621,   626,   629,
     630,   633,   636,   639,   642,   645,   648,   650,   655,   658,
     659,   662,   665,   668,   671,   674,   676,   681,   684,   685,
     687,   690,   693,   696,   699,   702,   705,   708,   711,   714,
     717,   720,   723,   726,   729,   732,   735,   738,   741,   744,
     746,   751,   754,   755,   757,   759,   761,   763,   766,   769,
     772,   775,   778,   781,   784,   787,   790,   793,   795,   800,
     803,   804,   807,   810,   813,   815,   820,   823,   824,   826,
     829,   832,   835,   838,   841,   844,   847,   850,   853,   856,
     860,   863,   864,   866,   868,   870,   872,   877,   880,   881,
     883,   886,   889,   892,   894,   899,   902,   903,   905,   907,
     909,   911,   914,   917,   920,   923,   926,   929,   932,   935,
     938,   941,   944,   947,   950,   953,   956,   959,   962,   965,
     968,   971,   974,   977,   979,   984,   987,   988,   991,   994,
     997,  1000,  1003,  1006,  1009,  1012,  1015,  1018,  1022,  1025,
    1026,  1028,  1031,  1033,  1035,  1037,  1042,  1045,  1046,  1048,
    1050,  1053,  1056,  1059,  1064,  1066,  1071,  1074,  1075,  1078,
    1081,  1084,  1087,  1090,  1093,  1095,  1100,  1103,  1104,  1106,
    1109,  1112,  1115,  1118,  1121,  1124,  1127,  1129,  1134,  1137,
    1138,  1140,  1142,  1145,  1148,  1151,  1154,  1157,  1159,  1164,
    1167,  1168,  1171,  1173,  1178,  1181,  1182,  1185,  1188,  1191,
    1193,  1198,  1201,  1202,  1205,  1208,  1211,  1213,  1218,  1221,
    1222,  1224,  1229,  1232,  1235,  1237,  1242,  1245,  1246,  1249,
    1252,  1255,  1258,  1261,  1263,  1268,  1271,  1272,  1275,  1278,
    1281,  1284,  1287,  1290,  1293,  1295,  1300,  1303,  1304,  1307,
    1310,  1313,  1316,  1319,  1322,  1324,  1329,  1332,  1333,  1336,
    1339,  1342,  1344,  1349,  1352,  1353,  1356,  1359,  1362,  1364,
    1369,  1372,  1373,  1376,  1379,  1382,  1384,  1389,  1392,  1393,
    1395,  1398,  1401,  1404,  1408,  1412,  1416,  1420,  1424,  1428,
    1432,  1436,  1440,  1444,  1448,  1452,  1456,  1460,  1464,  1468,
    1472,  1476,  1480,  1484,  1488,  1492,  1496,  1500,  1504,  1508,
    1512,  1516,  1520,  1524,  1528,  1532,  1536,  1540,  1544,  1548,
    1553,  1558,  1563,  1568,  1573,  1578,  1584,  1589,  1594,  1599,
    1604,  1606,  1611,  1614,  1615,  1618,  1621,  1624,  1627,  1630,
    1633,  1635,  1640,  1643,  1644,  1647,  1650,  1653,  1656,  1659,
    1662,  1665,  1668,  1671,  1674,  1677,  1680,  1683,  1685,  1690,
    1693,  1694,  1697,  1699,  1704,  1707,  1708,  1711,  1714,  1717,
    1719,  1724,  1727,  1728,  1731,  1734,  1737,  1740,  1743,  1746,
    1749,  1752,  1755,  1758,  1760,  1765,  1768,  1769,  1771,  1774,
    1779,  1782,  1784,  1789,  1792,  1793,  1795,  1797,  1800,  1803,
    1806,  1809,  1812,  1815,  1817,  1822,  1825,  1826,  1829,  1831,
    1836,  1839,  1840,  1843,  1846,  1849,  1852,  1854,  1859,  1862,
    1863,  1866,  1869,  1872,  1875,  1878,  1880,  1885,  1888,  1889,
    1892,  1894,  1899,  1902,  1903,  1905,  1908,  1911,  1914,  1917,
    1920,  1922,  1924,  1926,  1931,  1934,  1935,  1938,  1941,  1944,
    1947,  1950,  1952,  1957,  1960,  1961,  1964,  1967,  1970,  1972,
    1977,  1980,  1981,  1983,  1986,  1989,  1991
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     227,     0,    -1,   228,    -1,     1,    -1,    25,    -1,   237,
     228,    -1,    -1,   271,    -1,   298,    -1,   309,    -1,   368,
      -1,   390,    -1,   396,    -1,   482,    -1,   508,    -1,   319,
      -1,   330,    -1,   419,    -1,   423,    -1,   444,    -1,   452,
      -1,   338,    -1,   432,    -1,   473,    -1,   448,    -1,   500,
      -1,   504,    -1,   514,    -1,   302,    -1,   323,    -1,   334,
      -1,   347,    -1,   355,    -1,   381,    -1,   385,    -1,   440,
      -1,   469,    -1,   488,    -1,   313,    -1,   326,    -1,   415,
      -1,   492,    -1,   275,    -1,   400,    -1,   342,    -1,   477,
      -1,   436,    -1,   280,    -1,   234,    -1,   457,    -1,   461,
      -1,   465,    -1,   520,    -1,   294,    -1,   359,    -1,   411,
      -1,   512,    -1,   235,    -1,   236,    -1,   364,    -1,   230,
      -1,   231,    -1,   229,    -1,   233,    -1,   220,    -1,     3,
      -1,     3,    -1,     5,    -1,     4,    -1,     3,    -1,     4,
      -1,     3,    -1,   241,   241,   241,    -1,   241,   241,   241,
     241,    -1,   238,   245,    -1,    -1,   241,   241,    -1,   241,
     241,   241,    -1,   243,    -1,   248,   243,    -1,   243,    -1,
     222,   223,    -1,   222,   248,   223,    -1,   238,    -1,   250,
     238,    -1,   238,    -1,   222,   223,    -1,   222,   250,   223,
      -1,   241,    -1,   252,   241,    -1,   241,    -1,   222,   223,
      -1,   222,   252,   223,    -1,   240,    -1,   254,   240,    -1,
     240,    -1,   222,   223,    -1,   222,   254,   223,    -1,   246,
      -1,   256,   246,    -1,   246,    -1,   222,   223,    -1,   222,
     256,   223,    -1,   247,    -1,   258,   247,    -1,   247,    -1,
     222,   223,    -1,   222,   258,   223,    -1,   244,    -1,   260,
     244,    -1,   244,    -1,   222,   223,    -1,   222,   260,   223,
      -1,   224,    -1,   225,    -1,   269,   264,    -1,    -1,    65,
      -1,    66,    -1,    97,   247,    -1,    96,   247,    -1,   304,
      -1,   194,   240,    -1,   265,   255,    -1,   266,   255,    -1,
     267,    -1,   268,    -1,     7,    -1,   270,   262,   264,   263,
      -1,   273,   272,    -1,    -1,    67,    19,    -1,    67,   400,
      -1,    67,   220,    -1,    69,    19,    -1,    69,   373,    -1,
      69,   405,    -1,    69,   428,    -1,    69,   220,    -1,    68,
      19,    -1,    68,   496,    -1,    68,   220,    -1,     8,    -1,
     274,   262,   272,   263,    -1,   278,   276,    -1,    -1,    66,
      -1,   194,   240,    -1,    70,   239,    -1,   129,   241,    -1,
      71,   242,    -1,    72,   242,    -1,   277,   255,    -1,     9,
      -1,   279,   262,   276,   263,    -1,   292,   281,    -1,    -1,
      77,    -1,    78,    -1,    79,    -1,    80,    -1,    81,    -1,
      82,    -1,    73,    -1,    74,    -1,    83,    -1,    84,    -1,
     288,   253,    -1,   289,   249,    -1,   282,   255,    -1,   283,
     255,    -1,   284,   255,    -1,   285,   255,    -1,   286,   255,
      -1,   287,   255,    -1,   290,   253,    -1,   291,   249,    -1,
      10,    -1,   293,   262,   281,   263,    -1,   296,   295,    -1,
      -1,   304,    -1,    85,   247,    -1,   267,    -1,   268,    -1,
      11,    -1,   297,   262,   295,   263,    -1,   300,   299,    -1,
      -1,   174,   247,    -1,    12,    -1,   301,   262,   299,   263,
      -1,   237,   303,    -1,    -1,    64,   222,   303,   223,    -1,
      64,   237,    -1,   307,   305,    -1,    -1,    88,    -1,   304,
      -1,    86,   239,    -1,   267,    -1,   268,    -1,    88,   220,
      -1,    88,    19,    -1,   306,   237,    -1,    13,    -1,   308,
     262,   305,   263,    -1,   311,   310,    -1,    -1,   172,   249,
      -1,    14,    -1,   312,   262,   310,   263,    -1,   317,   314,
      -1,    -1,   123,    -1,   124,    -1,   315,   253,    -1,   316,
     249,    -1,    15,    -1,   318,   262,   314,   263,    -1,   321,
     320,    -1,    -1,    89,   239,    -1,   128,   239,    -1,   163,
     241,    -1,   164,   241,    -1,    20,    -1,   322,   262,   320,
     263,    -1,   165,   259,    -1,    -1,    16,    -1,   325,   262,
     324,   263,    -1,   328,   327,    -1,    -1,   315,   253,    -1,
     316,   259,    -1,    17,    -1,   329,   262,   327,   263,    -1,
     332,   331,    -1,    -1,    89,   239,    -1,   128,   239,    -1,
     162,   239,    -1,   169,   241,    -1,   164,   241,    -1,    22,
      -1,   333,   262,   331,   263,    -1,   336,   335,    -1,    -1,
      90,   239,    -1,    91,   241,    -1,    92,   239,    -1,    93,
     241,    -1,    94,   241,    -1,    95,   241,    -1,    18,    -1,
     337,   262,   335,   263,    -1,   340,   339,    -1,    -1,   170,
     239,    -1,   171,   241,    -1,   172,   243,    -1,   173,   247,
      -1,    99,   241,    -1,    23,    -1,   341,   262,   339,   263,
      -1,   345,   343,    -1,    -1,   164,    -1,   172,    19,    -1,
     172,   313,    -1,   172,   220,    -1,   100,    19,    -1,   100,
     415,    -1,   100,   220,    -1,   101,    19,    -1,   101,   492,
      -1,   101,   220,    -1,   344,   253,    -1,   102,   239,    -1,
     104,   241,    -1,   122,   239,    -1,   103,   239,    -1,   105,
     239,    -1,   106,   238,    -1,   107,   241,    -1,   108,   238,
      -1,   109,   241,    -1,    27,    -1,   346,   262,   343,   263,
      -1,   353,   348,    -1,    -1,   111,    -1,   188,    -1,   155,
      -1,   113,    -1,   110,   239,    -1,   102,   239,    -1,   127,
     239,    -1,   104,   241,    -1,   122,   239,    -1,   349,   257,
      -1,   112,   239,    -1,   350,   261,    -1,   351,   257,    -1,
     352,   259,    -1,    26,    -1,   354,   262,   348,   263,    -1,
     357,   356,    -1,    -1,   172,   243,    -1,   114,   240,    -1,
     115,   241,    -1,    28,    -1,   358,   262,   356,   263,    -1,
     362,   360,    -1,    -1,   117,    -1,   175,   240,    -1,   116,
     239,    -1,   361,   255,    -1,   118,   240,    -1,   119,   239,
      -1,   174,   241,    -1,   167,   241,    -1,   176,   240,    -1,
     120,   239,    -1,    24,   262,    -1,   363,   360,   263,    -1,
     366,   365,    -1,    -1,   304,    -1,   267,    -1,   268,    -1,
      33,    -1,   367,   262,   365,   263,    -1,   371,   369,    -1,
      -1,    66,    -1,   370,   255,    -1,   125,   239,    -1,   126,
     239,    -1,   121,    -1,   372,   262,   369,   263,    -1,   379,
     374,    -1,    -1,   131,    -1,   132,    -1,   133,    -1,   160,
      -1,   172,    19,    -1,   172,   313,    -1,   172,   220,    -1,
     130,    19,    -1,   130,   326,    -1,   130,   220,    -1,   100,
      19,    -1,   100,   415,    -1,   100,   220,    -1,   101,    19,
      -1,   101,   492,    -1,   101,   220,    -1,   102,   239,    -1,
     127,   239,    -1,   122,   239,    -1,   104,   241,    -1,   375,
     251,    -1,   103,   239,    -1,   376,   251,    -1,   377,   251,
      -1,   378,   251,    -1,   105,   239,    -1,    34,    -1,   380,
     262,   374,   263,    -1,   383,   382,    -1,    -1,   172,    19,
      -1,   172,   313,    -1,   172,   220,    -1,   130,    19,    -1,
     130,   326,    -1,   130,   220,    -1,   103,   239,    -1,   375,
     251,    -1,   376,   251,    -1,    35,   262,    -1,   384,   382,
     263,    -1,   388,   386,    -1,    -1,    66,    -1,   387,   255,
      -1,   267,    -1,   268,    -1,    29,    -1,   389,   262,   386,
     263,    -1,   394,   391,    -1,    -1,   177,    -1,   180,    -1,
     392,   253,    -1,   178,   247,    -1,   393,   237,    -1,   393,
     222,   228,   223,    -1,    37,    -1,   395,   262,   391,   263,
      -1,   398,   397,    -1,    -1,    99,   241,    -1,   181,   243,
      -1,   183,   243,    -1,   184,   241,    -1,   182,   243,    -1,
     185,   241,    -1,    38,    -1,   399,   262,   397,   263,    -1,
     403,   401,    -1,    -1,    66,    -1,    70,   239,    -1,    75,
     241,    -1,    71,   242,    -1,    72,   242,    -1,   402,   255,
      -1,   125,   239,    -1,   126,   239,    -1,    30,    -1,   404,
     262,   401,   263,    -1,   409,   406,    -1,    -1,    76,    -1,
     168,    -1,   407,   253,    -1,   161,   239,    -1,    75,   241,
      -1,   408,   255,    -1,    98,   241,    -1,    31,    -1,   410,
     262,   406,   263,    -1,   413,   412,    -1,    -1,   186,   259,
      -1,    39,    -1,   414,   262,   412,   263,    -1,   417,   416,
      -1,    -1,   315,   253,    -1,   316,   259,    -1,   221,   247,
      -1,    57,    -1,   418,   262,   416,   263,    -1,   421,   420,
      -1,    -1,   315,   253,    -1,   316,   261,    -1,   221,   244,
      -1,    58,    -1,   422,   262,   420,   263,    -1,   426,   424,
      -1,    -1,   192,    -1,   425,   222,   245,   223,    -1,   125,
     239,    -1,   126,   239,    -1,    32,    -1,   427,   262,   424,
     263,    -1,   430,   429,    -1,    -1,    90,   239,    -1,    92,
     239,    -1,   134,   246,    -1,   135,   246,    -1,    95,   247,
      -1,    62,    -1,   431,   262,   429,   263,    -1,   434,   433,
      -1,    -1,    99,   241,    -1,   136,   247,    -1,   172,   243,
      -1,   171,   241,    -1,   189,   247,    -1,   170,   239,    -1,
     169,   241,    -1,    59,    -1,   435,   262,   433,   263,    -1,
     438,   437,    -1,    -1,   172,    19,    -1,   172,   313,    -1,
     172,   220,    -1,   130,    19,    -1,   130,   326,    -1,   130,
     220,    -1,    60,    -1,   439,   262,   437,   263,    -1,   442,
     441,    -1,    -1,   315,   253,    -1,   316,   259,    -1,   221,
     247,    -1,    40,    -1,   443,   262,   441,   263,    -1,   446,
     445,    -1,    -1,   178,   247,    -1,   174,   247,    -1,    92,
     239,    -1,    41,    -1,   447,   262,   445,   263,    -1,   450,
     449,    -1,    -1,   315,   253,    -1,   316,   253,    -1,   221,
     246,    -1,    42,    -1,   451,   262,   449,   263,    -1,   455,
     453,    -1,    -1,    66,    -1,   454,   255,    -1,   139,   239,
      -1,   140,   239,    -1,   218,   195,     6,    -1,   218,   196,
       6,    -1,   218,   197,     6,    -1,   218,   198,     6,    -1,
     218,   199,     6,    -1,   218,   201,     6,    -1,   218,   202,
       6,    -1,   218,   203,     6,    -1,   218,   204,     6,    -1,
     218,   205,     6,    -1,   218,   207,     6,    -1,   218,   208,
       6,    -1,   218,   209,     6,    -1,   218,   210,     6,    -1,
     218,   212,     6,    -1,   218,   214,     6,    -1,   218,   215,
       6,    -1,   218,   216,     6,    -1,   219,   195,     6,    -1,
     219,   196,     6,    -1,   219,   197,     6,    -1,   219,   198,
       6,    -1,   219,   199,     6,    -1,   219,   201,     6,    -1,
     219,   202,     6,    -1,   219,   203,     6,    -1,   219,   204,
       6,    -1,   219,   205,     6,    -1,   219,   207,     6,    -1,
     219,   208,     6,    -1,   219,   209,     6,    -1,   219,   210,
       6,    -1,   219,   212,     6,    -1,   219,   214,     6,    -1,
     219,   215,     6,    -1,   219,   216,     6,    -1,   217,   195,
       6,   239,    -1,   217,   196,     6,   241,    -1,   217,   197,
       6,   238,    -1,   217,   198,     6,   242,    -1,   217,   199,
       6,   244,    -1,   217,   200,     6,    19,    -1,   217,   200,
       6,   220,     6,    -1,   217,   201,     6,   243,    -1,   217,
     203,     6,   240,    -1,   217,   204,     6,   246,    -1,   217,
     205,     6,   247,    -1,    43,    -1,   456,   262,   453,   263,
      -1,   459,   458,    -1,    -1,   137,    19,    -1,   137,   275,
      -1,   137,   220,    -1,   138,    19,    -1,   138,   232,    -1,
     138,   220,    -1,    44,    -1,   460,   262,   458,   263,    -1,
     463,   462,    -1,    -1,   173,   247,    -1,   171,   241,    -1,
     189,   247,    -1,   141,   241,    -1,   143,   241,    -1,   142,
     241,    -1,   144,   241,    -1,   145,   241,    -1,   146,    19,
      -1,   146,   280,    -1,   146,   405,    -1,   146,   220,    -1,
     147,   239,    -1,    45,    -1,   464,   262,   462,   263,    -1,
     467,   466,    -1,    -1,   169,   241,    -1,    61,    -1,   468,
     262,   466,   263,    -1,   471,   470,    -1,    -1,    90,   239,
      -1,    92,   239,    -1,    95,   244,    -1,    47,    -1,   472,
     262,   470,   263,    -1,   475,   474,    -1,    -1,    99,   241,
      -1,   136,   247,    -1,   148,   241,    -1,   172,   243,    -1,
     190,   241,    -1,   173,   247,    -1,   171,   241,    -1,   189,
     247,    -1,   170,   239,    -1,   169,   241,    -1,    46,    -1,
     476,   262,   474,   263,    -1,   480,   478,    -1,    -1,   149,
      -1,   479,   237,    -1,   479,   222,   228,   223,    -1,   150,
     238,    -1,    52,    -1,   481,   262,   478,   263,    -1,   486,
     483,    -1,    -1,   166,    -1,   152,    -1,   484,   255,    -1,
     151,    19,    -1,   151,   364,    -1,   151,   220,    -1,   485,
     253,    -1,   153,   241,    -1,    48,    -1,   487,   262,   483,
     263,    -1,   490,   489,    -1,    -1,   165,   257,    -1,    49,
      -1,   491,   262,   489,   263,    -1,   494,   493,    -1,    -1,
     178,   246,    -1,   154,   241,    -1,   155,   246,    -1,   179,
     246,    -1,    50,    -1,   495,   262,   493,   263,    -1,   498,
     497,    -1,    -1,   156,   242,    -1,    92,   239,    -1,    70,
     239,    -1,    71,   242,    -1,    72,   242,    -1,    51,    -1,
     499,   262,   497,   263,    -1,   502,   501,    -1,    -1,    92,
     239,    -1,    53,    -1,   503,   262,   501,   263,    -1,   506,
     505,    -1,    -1,   304,    -1,   178,   247,    -1,   154,   244,
      -1,   155,   247,    -1,   193,   244,    -1,   179,   247,    -1,
     267,    -1,   268,    -1,    63,    -1,   507,   262,   505,   263,
      -1,   510,   509,    -1,    -1,   157,   241,    -1,   158,   239,
      -1,   188,   244,    -1,   187,   247,    -1,   194,   240,    -1,
      54,    -1,   511,   262,   509,   263,    -1,   514,   513,    -1,
      -1,   178,   247,    -1,    92,   239,    -1,   174,   247,    -1,
      55,    -1,   515,   262,   513,   263,    -1,   518,   516,    -1,
      -1,    36,    -1,   517,   255,    -1,   159,   240,    -1,    56,
      -1,   519,   262,   516,   263,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   108,   108,   109,   110,   114,   115,   119,   120,   121,
     122,   123,   124,   125,   126,   130,   131,   132,   133,   134,
     135,   139,   140,   141,   142,   143,   144,   145,   149,   150,
     151,   152,   153,   154,   155,   156,   157,   158,   162,   163,
     164,   165,   166,   167,   171,   172,   173,   177,   178,   179,
     180,   181,   182,   186,   187,   188,   189,   193,   194,   195,
     196,   197,   198,   199,   200,   204,   211,   215,   222,   226,
     234,   235,   239,   249,   260,   261,   279,   288,   298,   299,
     303,   304,   305,   309,   310,   314,   315,   316,   321,   322,
     326,   327,   328,   332,   333,   337,   338,   339,   343,   344,
     348,   349,   350,   354,   355,   359,   360,   361,   365,   366,
     370,   371,   372,   376,   380,   390,   391,   395,   402,   409,
     416,   423,   424,   429,   433,   437,   438,   442,   452,   467,
     468,   472,   473,   474,   475,   476,   477,   478,   479,   480,
     481,   482,   486,   496,   511,   512,   516,   523,   527,   531,
     535,   539,   543,   550,   560,   575,   576,   580,   587,   594,
     601,   608,   615,   622,   629,   636,   643,   650,   654,   658,
     662,   666,   670,   674,   678,   682,   686,   693,   703,   718,
     719,   723,   724,   728,   729,   733,   743,   758,   759,   763,
     770,   780,   795,   796,   800,   801,   811,   812,   816,   823,
     824,   828,   829,   830,   831,   832,   839,   849,   864,   865,
     869,   873,   883,   898,   899,   903,   910,   917,   921,   928,
     938,   953,   954,   958,   962,   966,   970,   977,   987,  1002,
    1003,  1007,  1017,  1032,  1033,  1037,  1041,  1048,  1058,  1073,
    1074,  1078,  1082,  1086,  1090,  1094,  1101,  1111,  1126,  1127,
    1131,  1135,  1139,  1143,  1147,  1151,  1159,  1169,  1184,  1185,
    1189,  1193,  1197,  1201,  1205,  1212,  1222,  1237,  1238,  1242,
    1250,  1251,  1252,  1253,  1254,  1255,  1256,  1257,  1258,  1259,
    1263,  1267,  1271,  1275,  1279,  1283,  1287,  1291,  1295,  1302,
    1312,  1327,  1328,  1332,  1339,  1346,  1353,  1360,  1364,  1368,
    1372,  1376,  1380,  1384,  1388,  1392,  1396,  1403,  1413,  1428,
    1429,  1433,  1437,  1441,  1448,  1458,  1473,  1474,  1478,  1485,
    1489,  1493,  1497,  1501,  1505,  1509,  1513,  1517,  1524,  1534,
    1549,  1550,  1554,  1555,  1556,  1560,  1570,  1585,  1586,  1590,
    1597,  1601,  1605,  1612,  1622,  1637,  1638,  1642,  1649,  1656,
    1663,  1670,  1671,  1672,  1673,  1674,  1675,  1676,  1677,  1678,
    1679,  1680,  1681,  1682,  1686,  1690,  1694,  1698,  1702,  1706,
    1710,  1714,  1718,  1725,  1735,  1750,  1751,  1755,  1756,  1757,
    1758,  1759,  1760,  1761,  1765,  1769,  1776,  1786,  1801,  1802,
    1806,  1813,  1817,  1818,  1822,  1832,  1847,  1848,  1852,  1860,
    1867,  1871,  1875,  1879,  1886,  1896,  1911,  1912,  1916,  1920,
    1924,  1928,  1932,  1936,  1943,  1953,  1968,  1969,  1973,  1980,
    1984,  1988,  1992,  1996,  2000,  2004,  2011,  2021,  2036,  2037,
    2041,  2048,  2055,  2059,  2063,  2067,  2071,  2078,  2088,  2103,
    2104,  2108,  2112,  2122,  2137,  2138,  2142,  2146,  2150,  2156,
    2166,  2181,  2182,  2186,  2190,  2194,  2200,  2210,  2225,  2226,
    2230,  2237,  2241,  2245,  2252,  2262,  2277,  2278,  2282,  2286,
    2290,  2294,  2298,  2305,  2315,  2331,  2332,  2336,  2340,  2344,
    2348,  2352,  2356,  2360,  2367,  2377,  2392,  2393,  2397,  2398,
    2399,  2400,  2401,  2402,  2407,  2417,  2432,  2433,  2437,  2441,
    2445,  2451,  2461,  2476,  2477,  2481,  2485,  2489,  2496,  2506,
    2521,  2522,  2526,  2530,  2534,  2540,  2550,  2565,  2566,  2570,
    2577,  2581,  2585,  2594,  2600,  2606,  2612,  2618,  2632,  2638,
    2644,  2650,  2656,  2667,  2673,  2679,  2685,  2699,  2705,  2711,
    2717,  2728,  2734,  2740,  2746,  2752,  2766,  2772,  2778,  2784,
    2790,  2801,  2807,  2813,  2819,  2833,  2839,  2845,  2851,  2862,
    2868,  2874,  2880,  2886,  2893,  2900,  2908,  2922,  2928,  2934,
    2944,  2954,  2970,  2971,  2975,  2976,  2977,  2978,  2979,  2980,
    2984,  2994,  3009,  3010,  3014,  3018,  3022,  3026,  3030,  3034,
    3038,  3042,  3046,  3047,  3048,  3049,  3050,  3057,  3067,  3082,
    3083,  3087,  3094,  3104,  3119,  3120,  3124,  3128,  3132,  3139,
    3149,  3164,  3165,  3169,  3173,  3177,  3181,  3185,  3189,  3193,
    3197,  3201,  3205,  3212,  3222,  3237,  3238,  3242,  3249,  3253,
    3257,  3265,  3275,  3290,  3291,  3295,  3302,  3309,  3313,  3314,
    3315,  3316,  3320,  3328,  3338,  3353,  3354,  3358,  3363,  3373,
    3388,  3389,  3393,  3397,  3401,  3405,  3413,  3423,  3438,  3439,
    3443,  3447,  3451,  3455,  3459,  3467,  3477,  3492,  3493,  3497,
    3504,  3514,  3529,  3530,  3534,  3535,  3539,  3543,  3547,  3551,
    3555,  3556,  3560,  3570,  3585,  3586,  3590,  3594,  3598,  3602,
    3606,  3613,  3623,  3638,  3639,  3643,  3647,  3651,  3658,  3668,
    3683,  3684,  3688,  3695,  3699,  3706,  3716
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "NUMBER", "FLOAT", "STRING", "NAME",
  "ANCHOR", "APPEARANCE", "AUDIOCLIP", "BACKGROUND", "BILLBOARD", "BOX",
  "COLLISION", "COLOR", "COLOR_INTERP", "COORDINATE", "COORDINATE_INTERP",
  "CYLINDER_SENSOR", "NULL_STRING", "CONE", "CUBE", "CYLINDER",
  "DIRECTIONALLIGHT", "FONTSTYLE", "ERROR", "EXTRUSION", "ELEVATION_GRID",
  "FOG", "INLINE", "MOVIE_TEXTURE", "NAVIGATION_INFO", "PIXEL_TEXTURE",
  "GROUP", "INDEXEDFACESET", "INDEXEDLINESET", "S_INFO", "LOD", "MATERIAL",
  "NORMAL", "POSITION_INTERP", "PROXIMITY_SENSOR", "SCALAR_INTERP",
  "SCRIPT", "SHAPE", "SOUND", "SPOTLIGHT", "SPHERE_SENSOR", "TEXT",
  "TEXTURE_COORDINATE", "TEXTURE_TRANSFORM", "TIME_SENSOR", "SWITCH",
  "TOUCH_SENSOR", "VIEWPOINT", "VISIBILITY_SENSOR", "WORLD_INFO",
  "NORMAL_INTERP", "ORIENTATION_INTERP", "POINTLIGHT", "POINTSET",
  "SPHERE", "PLANE_SENSOR", "TRANSFORM", "S_CHILDREN", "S_PARAMETER",
  "S_URL", "S_MATERIAL", "S_TEXTURETRANSFORM", "S_TEXTURE", "S_LOOP",
  "S_STARTTIME", "S_STOPTIME", "S_GROUNDANGLE", "S_GROUNDCOLOR", "S_SPEED",
  "S_AVATAR_SIZE", "S_BACKURL", "S_BOTTOMURL", "S_FRONTURL", "S_LEFTURL",
  "S_RIGHTURL", "S_TOPURL", "S_SKYANGLE", "S_SKYCOLOR",
  "S_AXIS_OF_ROTATION", "S_COLLIDE", "S_COLLIDETIME", "S_PROXY", "S_SIDE",
  "S_AUTO_OFFSET", "S_DISK_ANGLE", "S_ENABLED", "S_MAX_ANGLE",
  "S_MIN_ANGLE", "S_OFFSET", "S_BBOXSIZE", "S_BBOXCENTER",
  "S_VISIBILITY_LIMIT", "S_AMBIENT_INTENSITY", "S_NORMAL", "S_TEXCOORD",
  "S_CCW", "S_COLOR_PER_VERTEX", "S_CREASE_ANGLE", "S_NORMAL_PER_VERTEX",
  "S_XDIMENSION", "S_XSPACING", "S_ZDIMENSION", "S_ZSPACING",
  "S_BEGIN_CAP", "S_CROSS_SECTION", "S_END_CAP", "S_SPINE", "S_FOG_TYPE",
  "S_VISIBILITY_RANGE", "S_HORIZONTAL", "S_JUSTIFY", "S_LANGUAGE",
  "S_LEFT2RIGHT", "S_TOP2BOTTOM", "IMAGE_TEXTURE", "S_SOLID", "S_KEY",
  "S_KEYVALUE", "S_REPEAT_S", "S_REPEAT_T", "S_CONVEX", "S_BOTTOM",
  "S_PICTH", "S_COORD", "S_COLOR_INDEX", "S_COORD_INDEX", "S_NORMAL_INDEX",
  "S_MAX_POSITION", "S_MIN_POSITION", "S_ATTENUATION", "S_APPEARANCE",
  "S_GEOMETRY", "S_DIRECT_OUTPUT", "S_MUST_EVALUATE", "S_MAX_BACK",
  "S_MIN_BACK", "S_MAX_FRONT", "S_MIN_FRONT", "S_PRIORITY", "S_SOURCE",
  "S_SPATIALIZE", "S_BERM_WIDTH", "S_CHOICE", "S_WHICHCHOICE",
  "S_FONTSTYLE", "S_LENGTH", "S_MAX_EXTENT", "S_ROTATION", "S_SCALE",
  "S_CYCLE_INTERVAL", "S_FIELD_OF_VIEW", "S_JUMP", "S_TITLE",
  "S_TEXCOORD_INDEX", "S_HEADLIGHT", "S_TOP", "S_BOTTOMRADIUS", "S_HEIGHT",
  "S_POINT", "S_STRING", "S_SPACING", "S_TYPE", "S_RADIUS", "S_ON",
  "S_INTENSITY", "S_COLOR", "S_DIRECTION", "S_SIZE", "S_FAMILY", "S_STYLE",
  "S_RANGE", "S_CENTER", "S_TRANSLATION", "S_LEVEL", "S_DIFFUSECOLOR",
  "S_SPECULARCOLOR", "S_EMISSIVECOLOR", "S_SHININESS", "S_TRANSPARENCY",
  "S_VECTOR", "S_POSITION", "S_ORIENTATION", "S_LOCATION", "S_CUTOFFANGLE",
  "S_WHICHCHILD", "S_IMAGE", "S_SCALEORIENTATION", "S_DESCRIPTION",
  "SFBOOL", "SFFLOAT", "SFINT32", "SFTIME", "SFROTATION", "SFNODE",
  "SFCOLOR", "SFIMAGE", "SFSTRING", "SFVEC2F", "SFVEC3F", "MFBOOL",
  "MFFLOAT", "MFINT32", "MFTIME", "MFROTATION", "MFNODE", "MFCOLOR",
  "MFIMAGE", "MFSTRING", "MFVEC2F", "MFVEC3F", "FIELD", "EVENTIN",
  "EVENTOUT", "USE", "S_VALUE_CHANGED", "'['", "']'", "'{'", "'}'",
  "$accept", "Vrml", "VrmlNodes", "GroupingNode", "InterpolatorNode",
  "SensorNode", "GeometryNode", "GeometryInfoNode", "LightNode",
  "CommonNode", "BindableNode", "SFNode", "SFInt32", "SFBool", "SFString",
  "SFFloat", "SFTime", "SFColor", "SFRotation", "SFImageList", "SFVec2f",
  "SFVec3f", "SFColorList", "MFColor", "SFInt32List", "MFInt32",
  "SFFloatList", "MFFloat", "SFStringList", "MFString", "SFVec2fList",
  "MFVec2f", "SFVec3fList", "MFVec3f", "SFRotationList", "MFRotation",
  "NodeBegin", "NodeEnd", "AnchorElements", "AnchorElementParameterBegin",
  "AnchorElementURLBegin", "bboxCenter", "bboxSize", "AnchorElement",
  "AnchorBegin", "Anchor", "AppearanceNodes", "AppearanceNode",
  "AppearanceBegin", "Appearance", "AudioClipElements", "AudioClipURL",
  "AudioClipElement", "AudioClipBegin", "AudioClip", "BackGroundElements",
  "BackGroundBackURL", "BackGroundBottomURL", "BackGroundFrontURL",
  "BackGroundLeftURL", "BackGroundRightURL", "BackGroundTopURL",
  "BackGroundGroundAngle", "BackGroundGroundColor", "BackGroundSkyAngle",
  "BackGroundSkyColor", "BackGroundElement", "BackgroundBegin",
  "Background", "BillboardElements", "BillboardElement", "BillboardBegin",
  "Billboard", "BoxElements", "BoxElement", "BoxBegin", "Box",
  "childrenElements", "children", "CollisionElements",
  "CollisionElementProxyBegin", "CollisionElement", "CollisionBegin",
  "Collision", "ColorElements", "ColorElement", "ColorBegin", "Color",
  "ColorInterpElements", "InterpolateKey", "InterporlateKeyValue",
  "ColorInterpElement", "ColorInterpBegin", "ColorInterp", "ConeElements",
  "ConeElement", "ConeBegin", "Cone", "CoordinateElements",
  "CoordinateBegin", "Coordinate", "CoordinateInterpElements",
  "CoordinateInterpElement", "CoordinateInterpBegin", "CoordinateInterp",
  "CylinderElements", "CylinderElement", "CylinderBegin", "Cylinder",
  "CylinderSensorElements", "CylinderSensorElement", "CylinderSensorBegin",
  "CylinderSensor", "DirLightElements", "DirLightElement", "DirLightBegin",
  "DirLight", "ElevationGridElements", "ElevationGridHeight",
  "ElevationGridElement", "ElevationGridBegin", "ElevationGrid",
  "ExtrusionElements", "ExtrusionCrossSection", "ExtrusionOrientation",
  "ExtrusionScale", "ExtrusionSpine", "ExtrusionElement", "ExtrusionBegin",
  "Extrusion", "FogElements", "FogElement", "FogBegin", "Fog",
  "FontStyleElements", "FontStyleJustify", "FontStyleElement",
  "FontStyleBegin", "FontStyle", "GroupElements", "GroupElement",
  "GroupBegin", "Group", "ImgTexElements", "ImgTexURL", "ImgTexElement",
  "ImageTextureBegin", "ImageTexture", "IdxFacesetElements", "ColorIndex",
  "CoordIndex", "NormalIndex", "TextureIndex", "IdxFacesetElement",
  "IdxFacesetBegin", "IdxFaceset", "IdxLinesetElements",
  "IdxLinesetElement", "IdxLinesetBegin", "IdxLineset", "InlineElements",
  "InlineURL", "InlineElement", "InlineBegin", "Inline", "LodElements",
  "LodRange", "LodLevel", "LodElement", "LodBegin", "Lod",
  "MaterialElements", "MaterialElement", "MaterialBegin", "Material",
  "MovieTextureElements", "MovieTextureURL", "MovieTextureElement",
  "MovieTextureBegin", "MovieTexture", "NavigationInfoElements",
  "NavigationInfoAvatarSize", "NavigationInfoType",
  "NavigationInfoElement", "NavigationInfoBegin", "NavigationInfo",
  "NormalElements", "NormalElement", "NormalBegin", "Normal",
  "NormalInterpElements", "NormalInterpElement", "NormalInterpBegin",
  "NormalInterp", "OrientationInterpElements", "OrientationInterpElement",
  "OrientationInterpBegin", "OrientationInterp", "PixelTextureElements",
  "PixelTextureImage", "PixelTextureElement", "PixelTextureBegin",
  "PixelTexture", "PlaneSensorElements", "PlaneSensorElement",
  "PlaneSensorBegin", "PlaneSensor", "PointLightNodes", "PointLightNode",
  "PointLightBegin", "PointLight", "PointsetElements", "PointsetElement",
  "PointsetBegin", "Pointset", "PositionInterpElements",
  "PositionInterpElement", "PositionInterpBegin", "PositionInterp",
  "ProximitySensorElements", "ProximitySensorElement",
  "ProximitySensorBegin", "ProximitySensor", "ScalarInterpElements",
  "ScalarInterpElement", "ScalarInterpBegin", "ScalarInterp",
  "ScriptElements", "ScriptURL", "ScriptElement", "ScriptBegin", "Script",
  "SharpElements", "SharpElement", "ShapeBegin", "Shape", "SoundElements",
  "SoundElement", "SoundBegin", "Sound", "SphereElements", "SphereElement",
  "SphereBegin", "Sphere", "SphereSensorElements", "SphereSensorElement",
  "SphereSensorBegin", "SphereSensor", "SpotLightElements",
  "SpotLightElement", "SpotLightBegin", "SpotLight", "SwitchElements",
  "SwitchChoice", "SwitchElement", "SwitchBegin", "Switch", "TextElements",
  "TextString", "TextLength", "TextElement", "TextBegin", "Text",
  "TexCoordElements", "TexCoordElement", "TexCoordBegin", "TexCoordinate",
  "TextureTransformElements", "TextureTransformElement",
  "TexTransformBegin", "TexTransform", "TimeSensorElements",
  "TimeSensorElement", "TimeSensorBegin", "TimeSensor",
  "TouchSensorElements", "TouchSensorElement", "TouchSensorBegin",
  "TouchSensor", "TransformElements", "TransformElement", "TransformBegin",
  "Transform", "ViewpointElements", "ViewpointElement", "ViewpointBegin",
  "Viewpoint", "VisibilitySensors", "VisibilitySensor",
  "VisibilitySensorBegine", "WorldInfoElements", "WorldInfoInfo",
  "WorldInfoElement", "WorldInfoBegin", "WorldInfo", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,   329,   330,   331,   332,   333,   334,
     335,   336,   337,   338,   339,   340,   341,   342,   343,   344,
     345,   346,   347,   348,   349,   350,   351,   352,   353,   354,
     355,   356,   357,   358,   359,   360,   361,   362,   363,   364,
     365,   366,   367,   368,   369,   370,   371,   372,   373,   374,
     375,   376,   377,   378,   379,   380,   381,   382,   383,   384,
     385,   386,   387,   388,   389,   390,   391,   392,   393,   394,
     395,   396,   397,   398,   399,   400,   401,   402,   403,   404,
     405,   406,   407,   408,   409,   410,   411,   412,   413,   414,
     415,   416,   417,   418,   419,   420,   421,   422,   423,   424,
     425,   426,   427,   428,   429,   430,   431,   432,   433,   434,
     435,   436,   437,   438,   439,   440,   441,   442,   443,   444,
     445,   446,   447,   448,   449,   450,   451,   452,   453,   454,
     455,   456,   457,   458,   459,   460,   461,   462,   463,   464,
     465,   466,   467,   468,   469,   470,   471,   472,   473,   474,
     475,   476,    91,    93,   123,   125
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   226,   227,   227,   227,   228,   228,   229,   229,   229,
     229,   229,   229,   229,   229,   230,   230,   230,   230,   230,
     230,   231,   231,   231,   231,   231,   231,   231,   232,   232,
     232,   232,   232,   232,   232,   232,   232,   232,   233,   233,
     233,   233,   233,   233,   234,   234,   234,   235,   235,   235,
     235,   235,   235,   236,   236,   236,   236,   237,   237,   237,
     237,   237,   237,   237,   237,   238,   239,   240,   241,   241,
     242,   242,   243,   244,   245,   245,   246,   247,   248,   248,
     249,   249,   249,   250,   250,   251,   251,   251,   252,   252,
     253,   253,   253,   254,   254,   255,   255,   255,   256,   256,
     257,   257,   257,   258,   258,   259,   259,   259,   260,   260,
     261,   261,   261,   262,   263,   264,   264,   265,   266,   267,
     268,   269,   269,   269,   269,   269,   269,   270,   271,   272,
     272,   273,   273,   273,   273,   273,   273,   273,   273,   273,
     273,   273,   274,   275,   276,   276,   277,   278,   278,   278,
     278,   278,   278,   279,   280,   281,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   292,   292,
     292,   292,   292,   292,   292,   292,   292,   293,   294,   295,
     295,   296,   296,   296,   296,   297,   298,   299,   299,   300,
     301,   302,   303,   303,   304,   304,   305,   305,   306,   307,
     307,   307,   307,   307,   307,   307,   308,   309,   310,   310,
     311,   312,   313,   314,   314,   315,   316,   317,   317,   318,
     319,   320,   320,   321,   321,   321,   321,   322,   323,   324,
     324,   325,   326,   327,   327,   328,   328,   329,   330,   331,
     331,   332,   332,   332,   332,   332,   333,   334,   335,   335,
     336,   336,   336,   336,   336,   336,   337,   338,   339,   339,
     340,   340,   340,   340,   340,   341,   342,   343,   343,   344,
     345,   345,   345,   345,   345,   345,   345,   345,   345,   345,
     345,   345,   345,   345,   345,   345,   345,   345,   345,   346,
     347,   348,   348,   349,   350,   351,   352,   353,   353,   353,
     353,   353,   353,   353,   353,   353,   353,   354,   355,   356,
     356,   357,   357,   357,   358,   359,   360,   360,   361,   362,
     362,   362,   362,   362,   362,   362,   362,   362,   363,   364,
     365,   365,   366,   366,   366,   367,   368,   369,   369,   370,
     371,   371,   371,   372,   373,   374,   374,   375,   376,   377,
     378,   379,   379,   379,   379,   379,   379,   379,   379,   379,
     379,   379,   379,   379,   379,   379,   379,   379,   379,   379,
     379,   379,   379,   380,   381,   382,   382,   383,   383,   383,
     383,   383,   383,   383,   383,   383,   384,   385,   386,   386,
     387,   388,   388,   388,   389,   390,   391,   391,   392,   393,
     394,   394,   394,   394,   395,   396,   397,   397,   398,   398,
     398,   398,   398,   398,   399,   400,   401,   401,   402,   403,
     403,   403,   403,   403,   403,   403,   404,   405,   406,   406,
     407,   408,   409,   409,   409,   409,   409,   410,   411,   412,
     412,   413,   414,   415,   416,   416,   417,   417,   417,   418,
     419,   420,   420,   421,   421,   421,   422,   423,   424,   424,
     425,   426,   426,   426,   427,   428,   429,   429,   430,   430,
     430,   430,   430,   431,   432,   433,   433,   434,   434,   434,
     434,   434,   434,   434,   435,   436,   437,   437,   438,   438,
     438,   438,   438,   438,   439,   440,   441,   441,   442,   442,
     442,   443,   444,   445,   445,   446,   446,   446,   447,   448,
     449,   449,   450,   450,   450,   451,   452,   453,   453,   454,
     455,   455,   455,   455,   455,   455,   455,   455,   455,   455,
     455,   455,   455,   455,   455,   455,   455,   455,   455,   455,
     455,   455,   455,   455,   455,   455,   455,   455,   455,   455,
     455,   455,   455,   455,   455,   455,   455,   455,   455,   455,
     455,   455,   455,   455,   455,   455,   455,   455,   455,   455,
     456,   457,   458,   458,   459,   459,   459,   459,   459,   459,
     460,   461,   462,   462,   463,   463,   463,   463,   463,   463,
     463,   463,   463,   463,   463,   463,   463,   464,   465,   466,
     466,   467,   468,   469,   470,   470,   471,   471,   471,   472,
     473,   474,   474,   475,   475,   475,   475,   475,   475,   475,
     475,   475,   475,   476,   477,   478,   478,   479,   480,   480,
     480,   481,   482,   483,   483,   484,   485,   486,   486,   486,
     486,   486,   486,   487,   488,   489,   489,   490,   491,   492,
     493,   493,   494,   494,   494,   494,   495,   496,   497,   497,
     498,   498,   498,   498,   498,   499,   500,   501,   501,   502,
     503,   504,   505,   505,   506,   506,   506,   506,   506,   506,
     506,   506,   507,   508,   509,   509,   510,   510,   510,   510,
     510,   511,   512,   513,   513,   514,   514,   514,   515,   514,
     516,   516,   517,   518,   518,   519,   520
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     1,     1,     2,     0,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     3,     4,     2,     0,     2,     3,     1,     2,
       1,     2,     3,     1,     2,     1,     2,     3,     1,     2,
       1,     2,     3,     1,     2,     1,     2,     3,     1,     2,
       1,     2,     3,     1,     2,     1,     2,     3,     1,     2,
       1,     2,     3,     1,     1,     2,     0,     1,     1,     2,
       2,     1,     2,     2,     2,     1,     1,     1,     4,     2,
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     1,     4,     2,     0,     1,     2,     2,     2,
       2,     2,     2,     1,     4,     2,     0,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     1,     4,     2,
       0,     1,     2,     1,     1,     1,     4,     2,     0,     2,
       1,     4,     2,     0,     4,     2,     2,     0,     1,     1,
       2,     1,     1,     2,     2,     2,     1,     4,     2,     0,
       2,     1,     4,     2,     0,     1,     1,     2,     2,     1,
       4,     2,     0,     2,     2,     2,     2,     1,     4,     2,
       0,     1,     4,     2,     0,     2,     2,     1,     4,     2,
       0,     2,     2,     2,     2,     2,     1,     4,     2,     0,
       2,     2,     2,     2,     2,     2,     1,     4,     2,     0,
       2,     2,     2,     2,     2,     1,     4,     2,     0,     1,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     1,
       4,     2,     0,     1,     1,     1,     1,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     1,     4,     2,
       0,     2,     2,     2,     1,     4,     2,     0,     1,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     3,
       2,     0,     1,     1,     1,     1,     4,     2,     0,     1,
       2,     2,     2,     1,     4,     2,     0,     1,     1,     1,
       1,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     1,     4,     2,     0,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     3,     2,     0,
       1,     2,     1,     1,     1,     4,     2,     0,     1,     1,
       2,     2,     2,     4,     1,     4,     2,     0,     2,     2,
       2,     2,     2,     2,     1,     4,     2,     0,     1,     2,
       2,     2,     2,     2,     2,     2,     1,     4,     2,     0,
       1,     1,     2,     2,     2,     2,     2,     1,     4,     2,
       0,     2,     1,     4,     2,     0,     2,     2,     2,     1,
       4,     2,     0,     2,     2,     2,     1,     4,     2,     0,
       1,     4,     2,     2,     1,     4,     2,     0,     2,     2,
       2,     2,     2,     1,     4,     2,     0,     2,     2,     2,
       2,     2,     2,     2,     1,     4,     2,     0,     2,     2,
       2,     2,     2,     2,     1,     4,     2,     0,     2,     2,
       2,     1,     4,     2,     0,     2,     2,     2,     1,     4,
       2,     0,     2,     2,     2,     1,     4,     2,     0,     1,
       2,     2,     2,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     3,     3,     3,     3,     4,
       4,     4,     4,     4,     4,     5,     4,     4,     4,     4,
       1,     4,     2,     0,     2,     2,     2,     2,     2,     2,
       1,     4,     2,     0,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     1,     4,     2,
       0,     2,     1,     4,     2,     0,     2,     2,     2,     1,
       4,     2,     0,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     1,     4,     2,     0,     1,     2,     4,
       2,     1,     4,     2,     0,     1,     1,     2,     2,     2,
       2,     2,     2,     1,     4,     2,     0,     2,     1,     4,
       2,     0,     2,     2,     2,     2,     1,     4,     2,     0,
       2,     2,     2,     2,     2,     1,     4,     2,     0,     2,
       1,     4,     2,     0,     1,     2,     2,     2,     2,     2,
       1,     1,     1,     4,     2,     0,     2,     2,     2,     2,
       2,     1,     4,     2,     0,     2,     2,     2,     1,     4,
       2,     0,     1,     2,     2,     1,     4
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       0,     3,   127,   142,   153,   177,   185,   206,   211,   219,
     231,   237,   256,   265,     0,     4,   314,   394,   437,   335,
     404,   414,   442,   501,   508,   515,   570,   580,   597,   623,
     609,   648,   665,   631,   670,   691,   698,   705,   449,   456,
     484,   473,   682,     0,     0,     0,    64,     0,     2,    62,
      60,    61,    63,    48,    57,    58,     6,     0,     7,     0,
      42,     0,    47,     0,    53,     0,     8,     0,     9,     0,
      38,     0,    15,     0,    39,     0,    16,     0,    21,     0,
      44,     0,    54,   317,    59,     0,    10,     0,    11,     0,
      12,     0,    43,     0,    55,     0,    40,     0,    17,     0,
      18,     0,    22,     0,    46,     0,    19,     0,    24,     0,
      20,     0,    49,     0,    50,     0,    51,     0,    23,     0,
      45,     0,    13,     0,    41,     0,    25,     0,    26,     0,
      14,     0,    56,    27,     0,     0,    52,   113,   328,    66,
     696,    69,    68,     0,   697,   695,     1,     5,   116,   130,
     145,   156,   180,   197,   209,   214,   230,   234,   249,   259,
     310,     0,   318,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   317,   331,   389,   397,   407,   429,   440,   445,
     452,   467,   476,   497,   504,   511,   518,   573,   583,   605,
     612,   626,   646,   659,   668,   673,   685,   694,   701,     0,
       0,   117,   118,     0,     0,     0,     0,     0,     0,   125,
     126,   116,   121,     0,     0,     0,     0,   130,   146,     0,
       0,     0,     0,     0,     0,     0,   145,   163,   164,   157,
     158,   159,   160,   161,   162,   165,   166,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   156,     0,
     183,   184,     0,   180,   181,     0,   198,   201,   202,   199,
       0,     0,   197,     0,     0,   209,   215,   216,     0,     0,
       0,   214,     0,     0,     0,     0,     0,   234,     0,     0,
       0,     0,     0,     0,     0,   249,     0,     0,     0,     0,
       0,     0,   259,     0,     0,     0,     0,   310,   320,    67,
     322,   323,   327,   325,   324,   319,   326,   114,   329,     0,
      95,   321,   316,   333,   334,   332,     0,   331,   390,   392,
     393,     0,     0,   389,   398,     0,   399,     0,     0,     0,
     397,     0,     0,     0,     0,     0,     0,     0,   407,     0,
     430,     0,     0,   431,     0,     0,     0,   429,     0,     0,
     440,     0,     0,     0,     0,   445,     0,     0,     0,     0,
     452,     0,     0,     0,     0,     0,     0,   467,     0,     0,
       0,     0,     0,     0,     0,     0,   476,     0,     0,     0,
       0,   497,     0,     0,     0,     0,   504,     0,     0,     0,
       0,   511,   519,     0,     0,     0,     0,     0,     0,     0,
     518,     0,     0,     0,   573,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   583,     0,     0,     0,
       0,   605,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   612,   627,     0,     0,     0,   626,     0,
       0,   646,     0,     0,     0,     0,     0,     0,   659,     0,
       0,   668,     0,     0,     0,     0,     0,   680,   681,   674,
       0,   673,     0,     0,     0,     0,     0,     0,   685,     0,
     694,   702,     0,     0,     0,   701,    77,   193,   195,   120,
     119,   122,   128,   123,   124,   115,   131,   133,   132,   139,
     656,   141,     0,   140,   134,   426,   464,   343,   138,     0,
     135,     0,   136,     0,   137,   143,   129,   148,    71,    70,
     150,   151,   149,   147,   154,   152,   144,   178,   169,   170,
     171,   172,   173,   174,     0,    90,   167,     0,     0,    80,
     168,   175,   176,   155,   182,   186,   179,   200,   204,   203,
     207,   205,   196,   210,   212,   208,   220,   217,   218,   213,
       0,   105,   229,   232,   235,   236,   238,   233,   250,   251,
     252,   253,   254,   255,   257,   248,   264,   260,   261,   262,
     263,   266,   258,   312,   313,   311,   315,   309,    96,    93,
       0,   336,   330,   395,   391,   388,   401,   405,   400,     6,
     402,   396,   408,   409,   412,   410,   411,   413,   415,   406,
     434,   436,   433,   438,   432,   435,   428,   441,   443,   439,
     448,   446,   447,   450,   444,     0,   455,   453,     0,   110,
     454,   457,   451,   468,   469,   472,     0,   470,   471,   474,
     466,   477,   478,   483,   482,   480,   479,   481,   485,   475,
     500,   498,   499,   502,   496,   507,   506,   505,   509,   503,
     514,   512,   513,   516,   510,   521,   522,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   571,   520,   517,   574,   576,   575,   190,
     577,   227,   246,   307,   289,   373,     0,   643,   494,   602,
     579,   578,     0,    28,     0,    29,     0,    30,     0,    31,
       0,    32,     0,    33,   376,    34,     0,    35,     0,    36,
       0,    37,   581,   572,   587,   589,   588,   590,   591,   592,
     595,   593,   594,   596,   585,   584,   586,   598,   582,   606,
     607,   608,   610,   604,   613,   614,   615,   622,   621,   619,
     616,   618,   620,   617,   624,   611,    65,   630,   632,     6,
     628,   625,     0,   100,   647,   649,   645,   662,   663,   664,
     661,   660,   666,   658,   669,   671,   667,   676,   677,   675,
     679,   678,   683,   672,   686,   687,   689,   688,   690,   692,
     684,   699,   693,   704,   706,   703,   700,   193,     0,   651,
     338,   417,   459,    91,    88,     0,    81,    78,     0,     0,
     106,   103,     0,    97,    94,     0,     0,   111,   108,     0,
      76,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   523,   524,   525,   526,   527,   528,   529,   530,   531,
     532,   533,   534,   535,   536,   537,   538,   539,   540,   541,
     542,   543,   544,   545,   546,   547,   548,   549,   550,   551,
     552,   553,   554,   555,   556,   557,   558,   386,   188,   222,
     240,   268,   292,   346,     0,     0,   347,   348,     0,     0,
       0,     0,   376,   487,   600,   634,     0,   101,    98,     0,
     192,   194,     0,     0,     0,     0,     0,   651,   339,     0,
       0,     0,     0,   338,   418,     0,     0,     0,     0,     0,
       0,     0,     0,   417,     0,     0,   460,     0,     0,   459,
      92,    89,    82,    79,    72,   107,   104,   403,     0,   112,
     109,   559,   560,   561,   562,   563,   564,     0,   566,   567,
     568,   569,     0,     0,   188,     0,     0,     0,     0,     0,
     222,     0,     0,     0,     0,     0,     0,   240,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   269,
       0,     0,     0,   268,     0,     0,     0,   293,     0,   296,
       0,     0,   295,   294,     0,     0,     0,     0,     0,   292,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   349,
     350,     0,     0,     0,     0,     0,     0,   346,   383,   380,
     382,   381,   377,   379,   378,     0,    85,   384,   385,   387,
     375,     0,     0,     0,   487,     0,     0,   600,     0,   636,
       0,   635,     0,     0,     0,   634,   629,   102,    99,   653,
     654,   652,   655,   657,   650,   341,   342,   344,   340,   337,
     419,   421,   422,   420,   424,   425,   427,   423,   416,   462,
     463,   465,    75,   458,    73,   565,   189,   191,   187,   223,
     224,   225,   226,   228,   221,   241,   242,   243,   245,   244,
     247,   239,   273,   275,   274,   276,   278,   277,   280,   283,
     281,   284,   285,   286,   287,   288,   282,   270,   272,   271,
     290,   279,   267,   298,   300,   297,   303,   301,   299,   308,
     302,   304,   305,   306,   291,   357,   359,   358,   360,   362,
     361,   363,   368,   366,   372,   365,   364,   354,   356,   355,
     351,   353,   352,   374,   367,   369,   370,   371,   345,    86,
      83,     0,   491,   493,   492,   488,   490,   489,   495,   486,
     601,   603,   599,   638,   640,   639,   642,   644,   637,   641,
     633,    75,     0,    87,    84,    74,   461
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,    47,    48,    49,    50,    51,   721,    52,    53,    54,
      55,    56,  1036,   140,   310,   143,   510,   529,   619,  1182,
     783,   551,   828,   530,  1161,  1037,   825,   526,   580,   311,
     909,   784,   832,   552,   839,   620,   138,   308,   206,   207,
     208,   209,   210,   211,    57,    58,   216,   217,    59,    60,
     224,   225,   226,    61,    62,   237,   238,   239,   240,   241,
     242,   243,   244,   245,   246,   247,   248,    63,    64,   252,
     253,    65,    66,   963,   964,   722,   723,   818,   212,   260,
     261,   262,    67,    68,   264,   265,    69,    70,   268,   269,
     270,   271,    71,    72,   969,   970,   724,   725,   273,    73,
      74,   276,   277,    75,    76,   976,   977,   726,   727,   284,
     285,    77,    78,   291,   292,    79,    80,   991,   992,   993,
     728,   729,  1004,  1005,  1006,  1007,  1008,  1009,   730,   731,
     296,   297,    81,    82,   170,   171,   172,    83,    84,   316,
     317,    85,    86,   921,   922,   923,   499,   500,  1022,   899,
     900,  1025,  1026,  1027,   732,   733,   901,   902,   734,   735,
     321,   322,   323,    87,    88,   327,   328,   329,   330,    89,
      90,   337,   338,    91,    92,   931,   932,   933,   501,   502,
     344,   345,   346,   347,    93,    94,   349,   350,    95,    96,
     354,   355,    97,    98,   359,   360,    99,   100,   937,   938,
     939,   503,   504,   366,   367,   101,   102,   375,   376,   103,
     104,  1043,  1044,   736,   737,   380,   381,   105,   106,   385,
     386,   107,   108,   390,   391,   109,   110,   398,   399,   400,
     111,   112,   403,   404,   113,   114,   415,   416,   115,   116,
    1046,  1047,   738,   739,   420,   421,   117,   118,   432,   433,
     119,   120,   436,   437,   438,   121,   122,  1052,  1053,  1054,
    1055,   740,   741,   440,   441,   123,   124,   916,   917,   492,
     493,   447,   448,   125,   126,   450,   451,   127,   128,   460,
     461,   129,   130,   467,   468,   131,   132,   469,   133,   134,
     473,   474,   475,   135,   136
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -866
static const yytype_int16 yypact[] =
{
    1246,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -180,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,    57,   137,   137,  -866,    73,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  1447,  -180,  -866,  -180,
    -866,  -180,  -866,  -180,  -866,  -180,  -866,  -180,  -866,  -180,
    -866,  -180,  -866,  -180,  -866,  -180,  -866,  -180,  -866,  -180,
    -866,  -180,  -866,   588,  -866,  -180,  -866,  -180,  -866,  -180,
    -866,  -180,  -866,  -180,  -866,  -180,  -866,  -180,  -866,  -180,
    -866,  -180,  -866,  -180,  -866,  -180,  -866,  -180,  -866,  -180,
    -866,  -180,  -866,  -180,  -866,  -180,  -866,  -180,  -866,  -180,
    -866,  -180,  -866,  -180,  -866,  -180,  -866,  -180,  -866,  -180,
    -866,  -180,  -866,  -866,  -180,  -180,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,   137,  -866,  -866,  -866,  -866,   361,   392,
     459,   930,   478,   708,  -115,   100,   -33,   100,   849,   520,
      54,    57,  -866,   116,    57,    57,   137,   137,   116,   116,
     -74,    25,   588,   348,   212,   110,   504,   380,   -25,    31,
     230,   683,   485,   271,    -7,   279,   340,   194,   685,   413,
     461,   227,    30,   481,    83,   422,   507,   338,    68,   137,
     854,  -866,  -866,   137,   137,   116,   -74,    25,    25,  -866,
    -866,   361,  -866,   101,   161,   185,   -74,   392,  -866,    57,
     495,   495,   137,   116,   -74,    25,   459,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,   -74,    25,    25,
      25,    25,    25,    25,    80,   105,    80,   105,   930,   137,
    -866,  -866,   -74,   478,  -866,    57,     2,  -866,  -866,  -866,
     -74,  1447,   708,   105,   -74,  -115,  -866,  -866,   -74,    80,
     105,   100,   112,   -74,    80,   112,   -74,   100,    57,   137,
      57,   137,   137,   137,   -74,   849,   137,    57,   137,   137,
     137,   -74,   520,   116,   137,   137,   -74,    54,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,     8,
    -866,  -866,  -866,  -866,  -866,  -866,   -74,   348,  -866,  -866,
    -866,   -74,    25,   212,  -866,   137,  -866,   -74,    80,  1303,
     110,   137,   137,   137,   137,   137,   137,   -74,   504,   137,
    -866,   137,    57,  -866,   -74,    80,    25,   380,   112,   -74,
     -25,   137,    80,   112,   -74,    31,   137,    80,   126,   -74,
     230,    57,    57,   137,   137,   137,   -74,   683,   137,   137,
     137,    57,   137,   137,   137,   -74,   485,   137,    80,   112,
     -74,   271,    57,   137,   137,   -74,    -7,   137,    80,    80,
     -74,   279,  -866,    57,    57,   838,   758,   894,   -74,    25,
     340,    91,   462,   -74,   194,   137,   137,   137,   137,   137,
      42,    57,   137,   137,   137,   -74,   685,    57,    57,   137,
     -74,   413,   137,   137,   137,   137,    57,   137,   137,   137,
     137,   137,   -74,   461,  -866,   200,   -74,  1360,   227,   130,
     -74,    30,    57,   495,   495,    57,   495,   -74,   481,    57,
     -74,    83,   137,   137,   137,   137,   137,  -866,  -866,  -866,
     -74,   422,   137,    57,   137,   137,   116,   -74,   507,   -74,
     338,  -866,   116,   -74,    25,    68,  -866,  1447,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,  -180,  -866,  -866,  -866,  -866,  -866,  -866,  -180,
    -866,  -180,  -866,  -180,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,    12,  -866,  -866,    16,   137,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
      43,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
      28,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  1447,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,   137,  -866,  -866,    49,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,   137,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,   206,   214,   244,
     247,   253,   255,   263,   269,   295,   307,   309,   317,   320,
     341,   351,   355,   359,   367,   381,   383,   390,   394,   398,
     401,   408,   415,   417,   430,   433,   442,   447,   457,   463,
     465,   467,   471,   477,   484,   503,   505,   511,   530,   533,
     534,   539,   543,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -180,  -866,  -866,  -866,
    -866,  -866,  -180,  -866,  -180,  -866,  -180,  -866,  -180,  -866,
    -180,  -866,  -180,  -866,   541,  -866,  -180,  -866,  -180,  -866,
    -180,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  1447,
    -866,  -866,    60,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  1447,    40,   604,
     234,   655,    18,  -866,  -866,    74,  -866,  -866,    87,   137,
    -866,  -866,    89,  -866,  -866,   205,   137,  -866,  -866,    93,
    -866,    57,   137,   200,   495,   137,    50,   137,   116,   137,
     137,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,    59,   404,
     444,   826,   557,   638,    57,   108,  -866,  -866,   104,    20,
      20,   -74,   541,    36,   318,   633,   278,  -866,  -866,    97,
    -866,  -866,   137,   137,   137,   137,   -74,   604,  -866,    57,
      57,   -74,    25,   234,  -866,    57,   495,   495,   137,    57,
      57,   -74,    25,   655,    57,    57,  -866,   -74,    55,    18,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,   137,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,   555,  -866,  -866,
    -866,  -866,   137,   -74,    59,    57,    57,   137,   137,   -74,
     404,    57,    57,    57,   137,   137,   -74,   444,    48,   170,
      57,    57,   137,    57,   200,   137,   200,   137,    57,  -866,
     117,   -74,    80,   826,    57,   137,    57,  -866,    57,  -866,
      57,    57,  -866,  -866,   -74,   130,   126,   130,   112,   557,
     143,   172,    57,    57,   137,    57,    57,    57,   171,  -866,
    -866,   144,   -74,    20,    20,    20,    20,   638,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,     6,  -866,  -866,  -866,  -866,
    -866,   181,   151,   -74,    36,   137,   -74,   318,    56,  -866,
     137,  -866,   -74,    25,    80,   633,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,   200,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,    21,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,   200,   290,  -866,  -866,  -866,  -866
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -866,  -866,   -46,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
    -866,  -155,  -430,   153,  -137,    10,  -203,  -278,  -354,  -655,
    -328,   -44,  -866,   257,  -866,  -433,  -866,  -240,  -866,  -200,
    -866,  -826,  -866,  -272,  -866,  -441,  1114,  -202,   369,  -866,
    -866,  -125,    33,  -866,  -866,  -866,   349,  -866,  -866,   177,
     356,  -866,  -866,  -866,   173,   346,  -866,  -866,  -866,  -866,
    -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,  -866,   336,
    -866,  -866,  -866,  -368,  -866,  -866,  -866,  -218,    41,   342,
    -866,  -866,  -866,  -866,   345,  -866,  -866,  -514,   343,   267,
     285,  -866,  -866,  -866,  -363,  -866,  -866,  -866,  -866,  -866,
    -816,   334,  -866,  -866,  -866,  -365,  -866,  -866,  -866,   332,
    -866,  -866,  -866,   326,  -866,  -866,  -866,  -373,  -866,  -866,
    -866,  -866,  -386,  -866,  -866,  -866,  -866,  -866,  -866,  -866,
     327,  -866,  -866,  -866,   454,  -866,  -866,  -866,  -419,   321,
    -866,  -866,  -866,  -288,  -866,  -866,  -866,  -866,  -388,  -790,
    -767,  -866,  -866,  -866,  -866,  -866,  -260,  -866,  -866,  -866,
     323,  -866,  -866,  -866,  -866,   313,  -866,  -866,  -866,  -866,
    -866,   311,  -866,  -866,   434,  -273,  -866,  -866,  -866,   265,
     316,  -866,  -866,  -866,  -866,  -866,   328,  -866,  -866,  -865,
     322,  -866,  -866,  -866,   337,  -866,  -866,  -866,  -259,  -866,
    -866,  -866,  -866,   314,  -866,  -866,  -866,   324,  -866,  -866,
    -866,  -361,  -866,  -866,  -866,   315,  -866,  -866,  -866,   312,
    -866,  -866,  -866,   319,  -866,  -866,  -866,   302,  -866,  -866,
    -866,  -866,   299,  -866,  -866,  -866,   293,  -866,  -866,  -866,
    -332,  -866,  -866,  -866,   296,  -866,  -866,  -866,   283,  -866,
    -866,  -866,   280,  -866,  -866,  -866,  -866,  -335,  -866,  -866,
    -866,  -866,  -866,   287,  -866,  -866,  -795,  -183,  -866,  -866,
    -866,   289,  -866,  -866,  -866,   297,  -866,  -866,  -866,   286,
    -866,  -866,  -866,   276,  -866,  -866,  -866,   284,  -185,  -866,
     274,  -866,  -866,  -866,  -866
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -7
static const yytype_int16 yytable[] =
{
     144,   145,   616,   555,   482,   777,   531,   483,   484,   776,
     147,   569,   470,   299,   505,   141,   142,   575,   511,   141,
     142,   538,   514,   776,   776,   515,   300,   250,   257,   547,
     299,   305,   306,   299,   554,   517,   627,   628,   518,   519,
     520,   521,   522,   523,   137,   478,   141,   142,   313,   319,
     535,     4,   141,   142,   593,   594,   595,   263,   540,   650,
     139,   749,   544,   141,   142,   761,   546,  1102,   481,   956,
     457,   553,   495,   146,   556,  1173,   607,   141,   142,  1031,
      14,   612,   564,   141,   142,   382,   513,    22,   588,   571,
     141,   142,   141,   142,   576,   636,   141,   142,   797,     3,
     141,   142,   801,  1023,   471,   604,   541,   642,   141,   142,
     706,   807,   611,  1104,   581,   141,   142,   617,     8,   583,
     486,   299,   584,  1032,    10,   587,  1024,  1029,   250,   141,
     142,     8,   272,   141,   142,   598,  1117,   257,   641,    21,
     141,   142,   603,   934,   935,  1137,   605,   608,   651,   652,
     770,   307,   613,   199,   266,   267,   573,   621,     8,   479,
     480,   348,  1135,  1150,   629,     8,  1041,   383,   293,   294,
    1165,   384,   579,   638,   590,   449,   303,   304,   643,  1130,
     489,  1132,    22,   648,  1107,   251,   258,    10,   653,  1105,
    1147,  1138,   313,   254,   259,   439,   703,    10,   319,   704,
    1162,   742,  1149,   776,   494,   534,   314,   320,  1042,   476,
     936,   490,   841,   757,   315,   495,  1140,   496,   762,    31,
     842,    31,   539,   266,   267,  1164,   295,   472,   458,  1159,
     774,   578,   512,   962,   778,   823,   459,  1023,   785,   826,
     788,   789,  1035,   791,  1183,   792,   570,   309,   795,   827,
     843,   833,   351,   844,   525,   528,   525,   528,   802,   845,
    1024,   846,   750,   911,   838,   809,   830,   811,  1103,   847,
     957,   814,   837,   528,   815,   848,  1174,  1082,   318,   525,
     528,   586,   780,   907,   525,   470,   251,   324,   325,   559,
     326,   561,   562,   563,   254,   258,   566,   940,   568,   528,
     918,   849,   524,   259,   574,   528,   497,   610,   203,   204,
     942,   707,   945,   850,   298,   851,   949,   301,   302,   625,
    1057,   487,   817,   852,  1033,   632,   853,   527,  1030,   808,
     637,   401,   402,   640,   550,   813,   457,  1118,   525,   646,
     647,   592,   528,   528,   528,   596,   597,   854,   618,   600,
     314,   601,   782,   266,   267,   525,   320,   855,   315,   919,
     920,   856,   525,  1136,  1151,   857,   615,   525,   615,   755,
     756,  1166,   507,   858,   626,   626,   434,   435,   631,   765,
     633,   491,   635,   528,  1034,   771,   772,   859,   525,   860,
    1106,  1148,  1139,    36,   266,   267,   861,   626,   525,   525,
     862,  1163,   266,   267,   863,   498,   392,   864,   537,   798,
     799,   800,   200,   953,   865,   744,   745,   746,   747,   748,
     806,   866,   754,   867,   274,   200,   201,   202,   947,   615,
      43,   558,   764,   560,   766,   767,   868,   769,   528,   869,
     567,   773,   275,   834,   203,   204,   352,   357,   870,   626,
     378,   356,   388,   871,   908,   339,   340,   203,   204,   213,
     214,   215,   615,   872,   353,   358,   615,  1038,   379,   873,
     389,   874,   804,   875,   709,   615,  1119,   876,   341,   393,
     394,   710,   711,   877,   712,   950,   200,  1045,   713,   714,
     878,   955,   377,   965,   458,   602,   715,   716,   508,   509,
     387,  1056,   459,   417,   532,   418,   831,  1152,   419,   879,
     717,   880,    44,  1186,   623,   624,    45,   881,   203,   204,
     543,   960,   718,   719,   634,   218,  1185,   548,  1167,   219,
     220,   221,   966,   971,   824,   645,   882,   528,   829,   883,
     884,   342,   200,   835,   274,   885,   655,   656,   343,   886,
     943,   442,   443,   444,  1112,   205,  1114,   395,   396,   397,
     422,  1085,   275,   249,   753,  1131,   506,   967,   968,   958,
     759,   760,   972,   445,   203,   204,   452,   453,   708,   768,
     485,  1058,   516,   751,   368,  1060,  1061,  1062,   222,   536,
    1154,  1155,  1156,  1157,   533,   787,  1088,   423,   790,   910,
     454,   455,   794,   331,   542,  1160,   973,  1094,   974,   424,
     545,   557,  1101,   975,   549,   456,   805,   565,   572,   286,
    1122,   369,   352,  1134,   577,   836,   312,   357,   615,  1175,
     425,   426,   427,   428,   429,  1069,   840,   446,   582,  1158,
     353,   954,  1040,   591,   894,   358,   585,   488,   378,   599,
     430,   431,  1181,   223,   370,   371,   372,   373,   388,   994,
    1078,   995,   817,   606,   462,   463,   379,   996,   997,   998,
     999,   895,   896,   897,   374,   752,   389,   614,   609,  1000,
    1083,   630,   720,  1169,  1001,   332,   333,   334,   335,   336,
     287,   288,   289,   290,   464,   465,   644,   622,   649,  1039,
     639,   466,   705,   743,   161,   162,   163,   164,   165,   758,
     654,   959,  1002,   898,  1063,  1172,   775,   763,   781,  1067,
    1180,   924,  1068,  1071,  1072,   925,   926,   927,   786,  1076,
     928,  1184,  1077,   906,  1064,  1081,  1133,   793,  1010,  1011,
    1012,  1013,  1014,  1015,   810,  1003,     0,   803,   796,   816,
       0,  1181,  1121,     0,   812,   166,     0,     0,   912,   913,
    1016,  1087,   167,   168,   169,  1017,     0,  1093,  1018,   896,
     897,  1019,   200,   361,  1100,   362,     0,     0,   363,     0,
     929,   930,   914,   915,  1048,  1049,  1050,     0,   946,  1120,
       0,     0,   626,     0,   255,     0,   256,     0,  1020,  1051,
       0,     0,  1129,     0,   203,   204,   961,     0,     0,     0,
    1021,     0,     0,     0,  1179,     0,     0,   364,   365,     0,
    1153,     0,     0,     0,     0,     0,   405,   406,   407,   408,
     409,   410,   411,     0,     0,   941,     0,     0,   528,   944,
       0,  1168,     0,     0,  1171,     0,   948,     0,     0,   615,
    1177,     0,   952,  1178,     0,   615,   412,   528,   413,   626,
       0,     2,     3,     4,     5,     6,     0,     7,     8,     9,
      10,    11,    12,     0,   414,     0,     0,    13,    14,     0,
       0,     0,    16,    17,     0,    18,     0,    19,     0,     0,
       0,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,     0,    31,     0,    32,    33,    34,    35,    36,
      37,    38,    39,    40,     0,     0,    41,    42,  1086,   626,
       0,     0,  1059,   626,   626,   626,   978,   979,   980,   981,
     982,   983,   984,   985,   986,   987,     0,     0,  1073,   278,
     279,   280,   281,   282,   283,     0,    43,     0,   988,     0,
       0,     0,     0,   667,   668,   669,   670,   671,  1084,   672,
     673,   674,   675,   676,     0,   677,   678,   679,   680,     0,
     681,     0,   682,   683,   684,     0,     0,  1091,  1092,     0,
       0,     0,     0,     0,  1098,  1099,     0,     0,     0,     0,
     989,     0,  1110,     0,   951,  1113,     0,  1115,   990,     0,
       0,     0,   525,   227,   228,  1124,     0,   229,   230,   231,
     232,   233,   234,   235,   236,   626,   615,   626,     0,     0,
       0,     0,     0,     0,  1143,     0,     0,     0,    44,     0,
       0,     0,    45,   657,   658,   659,   660,   661,   662,   663,
       0,   664,   665,   666,     0,     0,     0,  1028,     0,     0,
       0,     0,     0,     0,     0,  1170,     0,     0,     0,     0,
    1176,     0,     0,     0,   525,     0,     0,     0,     0,     0,
       0,     0,  1065,  1066,    46,     0,   477,     0,  1070,     0,
       0,     0,  1074,  1075,     0,     0,     0,  1079,  1080,   685,
     686,   687,   688,   689,     0,   690,   691,   692,   693,   694,
       0,   695,   696,   697,   698,     0,   699,     0,   700,   701,
     702,     0,     0,     0,     0,     0,     0,     0,  1089,  1090,
       0,     0,     0,     0,  1095,  1096,  1097,     0,     0,     0,
       0,     0,     0,  1108,  1109,     0,  1111,     0,     0,     0,
       0,  1116,     0,     0,     0,     0,     0,  1123,     0,  1125,
       0,  1126,     0,  1127,  1128,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,  1141,  1142,     0,  1144,  1145,
    1146,   148,     0,   149,     0,   150,     0,   151,     0,   152,
       0,   153,     0,   154,     0,   155,     0,   156,     0,   157,
       0,   158,     0,   159,     0,   160,     0,     0,     0,   173,
       0,   174,     0,   175,     0,   176,     0,   177,     0,   178,
       0,   179,     0,   180,     0,   181,     0,   182,     0,   183,
       0,   184,     0,   185,     0,   186,     0,   187,     0,   188,
       0,   189,     0,   190,     0,   191,     0,   192,     0,   193,
       0,   194,     0,   195,     0,   196,    -6,     1,   197,   198,
       0,     0,     0,     2,     3,     4,     5,     6,     0,     7,
       8,     9,    10,    11,    12,     0,     0,     0,     0,    13,
      14,    15,     0,     0,    16,    17,     0,    18,     0,    19,
       0,     0,     0,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,     0,    31,     0,    32,    33,    34,
      35,    36,    37,    38,    39,    40,     0,     0,    41,    42,
       2,     3,     4,     5,     6,     0,     7,     8,     9,    10,
      11,    12,     0,     0,     0,     0,    13,    14,     0,     0,
       0,    16,    17,     0,    18,     0,    19,     0,    43,     0,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,     0,    31,     0,    32,    33,    34,    35,    36,    37,
      38,    39,    40,     0,     0,    41,    42,     2,     3,     4,
       5,     6,     0,     7,     8,     9,    10,    11,    12,     0,
       0,     0,     0,    13,    14,     0,     0,     0,    16,    17,
       0,    18,     0,    19,     0,    43,     0,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,     0,    31,
       0,    32,    33,    34,    35,    36,    37,    38,    39,    40,
      44,     0,    41,    42,    45,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    43,     0,     2,     3,     4,     5,     6,     0,
       7,     8,     9,    10,    11,    12,    46,     0,     0,     0,
      13,    14,     0,     0,     0,    16,    17,    44,    18,     0,
      19,    45,     0,     0,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,     0,    31,     0,    32,    33,
      34,    35,    36,    37,    38,    39,    40,     0,     0,    41,
      42,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    46,     0,   589,     0,     0,     0,     0,
       0,     0,     0,     0,    44,     0,     0,     0,    45,    43,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      46,     0,   779,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   819,     0,     0,     0,
       0,     0,     0,   820,     0,   821,     0,   822,     0,     0,
       0,    44,     0,     0,     0,    45,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     887,     0,     0,     0,     0,     0,   888,     0,   889,     0,
     890,     0,   891,     0,   892,     0,   893,     0,     0,     0,
     903,     0,   904,     0,   905
};

static const yytype_int16 yycheck[] =
{
      44,    45,   356,   275,   206,   435,   246,   207,   208,     3,
      56,   289,   197,     5,   216,     3,     4,   295,   221,     3,
       4,    19,   224,     3,     3,   225,   163,   152,   153,   269,
       5,   168,   169,     5,   274,   237,   364,   365,   238,   239,
     240,   241,   242,   243,   224,   200,     3,     4,   173,   174,
     252,     9,     3,     4,   332,   333,   334,   172,   260,   387,
       3,    19,   264,     3,     4,   419,   268,    19,   205,    19,
     195,   273,    30,     0,   276,    19,   348,     3,     4,   895,
      24,   353,   284,     3,     4,    92,   223,    39,   328,   291,
       3,     4,     3,     4,   296,   373,     3,     4,   452,     8,
       3,     4,   456,   893,    36,   345,   261,   379,     3,     4,
      19,   465,   352,   978,   316,     3,     4,   357,    14,   321,
      19,     5,   322,    19,    16,   327,   893,    19,   253,     3,
       4,    14,   165,     3,     4,   337,    19,   262,   378,    38,
       3,     4,   344,   125,   126,  1010,   346,   349,   388,   389,
     428,   225,   354,   143,   123,   124,   293,   359,    14,   203,
     204,   186,    19,    19,   366,    14,   130,   174,   114,   115,
      19,   178,   309,   375,   329,    92,   166,   167,   380,  1005,
      19,  1007,    39,   385,   979,   152,   153,    16,   390,    19,
      19,    19,   317,   152,   153,   165,   398,    16,   323,   399,
      19,   403,  1018,     3,    19,   249,   173,   174,   172,   199,
     192,    50,     6,   415,   173,    30,  1011,    32,   420,    49,
       6,    49,   220,   123,   124,  1041,   172,   159,   195,   223,
     432,   223,   222,   174,   436,   223,   195,  1027,   440,   223,
     443,   444,   222,   446,   223,   447,   290,   222,   450,   527,
       6,   223,   221,     6,   244,   245,   246,   247,   460,     6,
    1027,     6,   220,   223,   618,   467,   223,   469,   220,     6,
     220,   473,   223,   263,   474,     6,   220,   222,    66,   269,
     270,   325,   437,   223,   274,   470,   253,   177,   178,   279,
     180,   281,   282,   283,   253,   262,   286,   223,   288,   289,
      66,     6,   222,   262,   294,   295,   121,   351,    96,    97,
     223,   220,   223,     6,   161,     6,   223,   164,   165,   363,
     223,   220,   477,     6,   220,   369,     6,   222,   220,   466,
     374,   137,   138,   377,   222,   472,   461,   220,   328,   383,
     384,   331,   332,   333,   334,   335,   336,     6,   222,   339,
     317,   341,   222,   123,   124,   345,   323,     6,   317,   125,
     126,     6,   352,   220,   220,     6,   356,   357,   358,   413,
     414,   220,   219,     6,   364,   365,   149,   150,   368,   423,
     370,   220,   372,   373,   898,   429,   430,     6,   378,     6,
     220,   220,   220,    55,   123,   124,     6,   387,   388,   389,
       6,   220,   123,   124,     6,   220,    66,     6,   255,   453,
     454,   455,    64,   843,     6,   405,   406,   407,   408,   409,
     464,     6,   412,     6,   157,    64,    65,    66,   223,   419,
      92,   278,   422,   280,   424,   425,     6,   427,   428,     6,
     287,   431,   157,   580,    96,    97,   179,   180,     6,   439,
     183,   221,   185,     6,   782,    75,    76,    96,    97,    67,
      68,    69,   452,     6,   179,   180,   456,   900,   183,     6,
     185,     6,   462,     6,    12,   465,   990,     6,    98,   139,
     140,    19,    20,     6,    22,   839,    64,   169,    26,    27,
       6,   845,   221,    89,   461,   342,    34,    35,     3,     4,
     221,   223,   461,    90,   247,    92,   550,  1021,    95,     6,
      48,     6,   174,   223,   361,   362,   178,     6,    96,    97,
     263,   849,    60,    61,   371,    66,  1181,   270,  1042,    70,
      71,    72,   128,    89,   524,   382,     6,   527,   528,     6,
       6,   161,    64,   589,   277,     6,   393,   394,   168,     6,
     828,    70,    71,    72,   984,   194,   986,   217,   218,   219,
      99,     6,   277,    85,   411,  1006,   217,   163,   164,   847,
     417,   418,   128,    92,    96,    97,   154,   155,   401,   426,
     211,   909,   226,   410,    99,   913,   914,   915,   129,   253,
    1023,  1024,  1025,  1026,   248,   442,   964,   136,   445,   817,
     178,   179,   449,    99,   262,  1035,   162,   970,   164,   148,
     265,   277,   977,   169,   271,   193,   463,   285,   292,    99,
     993,   136,   355,  1009,   297,   615,   172,   360,   618,  1048,
     169,   170,   171,   172,   173,   923,   626,   156,   317,  1027,
     355,   844,   902,   330,   103,   360,   323,   213,   381,   338,
     189,   190,  1082,   194,   169,   170,   171,   172,   391,   102,
     933,   104,   817,   347,   157,   158,   381,   110,   111,   112,
     113,   130,   131,   132,   189,   410,   391,   355,   350,   122,
     939,   367,   220,  1044,   127,   181,   182,   183,   184,   185,
     170,   171,   172,   173,   187,   188,   381,   360,   386,   901,
     376,   194,   400,   404,   116,   117,   118,   119,   120,   416,
     391,   848,   155,   172,   916,  1047,   433,   421,   438,   921,
    1055,    66,   922,   926,   927,    70,    71,    72,   441,   931,
      75,  1161,   932,   779,   917,   937,  1008,   448,   100,   101,
     102,   103,   104,   105,   468,   188,    -1,   461,   451,   475,
      -1,  1181,   992,    -1,   470,   167,    -1,    -1,   154,   155,
     122,   963,   174,   175,   176,   127,    -1,   969,   130,   131,
     132,   133,    64,    90,   976,    92,    -1,    -1,    95,    -1,
     125,   126,   178,   179,   151,   152,   153,    -1,   832,   991,
      -1,    -1,   782,    -1,    86,    -1,    88,    -1,   160,   166,
      -1,    -1,  1004,    -1,    96,    97,   850,    -1,    -1,    -1,
     172,    -1,    -1,    -1,  1054,    -1,    -1,   134,   135,    -1,
    1022,    -1,    -1,    -1,    -1,    -1,   141,   142,   143,   144,
     145,   146,   147,    -1,    -1,   825,    -1,    -1,   828,   829,
      -1,  1043,    -1,    -1,  1046,    -1,   836,    -1,    -1,   839,
    1052,    -1,   842,  1053,    -1,   845,   171,   847,   173,   849,
      -1,     7,     8,     9,    10,    11,    -1,    13,    14,    15,
      16,    17,    18,    -1,   189,    -1,    -1,    23,    24,    -1,
      -1,    -1,    28,    29,    -1,    31,    -1,    33,    -1,    -1,
      -1,    37,    38,    39,    40,    41,    42,    43,    44,    45,
      46,    47,    -1,    49,    -1,    51,    52,    53,    54,    55,
      56,    57,    58,    59,    -1,    -1,    62,    63,   962,   909,
      -1,    -1,   912,   913,   914,   915,   100,   101,   102,   103,
     104,   105,   106,   107,   108,   109,    -1,    -1,   928,    90,
      91,    92,    93,    94,    95,    -1,    92,    -1,   122,    -1,
      -1,    -1,    -1,   195,   196,   197,   198,   199,   948,   201,
     202,   203,   204,   205,    -1,   207,   208,   209,   210,    -1,
     212,    -1,   214,   215,   216,    -1,    -1,   967,   968,    -1,
      -1,    -1,    -1,    -1,   974,   975,    -1,    -1,    -1,    -1,
     164,    -1,   982,    -1,   841,   985,    -1,   987,   172,    -1,
      -1,    -1,   992,    73,    74,   995,    -1,    77,    78,    79,
      80,    81,    82,    83,    84,  1005,  1006,  1007,    -1,    -1,
      -1,    -1,    -1,    -1,  1014,    -1,    -1,    -1,   174,    -1,
      -1,    -1,   178,   195,   196,   197,   198,   199,   200,   201,
      -1,   203,   204,   205,    -1,    -1,    -1,   894,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,  1045,    -1,    -1,    -1,    -1,
    1050,    -1,    -1,    -1,  1054,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   919,   920,   220,    -1,   222,    -1,   925,    -1,
      -1,    -1,   929,   930,    -1,    -1,    -1,   934,   935,   195,
     196,   197,   198,   199,    -1,   201,   202,   203,   204,   205,
      -1,   207,   208,   209,   210,    -1,   212,    -1,   214,   215,
     216,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   965,   966,
      -1,    -1,    -1,    -1,   971,   972,   973,    -1,    -1,    -1,
      -1,    -1,    -1,   980,   981,    -1,   983,    -1,    -1,    -1,
      -1,   988,    -1,    -1,    -1,    -1,    -1,   994,    -1,   996,
      -1,   998,    -1,  1000,  1001,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,  1012,  1013,    -1,  1015,  1016,
    1017,    57,    -1,    59,    -1,    61,    -1,    63,    -1,    65,
      -1,    67,    -1,    69,    -1,    71,    -1,    73,    -1,    75,
      -1,    77,    -1,    79,    -1,    81,    -1,    -1,    -1,    85,
      -1,    87,    -1,    89,    -1,    91,    -1,    93,    -1,    95,
      -1,    97,    -1,    99,    -1,   101,    -1,   103,    -1,   105,
      -1,   107,    -1,   109,    -1,   111,    -1,   113,    -1,   115,
      -1,   117,    -1,   119,    -1,   121,    -1,   123,    -1,   125,
      -1,   127,    -1,   129,    -1,   131,     0,     1,   134,   135,
      -1,    -1,    -1,     7,     8,     9,    10,    11,    -1,    13,
      14,    15,    16,    17,    18,    -1,    -1,    -1,    -1,    23,
      24,    25,    -1,    -1,    28,    29,    -1,    31,    -1,    33,
      -1,    -1,    -1,    37,    38,    39,    40,    41,    42,    43,
      44,    45,    46,    47,    -1,    49,    -1,    51,    52,    53,
      54,    55,    56,    57,    58,    59,    -1,    -1,    62,    63,
       7,     8,     9,    10,    11,    -1,    13,    14,    15,    16,
      17,    18,    -1,    -1,    -1,    -1,    23,    24,    -1,    -1,
      -1,    28,    29,    -1,    31,    -1,    33,    -1,    92,    -1,
      37,    38,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    -1,    49,    -1,    51,    52,    53,    54,    55,    56,
      57,    58,    59,    -1,    -1,    62,    63,     7,     8,     9,
      10,    11,    -1,    13,    14,    15,    16,    17,    18,    -1,
      -1,    -1,    -1,    23,    24,    -1,    -1,    -1,    28,    29,
      -1,    31,    -1,    33,    -1,    92,    -1,    37,    38,    39,
      40,    41,    42,    43,    44,    45,    46,    47,    -1,    49,
      -1,    51,    52,    53,    54,    55,    56,    57,    58,    59,
     174,    -1,    62,    63,   178,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    92,    -1,     7,     8,     9,    10,    11,    -1,
      13,    14,    15,    16,    17,    18,   220,    -1,    -1,    -1,
      23,    24,    -1,    -1,    -1,    28,    29,   174,    31,    -1,
      33,   178,    -1,    -1,    37,    38,    39,    40,    41,    42,
      43,    44,    45,    46,    47,    -1,    49,    -1,    51,    52,
      53,    54,    55,    56,    57,    58,    59,    -1,    -1,    62,
      63,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   220,    -1,   222,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   174,    -1,    -1,    -1,   178,    92,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     220,    -1,   222,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   492,    -1,    -1,    -1,
      -1,    -1,    -1,   499,    -1,   501,    -1,   503,    -1,    -1,
      -1,   174,    -1,    -1,    -1,   178,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   220,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     716,    -1,    -1,    -1,    -1,    -1,   722,    -1,   724,    -1,
     726,    -1,   728,    -1,   730,    -1,   732,    -1,    -1,    -1,
     736,    -1,   738,    -1,   740
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,     1,     7,     8,     9,    10,    11,    13,    14,    15,
      16,    17,    18,    23,    24,    25,    28,    29,    31,    33,
      37,    38,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    49,    51,    52,    53,    54,    55,    56,    57,    58,
      59,    62,    63,    92,   174,   178,   220,   227,   228,   229,
     230,   231,   233,   234,   235,   236,   237,   270,   271,   274,
     275,   279,   280,   293,   294,   297,   298,   308,   309,   312,
     313,   318,   319,   325,   326,   329,   330,   337,   338,   341,
     342,   358,   359,   363,   364,   367,   368,   389,   390,   395,
     396,   399,   400,   410,   411,   414,   415,   418,   419,   422,
     423,   431,   432,   435,   436,   443,   444,   447,   448,   451,
     452,   456,   457,   460,   461,   464,   465,   472,   473,   476,
     477,   481,   482,   491,   492,   499,   500,   503,   504,   507,
     508,   511,   512,   514,   515,   519,   520,   224,   262,     3,
     239,     3,     4,   241,   247,   247,     0,   228,   262,   262,
     262,   262,   262,   262,   262,   262,   262,   262,   262,   262,
     262,   116,   117,   118,   119,   120,   167,   174,   175,   176,
     360,   361,   362,   262,   262,   262,   262,   262,   262,   262,
     262,   262,   262,   262,   262,   262,   262,   262,   262,   262,
     262,   262,   262,   262,   262,   262,   262,   262,   262,   241,
      64,    65,    66,    96,    97,   194,   264,   265,   266,   267,
     268,   269,   304,    67,    68,    69,   272,   273,    66,    70,
      71,    72,   129,   194,   276,   277,   278,    73,    74,    77,
      78,    79,    80,    81,    82,    83,    84,   281,   282,   283,
     284,   285,   286,   287,   288,   289,   290,   291,   292,    85,
     267,   268,   295,   296,   304,    86,    88,   267,   268,   304,
     305,   306,   307,   172,   310,   311,   123,   124,   314,   315,
     316,   317,   165,   324,   315,   316,   327,   328,    90,    91,
      92,    93,    94,    95,   335,   336,    99,   170,   171,   172,
     173,   339,   340,   114,   115,   172,   356,   357,   239,     5,
     240,   239,   239,   241,   241,   240,   240,   225,   263,   222,
     240,   255,   360,   267,   268,   304,   365,   366,    66,   267,
     268,   386,   387,   388,   177,   178,   180,   391,   392,   393,
     394,    99,   181,   182,   183,   184,   185,   397,   398,    75,
      76,    98,   161,   168,   406,   407,   408,   409,   186,   412,
     413,   221,   315,   316,   416,   417,   221,   315,   316,   420,
     421,    90,    92,    95,   134,   135,   429,   430,    99,   136,
     169,   170,   171,   172,   189,   433,   434,   221,   315,   316,
     441,   442,    92,   174,   178,   445,   446,   221,   315,   316,
     449,   450,    66,   139,   140,   217,   218,   219,   453,   454,
     455,   137,   138,   458,   459,   141,   142,   143,   144,   145,
     146,   147,   171,   173,   189,   462,   463,    90,    92,    95,
     470,   471,    99,   136,   148,   169,   170,   171,   172,   173,
     189,   190,   474,   475,   149,   150,   478,   479,   480,   165,
     489,   490,    70,    71,    72,    92,   156,   497,   498,    92,
     501,   502,   154,   155,   178,   179,   193,   267,   268,   304,
     505,   506,   157,   158,   187,   188,   194,   509,   510,   513,
     514,    36,   159,   516,   517,   518,   241,   222,   237,   247,
     247,   240,   263,   255,   255,   264,    19,   220,   400,    19,
      50,   220,   495,   496,    19,    30,    32,   121,   220,   372,
     373,   404,   405,   427,   428,   263,   272,   239,     3,     4,
     242,   242,   241,   240,   263,   255,   276,   263,   255,   255,
     255,   255,   255,   255,   222,   241,   253,   222,   241,   243,
     249,   253,   249,   281,   247,   263,   295,   239,    19,   220,
     263,   237,   305,   249,   263,   310,   263,   253,   249,   314,
     222,   247,   259,   263,   253,   259,   263,   327,   239,   241,
     239,   241,   241,   241,   263,   335,   241,   239,   241,   243,
     247,   263,   339,   240,   241,   243,   263,   356,   223,   240,
     254,   263,   365,   263,   255,   386,   247,   263,   253,   222,
     237,   391,   241,   243,   243,   243,   241,   241,   263,   397,
     241,   241,   239,   263,   253,   255,   406,   259,   263,   412,
     247,   253,   259,   263,   416,   241,   244,   253,   222,   244,
     261,   263,   420,   239,   239,   247,   241,   246,   246,   263,
     429,   241,   247,   241,   239,   241,   243,   247,   263,   433,
     247,   253,   259,   263,   441,   239,   247,   247,   263,   445,
     246,   253,   253,   263,   449,   239,   239,   195,   196,   197,
     198,   199,   200,   201,   203,   204,   205,   195,   196,   197,
     198,   199,   201,   202,   203,   204,   205,   207,   208,   209,
     210,   212,   214,   215,   216,   195,   196,   197,   198,   199,
     201,   202,   203,   204,   205,   207,   208,   209,   210,   212,
     214,   215,   216,   263,   255,   453,    19,   220,   275,    12,
      19,    20,    22,    26,    27,    34,    35,    48,    60,    61,
     220,   232,   301,   302,   322,   323,   333,   334,   346,   347,
     354,   355,   380,   381,   384,   385,   439,   440,   468,   469,
     487,   488,   263,   458,   241,   241,   241,   241,   241,    19,
     220,   280,   405,   239,   241,   247,   247,   263,   462,   239,
     239,   244,   263,   470,   241,   247,   241,   241,   239,   241,
     243,   247,   247,   241,   263,   474,     3,   238,   263,   222,
     237,   478,   222,   246,   257,   263,   489,   239,   242,   242,
     239,   242,   263,   497,   239,   263,   501,   244,   247,   247,
     247,   244,   263,   505,   241,   239,   247,   244,   240,   263,
     509,   263,   513,   240,   263,   255,   516,   237,   303,   262,
     262,   262,   262,   223,   241,   252,   223,   243,   248,   241,
     223,   247,   258,   223,   240,   228,   241,   223,   244,   260,
     241,     6,     6,     6,     6,     6,     6,     6,     6,     6,
       6,     6,     6,     6,     6,     6,     6,     6,     6,     6,
       6,     6,     6,     6,     6,     6,     6,     6,     6,     6,
       6,     6,     6,     6,     6,     6,     6,     6,     6,     6,
       6,     6,     6,     6,     6,     6,     6,   262,   262,   262,
     262,   262,   262,   262,   103,   130,   131,   132,   172,   375,
     376,   382,   383,   262,   262,   262,   228,   223,   246,   256,
     303,   223,   154,   155,   178,   179,   493,   494,    66,   125,
     126,   369,   370,   371,    66,    70,    71,    72,    75,   125,
     126,   401,   402,   403,   125,   126,   192,   424,   425,   426,
     223,   241,   223,   243,   241,   223,   247,   223,   241,   223,
     244,   239,   241,   238,   242,   244,    19,   220,   243,   240,
     246,   247,   174,   299,   300,    89,   128,   163,   164,   320,
     321,    89,   128,   162,   164,   169,   331,   332,   100,   101,
     102,   103,   104,   105,   106,   107,   108,   109,   122,   164,
     172,   343,   344,   345,   102,   104,   110,   111,   112,   113,
     122,   127,   155,   188,   348,   349,   350,   351,   352,   353,
     100,   101,   102,   103,   104,   105,   122,   127,   130,   133,
     160,   172,   374,   375,   376,   377,   378,   379,   239,    19,
     220,   326,    19,   220,   313,   222,   238,   251,   251,   263,
     382,   130,   172,   437,   438,   169,   466,   467,   151,   152,
     153,   166,   483,   484,   485,   486,   223,   223,   246,   241,
     246,   246,   246,   263,   493,   239,   239,   263,   255,   369,
     239,   242,   242,   241,   239,   239,   263,   255,   401,   239,
     239,   263,   222,   424,   241,     6,   247,   263,   299,   239,
     239,   241,   241,   263,   320,   239,   239,   239,   241,   241,
     263,   331,    19,   220,   415,    19,   220,   492,   239,   239,
     241,   239,   238,   241,   238,   241,   239,    19,   220,   313,
     263,   253,   343,   239,   241,   239,   239,   239,   239,   263,
     257,   261,   257,   259,   348,    19,   220,   415,    19,   220,
     492,   239,   239,   241,   239,   239,   239,    19,   220,   326,
      19,   220,   313,   263,   251,   251,   251,   251,   374,   223,
     238,   250,    19,   220,   326,    19,   220,   313,   263,   437,
     241,   263,   466,    19,   220,   364,   241,   263,   255,   253,
     483,   238,   245,   223,   238,   245,   223
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto yyerrlab

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      yytoken = YYTRANSLATE (yychar);				\
      YYPOPSTACK (1);						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (YYID (N))                                                    \
	{								\
	  (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;	\
	  (Current).first_column = YYRHSLOC (Rhs, 1).first_column;	\
	  (Current).last_line    = YYRHSLOC (Rhs, N).last_line;		\
	  (Current).last_column  = YYRHSLOC (Rhs, N).last_column;	\
	}								\
      else								\
	{								\
	  (Current).first_line   = (Current).last_line   =		\
	    YYRHSLOC (Rhs, 0).last_line;				\
	  (Current).first_column = (Current).last_column =		\
	    YYRHSLOC (Rhs, 0).last_column;				\
	}								\
    while (YYID (0))
#endif


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if YYLTYPE_IS_TRIVIAL
#  define YY_LOCATION_PRINT(File, Loc)			\
     fprintf (File, "%d.%d-%d.%d",			\
	      (Loc).first_line, (Loc).first_column,	\
	      (Loc).last_line,  (Loc).last_column)
# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (YYLEX_PARAM)
#else
# define YYLEX yylex ()
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      yy_symbol_print (stderr,						  \
		  Type, Value); \
      YYFPRINTF (stderr, "\n");						  \
    }									  \
} while (YYID (0))


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  switch (yytype)
    {
      default:
	break;
    }
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print (yytype_int16 *bottom, yytype_int16 *top)
#else
static void
yy_stack_print (bottom, top)
    yytype_int16 *bottom;
    yytype_int16 *top;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; bottom <= top; ++bottom)
    YYFPRINTF (stderr, " %d", *bottom);
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print (YYSTYPE *yyvsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yyrule)
    YYSTYPE *yyvsp;
    int yyrule;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      fprintf (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       		       );
      fprintf (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif



#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen (const char *yystr)
#else
static YYSIZE_T
yystrlen (yystr)
    const char *yystr;
#endif
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy (char *yydest, const char *yysrc)
#else
static char *
yystpcpy (yydest, yysrc)
    char *yydest;
    const char *yysrc;
#endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
	switch (*++yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (yyres)
	      yyres[yyn] = *yyp;
	    yyn++;
	    break;

	  case '"':
	    if (yyres)
	      yyres[yyn] = '\0';
	    return yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into YYRESULT an error message about the unexpected token
   YYCHAR while in state YYSTATE.  Return the number of bytes copied,
   including the terminating null byte.  If YYRESULT is null, do not
   copy anything; just return the number of bytes that would be
   copied.  As a special case, return 0 if an ordinary "syntax error"
   message will do.  Return YYSIZE_MAXIMUM if overflow occurs during
   size calculation.  */
static YYSIZE_T
yysyntax_error (char *yyresult, int yystate, int yychar)
{
  int yyn = yypact[yystate];

  if (! (YYPACT_NINF < yyn && yyn <= YYLAST))
    return 0;
  else
    {
      int yytype = YYTRANSLATE (yychar);
      YYSIZE_T yysize0 = yytnamerr (0, yytname[yytype]);
      YYSIZE_T yysize = yysize0;
      YYSIZE_T yysize1;
      int yysize_overflow = 0;
      enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
      char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
      int yyx;

# if 0
      /* This is so xgettext sees the translatable formats that are
	 constructed on the fly.  */
      YY_("syntax error, unexpected %s");
      YY_("syntax error, unexpected %s, expecting %s");
      YY_("syntax error, unexpected %s, expecting %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
# endif
      char *yyfmt;
      char const *yyf;
      static char const yyunexpected[] = "syntax error, unexpected %s";
      static char const yyexpecting[] = ", expecting %s";
      static char const yyor[] = " or %s";
      char yyformat[sizeof yyunexpected
		    + sizeof yyexpecting - 1
		    + ((YYERROR_VERBOSE_ARGS_MAXIMUM - 2)
		       * (sizeof yyor - 1))];
      char const *yyprefix = yyexpecting;

      /* Start YYX at -YYN if negative to avoid negative indexes in
	 YYCHECK.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;

      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yycount = 1;

      yyarg[0] = yytname[yytype];
      yyfmt = yystpcpy (yyformat, yyunexpected);

      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
	if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	  {
	    if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
	      {
		yycount = 1;
		yysize = yysize0;
		yyformat[sizeof yyunexpected - 1] = '\0';
		break;
	      }
	    yyarg[yycount++] = yytname[yyx];
	    yysize1 = yysize + yytnamerr (0, yytname[yyx]);
	    yysize_overflow |= (yysize1 < yysize);
	    yysize = yysize1;
	    yyfmt = yystpcpy (yyfmt, yyprefix);
	    yyprefix = yyor;
	  }

      yyf = YY_(yyformat);
      yysize1 = yysize + yystrlen (yyf);
      yysize_overflow |= (yysize1 < yysize);
      yysize = yysize1;

      if (yysize_overflow)
	return YYSIZE_MAXIMUM;

      if (yyresult)
	{
	  /* Avoid sprintf, as that infringes on the user's name space.
	     Don't have undefined behavior even if the translation
	     produced a string with the wrong number of "%s"s.  */
	  char *yyp = yyresult;
	  int yyi = 0;
	  while ((*yyp = *yyf) != '\0')
	    {
	      if (*yyp == '%' && yyf[1] == 's' && yyi < yycount)
		{
		  yyp += yytnamerr (yyp, yyarg[yyi++]);
		  yyf += 2;
		}
	      else
		{
		  yyp++;
		  yyf++;
		}
	    }
	}
      return yysize;
    }
}
#endif /* YYERROR_VERBOSE */


/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yymsg, yytype, yyvaluep)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  YYUSE (yyvaluep);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
	break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */

#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse (void *YYPARSE_PARAM);
#else
int yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */



/* The look-ahead symbol.  */
int yychar;

/* The semantic value of the look-ahead symbol.  */
YYSTYPE yylval;

/* Number of syntax errors so far.  */
int yynerrs;



/*----------.
| yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void *YYPARSE_PARAM)
#else
int
yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{
  
  int yystate;
  int yyn;
  int yyresult;
  /* Number of tokens to shift before error messages enabled.  */
  int yyerrstatus;
  /* Look-ahead token as an internal (translated) token number.  */
  int yytoken = 0;
#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

  /* Three stacks and their tools:
     `yyss': related to states,
     `yyvs': related to semantic values,
     `yyls': related to locations.

     Refer to the stacks thru separate pointers, to allow yyoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  yytype_int16 yyssa[YYINITDEPTH];
  yytype_int16 *yyss = yyssa;
  yytype_int16 *yyssp;

  /* The semantic value stack.  */
  YYSTYPE yyvsa[YYINITDEPTH];
  YYSTYPE *yyvs = yyvsa;
  YYSTYPE *yyvsp;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  YYSIZE_T yystacksize = YYINITDEPTH;

  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;


  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY;		/* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  yyssp = yyss;
  yyvsp = yyvs;

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	yytype_int16 *yyss1 = yyss;


	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),

		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	yytype_int16 *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyexhaustedlab;
	YYSTACK_RELOCATE (yyss);
	YYSTACK_RELOCATE (yyvs);

#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;


      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     look-ahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to look-ahead token.  */
  yyn = yypact[yystate];
  if (yyn == YYPACT_NINF)
    goto yydefault;

  /* Not known => get a look-ahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid look-ahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yyn == 0 || yyn == YYTABLE_NINF)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the look-ahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token unless it is eof.  */
  if (yychar != YYEOF)
    yychar = YYEMPTY;

  yystate = yyn;
  *++yyvsp = yylval;

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 3:
#line 109 "VRML97.y"
    {YYABORT;;}
    break;

  case 4:
#line 110 "VRML97.y"
    {YYABORT;;}
    break;

  case 65:
#line 205 "VRML97.y"
    {
			AddSFInt32((yyvsp[(1) - (1)].ival));
		;}
    break;

  case 67:
#line 216 "VRML97.y"
    {
			AddSFString((yyvsp[(1) - (1)].sval));
		;}
    break;

  case 68:
#line 223 "VRML97.y"
    {
			AddSFFloat((yyvsp[(1) - (1)].fval));
		;}
    break;

  case 69:
#line 227 "VRML97.y"
    {
			(yyval.fval) = (float)(yyvsp[(1) - (1)].ival);
			AddSFFloat((float)(yyvsp[(1) - (1)].ival));
		;}
    break;

  case 71:
#line 235 "VRML97.y"
    {(yyval.fval) = (float)(yyvsp[(1) - (1)].ival);;}
    break;

  case 72:
#line 240 "VRML97.y"
    {
			gColor[0] = (yyvsp[(1) - (3)].fval);
			gColor[1] = (yyvsp[(2) - (3)].fval);
			gColor[2] = (yyvsp[(3) - (3)].fval);
			AddSFColor(gColor);
	    ;}
    break;

  case 73:
#line 250 "VRML97.y"
    {
			gRotation[0] = (yyvsp[(1) - (4)].fval);
			gRotation[1] = (yyvsp[(2) - (4)].fval);
			gRotation[2] = (yyvsp[(3) - (4)].fval);
			gRotation[3] = (yyvsp[(4) - (4)].fval);
			AddSFRotation(gRotation);
		;}
    break;

  case 74:
#line 260 "VRML97.y"
    {;}
    break;

  case 76:
#line 280 "VRML97.y"
    {
			gVec2f[0] = (yyvsp[(1) - (2)].fval);
			gVec2f[1] = (yyvsp[(2) - (2)].fval);
			AddSFVec2f(gVec2f);
		;}
    break;

  case 77:
#line 289 "VRML97.y"
    {
			gVec3f[0] = (yyvsp[(1) - (3)].fval);
			gVec3f[1] = (yyvsp[(2) - (3)].fval);
			gVec3f[2] = (yyvsp[(3) - (3)].fval);
			AddSFVec3f(gVec3f);
		;}
    break;

  case 83:
#line 309 "VRML97.y"
    {;}
    break;

  case 84:
#line 310 "VRML97.y"
    {;}
    break;

  case 85:
#line 314 "VRML97.y"
    {;}
    break;

  case 86:
#line 315 "VRML97.y"
    {;}
    break;

  case 87:
#line 316 "VRML97.y"
    {;}
    break;

  case 88:
#line 321 "VRML97.y"
    {;}
    break;

  case 89:
#line 322 "VRML97.y"
    {;}
    break;

  case 90:
#line 326 "VRML97.y"
    {;}
    break;

  case 91:
#line 327 "VRML97.y"
    {;}
    break;

  case 92:
#line 328 "VRML97.y"
    {;}
    break;

  case 93:
#line 332 "VRML97.y"
    {;}
    break;

  case 94:
#line 333 "VRML97.y"
    {;}
    break;

  case 95:
#line 337 "VRML97.y"
    {;}
    break;

  case 96:
#line 338 "VRML97.y"
    {;}
    break;

  case 97:
#line 339 "VRML97.y"
    {;}
    break;

  case 117:
#line 396 "VRML97.y"
    {
			ParserPushNode(VRML97_ANCHOR_PARAMETER, ParserGetCurrentNode());
		;}
    break;

  case 118:
#line 403 "VRML97.y"
    {
			ParserPushNode(VRML97_ANCHOR_URL, ParserGetCurrentNode());
		;}
    break;

  case 119:
#line 410 "VRML97.y"
    {
			((AnchorNode *)ParserGetCurrentNode())->setBoundingBoxCenter(gVec3f);
		;}
    break;

  case 120:
#line 417 "VRML97.y"
    {
			((AnchorNode *)ParserGetCurrentNode())->setBoundingBoxSize(gVec3f);
		;}
    break;

  case 122:
#line 425 "VRML97.y"
    {
			((AnchorNode *)ParserGetCurrentNode())->setDescription((yyvsp[(2) - (2)].sval));
		;}
    break;

  case 123:
#line 430 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 124:
#line 434 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 127:
#line 443 "VRML97.y"
    {   
			AnchorNode	*anchor = new AnchorNode();
			anchor->setName(GetDEFName());
			ParserAddNode(anchor);
			ParserPushNode(VRML97_ANCHOR, anchor);
		;}
    break;

  case 128:
#line 453 "VRML97.y"
    {
			AnchorNode *anchor = (AnchorNode *)ParserGetCurrentNode();
			anchor->initialize();
			ParserPopNode();
		;}
    break;

  case 142:
#line 487 "VRML97.y"
    {
			AppearanceNode	*appearance = new AppearanceNode();
			appearance->setName(GetDEFName());
			ParserAddNode(appearance);
			ParserPushNode(VRML97_APPEARANCE, appearance);
		;}
    break;

  case 143:
#line 497 "VRML97.y"
    {
			AppearanceNode	*appearance = (AppearanceNode *)ParserGetCurrentNode();
			appearance->initialize();
			ParserPopNode();
		;}
    break;

  case 146:
#line 517 "VRML97.y"
    {
			ParserPushNode(VRML97_AUDIOCLIP_URL, ParserGetCurrentNode());
		;}
    break;

  case 147:
#line 524 "VRML97.y"
    {
			((AudioClipNode *)ParserGetCurrentNode())->setDescription((yyvsp[(2) - (2)].sval));
		;}
    break;

  case 148:
#line 528 "VRML97.y"
    {
			((AudioClipNode *)ParserGetCurrentNode())->setLoop((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 149:
#line 532 "VRML97.y"
    {
			((AudioClipNode *)ParserGetCurrentNode())->setPitch((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 150:
#line 536 "VRML97.y"
    {
			((AudioClipNode *)ParserGetCurrentNode())->setStartTime((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 151:
#line 540 "VRML97.y"
    {
			((AudioClipNode *)ParserGetCurrentNode())->setStopTime((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 152:
#line 544 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 153:
#line 551 "VRML97.y"
    {
			AudioClipNode	*audioClip = new AudioClipNode();
			audioClip->setName(GetDEFName());
			ParserAddNode(audioClip);
			ParserPushNode(VRML97_AUDIOCLIP, audioClip);
		;}
    break;

  case 154:
#line 561 "VRML97.y"
    {
			AudioClipNode *audioClip = (AudioClipNode *)ParserGetCurrentNode();
			audioClip->initialize();
			ParserPopNode();
		;}
    break;

  case 157:
#line 581 "VRML97.y"
    {
			ParserPushNode(VRML97_BACKGROUND_BACKURL, ParserGetCurrentNode());
		;}
    break;

  case 158:
#line 588 "VRML97.y"
    {
			ParserPushNode(VRML97_BACKGROUND_BOTTOMURL, ParserGetCurrentNode());
		;}
    break;

  case 159:
#line 595 "VRML97.y"
    {
			ParserPushNode(VRML97_BACKGROUND_FRONTURL, ParserGetCurrentNode());
		;}
    break;

  case 160:
#line 602 "VRML97.y"
    {
			ParserPushNode(VRML97_BACKGROUND_LEFTURL, ParserGetCurrentNode());
		;}
    break;

  case 161:
#line 609 "VRML97.y"
    {
			ParserPushNode(VRML97_BACKGROUND_RIGHTURL, ParserGetCurrentNode());
		;}
    break;

  case 162:
#line 616 "VRML97.y"
    {
			ParserPushNode(VRML97_BACKGROUND_TOPURL, ParserGetCurrentNode());
		;}
    break;

  case 163:
#line 623 "VRML97.y"
    {
			ParserPushNode(VRML97_BACKGROUND_GROUNDANGLE, ParserGetCurrentNode());
		;}
    break;

  case 164:
#line 630 "VRML97.y"
    {
			ParserPushNode(VRML97_BACKGROUND_GROUNDCOLOR, ParserGetCurrentNode());
		;}
    break;

  case 165:
#line 637 "VRML97.y"
    {
			ParserPushNode(VRML97_BACKGROUND_SKYANGLE, ParserGetCurrentNode());
		;}
    break;

  case 166:
#line 644 "VRML97.y"
    {
			ParserPushNode(VRML97_BACKGROUND_SKYCOLOR, ParserGetCurrentNode());
		;}
    break;

  case 167:
#line 651 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 168:
#line 655 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 169:
#line 659 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 170:
#line 663 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 171:
#line 667 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 172:
#line 671 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 173:
#line 675 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 174:
#line 679 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 175:
#line 683 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 176:
#line 687 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 177:
#line 694 "VRML97.y"
    {
			BackgroundNode *bg = new BackgroundNode();
			bg->setName(GetDEFName());
			ParserAddNode(bg);
			ParserPushNode(VRML97_BACKGROUND, bg);
		;}
    break;

  case 178:
#line 704 "VRML97.y"
    {
			BackgroundNode *bg = (BackgroundNode *)ParserGetCurrentNode();
			bg->initialize();
			ParserPopNode();
		;}
    break;

  case 182:
#line 725 "VRML97.y"
    {
			((BillboardNode *)ParserGetCurrentNode())->setAxisOfRotation(gVec3f);
		;}
    break;

  case 185:
#line 734 "VRML97.y"
    {   
			BillboardNode *billboard = new BillboardNode();
			billboard->setName(GetDEFName());
			ParserAddNode(billboard);
			ParserPushNode(VRML97_BILLBOARD, billboard);
		;}
    break;

  case 186:
#line 744 "VRML97.y"
    {
			BillboardNode *billboard = (BillboardNode *)ParserGetCurrentNode();
			billboard->initialize();
			ParserPopNode();
		;}
    break;

  case 189:
#line 764 "VRML97.y"
    {
			((BoxNode *)ParserGetCurrentNode())->setSize(gVec3f);
		;}
    break;

  case 190:
#line 771 "VRML97.y"
    {
			BoxNode *box = new BoxNode();
			box->setName(GetDEFName());
			ParserAddNode(box);
			ParserPushNode(VRML97_BOX, box);
		;}
    break;

  case 191:
#line 781 "VRML97.y"
    {
			BoxNode *box = (BoxNode *)ParserGetCurrentNode();
			box->initialize();
			ParserPopNode();
		;}
    break;

  case 198:
#line 817 "VRML97.y"
    {
			ParserPushNode(VRML97_COLLISION_PROXY, ParserGetCurrentNode());
		;}
    break;

  case 200:
#line 825 "VRML97.y"
    {
			((CollisionNode *)ParserGetCurrentNode())->setCollide((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 205:
#line 833 "VRML97.y"
    {
			ParserPopNode();							
		;}
    break;

  case 206:
#line 840 "VRML97.y"
    {   
			CollisionNode *collision = new CollisionNode();
			collision->setName(GetDEFName());
			ParserAddNode(collision);
			ParserPushNode(VRML97_BOX, collision);
		;}
    break;

  case 207:
#line 850 "VRML97.y"
    {
			CollisionNode *collision = (CollisionNode *)ParserGetCurrentNode();
			collision->initialize();
			ParserPopNode();
		;}
    break;

  case 211:
#line 874 "VRML97.y"
    {
			ColorNode *color = new ColorNode();
			color->setName(GetDEFName());
			ParserAddNode(color);
			ParserPushNode(VRML97_COLOR, color);
		;}
    break;

  case 212:
#line 884 "VRML97.y"
    {
			ColorNode *color = (ColorNode *)ParserGetCurrentNode();
			color->initialize();
			ParserPopNode();
		;}
    break;

  case 215:
#line 904 "VRML97.y"
    {
			ParserPushNode(VRML97_INTERPOLATOR_KEY, ParserGetCurrentNode());
		;}
    break;

  case 216:
#line 911 "VRML97.y"
    {
			ParserPushNode(VRML97_INTERPOLATOR_KEYVALUE, ParserGetCurrentNode());
		;}
    break;

  case 217:
#line 918 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 218:
#line 922 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 219:
#line 929 "VRML97.y"
    {
			ColorInterpolatorNode *colInterp = new ColorInterpolatorNode();
			colInterp->setName(GetDEFName());
			ParserAddNode(colInterp);
			ParserPushNode(VRML97_COLORINTERPOLATOR, colInterp);
		;}
    break;

  case 220:
#line 939 "VRML97.y"
    {
			ColorInterpolatorNode *colInterp = (ColorInterpolatorNode *)ParserGetCurrentNode();
			colInterp->initialize();
			ParserPopNode();
		;}
    break;

  case 223:
#line 959 "VRML97.y"
    {
			((ConeNode *)ParserGetCurrentNode())->setSide((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 224:
#line 963 "VRML97.y"
    {
			((ConeNode *)ParserGetCurrentNode())->setBottom((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 225:
#line 967 "VRML97.y"
    {
			((ConeNode *)ParserGetCurrentNode())->setBottomRadius((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 226:
#line 971 "VRML97.y"
    {
			((ConeNode *)ParserGetCurrentNode())->setHeight((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 227:
#line 978 "VRML97.y"
    {
			ConeNode *cone = new ConeNode();
			cone->setName(GetDEFName());
			ParserAddNode(cone);
			ParserPushNode(VRML97_CONE, cone);
		;}
    break;

  case 228:
#line 988 "VRML97.y"
    {
			ConeNode *cone = (ConeNode *)ParserGetCurrentNode();
			cone->initialize();
			ParserPopNode();
		;}
    break;

  case 231:
#line 1008 "VRML97.y"
    {
			CoordinateNode *coord = new CoordinateNode();
			coord->setName(GetDEFName());
			ParserAddNode(coord);
			ParserPushNode(VRML97_COORDINATE, coord);
		;}
    break;

  case 232:
#line 1018 "VRML97.y"
    {
			CoordinateNode *coord = (CoordinateNode *)ParserGetCurrentNode();
			coord->initialize();
			ParserPopNode();
		;}
    break;

  case 235:
#line 1038 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 236:
#line 1042 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 237:
#line 1049 "VRML97.y"
    {
			CoordinateInterpolatorNode *coordInterp = new CoordinateInterpolatorNode();
			coordInterp->setName(GetDEFName());
			ParserAddNode(coordInterp);
			ParserPushNode(VRML97_COORDINATEINTERPOLATOR, coordInterp);
		;}
    break;

  case 238:
#line 1059 "VRML97.y"
    {
			CoordinateInterpolatorNode *coordInterp = (CoordinateInterpolatorNode *)ParserGetCurrentNode();
			coordInterp->initialize();
			ParserPopNode();
		;}
    break;

  case 241:
#line 1079 "VRML97.y"
    {
			((CylinderNode *)ParserGetCurrentNode())->setSide((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 242:
#line 1083 "VRML97.y"
    {
			((CylinderNode *)ParserGetCurrentNode())->setBottom((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 243:
#line 1087 "VRML97.y"
    {
			((CylinderNode *)ParserGetCurrentNode())->setTop((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 244:
#line 1091 "VRML97.y"
    {
			((CylinderNode *)ParserGetCurrentNode())->setRadius((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 245:
#line 1095 "VRML97.y"
    {
			((CylinderNode *)ParserGetCurrentNode())->setHeight((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 246:
#line 1102 "VRML97.y"
    {
			CylinderNode *cylinder = new CylinderNode();
			cylinder->setName(GetDEFName());
			ParserAddNode(cylinder);
			ParserPushNode(VRML97_CYLINDER, cylinder);
		;}
    break;

  case 247:
#line 1112 "VRML97.y"
    {
			CylinderNode *cylinder = (CylinderNode *)ParserGetCurrentNode();
			cylinder->initialize();
			ParserPopNode();
		;}
    break;

  case 250:
#line 1132 "VRML97.y"
    {
			((CylinderSensorNode *)ParserGetCurrentNode())->setAutoOffset((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 251:
#line 1136 "VRML97.y"
    {
			((CylinderSensorNode *)ParserGetCurrentNode())->setDiskAngle((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 252:
#line 1140 "VRML97.y"
    {
			((CylinderSensorNode *)ParserGetCurrentNode())->setEnabled((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 253:
#line 1144 "VRML97.y"
    {
			((CylinderSensorNode *)ParserGetCurrentNode())->setMaxAngle((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 254:
#line 1148 "VRML97.y"
    {
			((CylinderSensorNode *)ParserGetCurrentNode())->setMinAngle((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 255:
#line 1152 "VRML97.y"
    {
			((CylinderSensorNode *)ParserGetCurrentNode())->setOffset((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 256:
#line 1160 "VRML97.y"
    {
			CylinderSensorNode *cysensor = new CylinderSensorNode();
			cysensor->setName(GetDEFName());
			ParserAddNode(cysensor);
			ParserPushNode(VRML97_CYLINDERSENSOR, cysensor);
		;}
    break;

  case 257:
#line 1170 "VRML97.y"
    {
			CylinderSensorNode *cysensor = (CylinderSensorNode *)ParserGetCurrentNode();
			cysensor->initialize();
			ParserPopNode();
		;}
    break;

  case 260:
#line 1190 "VRML97.y"
    {
			((DirectionalLightNode *)ParserGetCurrentNode())->setOn((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 261:
#line 1194 "VRML97.y"
    {
			((DirectionalLightNode *)ParserGetCurrentNode())->setIntensity((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 262:
#line 1198 "VRML97.y"
    {
			((DirectionalLightNode *)ParserGetCurrentNode())->setColor(gColor);
		;}
    break;

  case 263:
#line 1202 "VRML97.y"
    {
			((DirectionalLightNode *)ParserGetCurrentNode())->setDirection(gVec3f);
		;}
    break;

  case 264:
#line 1206 "VRML97.y"
    {
			((DirectionalLightNode *)ParserGetCurrentNode())->setAmbientIntensity((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 265:
#line 1213 "VRML97.y"
    {
			DirectionalLightNode *dirLight = new DirectionalLightNode();
			dirLight->setName(GetDEFName());
			ParserAddNode(dirLight);
			ParserPushNode(VRML97_DIRECTIONALLIGHT, dirLight);
		;}
    break;

  case 266:
#line 1223 "VRML97.y"
    {
			DirectionalLightNode *dirLight = (DirectionalLightNode *)ParserGetCurrentNode();
			dirLight->initialize();
			ParserPopNode();
		;}
    break;

  case 269:
#line 1243 "VRML97.y"
    {
			ParserPushNode(VRML97_ELEVATIONGRID_HEIGHT, ParserGetCurrentNode());
		;}
    break;

  case 279:
#line 1260 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 280:
#line 1264 "VRML97.y"
    {
			((ElevationGridNode *)ParserGetCurrentNode())->setCCW((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 281:
#line 1268 "VRML97.y"
    {
			((ElevationGridNode *)ParserGetCurrentNode())->setCreaseAngle((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 282:
#line 1272 "VRML97.y"
    {
			((ElevationGridNode *)ParserGetCurrentNode())->setSolid((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 283:
#line 1276 "VRML97.y"
    {
			((ElevationGridNode *)ParserGetCurrentNode())->setColorPerVertex((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 284:
#line 1280 "VRML97.y"
    {
			((ElevationGridNode *)ParserGetCurrentNode())->setNormalPerVertex((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 285:
#line 1284 "VRML97.y"
    {
			((ElevationGridNode *)ParserGetCurrentNode())->setXDimension((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 286:
#line 1288 "VRML97.y"
    {
			((ElevationGridNode *)ParserGetCurrentNode())->setXSpacing((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 287:
#line 1292 "VRML97.y"
    {
			((ElevationGridNode *)ParserGetCurrentNode())->setZDimension((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 288:
#line 1296 "VRML97.y"
    {
			((ElevationGridNode *)ParserGetCurrentNode())->setZSpacing((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 289:
#line 1303 "VRML97.y"
    {
			ElevationGridNode *elev = new ElevationGridNode();
			elev->setName(GetDEFName());
			ParserAddNode(elev);
			ParserPushNode(VRML97_ELEVATIONGRID, elev);
		;}
    break;

  case 290:
#line 1313 "VRML97.y"
    {
			ElevationGridNode *elev = (ElevationGridNode *)ParserGetCurrentNode();
			elev->initialize();
			ParserPopNode();
		;}
    break;

  case 293:
#line 1333 "VRML97.y"
    {
			ParserPushNode(VRML97_EXTRUSION_CROSSSECTION, ParserGetCurrentNode());
		;}
    break;

  case 294:
#line 1340 "VRML97.y"
    {
			ParserPushNode(VRML97_EXTRUSION_ORIENTATION, ParserGetCurrentNode());
		;}
    break;

  case 295:
#line 1347 "VRML97.y"
    {
			ParserPushNode(VRML97_EXTRUSION_SCALE, ParserGetCurrentNode());
		;}
    break;

  case 296:
#line 1354 "VRML97.y"
    {
			ParserPushNode(VRML97_EXTRUSION_SPINE, ParserGetCurrentNode());
		;}
    break;

  case 297:
#line 1361 "VRML97.y"
    {
			((ExtrusionNode *)ParserGetCurrentNode())->setBeginCap((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 298:
#line 1365 "VRML97.y"
    {
			((ExtrusionNode *)ParserGetCurrentNode())->setCCW((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 299:
#line 1369 "VRML97.y"
    {
			((ExtrusionNode *)ParserGetCurrentNode())->setConvex((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 300:
#line 1373 "VRML97.y"
    {
			((ExtrusionNode *)ParserGetCurrentNode())->setCreaseAngle((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 301:
#line 1377 "VRML97.y"
    {
			((ExtrusionNode *)ParserGetCurrentNode())->setSolid((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 302:
#line 1381 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 303:
#line 1385 "VRML97.y"
    {
			((ExtrusionNode *)ParserGetCurrentNode())->setEndCap((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 304:
#line 1389 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 305:
#line 1393 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 306:
#line 1397 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 307:
#line 1404 "VRML97.y"
    {
			ExtrusionNode *ex = new ExtrusionNode();
			ex->setName(GetDEFName());
			ParserAddNode(ex);
			ParserPushNode(VRML97_EXTRUSION, ex);
		;}
    break;

  case 308:
#line 1414 "VRML97.y"
    {
			ExtrusionNode *ex = (ExtrusionNode *)ParserGetCurrentNode();
			ex->initialize();
			ParserPopNode();
		;}
    break;

  case 311:
#line 1434 "VRML97.y"
    {
			((FogNode *)ParserGetCurrentNode())->setColor(gColor);
		;}
    break;

  case 312:
#line 1438 "VRML97.y"
    {
			((FogNode *)ParserGetCurrentNode())->setFogType((yyvsp[(2) - (2)].sval));
		;}
    break;

  case 313:
#line 1442 "VRML97.y"
    {
			((FogNode *)ParserGetCurrentNode())->setVisibilityRange((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 314:
#line 1449 "VRML97.y"
    {
			FogNode *fog= new FogNode();
			fog->setName(GetDEFName());
			ParserAddNode(fog);
			ParserPushNode(VRML97_FOG, fog);
		;}
    break;

  case 315:
#line 1459 "VRML97.y"
    {
			FogNode *fog= (FogNode *)ParserGetCurrentNode();
			fog->initialize();
			ParserPopNode();
		;}
    break;

  case 318:
#line 1479 "VRML97.y"
    {
			ParserPushNode(VRML97_FONTSTYLE_JUSTIFY, ParserGetCurrentNode());
		;}
    break;

  case 319:
#line 1486 "VRML97.y"
    {
			((FontStyleNode *)ParserGetCurrentNode())->setFamily((yyvsp[(2) - (2)].sval));
		;}
    break;

  case 320:
#line 1490 "VRML97.y"
    {
			((FontStyleNode *)ParserGetCurrentNode())->setHorizontal((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 321:
#line 1494 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 322:
#line 1498 "VRML97.y"
    {
			((FontStyleNode *)ParserGetCurrentNode())->setLanguage((yyvsp[(2) - (2)].sval));
		;}
    break;

  case 323:
#line 1502 "VRML97.y"
    {
			((FontStyleNode *)ParserGetCurrentNode())->setLeftToRight((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 324:
#line 1506 "VRML97.y"
    {
			((FontStyleNode *)ParserGetCurrentNode())->setSize((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 325:
#line 1510 "VRML97.y"
    {
			((FontStyleNode *)ParserGetCurrentNode())->setSpacing((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 326:
#line 1514 "VRML97.y"
    {
			((FontStyleNode *)ParserGetCurrentNode())->setStyle((yyvsp[(2) - (2)].sval));
		;}
    break;

  case 327:
#line 1518 "VRML97.y"
    {
			((FontStyleNode *)ParserGetCurrentNode())->setTopToBottom((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 328:
#line 1525 "VRML97.y"
    {
			FontStyleNode *fs = new FontStyleNode();
			fs->setName(GetDEFName());
			ParserAddNode(fs);
			ParserPushNode(VRML97_FONTSTYLE, fs);
		;}
    break;

  case 329:
#line 1535 "VRML97.y"
    {
			FontStyleNode *fs = (FontStyleNode *)ParserGetCurrentNode();
			fs->initialize();
			ParserPopNode();
		;}
    break;

  case 335:
#line 1561 "VRML97.y"
    {   
			GroupNode *group = new GroupNode();
			group->setName(GetDEFName());
			ParserAddNode(group);
			ParserPushNode(VRML97_GROUP, group);
		;}
    break;

  case 336:
#line 1571 "VRML97.y"
    {
			GroupNode *group = (GroupNode *)ParserGetCurrentNode();
			group->initialize();
			ParserPopNode();
		;}
    break;

  case 339:
#line 1591 "VRML97.y"
    {
			ParserPushNode(VRML97_IMAGETEXTURE_URL, ParserGetCurrentNode());
		;}
    break;

  case 340:
#line 1598 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 341:
#line 1602 "VRML97.y"
    {
			((ImageTextureNode *)ParserGetCurrentNode())->setRepeatS((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 342:
#line 1606 "VRML97.y"
    {
			((ImageTextureNode *)ParserGetCurrentNode())->setRepeatT((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 343:
#line 1613 "VRML97.y"
    {
			ImageTextureNode *imgTexture = new ImageTextureNode();
			imgTexture->setName(GetDEFName());
			ParserAddNode(imgTexture);
			ParserPushNode(VRML97_IMAGETEXTURE, imgTexture);
		;}
    break;

  case 344:
#line 1623 "VRML97.y"
    {
			ImageTextureNode *imgTexture = (ImageTextureNode *)ParserGetCurrentNode();
			imgTexture->initialize();
			ParserPopNode();
		;}
    break;

  case 347:
#line 1643 "VRML97.y"
    {
			ParserPushNode(VRML97_COLOR_INDEX, ParserGetCurrentNode());
		;}
    break;

  case 348:
#line 1650 "VRML97.y"
    {
			ParserPushNode(VRML97_COORDINATE_INDEX, ParserGetCurrentNode());
		;}
    break;

  case 349:
#line 1657 "VRML97.y"
    {
			ParserPushNode(VRML97_NORMAL_INDEX, ParserGetCurrentNode());
		;}
    break;

  case 350:
#line 1664 "VRML97.y"
    {
			ParserPushNode(VRML97_TEXTURECOODINATE_INDEX, ParserGetCurrentNode());
		;}
    break;

  case 363:
#line 1683 "VRML97.y"
    {
			((IndexedFaceSetNode *)ParserGetCurrentNode())->setCCW((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 364:
#line 1687 "VRML97.y"
    {
			((IndexedFaceSetNode *)ParserGetCurrentNode())->setConvex((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 365:
#line 1691 "VRML97.y"
    {
			((IndexedFaceSetNode *)ParserGetCurrentNode())->setSolid((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 366:
#line 1695 "VRML97.y"
    {
			((IndexedFaceSetNode *)ParserGetCurrentNode())->setCreaseAngle((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 367:
#line 1699 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 368:
#line 1703 "VRML97.y"
    {
			((IndexedFaceSetNode *)ParserGetCurrentNode())->setColorPerVertex((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 369:
#line 1707 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 370:
#line 1711 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 371:
#line 1715 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 372:
#line 1719 "VRML97.y"
    {
			((IndexedFaceSetNode *)ParserGetCurrentNode())->setNormalPerVertex((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 373:
#line 1726 "VRML97.y"
    {
			IndexedFaceSetNode	*idxFaceset = new IndexedFaceSetNode();
			idxFaceset->setName(GetDEFName());
			ParserAddNode(idxFaceset);
			ParserPushNode(VRML97_INDEXEDFACESET, idxFaceset);
		;}
    break;

  case 374:
#line 1736 "VRML97.y"
    {
			IndexedFaceSetNode *idxFaceset = (IndexedFaceSetNode *)ParserGetCurrentNode();
			idxFaceset->initialize();
			ParserPopNode();
		;}
    break;

  case 383:
#line 1762 "VRML97.y"
    {
			((IndexedLineSetNode *)ParserGetCurrentNode())->setColorPerVertex((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 384:
#line 1766 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 385:
#line 1770 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 386:
#line 1777 "VRML97.y"
    {
			IndexedLineSetNode	*idxLineset = new IndexedLineSetNode();
			idxLineset->setName(GetDEFName());
			ParserAddNode(idxLineset);
			ParserPushNode(VRML97_INDEXEDLINESET, idxLineset);
		;}
    break;

  case 387:
#line 1787 "VRML97.y"
    {
			IndexedLineSetNode *idxLineset = (IndexedLineSetNode *)ParserGetCurrentNode();
			idxLineset->initialize();
			ParserPopNode();
		;}
    break;

  case 390:
#line 1807 "VRML97.y"
    {
			ParserPushNode(VRML97_INLINE_URL, ParserGetCurrentNode());
		;}
    break;

  case 391:
#line 1814 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 394:
#line 1823 "VRML97.y"
    {   
			InlineNode *inlineNode = new InlineNode();
			inlineNode->setName(GetDEFName());
			ParserAddNode(inlineNode);
			ParserPushNode(VRML97_INLINE, inlineNode);
		;}
    break;

  case 395:
#line 1833 "VRML97.y"
    {
			InlineNode *inlineNode = (InlineNode *)ParserGetCurrentNode();
			//inlineNode->initialize();
			ParserPopNode();
		;}
    break;

  case 398:
#line 1853 "VRML97.y"
    {
			ParserPushNode(VRML97_LOD_RANGE, ParserGetCurrentNode());
		;}
    break;

  case 399:
#line 1861 "VRML97.y"
    {
			ParserPushNode(VRML97_LOD_LEVEL, ParserGetCurrentNode());
		;}
    break;

  case 400:
#line 1868 "VRML97.y"
    {
			ParserPopNode();							
		;}
    break;

  case 401:
#line 1872 "VRML97.y"
    {
			((LODNode *)ParserGetCurrentNode())->setCenter(gVec3f);
		;}
    break;

  case 402:
#line 1876 "VRML97.y"
    {
			ParserPopNode();							
		;}
    break;

  case 403:
#line 1880 "VRML97.y"
    {
			ParserPopNode();							
		;}
    break;

  case 404:
#line 1887 "VRML97.y"
    {   
			LODNode	*lod = new LODNode();
			lod->setName(GetDEFName());
			ParserAddNode(lod);
			ParserPushNode(VRML97_INLINE, lod);
		;}
    break;

  case 405:
#line 1897 "VRML97.y"
    {
			LODNode	*lod = (LODNode *)ParserGetCurrentNode();
			lod->initialize();
			ParserPopNode();
		;}
    break;

  case 408:
#line 1917 "VRML97.y"
    {
			((MaterialNode *)ParserGetCurrentNode())->setAmbientIntensity((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 409:
#line 1921 "VRML97.y"
    {
			((MaterialNode *)ParserGetCurrentNode())->setDiffuseColor(gColor);
		;}
    break;

  case 410:
#line 1925 "VRML97.y"
    {
			((MaterialNode *)ParserGetCurrentNode())->setEmissiveColor(gColor);
		;}
    break;

  case 411:
#line 1929 "VRML97.y"
    {
			((MaterialNode *)ParserGetCurrentNode())->setShininess((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 412:
#line 1933 "VRML97.y"
    {
			((MaterialNode *)ParserGetCurrentNode())->setSpecularColor(gColor);
		;}
    break;

  case 413:
#line 1937 "VRML97.y"
    {
			((MaterialNode *)ParserGetCurrentNode())->setTransparency((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 414:
#line 1944 "VRML97.y"
    {
			MaterialNode *material = new MaterialNode();
			material->setName(GetDEFName());
			ParserAddNode(material);
			ParserPushNode(VRML97_MATERIAL, material);
		;}
    break;

  case 415:
#line 1954 "VRML97.y"
    {
			MaterialNode *material = (MaterialNode *)ParserGetCurrentNode();
			material->initialize();
			ParserPopNode();
		;}
    break;

  case 418:
#line 1974 "VRML97.y"
    {
			ParserPushNode(VRML97_MOVIETEXTURE_URL, ParserGetCurrentNode());
		;}
    break;

  case 419:
#line 1981 "VRML97.y"
    {
			((MovieTextureNode *)ParserGetCurrentNode())->setLoop((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 420:
#line 1985 "VRML97.y"
    {
			((MovieTextureNode *)ParserGetCurrentNode())->setSpeed((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 421:
#line 1989 "VRML97.y"
    {
			((MovieTextureNode *)ParserGetCurrentNode())->setStartTime((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 422:
#line 1993 "VRML97.y"
    {
			((MovieTextureNode *)ParserGetCurrentNode())->setStopTime((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 423:
#line 1997 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 424:
#line 2001 "VRML97.y"
    {
			((MovieTextureNode *)ParserGetCurrentNode())->setRepeatS((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 425:
#line 2005 "VRML97.y"
    {
			((MovieTextureNode *)ParserGetCurrentNode())->setRepeatT((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 426:
#line 2012 "VRML97.y"
    {
			MovieTextureNode *movieTexture = new MovieTextureNode();
			movieTexture->setName(GetDEFName());
			ParserAddNode(movieTexture);
			ParserPushNode(VRML97_MOVIETEXTURE, movieTexture);
		;}
    break;

  case 427:
#line 2022 "VRML97.y"
    {
			MovieTextureNode *movieTexture = (MovieTextureNode *)ParserGetCurrentNode();
			movieTexture->initialize();
			ParserPopNode();
		;}
    break;

  case 430:
#line 2042 "VRML97.y"
    {
			ParserPushNode(VRML97_NAVIGATIONINFO_AVATARSIZE, ParserGetCurrentNode());
		;}
    break;

  case 431:
#line 2049 "VRML97.y"
    {
			ParserPushNode(VRML97_NAVIGATIONINFO_TYPE, ParserGetCurrentNode());
		;}
    break;

  case 432:
#line 2056 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 433:
#line 2060 "VRML97.y"
    {
			((NavigationInfoNode *)ParserGetCurrentNode())->setHeadlight((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 434:
#line 2064 "VRML97.y"
    {
			((NavigationInfoNode *)ParserGetCurrentNode())->setSpeed((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 435:
#line 2068 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 436:
#line 2072 "VRML97.y"
    {
			((NavigationInfoNode *)ParserGetCurrentNode())->setVisibilityLimit((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 437:
#line 2079 "VRML97.y"
    {
			NavigationInfoNode *navInfo = new NavigationInfoNode();
			navInfo->setName(GetDEFName());
			ParserAddNode(navInfo);
			ParserPushNode(VRML97_NAVIGATIONINFO, navInfo);
		;}
    break;

  case 438:
#line 2089 "VRML97.y"
    {
			NavigationInfoNode *navInfo = (NavigationInfoNode *)ParserGetCurrentNode();
			navInfo->initialize();
			ParserPopNode();
		;}
    break;

  case 442:
#line 2113 "VRML97.y"
    {
			NormalNode *normal = new NormalNode();
			normal->setName(GetDEFName());
			ParserAddNode(normal);
			ParserPushNode(VRML97_NORMAL, normal);
		;}
    break;

  case 443:
#line 2123 "VRML97.y"
    {
			NormalNode *normal = (NormalNode *)ParserGetCurrentNode();
			normal->initialize();
			ParserPopNode();
		;}
    break;

  case 446:
#line 2143 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 447:
#line 2147 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 448:
#line 2151 "VRML97.y"
    {
		;}
    break;

  case 449:
#line 2157 "VRML97.y"
    {
			NormalInterpolatorNode *normInterp = new NormalInterpolatorNode();
			normInterp->setName(GetDEFName());
			ParserAddNode(normInterp);
			ParserPushNode(VRML97_NORMALINTERPOLATOR, normInterp);
		;}
    break;

  case 450:
#line 2167 "VRML97.y"
    {
			NormalInterpolatorNode *normInterp = (NormalInterpolatorNode *)ParserGetCurrentNode();
			normInterp->initialize();
			ParserPopNode();
		;}
    break;

  case 453:
#line 2187 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 454:
#line 2191 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 455:
#line 2195 "VRML97.y"
    {
		;}
    break;

  case 456:
#line 2201 "VRML97.y"
    {
			OrientationInterpolatorNode *oriInterp = new OrientationInterpolatorNode();
			oriInterp->setName(GetDEFName());
			ParserAddNode(oriInterp);
			ParserPushNode(VRML97_ORIENTATIONINTERPOLATOR, oriInterp);
		;}
    break;

  case 457:
#line 2211 "VRML97.y"
    {
			OrientationInterpolatorNode *oriInterp = (OrientationInterpolatorNode *)ParserGetCurrentNode();
			oriInterp->initialize();
			ParserPopNode();
		;}
    break;

  case 460:
#line 2231 "VRML97.y"
    {
			ParserPushNode(VRML97_PIXELTEXTURE_IMAGE, ParserGetCurrentNode());
		;}
    break;

  case 461:
#line 2238 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 462:
#line 2242 "VRML97.y"
    {
			((PixelTextureNode *)ParserGetCurrentNode())->setRepeatS((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 463:
#line 2246 "VRML97.y"
    {
			((PixelTextureNode *)ParserGetCurrentNode())->setRepeatT((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 464:
#line 2253 "VRML97.y"
    {
			PixelTextureNode *pixTexture = new PixelTextureNode();
			pixTexture->setName(GetDEFName());
			ParserAddNode(pixTexture);
			ParserPushNode(VRML97_PIXELTEXTURE, pixTexture);
		;}
    break;

  case 465:
#line 2263 "VRML97.y"
    {
			PixelTextureNode *pixTexture = (PixelTextureNode *)ParserGetCurrentNode();
			pixTexture->initialize();
			ParserPopNode();
		;}
    break;

  case 468:
#line 2283 "VRML97.y"
    {
			((PlaneSensorNode *)ParserGetCurrentNode())->setAutoOffset((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 469:
#line 2287 "VRML97.y"
    {
			((PlaneSensorNode *)ParserGetCurrentNode())->setEnabled((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 470:
#line 2291 "VRML97.y"
    {
			((PlaneSensorNode *)ParserGetCurrentNode())->setMaxPosition(gVec2f);
		;}
    break;

  case 471:
#line 2295 "VRML97.y"
    {
			((PlaneSensorNode *)ParserGetCurrentNode())->setMinPosition(gVec2f);
		;}
    break;

  case 472:
#line 2299 "VRML97.y"
    {
			((PlaneSensorNode *)ParserGetCurrentNode())->setOffset(gVec3f);
		;}
    break;

  case 473:
#line 2306 "VRML97.y"
    {
			PlaneSensorNode *psensor = new PlaneSensorNode();
			psensor->setName(GetDEFName());
			ParserAddNode(psensor);
			ParserPushNode(VRML97_PLANESENSOR, psensor);
		;}
    break;

  case 474:
#line 2316 "VRML97.y"
    {
			PlaneSensorNode *psensor = (PlaneSensorNode *)ParserGetCurrentNode();
			psensor->initialize();
			ParserPopNode();
		;}
    break;

  case 477:
#line 2337 "VRML97.y"
    {
			((PointLightNode *)ParserGetCurrentNode())->setAmbientIntensity((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 478:
#line 2341 "VRML97.y"
    {
			((PointLightNode *)ParserGetCurrentNode())->setAttenuation(gVec3f);
		;}
    break;

  case 479:
#line 2345 "VRML97.y"
    {
			((PointLightNode *)ParserGetCurrentNode())->setColor(gColor);
		;}
    break;

  case 480:
#line 2349 "VRML97.y"
    {
			((PointLightNode *)ParserGetCurrentNode())->setIntensity((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 481:
#line 2353 "VRML97.y"
    {
			((PointLightNode *)ParserGetCurrentNode())->setLocation(gVec3f);
		;}
    break;

  case 482:
#line 2357 "VRML97.y"
    {
			((PointLightNode *)ParserGetCurrentNode())->setOn((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 483:
#line 2361 "VRML97.y"
    {
			((PointLightNode *)ParserGetCurrentNode())->setRadius((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 484:
#line 2368 "VRML97.y"
    {
			PointLightNode *pointLight = new PointLightNode();
			pointLight->setName(GetDEFName());
			ParserAddNode(pointLight);
			ParserPushNode(VRML97_POINTLIGHT, pointLight);
		;}
    break;

  case 485:
#line 2378 "VRML97.y"
    {
			PointLightNode *pointLight = (PointLightNode *)ParserGetCurrentNode();
			pointLight->initialize();
			ParserPopNode();
		;}
    break;

  case 494:
#line 2408 "VRML97.y"
    {
			PointSetNode *pset = new PointSetNode();
			pset->setName(GetDEFName());
			ParserAddNode(pset);
			ParserPushNode(VRML97_POINTSET, pset);
		;}
    break;

  case 495:
#line 2418 "VRML97.y"
    {
			PointSetNode *pset = (PointSetNode *)ParserGetCurrentNode();
			pset->initialize();
			ParserPopNode();
		;}
    break;

  case 498:
#line 2438 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 499:
#line 2442 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 500:
#line 2446 "VRML97.y"
    {
		;}
    break;

  case 501:
#line 2452 "VRML97.y"
    {
			PositionInterpolatorNode *posInterp = new PositionInterpolatorNode();
			posInterp->setName(GetDEFName());
			ParserAddNode(posInterp);
			ParserPushNode(VRML97_POSITIONINTERPOLATOR, posInterp);
		;}
    break;

  case 502:
#line 2462 "VRML97.y"
    {
			PositionInterpolatorNode *posInterp = (PositionInterpolatorNode *)ParserGetCurrentNode();
			posInterp->initialize();
			ParserPopNode();
		;}
    break;

  case 505:
#line 2482 "VRML97.y"
    {
			((ProximitySensorNode *)ParserGetCurrentNode())->setCenter(gVec3f);
		;}
    break;

  case 506:
#line 2486 "VRML97.y"
    {
			((ProximitySensorNode *)ParserGetCurrentNode())->setSize(gVec3f);
		;}
    break;

  case 507:
#line 2490 "VRML97.y"
    {
			((ProximitySensorNode *)ParserGetCurrentNode())->setEnabled((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 508:
#line 2497 "VRML97.y"
    {
			ProximitySensorNode *psensor = new ProximitySensorNode();
			psensor->setName(GetDEFName());
			ParserAddNode(psensor);
			ParserPushNode(VRML97_PROXIMITYSENSOR, psensor);
		;}
    break;

  case 509:
#line 2507 "VRML97.y"
    {
			ProximitySensorNode *psensor = (ProximitySensorNode *)ParserGetCurrentNode();
			psensor->initialize();
			ParserPopNode();
		;}
    break;

  case 512:
#line 2527 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 513:
#line 2531 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 514:
#line 2535 "VRML97.y"
    {
		;}
    break;

  case 515:
#line 2541 "VRML97.y"
    {
			ScalarInterpolatorNode *scalarInterp = new ScalarInterpolatorNode();
			scalarInterp->setName(GetDEFName());
			ParserAddNode(scalarInterp);
			ParserPushNode(VRML97_SCALARINTERPOLATOR, scalarInterp);
		;}
    break;

  case 516:
#line 2551 "VRML97.y"
    {
			ScalarInterpolatorNode *scalarInterp = (ScalarInterpolatorNode *)ParserGetCurrentNode();
			scalarInterp->initialize();
			ParserPopNode();
		;}
    break;

  case 519:
#line 2571 "VRML97.y"
    {
			ParserPushNode(VRML97_SCRIPT_URL, ParserGetCurrentNode());
		;}
    break;

  case 520:
#line 2578 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 521:
#line 2582 "VRML97.y"
    {
			((ScriptNode *)ParserGetCurrentNode())->setDirectOutput((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 522:
#line 2586 "VRML97.y"
    {
			((ScriptNode *)ParserGetCurrentNode())->setMustEvaluate((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 523:
#line 2595 "VRML97.y"
    {
			SFBool *value = new SFBool();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 524:
#line 2601 "VRML97.y"
    {
			SFFloat *value = new SFFloat();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 525:
#line 2607 "VRML97.y"
    {
			SFInt32 *value = new SFInt32();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 526:
#line 2613 "VRML97.y"
    {
			SFTime *value = new SFTime();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 527:
#line 2619 "VRML97.y"
    {
			SFRotation *value = new SFRotation();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 528:
#line 2633 "VRML97.y"
    {
			SFColor *value = new SFColor();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 529:
#line 2639 "VRML97.y"
    {
			SFImage *value = new SFImage();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 530:
#line 2645 "VRML97.y"
    {
			SFString *value = new SFString();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 531:
#line 2651 "VRML97.y"
    {
			SFVec2f *value = new SFVec2f();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 532:
#line 2657 "VRML97.y"
    {
			SFVec3f *value = new SFVec3f();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 533:
#line 2668 "VRML97.y"
    {
			MFFloat *value = new MFFloat();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 534:
#line 2674 "VRML97.y"
    {
			MFInt32 *value = new MFInt32();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 535:
#line 2680 "VRML97.y"
    {
			MFTime *value = new MFTime();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 536:
#line 2686 "VRML97.y"
    {
			MFRotation *value = new MFRotation();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 537:
#line 2700 "VRML97.y"
    {
			MFColor *value = new MFColor();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 538:
#line 2706 "VRML97.y"
    {
			MFString *value = new MFString();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 539:
#line 2712 "VRML97.y"
    {
			MFVec2f *value = new MFVec2f();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 540:
#line 2718 "VRML97.y"
    {
			MFVec3f *value = new MFVec3f();
			((ScriptNode *)ParserGetCurrentNode())->addEventIn((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 541:
#line 2729 "VRML97.y"
    {
			SFBool *value = new SFBool();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 542:
#line 2735 "VRML97.y"
    {
			SFFloat *value = new SFFloat();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 543:
#line 2741 "VRML97.y"
    {
			SFInt32 *value = new SFInt32();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 544:
#line 2747 "VRML97.y"
    {
			SFTime *value = new SFTime();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 545:
#line 2753 "VRML97.y"
    {
			SFRotation *value = new SFRotation();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 546:
#line 2767 "VRML97.y"
    {
			SFColor *value = new SFColor();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 547:
#line 2773 "VRML97.y"
    {
			SFImage *value = new SFImage();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 548:
#line 2779 "VRML97.y"
    {
			SFString *value = new SFString();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 549:
#line 2785 "VRML97.y"
    {
			SFVec2f *value = new SFVec2f();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 550:
#line 2791 "VRML97.y"
    {
			SFVec3f *value = new SFVec3f();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 551:
#line 2802 "VRML97.y"
    {
			MFFloat *value = new MFFloat();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 552:
#line 2808 "VRML97.y"
    {
			MFInt32 *value = new MFInt32();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 553:
#line 2814 "VRML97.y"
    {
			MFTime *value = new MFTime();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 554:
#line 2820 "VRML97.y"
    {
			MFRotation *value = new MFRotation();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 555:
#line 2834 "VRML97.y"
    {
			MFColor *value = new MFColor();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 556:
#line 2840 "VRML97.y"
    {
			MFString *value = new MFString();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 557:
#line 2846 "VRML97.y"
    {
			MFVec2f *value = new MFVec2f();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 558:
#line 2852 "VRML97.y"
    {
			MFVec3f *value = new MFVec3f();
			((ScriptNode *)ParserGetCurrentNode())->addEventOut((yyvsp[(3) - (3)].sval), value);
			delete[] (yyvsp[(3) - (3)].sval);
		;}
    break;

  case 559:
#line 2863 "VRML97.y"
    {
			SFBool *value = new SFBool((yyvsp[(4) - (4)].ival));
			((ScriptNode *)ParserGetCurrentNode())->addField((yyvsp[(3) - (4)].sval), value);
			delete[] (yyvsp[(3) - (4)].sval);
		;}
    break;

  case 560:
#line 2869 "VRML97.y"
    {
			SFFloat *value = new SFFloat((yyvsp[(4) - (4)].fval));
			((ScriptNode *)ParserGetCurrentNode())->addField((yyvsp[(3) - (4)].sval), value);
			delete[] (yyvsp[(3) - (4)].sval);
		;}
    break;

  case 561:
#line 2875 "VRML97.y"
    {
			SFInt32 *value = new SFInt32((yyvsp[(4) - (4)].ival));
			((ScriptNode *)ParserGetCurrentNode())->addField((yyvsp[(3) - (4)].sval), value);
			delete[] (yyvsp[(3) - (4)].sval);
		;}
    break;

  case 562:
#line 2881 "VRML97.y"
    {
			SFTime *value = new SFTime((yyvsp[(4) - (4)].fval));
			((ScriptNode *)ParserGetCurrentNode())->addField((yyvsp[(3) - (4)].sval), value);
			delete[] (yyvsp[(3) - (4)].sval);
		;}
    break;

  case 563:
#line 2887 "VRML97.y"
    {
			SFRotation *value = new SFRotation(gRotation);
			((ScriptNode *)ParserGetCurrentNode())->addField((yyvsp[(3) - (4)].sval), value);
			delete[] (yyvsp[(3) - (4)].sval);
		;}
    break;

  case 564:
#line 2894 "VRML97.y"
    {
			SFNode *value = new SFNode();
			((ScriptNode *)ParserGetCurrentNode())->addField((yyvsp[(3) - (4)].sval), value);
			delete[] (yyvsp[(3) - (4)].sval);
		;}
    break;

  case 565:
#line 2901 "VRML97.y"
    {
			Node *node = GetParserObject()->findNode((yyvsp[(5) - (5)].sval));
			SFNode *value = new SFNode(node);
			((ScriptNode *)ParserGetCurrentNode())->addField((yyvsp[(3) - (5)].sval), value);
			delete[] (yyvsp[(3) - (5)].sval); delete[] (yyvsp[(5) - (5)].sval);
		;}
    break;

  case 566:
#line 2909 "VRML97.y"
    {
			SFColor *value = new SFColor(gColor);
			((ScriptNode *)ParserGetCurrentNode())->addField((yyvsp[(3) - (4)].sval), value);
			delete[] (yyvsp[(3) - (4)].sval);
		;}
    break;

  case 567:
#line 2923 "VRML97.y"
    {
			SFString *value = new SFString((yyvsp[(4) - (4)].sval));
			((ScriptNode *)ParserGetCurrentNode())->addField((yyvsp[(3) - (4)].sval), value);
			delete[] (yyvsp[(3) - (4)].sval);
		;}
    break;

  case 568:
#line 2929 "VRML97.y"
    {
			SFVec2f *value = new SFVec2f(gVec2f);
			((ScriptNode *)ParserGetCurrentNode())->addField((yyvsp[(3) - (4)].sval), value);
			delete[] (yyvsp[(3) - (4)].sval);
		;}
    break;

  case 569:
#line 2935 "VRML97.y"
    {
			SFVec3f *value = new SFVec3f(gVec3f);
			((ScriptNode *)ParserGetCurrentNode())->addField((yyvsp[(3) - (4)].sval), value);
			delete[] (yyvsp[(3) - (4)].sval);
		;}
    break;

  case 570:
#line 2945 "VRML97.y"
    {
			ScriptNode *script = new ScriptNode();
			script->setName(GetDEFName());
			ParserAddNode(script);
			ParserPushNode(VRML97_SCRIPT, script);
		;}
    break;

  case 571:
#line 2955 "VRML97.y"
    {
			ScriptNode *script = (ScriptNode *)ParserGetCurrentNode();
			script->initialize();
			ParserPopNode();
		;}
    break;

  case 580:
#line 2985 "VRML97.y"
    {
			ShapeNode *shape = new ShapeNode();
			shape->setName(GetDEFName());
			ParserAddNode(shape);
			ParserPushNode(VRML97_SHAPE, shape);
		;}
    break;

  case 581:
#line 2995 "VRML97.y"
    {
			ShapeNode *shape = (ShapeNode *)ParserGetCurrentNode();
			shape->initialize();
			ParserPopNode();
		;}
    break;

  case 584:
#line 3015 "VRML97.y"
    {
			((SoundNode *)ParserGetCurrentNode())->setDirection(gVec3f);
		;}
    break;

  case 585:
#line 3019 "VRML97.y"
    {
			((SoundNode *)ParserGetCurrentNode())->setIntensity((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 586:
#line 3023 "VRML97.y"
    {
			((SoundNode *)ParserGetCurrentNode())->setLocation(gVec3f);
		;}
    break;

  case 587:
#line 3027 "VRML97.y"
    {
			((SoundNode *)ParserGetCurrentNode())->setMinBack((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 588:
#line 3031 "VRML97.y"
    {
			((SoundNode *)ParserGetCurrentNode())->setMaxFront((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 589:
#line 3035 "VRML97.y"
    {
			((SoundNode *)ParserGetCurrentNode())->setMinBack((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 590:
#line 3039 "VRML97.y"
    {
			((SoundNode *)ParserGetCurrentNode())->setMinFront((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 591:
#line 3043 "VRML97.y"
    {
			((SoundNode *)ParserGetCurrentNode())->setPriority((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 596:
#line 3051 "VRML97.y"
    {
			((SoundNode *)ParserGetCurrentNode())->setSpatialize((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 597:
#line 3058 "VRML97.y"
    {
			SoundNode *sound = new SoundNode();
			sound->setName(GetDEFName());
			ParserAddNode(sound);
			ParserPushNode(VRML97_SOUND, sound);
		;}
    break;

  case 598:
#line 3068 "VRML97.y"
    {
			SoundNode *sound = (SoundNode *)ParserGetCurrentNode();
			sound->initialize();
			ParserPopNode();
		;}
    break;

  case 601:
#line 3088 "VRML97.y"
    {
			((SphereNode *)ParserGetCurrentNode())->setRadius((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 602:
#line 3095 "VRML97.y"
    {
			SphereNode *sphere = new SphereNode();
			sphere->setName(GetDEFName());
			ParserAddNode(sphere);
			ParserPushNode(VRML97_SPHERE, sphere);
		;}
    break;

  case 603:
#line 3105 "VRML97.y"
    {
			SphereNode *sphere = (SphereNode *)ParserGetCurrentNode();
			sphere->initialize();
			ParserPopNode();
		;}
    break;

  case 606:
#line 3125 "VRML97.y"
    {
			((SphereSensorNode *)ParserGetCurrentNode())->setAutoOffset((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 607:
#line 3129 "VRML97.y"
    {
			((SphereSensorNode *)ParserGetCurrentNode())->setEnabled((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 608:
#line 3133 "VRML97.y"
    {
			((SphereSensorNode *)ParserGetCurrentNode())->setOffset(gRotation);
		;}
    break;

  case 609:
#line 3140 "VRML97.y"
    {
			SphereSensorNode *spsensor = new SphereSensorNode();
			spsensor->setName(GetDEFName());
			ParserAddNode(spsensor);
			ParserPushNode(VRML97_SPHERESENSOR, spsensor);
		;}
    break;

  case 610:
#line 3150 "VRML97.y"
    {
			SphereSensorNode *spsensor = (SphereSensorNode *)ParserGetCurrentNode();
			spsensor->initialize();
			ParserPopNode();
		;}
    break;

  case 613:
#line 3170 "VRML97.y"
    {
			((SpotLightNode *)ParserGetCurrentNode())->setAmbientIntensity((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 614:
#line 3174 "VRML97.y"
    {
			((SpotLightNode *)ParserGetCurrentNode())->setAttenuation(gVec3f);
		;}
    break;

  case 615:
#line 3178 "VRML97.y"
    {
			((SpotLightNode *)ParserGetCurrentNode())->setBeamWidth((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 616:
#line 3182 "VRML97.y"
    {
			((SpotLightNode *)ParserGetCurrentNode())->setColor(gColor);
		;}
    break;

  case 617:
#line 3186 "VRML97.y"
    {
			((SpotLightNode *)ParserGetCurrentNode())->setCutOffAngle((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 618:
#line 3190 "VRML97.y"
    {
			((SpotLightNode *)ParserGetCurrentNode())->setDirection(gVec3f);
		;}
    break;

  case 619:
#line 3194 "VRML97.y"
    {
			((SpotLightNode *)ParserGetCurrentNode())->setIntensity((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 620:
#line 3198 "VRML97.y"
    {
			((SpotLightNode *)ParserGetCurrentNode())->setLocation(gVec3f);
		;}
    break;

  case 621:
#line 3202 "VRML97.y"
    {
			((SpotLightNode *)ParserGetCurrentNode())->setOn((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 622:
#line 3206 "VRML97.y"
    {
			((SpotLightNode *)ParserGetCurrentNode())->setRadius((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 623:
#line 3213 "VRML97.y"
    {
			SpotLightNode *spotLight = new SpotLightNode();
			spotLight->setName(GetDEFName());
			ParserAddNode(spotLight);
			ParserPushNode(VRML97_SPOTLIGHT, spotLight);
		;}
    break;

  case 624:
#line 3223 "VRML97.y"
    {
			SpotLightNode *spotLight = (SpotLightNode *)ParserGetCurrentNode();
			spotLight->initialize();
			ParserPopNode();
		;}
    break;

  case 627:
#line 3243 "VRML97.y"
    {
			ParserPushNode(VRML97_SWITCH_CHOICE, ParserGetCurrentNode());
		;}
    break;

  case 628:
#line 3250 "VRML97.y"
    {
			ParserPopNode();							
		;}
    break;

  case 629:
#line 3254 "VRML97.y"
    {
			ParserPopNode();							
		;}
    break;

  case 630:
#line 3258 "VRML97.y"
    {
			((SwitchNode *)ParserGetCurrentNode())->setWhichChoice((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 631:
#line 3266 "VRML97.y"
    {   
			SwitchNode *switchNode = new SwitchNode();
			switchNode->setName(GetDEFName());
			ParserAddNode(switchNode);
			ParserPushNode(VRML97_SWITCH, switchNode);
		;}
    break;

  case 632:
#line 3276 "VRML97.y"
    {
			SwitchNode *switchNode = (SwitchNode *)ParserGetCurrentNode();
			switchNode->initialize();
			ParserPopNode();
		;}
    break;

  case 635:
#line 3296 "VRML97.y"
    {
			ParserPushNode(VRML97_TEXT_STRING, ParserGetCurrentNode());
		;}
    break;

  case 636:
#line 3303 "VRML97.y"
    {
			ParserPushNode(VRML97_TEXT_LENGTH, ParserGetCurrentNode());
		;}
    break;

  case 637:
#line 3310 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 641:
#line 3317 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 642:
#line 3321 "VRML97.y"
    {
			((TextNode *)ParserGetCurrentNode())->setMaxExtent((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 643:
#line 3329 "VRML97.y"
    {
			TextNode *text = new TextNode();
			text->setName(GetDEFName());
			ParserAddNode(text);
			ParserPushNode(VRML97_TEXT, text);
		;}
    break;

  case 644:
#line 3339 "VRML97.y"
    {
			TextNode *text = (TextNode *)ParserGetCurrentNode();
			text->initialize();
			ParserPopNode();
		;}
    break;

  case 648:
#line 3364 "VRML97.y"
    {
			TextureCoordinateNode *texCoord = new TextureCoordinateNode();
			texCoord->setName(GetDEFName());
			ParserAddNode(texCoord);
			ParserPushNode(VRML97_TEXTURECOODINATE, texCoord);
		;}
    break;

  case 649:
#line 3374 "VRML97.y"
    {
			TextureCoordinateNode *texCoord = (TextureCoordinateNode *)ParserGetCurrentNode();
			texCoord->initialize();
			ParserPopNode();
		;}
    break;

  case 652:
#line 3394 "VRML97.y"
    {
			((TextureTransformNode *)ParserGetCurrentNode())->setCenter(gVec2f);
		;}
    break;

  case 653:
#line 3398 "VRML97.y"
    {
			((TextureTransformNode *)ParserGetCurrentNode())->setRotation((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 654:
#line 3402 "VRML97.y"
    {
			((TextureTransformNode *)ParserGetCurrentNode())->setScale(gVec2f);
		;}
    break;

  case 655:
#line 3406 "VRML97.y"
    {
			((TextureTransformNode *)ParserGetCurrentNode())->setTranslation(gVec2f);
		;}
    break;

  case 656:
#line 3414 "VRML97.y"
    {
			TextureTransformNode *textureTransform = new TextureTransformNode();
			textureTransform->setName(GetDEFName());
			ParserAddNode(textureTransform);
			ParserPushNode(VRML97_TEXTURETRANSFORM, textureTransform);
		;}
    break;

  case 657:
#line 3424 "VRML97.y"
    {
			TextureTransformNode *textureTransform = (TextureTransformNode *)ParserGetCurrentNode();
			textureTransform->initialize();
			ParserPopNode();
		;}
    break;

  case 660:
#line 3444 "VRML97.y"
    {
			((TimeSensorNode *)ParserGetCurrentNode())->setCycleInterval((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 661:
#line 3448 "VRML97.y"
    {
			((TimeSensorNode *)ParserGetCurrentNode())->setEnabled((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 662:
#line 3452 "VRML97.y"
    {
			((TimeSensorNode *)ParserGetCurrentNode())->setLoop((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 663:
#line 3456 "VRML97.y"
    {
			((TimeSensorNode *)ParserGetCurrentNode())->setStartTime((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 664:
#line 3460 "VRML97.y"
    {
			((TimeSensorNode *)ParserGetCurrentNode())->setStopTime((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 665:
#line 3468 "VRML97.y"
    {
			TimeSensorNode *tsensor = new TimeSensorNode();
			tsensor->setName(GetDEFName());
			ParserAddNode(tsensor);
			ParserPushNode(VRML97_TIMESENSOR, tsensor);
		;}
    break;

  case 666:
#line 3478 "VRML97.y"
    {
			TimeSensorNode *tsensor = (TimeSensorNode *)ParserGetCurrentNode();
			tsensor->initialize();
			ParserPopNode();
		;}
    break;

  case 669:
#line 3498 "VRML97.y"
    {
			((TouchSensorNode *)ParserGetCurrentNode())->setEnabled((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 670:
#line 3505 "VRML97.y"
    {
			TouchSensorNode *touchSensor = new TouchSensorNode();
			touchSensor->setName(GetDEFName());
			ParserAddNode(touchSensor);
			ParserPushNode(VRML97_TOUCHSENSOR, touchSensor);
		;}
    break;

  case 671:
#line 3515 "VRML97.y"
    {
			TouchSensorNode *touchSensor = (TouchSensorNode *)ParserGetCurrentNode();
			touchSensor->initialize();
			ParserPopNode();
		;}
    break;

  case 675:
#line 3536 "VRML97.y"
    {
			((TransformNode *)ParserGetCurrentNode())->setCenter(gVec3f);
		;}
    break;

  case 676:
#line 3540 "VRML97.y"
    {
			((TransformNode *)ParserGetCurrentNode())->setRotation(gRotation);
		;}
    break;

  case 677:
#line 3544 "VRML97.y"
    {
			((TransformNode *)ParserGetCurrentNode())->setScale(gVec3f);
		;}
    break;

  case 678:
#line 3548 "VRML97.y"
    {
			((TransformNode *)ParserGetCurrentNode())->setScaleOrientation(gRotation);
		;}
    break;

  case 679:
#line 3552 "VRML97.y"
    {
			((TransformNode *)ParserGetCurrentNode())->setTranslation(gVec3f);
		;}
    break;

  case 682:
#line 3561 "VRML97.y"
    {
			TransformNode *transform = new TransformNode();
			transform->setName(GetDEFName());
			ParserAddNode(transform);
			ParserPushNode(VRML97_TRANSFORM, transform);
		;}
    break;

  case 683:
#line 3571 "VRML97.y"
    {
			TransformNode *transform = (TransformNode *)ParserGetCurrentNode();
			transform->initialize();
			ParserPopNode();
		;}
    break;

  case 686:
#line 3591 "VRML97.y"
    {
			((ViewpointNode *)ParserGetCurrentNode())->setFieldOfView((yyvsp[(2) - (2)].fval));
		;}
    break;

  case 687:
#line 3595 "VRML97.y"
    {
			((ViewpointNode *)ParserGetCurrentNode())->setJump((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 688:
#line 3599 "VRML97.y"
    {
			((ViewpointNode *)ParserGetCurrentNode())->setOrientation(gRotation);
		;}
    break;

  case 689:
#line 3603 "VRML97.y"
    {
			((ViewpointNode *)ParserGetCurrentNode())->setPosition(gVec3f);
		;}
    break;

  case 690:
#line 3607 "VRML97.y"
    {
			((ViewpointNode *)ParserGetCurrentNode())->setDescription((yyvsp[(2) - (2)].sval));
		;}
    break;

  case 691:
#line 3614 "VRML97.y"
    {
			ViewpointNode *viewpoint = new ViewpointNode();
			viewpoint->setName(GetDEFName());
			ParserAddNode(viewpoint);
			ParserPushNode(VRML97_VIEWPOINT, viewpoint);
		;}
    break;

  case 692:
#line 3624 "VRML97.y"
    {
			ViewpointNode *viewpoint = (ViewpointNode *)ParserGetCurrentNode();
			viewpoint->initialize();
			ParserPopNode();
		;}
    break;

  case 695:
#line 3644 "VRML97.y"
    {
			((VisibilitySensorNode *)ParserGetCurrentNode())->setCenter(gVec3f);
		;}
    break;

  case 696:
#line 3648 "VRML97.y"
    {
			((VisibilitySensorNode *)ParserGetCurrentNode())->setEnabled((yyvsp[(2) - (2)].ival));
		;}
    break;

  case 697:
#line 3652 "VRML97.y"
    {
			((VisibilitySensorNode *)ParserGetCurrentNode())->setSize(gVec3f);
		;}
    break;

  case 698:
#line 3659 "VRML97.y"
    {
			VisibilitySensorNode *vsensor = new VisibilitySensorNode();
			vsensor->setName(GetDEFName());
			ParserAddNode(vsensor);
			ParserPushNode(VRML97_VISIBILITYSENSOR, vsensor);
		;}
    break;

  case 699:
#line 3669 "VRML97.y"
    {
			VisibilitySensorNode *vsensor = (VisibilitySensorNode *)ParserGetCurrentNode();
			vsensor->initialize();
			ParserPopNode();
		;}
    break;

  case 702:
#line 3689 "VRML97.y"
    {
			ParserPushNode(VRML97_WORLDINFO_INFO, ParserGetCurrentNode());
		;}
    break;

  case 703:
#line 3696 "VRML97.y"
    {
			ParserPopNode();
		;}
    break;

  case 704:
#line 3700 "VRML97.y"
    {
			((WorldInfoNode *)ParserGetCurrentNode())->setTitle((yyvsp[(2) - (2)].sval));
		;}
    break;

  case 705:
#line 3707 "VRML97.y"
    {
			WorldInfoNode *worldInfo = new WorldInfoNode();
			worldInfo->setName(GetDEFName());
			ParserAddNode(worldInfo);
			ParserPushNode(VRML97_WORLDINFO, worldInfo);
		;}
    break;

  case 706:
#line 3717 "VRML97.y"
    {
			WorldInfoNode *worldInfo = (WorldInfoNode *)ParserGetCurrentNode();
			worldInfo->initialize();
			ParserPopNode();
		;}
    break;


/* Line 1267 of yacc.c.  */
#line 6529 "VRML97.tab.cpp"
      default: break;
    }
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;


  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
      {
	YYSIZE_T yysize = yysyntax_error (0, yystate, yychar);
	if (yymsg_alloc < yysize && yymsg_alloc < YYSTACK_ALLOC_MAXIMUM)
	  {
	    YYSIZE_T yyalloc = 2 * yysize;
	    if (! (yysize <= yyalloc && yyalloc <= YYSTACK_ALLOC_MAXIMUM))
	      yyalloc = YYSTACK_ALLOC_MAXIMUM;
	    if (yymsg != yymsgbuf)
	      YYSTACK_FREE (yymsg);
	    yymsg = (char *) YYSTACK_ALLOC (yyalloc);
	    if (yymsg)
	      yymsg_alloc = yyalloc;
	    else
	      {
		yymsg = yymsgbuf;
		yymsg_alloc = sizeof yymsgbuf;
	      }
	  }

	if (0 < yysize && yysize <= yymsg_alloc)
	  {
	    (void) yysyntax_error (yymsg, yystate, yychar);
	    yyerror (yymsg);
	  }
	else
	  {
	    yyerror (YY_("syntax error"));
	    if (yysize != 0)
	      goto yyexhaustedlab;
	  }
      }
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse look-ahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  yydestruct ("Error: discarding",
		      yytoken, &yylval);
	  yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse look-ahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (yyn != YYPACT_NINF)
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;


      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  *++yyvsp = yylval;


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#ifndef yyoverflow
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEOF && yychar != YYEMPTY)
     yydestruct ("Cleanup: discarding lookahead",
		 yytoken, &yylval);
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}


#line 3724 "VRML97.y"


