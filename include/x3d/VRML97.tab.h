/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton interface for Bison's Yacc-like parsers in C

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




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
#line 20 "VRML97.y"
{
int		ival;
float	fval;
char	*sval;
}
/* Line 1489 of yacc.c.  */
#line 497 "VRML97.tab.hpp"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif

extern YYSTYPE yylval;

