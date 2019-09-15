/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2010 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
 * Coypright (c) 2017 Andy Somogyi (somogyie at indiana dot edu)
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 ******************************************************************************/

#ifndef INCLUDE_ERRS_H_
#define INCLUDE_ERRS_H_
#include "platform.h"
#include "stdio.h"


/* Some defines. */
#define errs_maxstack                           100

#define errs_err_ok                             0
#define errs_err_io                             -1

MDCORE_BEGIN_DECLS


/* Global variables. */
extern int errs_err;
extern const char *errs_err_msg[];


/* Functions. */

int errs_register( int id , const char *msg , int line , const char *func , char *file );
int errs_dump(FILE *out );
void errs_clear( );


MDCORE_END_DECLS

#endif // INCLUDE_ERRS_H_
