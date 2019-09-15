/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2011 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
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


/* include some standard header files */
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <alloca.h>

/* include local headers */
#include "config.h"
#include "errs.h"
#include "reader.h"


/* the last error */
int reader_err = reader_err_ok;


/* list of error messages. */
char *reader_err_msg[6] = {
	"Nothing bad happened.",
    "An unexpected NULL pointer was encountered.",
    "A call to malloc failed, probably due to insufficient memory.",
    "An input-output error occurred.",
    "Maximum buffer size reached.",
    "End of file reached.",
	};
char reader_buff[100];
    
/* the error macro. */
#define error(r,id)				reader_error( r , id , __LINE__ , __FUNCTION__ )
int reader_error ( struct reader *r , int id , int line , const char *function ) {
    sprintf( reader_buff , "reading line %i, col %i: %s" , r->line , r->col , reader_err_msg[-id] );
    return reader_err = errs_register( id , reader_buff , line , function , __FILE__ );
    }


/**
 * @brief wrap-up a reader.
 */
 
void reader_close ( struct reader *r ) {
    
    /* Free the character buffer. */
    if ( r->buff != NULL )
        free( r->buff );
        
    /* Flag as not ready. */
    r->flags = reader_flag_none;
    
    }


/**
 * @brief Read the next char.
 *
 * @param r The #reader.
 *
 * @return The next character or < 0 on error (see #reader_err).
 */
 
int reader_getc ( struct reader *r ) {

    /* Do we need to fill the buffer? */
    if ( r->first == r->last ) {
        if ( ( r->last = read( r->fd , r->buff , r->size ) ) < 0 )
            return error(r,reader_err_io);
        r->first = 0;
        }
    
    /* Get the next char. */
    if ( r->first < r->last ) {
        r->c = r->buff[ r->first ];
        r->first += 1;
        }
    else {
        r->c = EOF;
        r->flags |= reader_flag_eof;
        }

    /* End of line? */
    if ( r->c == '\n' || r->c == '\r' ) {
        r->line += 1;
        r->col = 0;
        }
        
    /* Otherwise, just advance. */
    else
        r->col += 1;
        
    /* Return the char. */
    return r->c;

    }
    

/**
 * @brief Read until a newline.
 *
 * @param r The #reader.
 * @param buff Pointer to @c char at which to store the token.
 * @param buff_size Size of the @c buff.
 *
 * @return The number of read characters or < 0 on error (see #reader_err).
 */
 
int reader_getline ( struct reader *r , char *buff , int buff_size ) {

    int k = 0;

    /* Check inputs. */
    if ( r == NULL || buff == NULL )
        return error(r,reader_err_null);
        
    /* Has an EOF already been reached? */
    if ( r->flags & reader_flag_eof )
        return error(r,reader_err_eof);
        
    /* Skip any input util a comment_start is hit. */
    while ( r->c != EOF && r->c != '\n' && r->c != '\r' ) {
    
        /* Is there room in the buffer? */
        if ( k < buff_size-1 ) {
        
            /* Store this character. */
            buff[k] = r->c;
            k += 1;
            
            /* Get the next char. */
            reader_getc( r );
                
            }
            
        /* Otherwise, buffer full. */
        else
            return error(r,reader_err_buff);
            
        }
        
    /* Skip the newline character. */
    reader_getc( r );
        
    /* Terminate the buffer. */
    buff[k] = 0;
    
    /* Return the number of read characters. */
    return k;
    
    }


/**
 * @brief Skip until a newline.
 *
 * @param r The #reader.
 *
 * @return The number of read characters or < 0 on error (see #reader_err).
 */
 
int reader_skipline ( struct reader *r ) {

    int k = 0;

    /* Check inputs. */
    if ( r == NULL )
        return error(r,reader_err_null);
        
    /* Has an EOF already been reached? */
    if ( r->flags & reader_flag_eof )
        return error(r,reader_err_eof);
        
    /* Skip any input util a comment_start is hit. */
    while ( r->c != EOF && r->c != '\n' && r->c != '\r' ) {
    
        /* Get the next char. */
        k += 1;
        reader_getc( r );

        }
        
    /* Skip the newline character. */
    reader_getc( r );
        
    /* Return the number of read characters. */
    return k;
    
    }


/**
 * @brief Check if a character is a comment stop.
 *
 * @param r The #reader.
 * @param c The character to verify.
 *
 * @return 1 if @c c is in the comm_stop of the #reader @c r or 0
 *      otherwise.
 */
 
int reader_iscomm_stop ( struct reader *r  , int c ) {

    int k;
    
    /* Is c in the whitespace? */
    for ( k = 0 ; k < r->nr_comm_stop ; k++ )
        if ( c == r->comm_stop[k] )
            return 1;
            
    /* Otherwise... */
    return 0;

    }
    
    
/**
 * @brief Check if a character is a comment start.
 *
 * @param r The #reader.
 * @param c The character to verify.
 *
 * @return 1 if @c c is in the comm_start of the #reader @c r or 0
 *      otherwise.
 */
 
int reader_iscomm_start ( struct reader *r  , int c ) {

    int k;
    
    /* Is c in the whitespace? */
    for ( k = 0 ; k < r->nr_comm_start ; k++ )
        if ( c == r->comm_start[k] )
            return 1;
            
    /* Otherwise... */
    return 0;

    }
    
    
/**
 * @brief Check if a character is whitespace.
 *
 * @param r The #reader.
 * @param c The character to verify.
 *
 * @return 1 if @c c is in the comm_stop of the #reader @c r or 0
 *      otherwise.
 */
 
int reader_isws ( struct reader *r  , int c ) {

    int k;
    
    /* Is c in the whitespace? */
    for ( k = 0 ; k < r->nr_ws ; k++ )
        if ( c == r->ws[k] )
            return 1;
            
    /* Otherwise... */
    return 0;

    }
    
    
/**
 * @brief Read the next comment from the given reader.
 *
 * @param r the #reader.
 * @param buff Pointer to @c char at which to store the token.
 * @param buff_size Size of the @c buff.
 *
 * @return The number of read characters or < 0 on error (see #reader_err).
 */

int reader_getcomment ( struct reader *r , char *buff , int buff_size ) {

    int k = 0;

    /* Check inputs. */
    if ( r == NULL || buff == NULL )
        return error(r,reader_err_null);
        
    /* Has an EOF already been reached? */
    if ( r->flags & reader_flag_eof )
        return error(r,reader_err_eof);
        
    /* Skip any input util a comment_start is hit. */
    while ( !reader_iscomm_start( r , r->c ) )
        if ( reader_getc( r ) == EOF )
            return error(r,reader_err_eof);
            
    /* Skip the comment start character. */
    if ( reader_getc( r ) == EOF ) {
        buff[0] = 0;
        return 0;
        }
            
    /* Write the input to the buffer until a comm_stop is reached. */
    while ( r->c != EOF && !reader_iscomm_stop( r , r->c ) ) {
    
        /* Check buffer length. */
        if ( k < buff_size-1 ) {
        
            /* Store the current char to the buff. */
            buff[k] = r->c;
            k += 1;
            
            /* Get the next char. */
            reader_getc( r );
                
            }
        else
            return error(r,reader_err_buff);
    
        } /* Read comment into buffer. */
        
    /* Terminate the comment. */
    buff[k] = 0;
        
    /* Read the next char. */
    reader_getc( r );
        
    /* Return the comment length. */
    return k;
        
    }


/**
 * @brief Skip the next comment from the given reader.
 *
 * @param r the #reader.
 *
 * @return The number of read characters or < 0 on error (see #reader_err).
 */

int reader_skipcomment ( struct reader *r ) {

    int k = 0;

    /* Check inputs. */
    if ( r == NULL )
        return error(r,reader_err_null);
        
    /* Has an EOF already been reached? */
    if ( r->flags & reader_flag_eof )
        return error(r,reader_err_eof);
        
    /* Skip any input util a comment_start is hit. */
    while ( !reader_iscomm_start( r , r->c ) )
        if ( reader_getc( r ) == EOF )
            return error(r,reader_err_eof);
            
    /* Skip the comment start character. */
    if ( reader_getc( r ) == EOF )
        return 0;
            
    /* Read the input until a comm_stop is reached. */
    while ( r->c != EOF && !reader_iscomm_stop( r , r->c ) ) {
    
        /* Get the next char. */
        k += 1;
        reader_getc( r );
                
        } /* Read comment into buffer. */
        
    /* Read the next char. */
    reader_getc( r );
        
    /* Return the comment length. */
    return k;
        
    }


/**
 * @brief Read a token from the given reader.
 * 
 * @param r The #reader.
 * @param buff Pointer to @c char at which to store the token.
 * @param buff_size Size of the @c buff.
 *
 * @return The number of read characters or < 0 on error (see #reader_err).
 */

int reader_gettoken ( struct reader *r , char *buff , int buff_size ) {

    int k = 0;

    /* Check inputs. */
    if ( r == NULL || buff == NULL )
        return error(r,reader_err_null);
        
    /* Has an EOF already been reached? */
    if ( r->flags & reader_flag_eof )
        return error(r,reader_err_eof);
        
    /* Skim-off whitespace and/or comments. */
    while ( 1 ) {
    
        /* Skip whitespace. */
        if ( reader_isws( r , r->c ) ) {
            if ( reader_getc( r ) == EOF )
                return error(r,reader_err_eof);
            }
            
        /* Skip comments. */
        else if ( reader_iscomm_start( r , r->c ) ) {
            do {
                if ( reader_getc( r ) == EOF )
                    return error(r,reader_err_eof);
                } while ( !reader_iscomm_stop( r , r->c ) );
            if ( reader_getc( r ) == EOF )
                return error(r,reader_err_eof);
            }
            
        else
            break;
            
        } /* get ws and comments. */
        
    /* Read the token. */
    while ( r->c != EOF && !reader_isws( r , r->c ) && !reader_iscomm_start( r , r->c ) ) {
    
        /* Check buffer length. */
        if ( k < buff_size-1 ) {
        
            /* Store the current char to the buff. */
            buff[k] = r->c;
            k += 1;
            
            /* Get the next char. */
            reader_getc( r );
                
            }
        else
            return error(r,reader_err_buff);
    
        } /* Read the token. */
        
    /* Terminate the token. */
    buff[k] = 0;
        
    /* Return the token length. */
    return k;
        
    }
    

/**
 * @brief Skip a token from the given reader.
 * 
 * @param r The #reader.
 
 * @return The number of read characters or < 0 on error (see #reader_err).
 */

int reader_skiptoken ( struct reader *r ) {

    int k = 0;

    /* Check inputs. */
    if ( r == NULL )
        return error(r,reader_err_null);
        
    /* Has an EOF already been reached? */
    if ( r->flags & reader_flag_eof )
        return error(r,reader_err_eof);
        
    /* Skim-off whitespace and/or comments. */
    while ( 1 ) {
    
        /* Skip whitespace. */
        if ( reader_isws( r , r->c ) ) {
            if ( reader_getc( r ) == EOF )
                return error(r,reader_err_eof);
            }
            
        /* Skip comments. */
        else if ( reader_iscomm_start( r , r->c ) ) {
            do {
                if ( reader_getc( r ) == EOF )
                    return error(r,reader_err_eof);
                } while ( !reader_iscomm_stop( r , r->c ) );
            if ( reader_getc( r ) == EOF )
                return error(r,reader_err_eof);
            }
            
        else
            break;
            
        } /* get ws and comments. */
        
    /* Read the token. */
    while ( r->c != EOF && !reader_isws( r , r->c ) && !reader_iscomm_start( r , r->c ) ) {
    
        /* Get the next char. */
        k += 1;
        reader_getc( r );
                
        } /* Read the token. */
        
    /* Return the token length. */
    return k;
        
    }
    

/**
 * @brief Initialize the reader.
 *
 * @param r The #reader structure.
 * @param file The @c FILE with which the #reader should be associated.
 * @param ws String containing the accepted whitespace characters.
 * @param comm_start String containing characters indicating the start
 *      of a comment.
 * @param comm_stop String containing characters indicating the end
 *      of a comment.
 *
 * The @c FILE supplied should be open and will be read as of its
 * current position.
 *
 * @return #reader_err_ok or < 0 on error (see #reader_err).
 */
 
int reader_init ( struct reader *r , int fd , char *ws , char *comm_start , char *comm_stop , int buffsize ) {

    /* Check inputs. */
    if ( r == NULL )
        return error(r,reader_err_null);
        
    /* Init the flags. */
    r->flags = reader_flag_none;
        
    /* Set the file. */
    r->fd = fd;
    
    /* Init the buffer. */
    if ( ( r->buff = (char *)malloc( buffsize ) ) == NULL )
        return error(r,reader_err_malloc);
    r->first = 0;
    r->last = 0;
    r->size = buffsize;
    
    /* Re-set the line and column counts. */
    r->line = 1;
    r->col = 0;
        
    /* Read the first character. */
    if ( reader_getc( r ) == EOF )
        r->flags = reader_flag_eof;
    else
        r->flags |= reader_flag_ready;
        
    /* Did the user supply whitespace? */
    if ( ws != NULL ) {
        r->ws = ws;
        for ( r->nr_ws = 0 ; ws[r->nr_ws] != 0 ; r->nr_ws++ );
        }
    else {
        r->ws = " \f\n\r\t\v";
        r->nr_ws = 6;
        }
    
    /* Did the user supply comment start/stop? */
    if ( comm_start != NULL ) {
        r->comm_start = comm_start;
        for ( r->nr_comm_start = 0 ; comm_start[r->nr_comm_start] != 0 ; r->nr_comm_start++ );
        }
    else {
        r->comm_start = "";
        r->nr_comm_start = 0;
        }
    if ( comm_stop != NULL ) {
        r->comm_stop = comm_stop;
        for ( r->nr_comm_stop = 0 ; comm_stop[r->nr_comm_stop] != 0 ; r->nr_comm_stop++ );
        }
    else {
        r->comm_stop = "\n\r";
        r->nr_comm_stop = 2;
        }
    
    /* We're all set. */
    return reader_err_ok;
    
    }


