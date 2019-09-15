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

/* Reader error codes */
#define reader_err_ok                    0
#define reader_err_null                  -1
#define reader_err_malloc                -2
#define reader_err_io                    -3
#define reader_err_buff                  -4
#define reader_err_eof                   -5


/** Flags. */
#define reader_flag_none                    0
#define reader_flag_ready                   1
#define reader_flag_eof                     2


/** ID of the last error */
extern int reader_err;


/** The reader structure */
struct reader {

    /** Status flags. */
    unsigned int flags;

    /** File to which this reader is associated. */
    int fd;
    
    /** Current character. */
    int c;
    
    /** Character buffer. */
    char *buff;
    int first, last, size;
    
    /** Current location in file. */
    int line, col;
    
    /** Characters defined as whitespace. */
    char *ws;
    int nr_ws;
    
    /** Characters defined as comments. */
    char *comm_start, *comm_stop;
    int nr_comm_start, nr_comm_stop;
    
    };
    

/* associated functions */
void reader_close ( struct reader *r );
int reader_getc ( struct reader *r );
int reader_init ( struct reader *r , int fd , char *ws , char *comm_start , char *comm_stop , int buffsize );
int reader_gettoken ( struct reader *r , char *buff , int buff_size );
int reader_getcomment ( struct reader *r , char *buff , int buff_size );
int reader_getline ( struct reader *r , char *buff , int buff_size );
int reader_skiptoken ( struct reader *r );
int reader_skipcomment ( struct reader *r );
int reader_skipline ( struct reader *r );
int reader_isws ( struct reader *r  , int c );
int reader_iscomm_start ( struct reader *r  , int c );
int reader_iscomm_stop ( struct reader *r  , int c );
