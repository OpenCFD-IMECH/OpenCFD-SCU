#ifndef __IO_WARP_H
#define __IO_WARP_H

#include "stdio.h"

#define FWRITE(ptr , size , num , stream) \
    {   \
        int _ENCODE_START = num*size;\
        int _ENCODE_END = num*size;\
        fwrite(&_ENCODE_START , sizeof(int) , 1 , stream);\
        fwrite(ptr , size , num , stream);\
        fwrite(&_ENCODE_END , sizeof(int) , 1 , stream);\
    }
    
#define FREAD(ptr , size , num , stream) \
    {   int tmp_buffer;\
        fread(&tmp_buffer , sizeof(int) , 1 , stream);\
        fread(ptr , size , num , stream);\
        fread(&tmp_buffer , sizeof(int) , 1 , stream);\
    }

#endif
    