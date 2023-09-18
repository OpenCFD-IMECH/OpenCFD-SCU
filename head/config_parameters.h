#ifndef __READ_PARAMETERS_H
#define __READ_PARAMETERS_H

#include "stdio.h"

#ifdef __cplusplus
extern "C"{
#endif

// config macro
#define DEBUG_MODE

typedef double REAL;
// boundary lap size
#define LAP 4
// unknown number
#define NVARS 5                         

#define hipWarpSize 64

typedef struct scheme_choose{
    char *invis;
    char *vis;
}SCHEME_CHOOSE;

#define Negatie_T_limit 100
#define NegT_Max 500

typedef struct TYPE_NegT_{
    int Num_NegT;
    int NT_Point[NegT_Max][3];
}TYPE_NegT;

typedef struct configItem_
{
    char name[1000];
    char value[1000];
} configItem;


int ExtarctItem(char *src, char *name, char *value);
int ItemNUM(FILE *file, char *Item_name, int *NameNUM);
int PartItem(char *src, char part[][1000]);
void ModifyItem(char *name, char *buff);
void RemovalNUM(char *buff);
void SearchItem(FILE *file, configItem *List, int configNum);
void Schemes_Choose_ID(SCHEME_CHOOSE *scheme);

#ifdef __cplusplus
}
#endif
#endif

