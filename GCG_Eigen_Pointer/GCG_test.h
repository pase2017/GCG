/*************************************************************************
	> File Name: GCG_test.h
	> Author: nzhang
	> Mail: zhangning114@lsec.cc.ac.cn
	> Created Time: 2018年04月26日 星期四 21时02分06秒
 ************************************************************************/

#ifndef __GCGTEST__
#define __GCGTEST__
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GCG_Int int
#define GCG_Double double 

#ifndef __MATRIX__
#define __MATRIX__
typedef struct GCG_MATRIX_ {	
    /** 下面是存储稀疏矩阵的行压缩形式 */
    /** number rof rows */
    GCG_Int N_Rows;
    /** number columns */
    GCG_Int N_Columns;
    /** number of matrix entries */
    GCG_Int N_Entries;
    /** in which column is the current entry */
    GCG_Int *KCol;
    /** index in KCol where each row starts */
    GCG_Int *RowPtr;   
    /** matrix elements in an array */
    GCG_Double *Entries;
}GCG_MATRIX;
#endif

#ifndef __VECTOR__
#define __VECTOR__
typedef struct GCG_VEC_ {
    GCG_Int    size;
    GCG_Double *Entries;
}GCG_VEC;
#endif

void GCG_GetRandomInitValue(void **V, GCG_Int n_vec);
void GCG_MatrixDotVec(void *Matrix, void *x, void *r);
void GCG_VecLinearAlg(GCG_Double a, void *x, GCG_Double b, void *y);
void GCG_VecInnerProduct(void *x, void *y, GCG_Double *xTy);
void GCG_BuildVectorsbyVector(void *init_vec, void ***Vectors, GCG_Int n_vec);
void GCG_BuildVectorsbyMatrix(void *Matrix, void ***Vectors, GCG_Int n_vec);
void GCG_FreeVectors(void ***Vectors, GCG_Int n_vec);
void GCG_CG(void *Matrix, void *b, void *x, void **V_tmp, 
        GCG_Double rate, GCG_Int max_it);

//从文件读入CSR格式矩阵
GCG_MATRIX *GCG_ReadMatrix(const char *filename);
void GCG_FreeMatrix(GCG_MATRIX **mat);

#endif
