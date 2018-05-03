/*************************************************************************
	> File Name: GCG_Eigen.h
	> Author: nzhang
	> Mail: zhangning114@lsec.cc.ac.cn
	> Created Time: 2017年12月28日 星期四 11时07分40秒
 ************************************************************************/
#ifndef __GCGEIGEN__
#define __GCGEIGEN__
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/unistd.h>
#include <string.h>

#include "GCG_test.h"

#include "lapacke.h"
#include "lapacke_utils.h"

extern lapack_int LAPACKE_dsyev( int matrix_order, char jobz, char uplo, 
        lapack_int n, double* a, lapack_int lda, double* w );

#define ABS_TOL 1
#define REL_TOL 2
#define A_ORTH 1
#define B_ORTH 2
#define EPS 2.220446e-16

typedef struct GCG_OPS_ {
	void (*GetRandomInitValue)   (void **V, GCG_Int n_vec);
	void (*MatrixDotVec)         (void *Matrix, void *x, void *r);
	void (*VecLinearAlg)         (GCG_Double a, void *x, GCG_Double b, void *y);
	void (*VecInnerProduct)      (void *x, void *y, GCG_Double *xTy);
	void (*BuildVectorsbyVector) (void *init_vec, void ***Vectors, GCG_Int n_vec);
	void (*BuildVectorsbyMatrix) (void *Matrix, void ***Vectors, GCG_Int n_vec);
	void (*FreeVectors)          (void ***Vectors, GCG_Int n_vec);
	void (*LinearSolver)         (void *Matrix, void *b, void *x, void **V_tmp, GCG_Double rate, GCG_Int max_it);
}GCG_OPS;

typedef struct GCG_PART_TIME_ {
    GCG_Double GetW_Time;
    GCG_Double GetX_Time;
    GCG_Double GetP_Time;
    GCG_Double Orth_Time;
    GCG_Double RayleighRitz_Time;
    GCG_Double Conv_Time;
    GCG_Double Subspace_Time;
}GCG_PART_TIME;

typedef struct GCG_OPS_TIME_COUNT_ {
    GCG_Double SPMV_Time;
    GCG_Int    SPMV_Count;
    GCG_Double VDot_Time;
    GCG_Int    VDot_Count;
    GCG_Double AXPY_Time;
    GCG_Int    AXPY_Count;
}GCG_OPS_TIME_COUNT;

typedef struct GCG_STATISTIC_PARA_ {
	GCG_PART_TIME *PartTimeTotal;
	GCG_PART_TIME *PartTimeInOneIte;
	GCG_OPS_TIME_COUNT *OpsTimeCountTotal;
	GCG_OPS_TIME_COUNT *OpsTimeCountInLinearSolver;
	GCG_Double ite_time;
	GCG_Double total_time;
}GCG_STATISTIC_PARA;

typedef struct GCG_ALGORITHM_ {
	//向量工作空间
	//V用于存储[X,P,W]基底
    GCG_VEC    **V;
	//V_tmp用作临时的向量存储空间
    GCG_VEC    **V_tmp;
	//RitzVec用于存储Ritz向量
    GCG_VEC    **RitzVec;
	//向量指针数组，用于交换向量指针的临时变量
    GCG_VEC    **Swap_tmp;

	//unlock用于存储未锁定的特征对编号(0-nev)
    GCG_Int    *unlock;
	//nunlock表示未锁定的特征对个数(0<=nunlock<=nev)
    GCG_Int    nunlock;
	//X中最多的向量个数
    GCG_Int    max_dim_x;
	//X中的向量个数
    GCG_Int    dim_x;
	//[X,P]中的向量个数
    GCG_Int    dim_xp;
	//[X,P,W]中的向量个数
    GCG_Int    dim_xpw;
	//上次GCG迭代中[X,P,W]中的向量个数
    GCG_Int    last_dim_xpw;
	//上次GCG迭代中X中的向量个数
    GCG_Int    last_dim_x;

	//最大残差
    GCG_Double max_res;
	//最小残差
    GCG_Double min_res;
	//残差和
    GCG_Double sum_res;
	//残差
    GCG_Double *res;

    //子空间操作用到的参数
	//子空间矩阵
    GCG_Double *subspace_matrix;
	//子空间特征向量
    GCG_Double *subspace_evec;
	//子空间矩阵的特征值
    GCG_Double *eval;
    //小规模的工作空间，原来的AA_sub等都放到work_space
    GCG_Double *work_space;

    //正交化用到的参数
	//用于正交化过程中存储非零向量的编号
    GCG_Int    *orth_ind;

	//GCG迭代次数,因为统计的是[X,P,W]都参与计算的迭代次数，所以从-1开始
    GCG_Int    niter;//初始化为-1

	GCG_STATISTIC_PARA *stat_para;
}GCG_ALGORITHM;

typedef struct GCG_PARA_ {
	//要求解的特征值个数，默认值为６
    GCG_Int    nev;
    //gcg最大迭代次数
    GCG_Int    ev_max_it;

    //基本参数
	//使用gcg或lobgcg,0:gcg,1:lobgcg
    GCG_Int    if_lobgcg;
	//用户是否给定初始特征向量,0:用户不给定,1:用户给定
    GCG_Int    given_init_evec;

	//用于统计时间
    GCG_STATISTIC_PARA *stat_para;

    //检查收敛性用到的参数
	//判定特征值已收敛的阈值
    GCG_Double ev_tol;
    //判定收敛的方式,1:绝对残差, 2:相对残差
    GCG_Int    conv_type;
	//是否每次ＧＣＧ迭代后打印特征值
    GCG_Int    print_eval;

    //正交化用到的参数
	//确定是用Ａ正交还是Ｂ正交,1:A_ORTH, 2:B_ORTH
    GCG_Int    orth_type;
	//确定正交化中出现０向量的阈值
    GCG_Double orth_zero_tol;
	//确定是否进行重正交化的阈值
    GCG_Double reorth_tol;
	//最大的重正交化次数
    GCG_Int    max_reorth_time;
    //print_orthzero确定是否打印正交化过程中出现0向量的信息
    //0:不打印，1:打印
    GCG_Int    print_orthzero;

    //判断特征值重数（或距离较近）的阈值
    GCG_Double multi_tol;//0.2

    //GCG内置LinearSolver(CG)参数
	//CG最大迭代次数
    GCG_Int    cg_max_it;
	//终止CG迭代残差下降比例
    GCG_Double cg_rate;
	//是否打印每次CG迭代的残差
    GCG_Int    print_cg_error;

    //是否打印GCG_PARA的参数信息
    GCG_Int    para_view;

}GCG_PARA;

typedef struct GCG_SOLVER_ {
    //参数：矩阵A,B
	void          *A;
	void          *B;
    void          **evec;
    GCG_Double    *eval;
	GCG_Int       nev;
    //矩阵向量操作与向量间操作函数
	GCG_OPS       *ops;
	GCG_PARA      *para;
	GCG_ALGORITHM *alg;
}GCG_SOLVER;

//GCG_SOLVER的一些操作

void GCG_Solve(GCG_SOLVER *solver);
void GCG_SOLVER_Create(GCG_SOLVER **solver);
void GCG_SOLVER_Free(GCG_SOLVER **solver);
GCG_OPS *GCG_OPS_Create(
	void (*GetRandomInitValue)   (void **V, GCG_Int n_vec),
	void (*MatrixDotVec)         (void *Matrix, void *x, void *r),
	void (*VecLinearAlg)         (GCG_Double a, void *x, GCG_Double b, void *y),
	void (*VecInnerProduct)      (void *x, void *y, GCG_Double *xTy),
	void (*BuildVectorsbyVector) (void *init_vec, void ***Vectors, GCG_Int n_vec),
	void (*BuildVectorsbyMatrix) (void *Matrix, void ***Vectors, GCG_Int n_vec),
	void (*FreeVectors)          (void ***Vectors, GCG_Int n_vec),
	void (*LinearSolver)         (void *Matrix, void *b, void *x, void **V_tmp, 
                                  GCG_Double rate, GCG_Int max_it));
void GCG_SOLVER_SetMatrix(GCG_SOLVER *solver, void *A, void *B);
void GCG_SOLVER_SetEigenpairs(GCG_SOLVER *solver, GCG_Double *eval, void **evec);

//GCG算法的一些函数
void GCG_Eigen(void *A, void *B, GCG_Double *eval, void **evec, GCG_SOLVER *solver);
//一些结构体的空间的创建与销毁
void GCG_ALGORITHM_Create(GCG_SOLVER *solver);
void GCG_STATISTIC_PARA_Create(GCG_STATISTIC_PARA **stat_para);
void GCG_PART_TIME_Create(GCG_PART_TIME **gcg_part_time);
void GCG_OPS_TIME_COUNT_Create(GCG_OPS_TIME_COUNT **gcg_ops_time_count);
void GCG_STATISTIC_PARA_Free(GCG_STATISTIC_PARA **stat_para);
void GCG_PARA_View(GCG_PARA *gcg_para);
void GCG_ALGORITHM_Free(GCG_SOLVER *solver);
void GCG_PARA_Create(GCG_PARA **para);

//GCG算法具体操作
void GCG_Orthogonal(void **V, void *B, GCG_Int start, GCG_Int *end, GCG_SOLVER *solver);
void GCG_GetSubspaceMatrix(void *A, void **V, GCG_Double *subspace_matrix, 
		GCG_SOLVER *solver);
void GCG_DenseVecsMatrixVecsSymmetric(GCG_Double *DenseMat, GCG_Double *RVecs, 
        GCG_Double *ProductMat, GCG_Int nr, GCG_Int dim, GCG_Double *tmp);
void GCG_MatDotVecSubspace(GCG_Double *DenseMat, GCG_Double *x, GCG_Double *b, 
		GCG_Int dim);
void GCG_SparseVecsMatrixVecsSymmetric(void *A, void **V, 
        GCG_Double *subspace_matrix, GCG_Int start, GCG_SOLVER *solver);
void GCG_ComputeSubspaceEigenpairs(GCG_Double *subspace_matrix, 
        GCG_Double *eval, GCG_Double *subspace_evec, GCG_SOLVER *solver);
void GCG_SortEigenpairs(GCG_Double *eval, GCG_Double *evec, GCG_ALGORITHM *alg);
void GCG_GetRitzVectors(void **V, GCG_Double *subspace_evec, void **RitzVec, 
		GCG_SOLVER *solver);
void GCG_SumSeveralVecs(void **V, GCG_Double *x, void *U, GCG_Int n_vec, 
		GCG_OPS *ops);
void GCG_CheckConvergence(void *A, void *B, GCG_Double *eval, void **evec, 
		GCG_SOLVER *solver);
void GCG_SwapVecs(void **V_1, void **V_2, GCG_Int size, GCG_ALGORITHM *alg);
void GCG_GetW(void *A, void *B, void **V, GCG_Double *eval, GCG_SOLVER *solver);
void GCG_GetP(GCG_Double *subspace_evec, void **V, GCG_SOLVER *solver);
void GCG_OrthogonalSubspace(GCG_Double *V, GCG_Double **B, GCG_Int start, GCG_Int *end, 
        GCG_Int dim, GCG_SOLVER *solver);
void GCG_CheckMultiplicity(GCG_Int start, GCG_Int end, GCG_Int *dim_x, GCG_Double *eval);
GCG_Double GCG_VecMatrixVec(void *a, void *Matrix, void *b, void *temp,
		GCG_OPS *ops);
GCG_Double GCG_VecNorm(void *x, GCG_OPS *ops);

//子空间线性代数操作
GCG_Double GCG_VecDotVecSubspace(GCG_Double *a, GCG_Double *b, GCG_Int n);
GCG_Double GCG_VecNormSubspace(GCG_Double *a, GCG_Int n);
void GCG_VecAXPBYSubspace(GCG_Double a, GCG_Double *x, GCG_Double b, GCG_Double *y, 
		GCG_Int n);
void GCG_VecScaleSubspace(GCG_Double alpha, GCG_Double *a, GCG_Int n);
GCG_Int GCG_max(GCG_Int a, GCG_Int b);
GCG_Double GCG_fmax(GCG_Double a, GCG_Double b);

//一些打印操作
void GCG_PrintConvInfo(GCG_SOLVER *solver);
void GCG_PrintFinalInfo(GCG_SOLVER *solver);
void GCG_PrintSubspaceEigenpairs(GCG_ALGORITHM *alg);

//一些其他操作
GCG_Double GCG_GetTime();
GCG_Int GCG_GetNevFromCommandLine(GCG_Int argc, char**argv);
void GCG_GetCommandLineInfo(GCG_Int argc, char **argv, GCG_PARA *para);
GCG_Int isnum(char s[]);


void GCG_ReadVecsFromFile(char *filename, GCG_VEC **vec, GCG_Int n_vec);
void GCG_WriteVecsToFile(char *filename, GCG_VEC **vec, GCG_Int n_vec);
#endif
