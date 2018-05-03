/*************************************************************************
  > File Name: GCG_Eigen.c
  > Author: nzhang
  > Mail: zhangning114@lsec.cc.ac.cn
  > Created Time: Wed Nov 29 15:46:26 2017
 ************************************************************************/

#include "GCG_Eigen.h"

void GCG_Solve(GCG_SOLVER *solver)
{
    //用户可以不给eval,evec分配空间
    if(solver->eval == NULL)
    {
		GCG_Int nev = solver->para->nev;
		solver->eval = (GCG_Double*)calloc(nev, sizeof(GCG_Double));
		solver->ops->BuildVectorsbyMatrix(solver->A, &(solver->evec), nev);
    }

	GCG_Eigen(solver->A, solver->B, solver->eval, solver->evec, 
            solver);

}

//创建GCG_SOLVER
void GCG_SOLVER_Create(GCG_SOLVER **solver)
{
	//给solver创建空间
	(*solver) = (GCG_SOLVER*)malloc(sizeof(GCG_SOLVER));
	//创建函数指针结构GCG_OPS
	(*solver)->ops = GCG_OPS_Create(GCG_GetRandomInitValue,
									GCG_MatrixDotVec,
			                        GCG_VecLinearAlg,
			                        GCG_VecInnerProduct,
                                    GCG_BuildVectorsbyVector,
                                    GCG_BuildVectorsbyMatrix,
                                    GCG_FreeVectors,
									GCG_CG);
	//创建GCG_PARA参数空间
	GCG_PARA_Create(&(*solver)->para);
}

//释放GCG_SOLVER空间
void GCG_SOLVER_Free(GCG_SOLVER **solver)
{
    free((*solver)->para);  (*solver)->para = NULL;
    free((*solver)->ops);   (*solver)->ops  = NULL;
    free(*solver);          *solver         = NULL;
}

//创建ops,函数指针
GCG_OPS *GCG_OPS_Create(
	void (*GetRandomInitValue)   (void **V, GCG_Int n_vec),
	void (*MatrixDotVec)         (void *Matrix, void *x, void *r),
	void (*VecLinearAlg)         (GCG_Double a, void *x, GCG_Double b, void *y),
	void (*VecInnerProduct)      (void *x, void *y, GCG_Double *xTy),
	void (*BuildVectorsbyVector) (void *init_vec, void ***Vectors, GCG_Int n_vec),
	void (*BuildVectorsbyMatrix) (void *Matrix, void ***Vectors, GCG_Int n_vec),
	void (*FreeVectors)          (void ***Vectors, GCG_Int n_vec),
	void (*LinearSolver)         (void *Matrix, void *b, void *x, void **V_tmp, GCG_Double rate, GCG_Int max_it))
{
	GCG_OPS *ops = (GCG_OPS*)malloc(sizeof(GCG_OPS));

	ops->GetRandomInitValue   = GetRandomInitValue;
	ops->MatrixDotVec         = MatrixDotVec;
	ops->VecLinearAlg         = VecLinearAlg;
	ops->VecInnerProduct      = VecInnerProduct;
	ops->BuildVectorsbyVector = BuildVectorsbyVector;
	ops->BuildVectorsbyMatrix = BuildVectorsbyMatrix;
	ops->FreeVectors          = FreeVectors;
	ops->LinearSolver         = LinearSolver;
	
	return ops;
}

//设置求解矩阵
void GCG_SOLVER_SetMatrix(GCG_SOLVER *solver, void *A, void *B)
{
    solver->A = A;
    if(B == NULL)
    {
        solver->B = NULL;
    }
    else
    {
        solver->B = B;
    }
}

//设置eval,evec
void GCG_SOLVER_SetEigenpairs(GCG_SOLVER *solver, GCG_Double *eval, void **evec)
{
    solver->eval = eval;
    solver->evec = evec;
}

//求解特征值调用GCG_Eigen函数
//A,B表示要求解特征值的矩阵GCG_MATRIX，evec是要求解的特征向量GCG_Vec
void GCG_Eigen(void *A, void *B, GCG_Double *eval, void **evec, GCG_SOLVER *solver)
{
    //统计GCG求解时间
    GCG_Double ttotal1 = GCG_GetTime();
    //给GCG_ALGORITHM结构中各个部分分配空间
    //GCG_ALGORITHM结构中包含GCG算法用到的各种工作空间
    GCG_ALGORITHM_Create(solver);

	GCG_PARA      *para = solver->para;
	GCG_ALGORITHM *alg  = solver->alg;
	GCG_OPS       *ops  = solver->ops;

    if(para->para_view == 1)
    {
        //打印GCG_PARA参数信息
        GCG_PARA_View(para);
    }

    GCG_Int    i, nev = para->nev, 
               max_dim_x = alg->max_dim_x,
               ev_max_it = para->ev_max_it;
    void       **V = alg->V,
               **RitzVec = alg->RitzVec;
    GCG_Double *subspace_matrix = alg->subspace_matrix, 
               *subspace_evec = alg->subspace_evec,
			   titer1, titer2;
    void       *Orth_mat, *RayleighRitz_mat;
    
    if(para->given_init_evec == 0)
    {
		//获取随机初值函数也由用户提供
        ops->GetRandomInitValue(evec, nev);
    }

    //把用户提供的evec初值copy给V
    for(i=0; i<nev; i++)
    {
        ops->VecLinearAlg(1.0, evec[i], 0.0, V[i]);
    }

    //如果使用B内积，那么用矩阵B对基向量组进行正交化，用矩阵A计算子空间矩阵
    //如果使用A内积，那么用矩阵A对基向量组进行正交化，用矩阵B计算子空间矩阵
    if(para->orth_type == B_ORTH)
    {
        Orth_mat = B;
        RayleighRitz_mat = A;
    }
    else
    {
        Orth_mat = A;
        RayleighRitz_mat = B;
    }

    //对初始近似特征向量(V的第0至第dim_x列)做B正交化,
    //如果正交化中出现0向量，dim_x将被更新为非零向量第个数
    GCG_Orthogonal(V, Orth_mat, 0, &(alg->dim_x), solver);
    //dim_xpw表示V=[X,P,W]的总向量个数，此时V中只有X
    alg->dim_xpw = alg->dim_x;
    alg->last_dim_x = alg->dim_x;
    //计算得到子空间矩阵subspace_matrix=V^TAV
    GCG_GetSubspaceMatrix(RayleighRitz_mat, V, subspace_matrix, solver);
    //计算子空间矩阵的特征对
    GCG_ComputeSubspaceEigenpairs(subspace_matrix, alg->eval, subspace_evec, solver);
    //用子空间特征向量与基向量 V 计算Ritz向量
    GCG_GetRitzVectors(V, subspace_evec, RitzVec, solver);
    //检查收敛性
    GCG_CheckConvergence(A, B, alg->eval, RitzVec, solver);
    //Ritz向量放到V中作为下次迭代中基向量的一部分,即为[X,P,W]中的X
    GCG_SwapVecs(V, RitzVec, alg->dim_x, alg);
    //dim_xp为X,P向量个数，也是V中W向量存放的初始位置
    alg->dim_xp = alg->dim_x;
    //用未收敛的特征向量作为X求解线性方程组 A W = \lambda B X
    GCG_GetW(A, B, V, alg->eval, solver);
    alg->last_dim_xpw = alg->dim_xpw;
    alg->dim_xpw = 2 * alg->dim_x;
    //对基向量组V的第dim_x列到dim_xpw列进行正交化
    GCG_Orthogonal(V, Orth_mat, alg->dim_x, &(alg->dim_xpw), solver);
    alg->dim_xp = 0;
    //计算子空间矩阵subspace_matrix = V^T*A*V
    GCG_GetSubspaceMatrix(RayleighRitz_mat, V, subspace_matrix, solver);
    //计算子空间矩阵的特征对
    GCG_ComputeSubspaceEigenpairs(subspace_matrix, alg->eval, subspace_evec, solver);
    //用子空间特征向量与基向量 V 计算Ritz向量
    GCG_GetRitzVectors(V, subspace_evec, RitzVec, solver);
    //检查收敛性
    GCG_CheckConvergence(A, B, alg->eval, RitzVec, solver);

	//--------------------开始循环--------------------------------------------
    while((alg->nunlock > 0)&&(alg->niter < ev_max_it))
	{
        //计算P=X_j-X_{j-1},这次迭代用的X减去上一次迭代用的X
        GCG_GetP(subspace_evec, V, solver);
        //Ritz向量放到V中作为下次迭代中基向量的一部分
		GCG_SwapVecs(V, RitzVec, alg->dim_x, alg);
        //保存上次迭代的dim_xpw备用
        alg->last_dim_xpw = alg->dim_xpw;
        alg->dim_xpw = alg->dim_xp+alg->nunlock;
        //用未收敛的特征向量作为X求解线性方程组 A W = \lambda B X
		GCG_GetW(A, B, V, alg->eval, solver);
        //对W与前dim_xp个向量进行正交化
        //对基向量组V的第dim_xp列到dim_xpw列，即是W部分进行正交化
		GCG_Orthogonal(V, Orth_mat, alg->dim_xp, &(alg->dim_xpw), solver);
        //计算子空间矩阵subspace_matrix = V^T*A*V
        GCG_GetSubspaceMatrix(RayleighRitz_mat, V, subspace_matrix, solver);
        //计算子空间矩阵的特征对
		GCG_ComputeSubspaceEigenpairs(subspace_matrix, alg->eval, subspace_evec, solver);
        alg->last_dim_x = alg->dim_x;
        //检查特征值重数,确定X空间的维数
        GCG_CheckMultiplicity(nev+2, max_dim_x, &(alg->dim_x), alg->eval);
        //用子空间特征向量与基向量 V 计算Ritz向量
		GCG_GetRitzVectors(V, subspace_evec, RitzVec, solver);
        //检查收敛性
        GCG_CheckConvergence(A, B, alg->eval, RitzVec, solver);
	}

    //把计算得到的近似特征对拷贝给eval,evec输出
	memcpy(eval, alg->eval, nev*sizeof(GCG_Double));
	for( i=0; i<nev; i++ )
    {
        ops->VecLinearAlg(1.0, RitzVec[i], 0.0, evec[i]);
    }

    GCG_Double ttotal2 = GCG_GetTime();
	solver->alg->stat_para->total_time = ttotal2-ttotal1;
    //输出总时间信息，以及最终求得的特征值与残差信息
    GCG_PrintFinalInfo(solver);

	//释放GCG工作空间
    GCG_ALGORITHM_Free(solver);
}

//给GCG_Para结构中各个部分分配空间
void GCG_ALGORITHM_Create(GCG_SOLVER *solver)
{
    GCG_Int       i, nev    = solver->para->nev,
                  max_dim_x = GCG_max((GCG_Int)((GCG_Double)(nev)*1.25), nev+2);
	GCG_OPS       *ops      = solver->ops;
	solver->alg = (GCG_ALGORITHM*)malloc(sizeof(GCG_ALGORITHM));
	GCG_ALGORITHM *alg      = solver->alg;

    //对GCG算法中用到的一些参数进行初始化
    alg->max_dim_x = max_dim_x;
    alg->dim_x     = nev;
    alg->dim_xp    = 0;
    alg->dim_xpw   = nev;
    alg->niter     = -2;

    //近似特征值
    alg->eval            = (GCG_Double*)calloc(3*max_dim_x, sizeof(GCG_Double));
    //小规模的临时工作空间
    alg->work_space      = (GCG_Double*)calloc(4*max_dim_x*max_dim_x+3*max_dim_x, sizeof(GCG_Double));
    //存储一次迭代中的残差
    alg->res             = (GCG_Double*)calloc(max_dim_x, sizeof(GCG_Double));
    //用于存储子空间矩阵
    alg->subspace_matrix = (GCG_Double*)calloc(9*max_dim_x*max_dim_x, sizeof(GCG_Double));
    //存储子空间矩阵的特征向量
    alg->subspace_evec   = (GCG_Double*)calloc(9*max_dim_x*max_dim_x, sizeof(GCG_Double));

    //存储前nev个特征对中未收敛的特征对编号
    alg->unlock   = (GCG_Int*)calloc(max_dim_x, sizeof(GCG_Int));
    //正交化时用到的临时GCG_Int*型空间,用于临时存储非0的编号
    alg->orth_ind = (GCG_Int*)calloc(3*max_dim_x, sizeof(GCG_Int));
	//Swap_tmp用于交换向量指针
    alg->Swap_tmp = (void**)malloc(max_dim_x*sizeof(void*));

	//V,V_tmp,RitzVec是向量工作空间
	void *init_vec = solver->evec[0];
	ops->BuildVectorsbyVector(init_vec, &(alg->V), 3*max_dim_x);
	ops->BuildVectorsbyVector(init_vec, &(alg->V_tmp), GCG_max(3*max_dim_x,4));
	ops->BuildVectorsbyVector(init_vec, &(alg->RitzVec), max_dim_x);

	//GCG_STATISTIC_Para用于统计各部分时间
    GCG_STATISTIC_PARA_Create(&(alg->stat_para));
}

//创建统计算法中时间信息的结构体空间
void GCG_STATISTIC_PARA_Create(GCG_STATISTIC_PARA **stat_para)
{
    (*stat_para) = (GCG_STATISTIC_PARA*)malloc(sizeof(GCG_STATISTIC_PARA));
	GCG_PART_TIME_Create(&((*stat_para)->PartTimeTotal));
	GCG_PART_TIME_Create(&((*stat_para)->PartTimeInOneIte));
	GCG_OPS_TIME_COUNT_Create(&((*stat_para)->OpsTimeCountTotal));
	GCG_OPS_TIME_COUNT_Create(&((*stat_para)->OpsTimeCountInLinearSolver));
	(*stat_para)->ite_time   = 0.0;
	(*stat_para)->total_time = 0.0;
}

//释放统计算法中时间信息的结构体空间
void GCG_STATISTIC_PARA_Free(GCG_STATISTIC_PARA **stat_para)
{
    free((*stat_para)->PartTimeTotal);
    (*stat_para)->PartTimeTotal = NULL;
    free((*stat_para)->PartTimeInOneIte);
    (*stat_para)->PartTimeInOneIte = NULL;
    free((*stat_para)->OpsTimeCountTotal);
    (*stat_para)->OpsTimeCountTotal = NULL;
    free((*stat_para)->OpsTimeCountInLinearSolver);
    (*stat_para)->OpsTimeCountInLinearSolver = NULL;
    free(*stat_para);
    *stat_para = NULL;
}

//统计GCG算法各部分时间信息的初始化
void GCG_PART_TIME_Create(GCG_PART_TIME **gcg_part_time)
{
	(*gcg_part_time) = (GCG_PART_TIME*)malloc(sizeof(GCG_PART_TIME));
    (*gcg_part_time)->GetW_Time = 0.0;
    (*gcg_part_time)->GetX_Time = 0.0;
    (*gcg_part_time)->GetP_Time = 0.0;
    (*gcg_part_time)->Orth_Time = 0.0;
    (*gcg_part_time)->RayleighRitz_Time = 0.0;
    (*gcg_part_time)->Conv_Time = 0.0;
    (*gcg_part_time)->Subspace_Time = 0.0;
}

//统计GCG各种操作时间信息的初始化
void GCG_OPS_TIME_COUNT_Create(GCG_OPS_TIME_COUNT **gcg_ops_time_count)
{
	(*gcg_ops_time_count) = (GCG_OPS_TIME_COUNT*)malloc(sizeof(GCG_OPS_TIME_COUNT));
    (*gcg_ops_time_count)->SPMV_Time  = 0.0;
    (*gcg_ops_time_count)->SPMV_Count = 0;
    (*gcg_ops_time_count)->VDot_Time  = 0.0;
    (*gcg_ops_time_count)->VDot_Count = 0;
    (*gcg_ops_time_count)->AXPY_Time  = 0.0;
    (*gcg_ops_time_count)->AXPY_Count = 0;
}

//打印GCG_Para信息
void GCG_PARA_View(GCG_PARA *gcg_para)
{
    printf("\nGCG_Para:\n");
    printf("nev: %d\n", gcg_para->nev);
    printf("ev_max_it: %d\n", gcg_para->ev_max_it);
    printf("ev_tol: %e\n", gcg_para->ev_tol);
    printf("converged type: ");
    if(gcg_para->conv_type == REL_TOL)
    {
        printf("relative\n");
    }
    else
    {
        printf("abosolute\n");
    }
    printf("gcg or lobgcg? ");
    if(gcg_para->if_lobgcg == 0)
    {
        printf("gcg\n");
    }
    else
    {
        printf("lobgcg\n");
    }
    printf("given init eigenvector? ");
    if(gcg_para->given_init_evec == 0)
    {
        printf("no\n");
    }
    else
    {
        printf("yes\n");
    }
    printf("orthogonal type: ");
    if(gcg_para->orth_type == A_ORTH)
    {
        printf("A_ORTH\n");
    }
    else
    {
        printf("B_ORTH\n");
    }
    printf("reorthogonal tol: %e\n", gcg_para->reorth_tol);
    printf("max reorthogonal time: %d\n", gcg_para->max_reorth_time);
    printf("multiplicity tol: %e\n", gcg_para->multi_tol);
    printf("cg_max_it: %d\n", gcg_para->cg_max_it);
    printf("cg_rate: %e\n", gcg_para->cg_rate);
    printf("print cg error information? ");
    if(gcg_para->print_cg_error == 0)
    {
        printf("no\n");
    }
    else
    {
        printf("yes\n");
    }
}

//对V的所有列向量做关于矩阵B的正交化，如果B=NULL，那么进行L2正交化
//全部正交化，则start=0
//V是要正交化的向量组(GCG_Vec),B是矩阵(GCG_MATRIX)
void GCG_Orthogonal(void **V, void *B, GCG_Int start, GCG_Int *end, GCG_SOLVER *solver)
{
	GCG_Double    t1 = GCG_GetTime();
	GCG_ALGORITHM *alg  = solver->alg;
	GCG_PARA      *para = solver->para;
	GCG_OPS       *ops  = solver->ops;
	GCG_Int       i, j, n_nonzero = 0, n_zero = 0, reorth_time,
                  *Ind = alg->orth_ind,
                  print_orthzero  = para->print_orthzero,
                  max_reorth_time = para->max_reorth_time;
	GCG_Double    vin, vout, tmp, dd, 
                  orth_zero_tol   = para->orth_zero_tol,
                  reorth_tol      = para->reorth_tol;
    void          **V_tmp         = alg->V_tmp;
    //L2正交化时
	if(B == NULL)
	{
        //从V中start位置开始进行正交化
		for( i=start; i<(*end); i++ )
		{
            //如果i==0,说明start=0,对第一个向量直接进行单位化
			if(i == 0)
			{
				dd = GCG_VecNorm(V[0], ops);
				if(dd > 10*orth_zero_tol)
				{
					//做归一化
					ops->VecLinearAlg(0.0, V[0], 1.0/dd, V[0]);
					Ind[0] = 0;
					n_nonzero += 1;
				}
			}
			else
			{
				vout = GCG_VecNorm(V[i], ops);
                reorth_time = 0;
                //进行最多max_reorth_time次重正交化
				do{
                    reorth_time += 1;
					vin = vout;
                    //先去掉V中前start个向量中的分量
					for( j=0; j<start; j++ )
					{
						ops->VecInnerProduct(V[i], V[j], &tmp);
						//a,x,b,y,y=a*x+b*y
						ops->VecLinearAlg(-tmp, V[j], 1.0, V[i]);
					}
                    //对start之后不为0的向量进行正交化
					for( j=0; j<n_nonzero; j++ )
					{
						ops->VecInnerProduct(V[i], V[Ind[j]], &tmp);
						ops->VecLinearAlg(-tmp, V[Ind[j]], 1.0, V[i]);
					}
					vout = GCG_VecNorm(V[i], ops);
                }while((vout/vin < reorth_tol)&&(reorth_time < max_reorth_time));
                //如果向量非0，进行单位化
				if(vout > 10*orth_zero_tol)
				{
					ops->VecLinearAlg(0.0, V[i], 1.0/vout, V[i]);
					Ind[n_nonzero++] = i;
				}
				else
				{
                    if(print_orthzero == 1)
                    {
					    printf("In Orthogonal, there is a zero vector!, "
                                "i = %d, start = %d, end = %d\n", i, start, end);
                    }
				}
			}
		}
	}
	else
	{
        //从V中start位置开始进行正交化
        //B正交化，先计算前start个向量的矩阵乘向量放到V_tmp中
		for( i=0; i<start; i++ )
		{
			ops->MatrixDotVec(B, V[i], V_tmp[i]);
		}
		for( i=start; i<(*end); i++ )
		{
            //如果i==0,说明start=0,对第一个向量直接进行单位化
			if(i == 0)
			{
				dd = sqrt(GCG_VecMatrixVec(V[0], B, V[0], V_tmp[0], ops));
				if(dd > 10*orth_zero_tol)
				{
					ops->VecLinearAlg(0.0, V[0], 1.0/dd, V[0]);
					Ind[n_nonzero++] = 0;
					ops->MatrixDotVec(B, V[0], V_tmp[0]);
				}
			}
			else
			{
				vout = sqrt(GCG_VecMatrixVec(V[i], B, V[i], V_tmp[start+n_nonzero], 
							ops));
                reorth_time = 0;
                //进行最多max_reorth_time次重正交化
				do{
                    reorth_time += 1;
					vin = vout;
                    //先去掉V中前start个向量中的分量
					for( j=0; j<start; j++ )
					{
						ops->VecInnerProduct(V[i], V_tmp[j], &tmp);
						ops->VecLinearAlg(-tmp, V[j], 1.0, V[i]);
					}
                    //对start之后不为0的向量进行正交化
					for( j=0; j<n_nonzero; j++ )
					{
						ops->VecInnerProduct(V[i], V_tmp[start+j], &tmp);
						ops->VecLinearAlg(-tmp, V[Ind[j]], 1.0, V[i]);
					}
					vout = sqrt(GCG_VecMatrixVec(V[i], B, V[i], V_tmp[start+n_nonzero],
								ops));
                }while((vout/vin < reorth_tol)&&(reorth_time < max_reorth_time));
                //如果向量非0，进行单位化
				if(vout > 10*orth_zero_tol)
				{
					ops->VecLinearAlg(0.0, V[i], 1.0/vout, V[i]);
					ops->MatrixDotVec(B, V[i], V_tmp[start+n_nonzero]);
					Ind[n_nonzero++] = i;
				}
				else
				{
                    if(print_orthzero == 1)
                    {
					    printf("In Orthogonal, there is a zero vector!, "
                               "i = %d, start = %d, end = %d\n", i, start, end);
                    }
					//Nonzero_Vec[n_zero++] = V[i];
				}
			}
		}
	}
	//接下来要把V的所有非零列向量存储在地址表格中靠前位置
	*end = start+n_nonzero;
	if(n_zero > 0)
	{
		for( i=0; i<n_nonzero; i++ )
        {
			V[start+i] = V[Ind[i]];
            ops->VecLinearAlg(1.0, V[Ind[i]], 0.0, V[start+i]);
        }
	}
	GCG_Double t2 = GCG_GetTime();
    //统计正交化时间
    alg->stat_para->PartTimeTotal->Orth_Time += t2-t1;
    alg->stat_para->PartTimeInOneIte->Orth_Time = t2-t1;
}

//计算子空间矩阵subspace_matrix=V^TAV
//A是计算子空间矩阵时用到的矩阵Ａ(GCG_MATRIX), V是用到的向量组(GCG_Vec)
void GCG_GetSubspaceMatrix(void *A, void **V, GCG_Double *subspace_matrix, 
		GCG_SOLVER *solver)
{
	GCG_ALGORITHM *alg  = solver->alg;
    GCG_Int    i, start = alg->dim_xp, dim = alg->dim_xpw;
    GCG_Double *work_space = alg->work_space, t1, t2;
    t1 = GCG_GetTime();
    if(start != 0)
    {
        //最后一个参数是临时存储空间
        GCG_DenseVecsMatrixVecsSymmetric(subspace_matrix, 
                alg->subspace_evec, work_space, 
                start, alg->last_dim_xpw, work_space+start*start);
        memset(subspace_matrix, 0.0, dim*dim*sizeof(GCG_Double));
        for( i=0; i<start; i++ )
            memcpy(subspace_matrix+i*dim, work_space+i*start, start*sizeof(GCG_Double));
    }
    GCG_SparseVecsMatrixVecsSymmetric(A, V, subspace_matrix, start, solver);
    t2 = GCG_GetTime();
    //统计RayleighRitz问题求解时间
    alg->stat_para->PartTimeTotal->RayleighRitz_Time += t2-t1;
    alg->stat_para->PartTimeInOneIte->RayleighRitz_Time = t2-t1;
}

//计算ProductMat = LVecs ^T * DenseMat * RVecs, 
//此函数处理LVecs=RVecs的对称情况，RVecs中有nr个列向量，dim为列向量的长度，
void GCG_DenseVecsMatrixVecsSymmetric(GCG_Double *DenseMat, GCG_Double *RVecs, 
        GCG_Double *ProductMat, GCG_Int nr, GCG_Int dim, GCG_Double *tmp)
{
    GCG_Int  i, j;
    for( i=0; i<nr; i++ )
    {
        GCG_MatDotVecSubspace(DenseMat, RVecs+i*dim, tmp, dim);
        for( j=0; j<i+1; j++ )
        {
            ProductMat[i*nr+j] = GCG_VecDotVecSubspace(RVecs+j*dim, tmp, dim);
            ProductMat[j*nr+i] = ProductMat[i*nr+j];
        }
    }
}

//右乘:b=Ax,A是方阵，按列优先存储
void GCG_MatDotVecSubspace(GCG_Double *DenseMat, GCG_Double *x, GCG_Double *b, 
		GCG_Int dim)
{
    GCG_Int i;
    memset(b, 0.0, dim*sizeof(GCG_Double));

    for( i=0; i<dim; i++ )
    {
        GCG_VecAXPBYSubspace(x[i], DenseMat+i*dim, 1.0, b, dim);
    }
}

//计算subspace_matrix = V^T*A*V
//A是计算中用到的矩阵(GCG_MATRIX),V是计算中用到的向量组(GCG_Vec)
void GCG_SparseVecsMatrixVecsSymmetric(void *A, void **V, 
        GCG_Double *subspace_matrix, GCG_Int start, GCG_SOLVER *solver)
{
    GCG_Int i, j, dim = solver->alg->dim_xpw;
    void    *tmp = solver->alg->V_tmp[0];
	GCG_OPS *ops = solver->ops;
    for( i=start; i<dim; i++ )
    {
        ops->MatrixDotVec(A, V[i], tmp);
        for( j=0; j<i+1; j++ )
        {
            ops->VecInnerProduct(V[j], tmp, subspace_matrix+i*dim+j);
            subspace_matrix[j*dim+i] = subspace_matrix[i*dim+j];
        }
    }
}

//用LAPACKE_dsyev求解子空间矩阵的特征对
void GCG_ComputeSubspaceEigenpairs(GCG_Double *subspace_matrix, 
        GCG_Double *eval, GCG_Double *subspace_evec, GCG_SOLVER *solver)
{
	GCG_ALGORITHM *alg  = solver->alg;
    GCG_Double    t1 = GCG_GetTime();
    GCG_Int       dim = alg->dim_xpw;
    memcpy(subspace_evec, subspace_matrix, dim*dim*sizeof(GCG_Double));
	LAPACKE_dsyev( 102, 'V', 'U', dim, subspace_evec, dim, eval );
    //用lapack_syev计算得到的特征值是按从小到大排列,
    //如果用A内积，需要把特征值取倒数后再按从小到大排列
    //由于后面需要用到的只有前dim_x个特征值，所以只把前dim_x个拿到前面来
    if(solver->para->orth_type == A_ORTH)
    {
        GCG_SortEigenpairs(eval, subspace_evec, alg);
    }
    GCG_Double t2 = GCG_GetTime();
    //统计RayleighRitz问题求解时间
    alg->stat_para->PartTimeTotal->RayleighRitz_Time += t2-t1;
    alg->stat_para->PartTimeInOneIte->RayleighRitz_Time = t2-t1;
}

//用A内积的话，特征值从小到大，就是原问题特征值倒数的从小到大，
//所以顺序反向，同时特征值取倒数
void GCG_SortEigenpairs(GCG_Double *eval, GCG_Double *evec, GCG_ALGORITHM *alg)
{
    GCG_Int    dim_x = alg->dim_x,
               dim = alg->dim_xpw,
               head = 0, tail = dim-1;
    GCG_Double *work = alg->work_space;
	for( head=0; head<dim_x; head++ )
	{
	    tail = dim-1-head;
	    if(head < tail)
	    {
			memcpy(work, evec+head*dim, dim*sizeof(GCG_Double));
			memcpy(evec+head*dim, evec+tail*dim, dim*sizeof(GCG_Double));
			memcpy(evec+tail*dim, work, dim*sizeof(GCG_Double));
			work[0] = eval[head];
			eval[head] = 1.0/eval[tail];
			eval[tail] = 1.0/work[0];
	    }
	    else
	    {
		    break;
	    }
	}
}

//计算RitzVec = V*subspace_evec
//V是用来计算Ritz向量的基底向量组(GCG_Vec),RitzVec是计算得到的Ritz向量(GCG_Vec)
void GCG_GetRitzVectors(void **V, GCG_Double *subspace_evec, void **RitzVec, 
		GCG_SOLVER *solver)
{
    GCG_Double    t1 = GCG_GetTime();
	GCG_ALGORITHM *alg  = solver->alg;
	GCG_OPS       *ops  = solver->ops;
    GCG_Int       i, dim = alg->dim_xpw;
    for( i=0; i < alg->dim_x; i++ )
    {
        GCG_SumSeveralVecs(V, subspace_evec+i*dim, RitzVec[i], dim, ops);
    }
    GCG_Double t2 = GCG_GetTime();
    //统计计算Rtiz向量时间
    alg->stat_para->PartTimeTotal->GetX_Time += t2-t1;
    alg->stat_para->PartTimeInOneIte->GetX_Time = t2-t1;
}

//由一组基向量组V与子空间向量x相乘得到一个长向量U
//V是用来计算基底向量组(GCG_Vec),U是计算得到的向量(GCG_Vec)
void GCG_SumSeveralVecs(void **V, GCG_Double *x, void *U, GCG_Int n_vec, 
		GCG_OPS *ops)
{
	GCG_Int i;
	ops->VecLinearAlg(0.0, U, 0.0, U);
	for( i=0; i<n_vec; i++ )
    {
		ops->VecLinearAlg(x[i], V[i], 1.0, U);
    }
}

//计算残差，检查收敛性，并获取未收敛的特征对编号及个数
//A,B是用来计算残差的矩阵(GCG_MATRIX), evec是特征向量(GCG_Vec)
void GCG_CheckConvergence(void *A, void *B, GCG_Double *eval, void **evec, 
		GCG_SOLVER *solver)
{
    GCG_Double    t1 = GCG_GetTime();
	GCG_ALGORITHM *alg  = solver->alg;
	GCG_PARA      *para = solver->para;
	GCG_OPS       *ops  = solver->ops;
	GCG_Int       i, nunlock = 0, *unlock = alg->unlock,
                  conv_type = para->conv_type;
    GCG_Double    res_norm, evec_norm, residual, 
                  max_res = 0.0, min_res = 0.0, sum_res = 0.0,
                  ev_tol = para->ev_tol,
                  *res = alg->res,
                  **V_tmp = alg->V_tmp;
	for( i=0; i<para->nev; i++ )
	{
        //计算残量
		ops->MatrixDotVec(A, evec[i], V_tmp[0]);
        if(B == NULL)
        {
		    ops->VecLinearAlg(-eval[i], evec[i], 1.0, V_tmp[0]);
        }
        else
        {
		    ops->MatrixDotVec(B, evec[i], V_tmp[1]);
		    ops->VecLinearAlg(-eval[i], V_tmp[1], 1.0, V_tmp[0]);
        }
		res_norm  = GCG_VecNorm(V_tmp[0], ops);
		evec_norm = GCG_VecNorm(evec[i], ops);
        //计算相对/绝对残差
        if(conv_type == REL_TOL)
        {
            residual  = res_norm/evec_norm/GCG_max(1.0,fabs(eval[i]));
        }
        else
        {
            residual  = res_norm/evec_norm;
        }
        res[i] = residual;
        //统计最大、最小、总残差
	    sum_res += residual;
        if(i == 0)
        {
            max_res = residual;
            min_res = residual;
        }
        else
        {
            if(residual > max_res)
                max_res = residual;
            if(residual < min_res)
                min_res = residual;
        }
        //统计未收敛的特征对编号unlock及数量nunlock
        if(residual > ev_tol)
        {
            unlock[nunlock] = i;
            nunlock += 1;
        }
	}
    alg->nunlock = nunlock;
    alg->max_res = max_res;
    alg->min_res = min_res;
    alg->sum_res = sum_res;

    alg->niter += 1;
    //打印收敛信息
    GCG_PrintConvInfo(solver);

    GCG_Double t2 = GCG_GetTime();
    alg->stat_para->PartTimeTotal->Conv_Time += t2-t1;
    alg->stat_para->PartTimeInOneIte->Conv_Time = t2-t1;
}

//打印收敛性信息
void GCG_PrintConvInfo(GCG_SOLVER *solver)
{
    GCG_Int i;
	GCG_ALGORITHM *alg  = solver->alg;
	GCG_PARA      *para = solver->para;
    printf("\nnunlock(%d)= %d; max_res(%d)= %e; min_res(%d)= %e;\n", 
           alg->niter, alg->nunlock, 
           alg->niter, alg->max_res,
           alg->niter, alg->min_res);
    if(para->print_eval == 1)
    {
        GCG_Double *eval = alg->eval, 
                   *res = alg->res;
        if(para->conv_type == REL_TOL)
        {
            for( i=0; i < para->nev; i++ )
            {
                printf("eval(%d,%3d) = %20.15lf; "
                       "relative_res(%d,%3d) =  %20.15e;\n", 
                       alg->niter, i+1, eval[i], 
                       alg->niter, i+1, res[i]);
            }
        }
        else
        {
            for( i=0; i < para->nev; i++ )
            {
                printf("eval(%d,%3d) = %20.15lf; "
                       "abosolute_res(%d,%3d) =  %20.15e;\n", 
                       alg->niter, i+1, eval[i], 
                       alg->niter, i+1, res[i]);
            }
        }
    }
    fflush(stdout);
}

//交换V_1与V_2的指针
//V_1,V_2是用于交换的两个向量组指针(GCG_Vec)
void GCG_SwapVecs(void **V_1, void **V_2, GCG_Int size, GCG_ALGORITHM *alg)
{
    void    **tmp = alg->Swap_tmp; 
    GCG_Int i;
    for(i=0; i<size; i++)
    {
        tmp[i] = V_1[i];
        V_1[i] = V_2[i];
        V_2[i] = tmp[i];
    }
}

//用未收敛的特征向量作为X求解线性方程组 A W = \lambda B X
//A,B表示矩阵A,B(GCG_MATRIX),V表示向量组(GCG_Vec)
void GCG_GetW(void *A, void *B, void **V, GCG_Double *eval, GCG_SOLVER *solver)
{
    GCG_Double    t1 = GCG_GetTime();
	GCG_ALGORITHM *alg  = solver->alg;
	GCG_PARA      *para = solver->para;
	GCG_OPS       *ops  = solver->ops;
    GCG_Int       i, j, start = alg->dim_xp,
                  *unlock = alg->unlock;
    void          **V_tmp = alg->V_tmp;
    for( i=0; i < alg->nunlock; i++ )
    {
        j = unlock[i];
        //初值V[start+i]=V[i]
        //Vtmp[0]=\lambda*B*V[i]作为右端项
        //计算V[start+i]=A^(-1)BV[i]
        //调用CG迭代来计算线性方程组
        if(para->if_lobgcg == 0)
        {
            //使用GCG
            ops->VecLinearAlg(1.0, V[j], 0.0, V[start+i]);
            if(B == NULL)
            {
                ops->VecLinearAlg(eval[j], V[j], 0.0, V_tmp[0]);
            }
            else
            {
                ops->MatrixDotVec(B, V[j], V_tmp[0]);
                ops->VecLinearAlg(0.0, V_tmp[0], eval[j], V_tmp[0]);
            }
        }
        else
        {
            //使用LOBGCG
			ops->MatrixDotVec(A, V[j], V_tmp[0]);
            if(B == NULL)
            {
                ops->VecLinearAlg(-eval[j], V[j], 1.0, V_tmp[0]);
            }
            else
            {
			    ops->MatrixDotVec(B, V[j], V_tmp[1]);
                ops->VecLinearAlg(-eval[j], V_tmp[1], 1.0, V_tmp[0]);
            }
            ops->VecLinearAlg(0.0, V[start+i], 0.0, V[start+i]);
        }
        ops->LinearSolver(A, V_tmp[0], V[start+i], V_tmp+1, para->cg_rate, para->cg_max_it);
    }
    GCG_Double t2 = GCG_GetTime();
    alg->stat_para->PartTimeTotal->GetW_Time += t2-t1;
    alg->stat_para->PartTimeInOneIte->GetW_Time = t2-t1;
}

//获取P向量组
//V表示用于计算的基底向量组(GCG_Vec)
void GCG_GetP(GCG_Double *subspace_evec, void **V, GCG_SOLVER *solver)
{
	GCG_Double    t1 = GCG_GetTime();
	GCG_ALGORITHM *alg  = solver->alg;
	GCG_OPS       *ops  = solver->ops;
    GCG_Int       i, n_vec, dim_x = alg->dim_x,
                  last_dim_x = alg->last_dim_x,
                  dim_xpw = alg->dim_xpw,
                  *unlock = alg->unlock;
    void          **V_tmp = alg->V_tmp;
    //小规模正交化，构造d,构造subspace_matrix_sub用于下次计算
    for( i=0; i< alg->nunlock; i++ )
    {
        memset(subspace_evec+(dim_x+i)*dim_xpw, 0.0, dim_xpw*sizeof(GCG_Double));
        memcpy(subspace_evec+(dim_x+i)*dim_xpw+last_dim_x, 
               subspace_evec+unlock[i]*dim_xpw+last_dim_x, 
               (dim_xpw-last_dim_x)*sizeof(GCG_Double));
    }
    //更新dim_xp,dim_xp表示[X,P]的向量个数
    alg->dim_xp = alg->dim_x+alg->nunlock;
    //小规模evec中，X部分是已经正交的（BB正交），所以对P部分正交化
    GCG_OrthogonalSubspace(subspace_evec, NULL, dim_x, &(alg->dim_xp), dim_xpw, solver);
    //用基向量组V计算P所对应的长向量，存在V_tmp中
    n_vec = alg->dim_xp - dim_x;
    for( i=0; i < n_vec; i++ )
    {
        GCG_SumSeveralVecs(V, subspace_evec+(dim_x+i)*dim_xpw, V_tmp[i], dim_xpw, ops);
    }
	//交换V_tmp与V[dim_x:dim_xp]的向量指针，使Ｐ存到V的dim_x到dim_xp列中
    GCG_SwapVecs(V+dim_x, V_tmp, n_vec, alg);
    GCG_Double t2 = GCG_GetTime();
    alg->stat_para->PartTimeTotal->GetP_Time += t2-t1;
    alg->stat_para->PartTimeInOneIte->GetP_Time = t2-t1;
}

//进行部分的正交化, 对V中start位置中后的向量与前start的向量做正交化，
//同时V的start之后的向量自己也做正交化,dim表示向量长度
void GCG_OrthogonalSubspace(GCG_Double *V, GCG_Double **B, GCG_Int start, GCG_Int *end, 
        GCG_Int dim, GCG_SOLVER *solver)
{
	GCG_ALGORITHM *alg  = solver->alg;
	GCG_PARA      *para = solver->para;
    GCG_Int       i, j, n_nonzero = 0, reorth_time = 0,
                  print_orthzero  = para->print_orthzero,
                  max_reorth_time = para->max_reorth_time,
                  *Ind = alg->orth_ind;
    GCG_Double    vin, vout, tmp, dd,
                  orth_zero_tol = para->orth_zero_tol, 
                  reorth_tol    = para->reorth_tol;

    if(B == NULL)
    {
        for( i=start; i<(*end); i++ )
        {
            if(i == 0)
            {
	            dd = GCG_VecNormSubspace(V, dim);
                if(dd > 10*orth_zero_tol)
                {
                    GCG_VecScaleSubspace(1.0/dd, V, dim);
                    Ind[0] = 0;
                    n_nonzero = 1;
                }
            }
	        else
	        {
         
                vout = GCG_VecNormSubspace(V+i*dim, dim);
                reorth_time = 0; 
                do{
                    vin = vout;
                    for(j = 0; j < start; j++)
                    {
                        tmp = GCG_VecDotVecSubspace(V+j*dim, V+i*dim, dim);
                        GCG_VecAXPBYSubspace(-tmp, V+j*dim, 1.0, V+i*dim, dim);
                    }
                    for(j = 0; j < n_nonzero; j++)
                    {
                        tmp = GCG_VecDotVecSubspace(V+Ind[j]*dim, V+i*dim, dim);
                        GCG_VecAXPBYSubspace(-tmp, V+Ind[j]*dim, 1.0, V+i*dim, dim);
                    }
                    vout = GCG_VecNormSubspace(V+i*dim, dim);
                    reorth_time += 1;
                }while((vout/vin < reorth_tol)&&(reorth_time < max_reorth_time));
         
                if(vout > 10*orth_zero_tol)
                {
                    GCG_VecScaleSubspace(1.0/vout, V+i*dim, dim);
                    Ind[n_nonzero++] = i;
                }
                else
                {
                    if(print_orthzero == 1)
                    {
                        printf("In OrthogonalSubspace, there is a zero vector!"
                               "i = %d, start = %d, end: %d\n", i, start, *end);
                    }
                }
	        }

        }
    }

    if(n_nonzero < (*end-start))
    {
        *end = start+n_nonzero;
        for( i=0; i<n_nonzero; i++ )
        {
            //printf("Ind[%d] = %d\n", i, Ind[i]);
            memcpy(V+(start+i)*dim, V+Ind[i]*dim, dim*sizeof(GCG_Double));
        }
    }
}

//检查特征值重数，更新dim_x
void GCG_CheckMultiplicity(GCG_Int start, GCG_Int end, GCG_Int *dim_x, GCG_Double *eval)
{
    GCG_Int tmp, i;
    tmp = start;
    //dsygv求出的特征值已经排序是ascending,从小到大
    //检查特征值的数值确定下次要进行计算的特征值个数
    for( i=start; i<end; i++ )
    {
        if((fabs(fabs(eval[tmp]/eval[tmp-1])-1))<0.2)
            tmp += 1;
        else
            break;
    }
    *dim_x = tmp;
}

//GCG迭代结束后打印特征值及收敛情况，时间信息
void GCG_PrintFinalInfo(GCG_SOLVER *solver)
{
    GCG_ALGORITHM *alg = solver->alg;
    GCG_STATISTIC_PARA *stat_para = alg->stat_para;

    GCG_Int    i, nev = solver->para->nev;
    GCG_Double *eval = alg->eval, 
               *res = alg->res;
    printf("\nFinal eigenvalues and residual information:\n");
    if(solver->para->conv_type == REL_TOL)
    {
        printf("\n      "
               "eigenvalue           "
               "relative residual\n\n");
    }
    else
    {
        printf("\n      "
               "eigenvalue           "
               "abosolute residual\n\n");
    }
    for( i=0; i < nev; i++ )
    {
        printf("%5d  %20.15lf  %20.15e\n",
               i+1, eval[i], res[i]);
    }
    printf("\nTotal gcg iteration: %d, total time: %12.3lf\n",
           alg->niter, (stat_para->total_time));
    printf("\nDetail time in total gcg iteration:\n"
            "         GetP"
            "         GetW"
            "   Orthogonal"
            " RayleighRitz"
            "         GetX\n"
            " %12.3lf %12.3lf %12.3lf %12.3lf %12.3lf\n\n", 
            stat_para->PartTimeTotal->GetP_Time, 
            stat_para->PartTimeTotal->GetW_Time, 
            stat_para->PartTimeTotal->Orth_Time, 
            stat_para->PartTimeTotal->RayleighRitz_Time,
            stat_para->PartTimeTotal->GetX_Time);
}

//计算a^T*Matrix*b,两个向量的内积
//a,b表示要计算内积的两个向量(GCG_Vec)，
//Matrix表示计算内积用到的矩阵(GCG_MATRIX),
//temp是临时向量存储空间(GCG_Vec)
GCG_Double GCG_VecMatrixVec(void *a, void *Matrix, void *b, void *temp,
		GCG_OPS *ops)
{
	GCG_Double   value=0.0;
	ops->MatrixDotVec(Matrix, b, temp);
	ops->VecInnerProduct(a, temp, &value);
	return value;
}

//释放GCG_Para结构中分配的空间
void GCG_ALGORITHM_Free(GCG_SOLVER *solver)
{
	GCG_ALGORITHM *alg = solver->alg;
    free(alg->unlock);
    alg->unlock = NULL;
    free(alg->res);
    alg->res = NULL;
    free(alg->subspace_matrix);
    alg->subspace_matrix = NULL;
    free(alg->subspace_evec);
    alg->subspace_evec = NULL;
    free(alg->eval);
    alg->eval = NULL;
    free(alg->work_space);
    alg->work_space = NULL;
    free(alg->orth_ind);
    alg->orth_ind = NULL;
    free(alg->Swap_tmp);
    alg->Swap_tmp = NULL;
    GCG_STATISTIC_PARA_Free(&(alg->stat_para));
    
    GCG_Int max_dim_x = alg->max_dim_x;
	GCG_OPS *ops = solver->ops;
    ops->FreeVectors(&(alg->V), 3*max_dim_x);
    ops->FreeVectors(&(alg->V_tmp), GCG_max(3*max_dim_x, 4));
    ops->FreeVectors(&(alg->RitzVec), max_dim_x);

    free(alg); alg = NULL;
}

//创建GCG_Para结构并进行初始化
void GCG_PARA_Create(GCG_PARA **para)
{
    (*para) = (GCG_PARA *)malloc(sizeof(GCG_PARA));
    //默认计算6个特征值
    (*para)->nev             = 6;
    (*para)->ev_max_it       = 30;
    (*para)->if_lobgcg       = 0;
    (*para)->given_init_evec = 0;
    (*para)->ev_tol          = 1e-8;
    (*para)->conv_type       = REL_TOL;
    (*para)->orth_type       = B_ORTH;
    (*para)->orth_zero_tol   = EPS;
    (*para)->reorth_tol      = 0.75;
    (*para)->max_reorth_time = 3;
    (*para)->print_orthzero  = 1;
    (*para)->multi_tol       = 0.2;
    (*para)->cg_max_it       = 20;
    (*para)->cg_rate         = 1e-2;
    (*para)->print_cg_error  = 0;
    (*para)->print_eval      = 1;
    (*para)->para_view       = 1;

}

//获取当前时刻时间
GCG_Double GCG_GetTime()
{
    struct rusage usage;
    GCG_Double ret;
      
    if(getrusage(RUSAGE_SELF, &usage) == -1) 
        printf("Error in GCG_GetTime!\n");
 
    ret = ((GCG_Double) usage.ru_utime.tv_usec)/1000000;
 
    ret += usage.ru_utime.tv_sec;
 
    return ret;
}

//子空间向量操作
//计算向量a和b的内积
GCG_Double GCG_VecDotVecSubspace(GCG_Double *a, GCG_Double *b, GCG_Int n)
{
	GCG_Int i;
	GCG_Double value = 0.0;
	for( i=0; i<n; i++ )
		value += a[i]*b[i];
	return value;
}

//计算向量a的范数
GCG_Double GCG_VecNormSubspace(GCG_Double *a, GCG_Int n)
{
	GCG_Int i;
	GCG_Double value = 0.0;
	for( i=0; i<n; i++ )
		value += a[i]*a[i];
	return sqrt(value);
}

//计算y=a*x+b*y
void GCG_VecAXPBYSubspace(GCG_Double a, GCG_Double *x, GCG_Double b, GCG_Double *y, 
		GCG_Int n)
{
	GCG_Int i;
	for( i=0; i<n; i++ )
		y[i] = a*x[i] + b*y[i];
}

//a=alpha*a
void GCG_VecScaleSubspace(GCG_Double alpha, GCG_Double *a, GCG_Int n)
{
	GCG_Int i;
	for( i=0; i<n; i++ )
		a[i] *= alpha;
}

//求整型最大值
GCG_Int GCG_max(GCG_Int a, GCG_Int b)
{
    return (a>b)?a:b;
}

//求GCG_Double型最大值
GCG_Double GCG_fmax(GCG_Double a, GCG_Double b)
{
    return (a>b)?a:b;
}

//打印子空间特征对
void GCG_PrintSubspaceEigenpairs(GCG_ALGORITHM *alg)
{
    GCG_Int    i, n = alg->dim_xpw;
    GCG_Double *eval = alg->eval,
               *evec = alg->subspace_evec;
    printf("eigenpairs, niter: %d\n", alg->niter);
    for(i=0; i<n; i++)
    {
        printf("%20.15lf\n", eval[i]);
    }
    for(i=0;  i<n*n; i++)
    {
        printf("%20.15lf\n", evec[i]);
    }
}

//从命令行读取nev
GCG_Int GCG_GetNevFromCommandLine(GCG_Int argc, char**argv)
{
    GCG_Int arg_index = 0, nev;
    while(arg_index < argc) 
    {
        if(0 == strcmp(argv[arg_index], "-nev")) 
        {
            arg_index++;
            nev = atoi(argv[arg_index++]);
			break;
        }
	}
	return nev;
}

void GCG_GetCommandLineInfo(GCG_Int argc, char **argv, GCG_PARA *para)
{
    GCG_Int arg_index = 0,
            print_usage = 0,
            myid;
  
    while(arg_index < argc) 
    {
		/*
        if(0 == strcmp(argv[arg_index], "-nev")) 
        {
            arg_index++;
            para->nev = atoi(argv[arg_index++]);
        }
		*/
        if(0 == strcmp(argv[arg_index], "-ev_max_it")) 
        {
            arg_index++;
            para->ev_max_it = atoi(argv[arg_index++]);
        }
        else if(0 == strcmp(argv[arg_index], "-if_lobgcg")) 
        {
            arg_index++;
            para->if_lobgcg = atoi(argv[arg_index++]);
        }
        else if(0 == strcmp(argv[arg_index], "-ev_tol")) 
        {
            arg_index++;
            para->ev_tol = atof(argv[arg_index++]);
        }
        else if(0 == strcmp(argv[arg_index], "-print_eval")) 
        {
            arg_index++;
            para->print_eval = atoi(argv[arg_index++]);
        }
        else if(0 == strcmp(argv[arg_index], "-cg_max_it")) 
        {
            arg_index++;
            para->cg_max_it = atoi(argv[arg_index++]);
        }
        else if(0 == strcmp(argv[arg_index], "-cg_rate")) 
        {
            arg_index++;
            para->cg_rate = atof(argv[arg_index++]);
        }
        else if(0 == strcmp(argv[arg_index], "-orth_type")) 
        {
            arg_index++;
            para->orth_type = atoi(argv[arg_index++]);
        }
        else
        {
            arg_index++;
        }
    }
  
}

//判断字符串是否是整型数字
GCG_Int isnum(char s[])
{
    GCG_Int i;
    for(i=0; i<strlen(s); i++)
    {
        if(s[i] < '0' || s[i] > '9')
        {
            return 0;
        }
    }
    return 1;
}

//计算向量范数
//x是向量(GCG_Vec)
GCG_Double GCG_VecNorm(void *x, GCG_OPS *ops)
{
    GCG_Double xTx;
	ops->VecInnerProduct(x, x, &xTx);
    return sqrt(xTx);
}

//下面两个函数调试时用
#if 0
//从文件读入向量
void GCG_ReadVecsFromFile(char *filename, GCG_VEC **vec, GCG_Int n_vec)
{
	FILE *fp = NULL;
	GCG_Int size = vec[0]->size;
	GCG_Int i, j;
	fp = fopen(filename,"r");
	if(!fp)
	{
		printf("open file error\n");
		return -1;
	}
	for( i=0; i<n_vec; i++ )
    {
		for( j=0; j<size; j++ )
			fscanf(fp, "%lf", &(vec[i]->Entries[j]));
    }
	fclose(fp);
}

//将向量写入文件
void GCG_WriteVecsToFile(char *filename, GCG_VEC **vec, GCG_Int n_vec)
{
	FILE *file = NULL;
	file = fopen(filename,"w");
	if(!file)
	{
		printf("\ncannot open the %s!\n", filename);
		exit(0);
	}
	GCG_Int i, j, size = vec[0]->size;

	for(i=0;i<n_vec;i++)
		for(j=0;j<size;j++)
			fprintf(file,"%18.15f\n", vec[i]->Entries[j]);

	fclose(file);
}  
#endif
