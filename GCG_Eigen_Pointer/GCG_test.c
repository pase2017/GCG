/*************************************************************************
	> File Name: test_QR.c
	> Author: nzhang
	> Mail: zhangning114@lsec.cc.ac.cn
	> Created Time: Wed Nov 15 13:25:11 2017
 ************************************************************************/

#include "GCG_Eigen.h"

int main(int argc, char* argv[])
{
    GCG_MATRIX *A, *B;
	printf("reading matrix ...\n");
	A = GCG_ReadMatrix("dat/Andrews.txt");
	B = GCG_ReadMatrix("dat/BI_60000.txt");
	printf("read matrix, done! \n");

	GCG_SOLVER *solver;
	GCG_SOLVER_Create(&solver);
    GCG_GetCommandLineInfo(argc, argv, solver->para);
	GCG_Int    nev = solver->para->nev;
	GCG_Double *eval = (GCG_Double*)calloc(nev, sizeof(GCG_Double));
	void **evec;
	GCG_BuildVectorsbyMatrix(A, &evec, nev);
	//Setevec,eval,A,B
    GCG_SOLVER_SetMatrix(solver, A, NULL);
    //GCG_SOLVER_SetMatrix(solver, A, B);
    //GCG_SOLVER_SetEigenpairs(solver, eval, evec);

	GCG_Solve(solver);

	GCG_SOLVER_Free(&solver);
	GCG_FreeVectors(&evec, nev);
	free(eval);  eval = NULL;
	GCG_FreeMatrix(&A);
	GCG_FreeMatrix(&B);
}


//--------------------------------------------------------------
//----------------- 下面的函数是要用户提供的 -------------------
//--------------------------------------------------------------

//获取n_vec个随机向量初值
void GCG_GetRandomInitValue(void **V, GCG_Int n_vec)
{
	GCG_VEC **VV = (GCG_VEC**)V;
	GCG_Int i, j, size = VV[0]->size;
	srand((unsigned)time(NULL));
	for( i=0; i<n_vec; i++ )
	{
		for( j=0; j<size; j++ )
        {
			VV[i]->Entries[j] = rand()%(size+1)/((GCG_Double)size);
        }
	}
    //Writevecs(V, "dat/init_vec", n_vec);
}

//稀疏矩阵乘向量
void GCG_MatrixDotVec(void *Matrix, void *x, void *r)
{
	GCG_MATRIX *M = (GCG_MATRIX*)Matrix;
	GCG_VEC    *xx = (GCG_VEC*)x;
	GCG_VEC    *rr = (GCG_VEC*)r;
	GCG_Int    N_Rows = M->N_Rows, 
               N_Columns = M->N_Columns,
               i, j, length, start, end,
               *RowPtr,*KCol;
	GCG_Double *Mat_Entries, *x_Entries, *r_Entries, tmp;
	RowPtr      = M->RowPtr;
	KCol        = M->KCol;
	Mat_Entries = M->Entries;
	x_Entries   = xx->Entries;
	r_Entries   = rr->Entries;

	memset(r_Entries,0.0,N_Rows*sizeof(GCG_Double));

	start = RowPtr[0];
	for(i=0;i<N_Rows;i++)
	{    
		end = RowPtr[i+1];
		length = end - start;
		for(j=0;j<length;j++)
		{
			r_Entries[i] += Mat_Entries[start+j]*x_Entries[KCol[start+j]];      
		}
		start = end;
	}
}

//向量的线性代数操作：y = a*x+b*y
//如果 a=0.0, 那么 y = b*y, 相当于VecScale
//如果 b=0.0, 那么 y = a*x, 相当于VecScalep2q
//如果 a=1.0, b=0.0, 那么 y = x, 相当于VecCopy
//如果 a=0.0, b=0.0, 那么 y = 0, 相当于VecSetZero
void GCG_VecLinearAlg(GCG_Double a, void *x, GCG_Double b, void *y)
{
	GCG_VEC    *xx = (GCG_VEC*)x;
	GCG_VEC    *yy = (GCG_VEC*)y;
    GCG_Int    i, size = xx->size;
    GCG_Double *x_Entries = xx->Entries, 
               *y_Entries = yy->Entries;
    if(a == 0.0)
    {
        for(i=0; i<size; i++)
        {
            y_Entries[i] = b * y_Entries[i];
        }
    }
    else if(b == 0.0)
    {
        if(a == 1.0)
        {
            for(i=0; i<size; i++)
            {
                y_Entries[i] = x_Entries[i];
            }
        }
        else
        {
            for(i=0; i<size; i++)
            {
                y_Entries[i] = a * x_Entries[i];
            }
        }
    }
    else
    {
        for(i=0; i<size; i++)
        {
            y_Entries[i] = a * x_Entries[i] + b * y_Entries[i];
        }
    }
}

//计算向量内积，如果计算向量范数，就先计算内积，再开平方
void GCG_VecInnerProduct(void *x, void *y, GCG_Double *xTy)
{
	GCG_VEC    *xx = (GCG_VEC*)x;
	GCG_VEC    *yy = (GCG_VEC*)y;
    GCG_Int    i, size = xx->size;
    GCG_Double *x_Entries = xx->Entries, 
               *y_Entries = yy->Entries,
               xy = 0.0;
    for(i=0; i<size; i++)
    {
        xy += x_Entries[i]*y_Entries[i];
    }
    //return xTy;
	*xTy = xy;
}

//由已给向量创建向量组
void GCG_BuildVectorsbyVector(void *init_vec, void ***Vectors, GCG_Int n_vec)
{
	GCG_VEC *iinit_vec = (GCG_VEC*)init_vec;
	GCG_VEC ***VVectors = (GCG_VEC***)Vectors;
    GCG_Int i, size = iinit_vec->size;
    (*VVectors) = (GCG_VEC **)malloc(n_vec * sizeof(GCG_VEC*));
    for(i=0; i<n_vec; i++)
    {
        (*VVectors)[i] = (GCG_VEC *)malloc(sizeof(GCG_VEC));
		(*VVectors)[i]->size = size;
        (*VVectors)[i]->Entries = (GCG_Double *)calloc(size, sizeof(GCG_Double));
    }
}

//由已给矩阵创建向量组
void GCG_BuildVectorsbyMatrix(void *Matrix, void ***Vectors, GCG_Int n_vec)
{
	GCG_MATRIX *M = (GCG_MATRIX*)Matrix;
	GCG_VEC    ***VVectors = (GCG_VEC***)Vectors;
    GCG_Int i, size = M->N_Rows;
    (*VVectors) = (GCG_VEC **)malloc(n_vec * sizeof(GCG_VEC*));
    for(i=0; i<n_vec; i++)
    {
        (*VVectors)[i] = (GCG_VEC *)malloc(sizeof(GCG_VEC));
        (*VVectors)[i]->size = size;
        (*VVectors)[i]->Entries = (GCG_Double *)calloc(size, sizeof(GCG_Double));
    }
}

//释放向量组空间
void GCG_FreeVectors(void ***Vectors, GCG_Int n_vec)
{
	GCG_VEC ***VVectors = (GCG_VEC***)Vectors;
    GCG_Int i;
    for(i=0; i<n_vec; i++)
    {
        free((*VVectors)[i]->Entries);  (*VVectors)[i]->Entries = NULL;
        free((*VVectors)[i]);           (*VVectors)[i] = NULL;
    }
    free(*VVectors);                    (*VVectors) = NULL;
}

//从文件读入CSR格式矩阵
GCG_MATRIX *GCG_ReadMatrix(const char *filename)
{
    GCG_Int status = 0;
    FILE *file;
    file = fopen(filename,"r");
    if(!file)
    {
        printf("\ncannot open %s!\n", filename);
		exit(0);
    }
    if(status) printf("Read matrix: %s\n", filename); 
      
    GCG_Int nrow, ncol, nnz;
    fscanf(file, "%d\n", &nrow);
    fscanf(file, "%d\n", &ncol);
    fscanf(file, "%d\n", &nnz);
    
    if(status) 
        printf( "(%10d, %10d, %10d)\n", nrow, ncol, nnz );
   
    GCG_Int *ia = (GCG_Int *)malloc( (nrow+1)*sizeof(GCG_Int) );
    GCG_Int *ja = (GCG_Int *)malloc( nnz*sizeof(GCG_Int) );
    GCG_Double *aa = (GCG_Double *)malloc( nnz*sizeof(GCG_Double) );
    
    GCG_Int i;
   
    for(i=0;i<nrow+1;i++)
    {
        fscanf(file, "%d\n", ia+i);
    }
    for(i=0;i<nnz;i++)
    {
        fscanf(file, "%d\n", ja+i);
    }
    for(i=0;i<nnz;i++)
    {
        fscanf(file, "%lf\n", aa+i);
    }
    
    GCG_MATRIX *matrix = malloc( sizeof(GCG_MATRIX) );
    
    matrix->N_Rows = nrow;
    matrix->N_Columns = ncol;
    matrix->N_Entries = nnz;
    
    matrix->RowPtr = ia;
    matrix->KCol = ja;
    matrix->Entries = aa;

    fclose(file);
    return matrix;
}

//释放矩阵空间
void GCG_FreeMatrix(GCG_MATRIX **mat)
{
	free((*mat)->RowPtr);    (*mat)->RowPtr  = NULL;
	free((*mat)->KCol);      (*mat)->KCol    = NULL;
	free((*mat)->Entries);   (*mat)->Entries = NULL;
	free(*mat);              *mat            = NULL;
}

//CG迭代求解Matrix * x = b
//Matrix是用于求解的矩阵(GCG_MATRIX),b是右端项向量(GCG_VEC),x是解向量(GCG_VEC)
//用时，V_tmp+1
void GCG_CG(void *Matrix, void *b, void *x, void **V_tmp, 
        GCG_Double rate, GCG_Int max_it)
{
	//临时向量
    void         *r, *p, *tmp;
	GCG_Double   tmp1, tmp2, alpha, beta, error, last_error;
    GCG_Int      niter = 0;

	//CG迭代中用到的临时向量
	r   = V_tmp[0];
	p   = V_tmp[1];
	tmp = V_tmp[2];
  
    GCG_MatrixDotVec(Matrix,x,p); //tmp1 = A*x0
 
	GCG_VecLinearAlg(1.0, b, 0.0, r);
    GCG_VecLinearAlg(-1.0, p, 1.0, r);//r=b-p=r-p
    GCG_VecInnerProduct(r, r, &error);//用残量的模来判断误差
    error = sqrt(error);
	GCG_VecLinearAlg(1.0, r, 0.0, p);

    do{
        GCG_MatrixDotVec(Matrix,p,tmp);//tmp = A*p
        GCG_VecInnerProduct(r,p,&tmp1);
        GCG_VecInnerProduct(p,tmp,&tmp2);    
		alpha = tmp1/tmp2;
        GCG_VecLinearAlg(alpha, p, 1.0, x);   
        GCG_VecLinearAlg(-alpha, tmp, 1.0, r);
        last_error = error;
        GCG_VecInnerProduct(r, r, &error);   //用残量的模来判断误差
        error = sqrt(error);
    
        if(error/last_error < rate)
        {
            printf("error: %e, last_error: %e, error/last_error: %e, rate: %e\n", 
                error, last_error, error/last_error, rate);
            break;
        }
    
		//beta = -(r,tmp)/(p,tmp)
        GCG_VecInnerProduct(r,tmp,&tmp1);
        GCG_VecInnerProduct(p,tmp,&tmp2);    
		beta = -tmp1/tmp2;
        GCG_VecLinearAlg(1.0, r, beta, p);    

        niter++; 
    }while((error/last_error >= rate)&&(niter<max_it));

}

