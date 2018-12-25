#include <iostream>
#include <cmath>
#include <algorithm>
using namespace std;
__global__ void RowOperation1(float* matrix_cu,int* rank_cu,float* inverse_cu, int* dim)
{
	int i = threadIdx.y + blockDim.y * blockIdx.y;
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	float pivot_cu = matrix_cu[i + dim[0] * rank_cu[0]];
	__syncthreads();
	if(i != rank_cu[0])
	{
		inverse_cu[i + dim[0] * j] -= pivot_cu * inverse_cu[rank_cu[0] + dim[0] * j];
		matrix_cu[i + dim[0] * j] -= pivot_cu * matrix_cu[rank_cu[0] + dim[0] * j];
	}
}
__global__ void RowOperation2(float* matrix_cu,int* rank_cu,float* inverse_cu, int* dim)
{
	int i = threadIdx.x;
	float pivot = matrix_cu[(dim[0]+1) * (rank_cu[0])];
	__syncthreads();
	matrix_cu[rank_cu[0] + dim[0] * i] /= pivot;
	inverse_cu[rank_cu[0] + dim[0] * i] /= pivot;
}


/** matrix inverse */
void inv(float* matrix, int row_dim, int col_dim,float* inverse)
{
	// check square matrix
	if(col_dim == row_dim)
	{
		int * dime = new int [1];
		int * rank =new int [1];
		dime[0]=col_dim;
		float * matrix_cu;
                float * inverse_cu;
                int * dim;
                int * rank_cu;
                int matrix_size = sizeof(float) * row_dim * col_dim;
		int int_size = sizeof(int);
		cudaError_t err;
                cudaMalloc(&matrix_cu,matrix_size);
                cudaMalloc(&inverse_cu,matrix_size);
		cudaMalloc(&rank_cu,int_size);
		err = cudaGetLastError();
		if (err != cudaSuccess) {cout<< "rank_malloc wrong";return;}
                for(int j = 0;j < col_dim; j++)
		cudaMalloc(&dim,int_size);
		cudaMemcpy(dim,dime,int_size,cudaMemcpyHostToDevice);
		err = cudaGetLastError();
                if (err != cudaSuccess) {cout<< "dim_copy wrong";return;}
		for(int j = 0;j < col_dim; j++)
		{
			rank[0]=j;
			//find max magnitude
			float tmp = 0;
			int p = -1;
			for(int i = j; i < row_dim; i++)
			{
				if(abs(matrix[i + row_dim * j]) != 0) 
				{
					tmp = abs(matrix[i + row_dim * j]);
					p = i;
					if(j != p)
					{
						for(int k=0;k < col_dim; k++)
						{
						swap(matrix[j + row_dim * k],matrix[p + row_dim * k]);
                                		swap(inverse[j + row_dim * k],inverse[p + row_dim * k]);
						}
					}
					break;
				}
			}
			
			// have zero row
			if(p == -1)
			{
				cout << "it's singular";
				return;
			}
			cudaMemcpy(rank_cu,rank,int_size,cudaMemcpyHostToDevice);
			err = cudaGetLastError();
                        if (err != cudaSuccess) {cout<< "rank_copy wrong";return;}
			cudaMemcpy(matrix_cu,matrix,matrix_size,cudaMemcpyHostToDevice);
			cudaMemcpy(inverse_cu,inverse,matrix_size,cudaMemcpyHostToDevice);
			err = cudaGetLastError();
                        if (err != cudaSuccess) {cout<< "inverse_copy wrong";return;}
			//row operation
			RowOperation2<<<1,col_dim>>>(matrix, rank_cu, inverse, dim);
			err = cudaGetLastError();
			if (err != cudaSuccess) {cout<< "2 wrong";return;}
			RowOperation1<<<row_dim,col_dim>>>(matrix, rank_cu, inverse, dim);
			err = cudaGetLastError();
                        if (err != cudaSuccess) {cout<< "1 wrong";return;}
			cudaMemcpy(matrix,matrix_cu,matrix_size,cudaMemcpyDeviceToHost);
			cudaMemcpy(inverse,inverse_cu,matrix_size,cudaMemcpyDeviceToHost);
			
		}
	}
	else
	{
		cout << "it isn't sqare matrix";
		return;
	}
}

/** matrix print */
void print(float* matrix, int row_dim, int col_dim)
{
	for(int i=0; i < row_dim; i++)
	{
		for(int j=0; j < col_dim; j++)
		{
			cout << matrix[i+ col_dim * j]<<" ";
		}
		cout<<endl;
	}
}

int main ()
{
	//random seed
	srand(0);
	
	//set dimention
	int row_dim = 6;
	int col_dim = 6;
	
	//initial array
	float* inverse = new float [row_dim * col_dim];
	float* result = new float [row_dim * col_dim];
    for(int i = 0; i < row_dim * col_dim; i++)
    {
        inverse[i] = rand()%9;
        result[i] = (i / col_dim == i % row_dim)?1.0f:0.0f;
        /*for(int j = 0;j < col_dim; j++)
        {
            inverse[i][j] = float(rand()%9);
            result[i][j] = (i == j)?1.0f:0.0f;
        }*/
    }
    
    //check input
    print(inverse, row_dim, col_dim);
    
    cout << "----------------------\n";
    
    //test inverse
    inv(inverse, row_dim, col_dim, result);
    
    //check result
    print(result, row_dim, col_dim);
    
    
	return 0;
}
