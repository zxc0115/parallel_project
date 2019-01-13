
#include <iostream>
using namespace std;
#define TILE_WIDTH 16  
 
__global__ void MatrixMulKernle(int m, int n, int k, float *A,float  *B, float *C)
{
	__shared__ float ds_A[TILE_WIDTH][TILE_WIDTH]; 
	__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
 
	int bx = blockIdx.x;		int by = blockIdx.y;
	int tx = threadIdx.x;		int ty = threadIdx.y;
 
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
 
	float Cvalue = 0;
 
	for (int t=0; t<(n-1)/TILE_WIDTH+1; ++t)
	{
	
		if (Row < m && t * TILE_WIDTH + tx < n)		
		    ds_A[tx][ty] = A[Row*n+t*TILE_WIDTH+tx];
		else
			ds_A[tx][ty] = 0.0;
 
		if (t * TILE_WIDTH + ty < n && Col < k)
            ds_B[tx][ty] = B[(t*TILE_WIDTH + ty)*k+Col];
		else
			ds_B[tx][ty] = 0.0;	
		__syncthreads();
		
		for (int i = 0; i < TILE_WIDTH; ++i)
            Cvalue += ds_A[i][ty] * ds_B[tx][i];
		__syncthreads();
 
		if(Row < m && Col < k)
			C[Row*k+Col]=Cvalue;		
	}
}
 
int main()
{
	int m=1,n=10000,k=10000;
 	
	float* A=new float [m*n];
	float* B=new float [n*k];
	float* C=new float [m*k];
	for(int i =0;i < m*n;i++)
		A[i]=rand();
	for(int i=0;i < n*k;i++)
		B[i]=rand();
	int size = sizeof(float);
	float *d_a;
	float *d_b;
	float *d_c;
	cudaMalloc((void**)&d_a,m*n*size);
	cudaMalloc((void**)&d_b,n*k*size);
	cudaMalloc((void**)&d_c,m*k*size);

	cudaMemcpy(d_a, A, size*m*n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, B, size*n*k, cudaMemcpyHostToDevice);

	dim3 dimGrid((k-1)/TILE_WIDTH+1,(m-1)/TILE_WIDTH+1,1);	
	dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);
 
	MatrixMulKernle<<<dimGrid,dimBlock>>>(m,n,k,d_a,d_b,d_c);
 
	cudaMemcpy(C, d_c, size*m*k, cudaMemcpyDeviceToHost);
 
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
 
	return 0;
}
