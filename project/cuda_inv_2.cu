#include <iostream>
#include <cmath>
#include <algorithm>
#include <fstream>
using namespace std;

#define MAXTHREAD 1024
#define DIM 1000

void print(float* matrix, int row_dim, int col_dim);

/* CUDA Variable */
float  *input_device,
       *result_device,
	   *temp_row,
	   *device_pivot;
int *temp_index, *pindex_device;
	   
__constant__ int device_i[1];
__constant__ int device_j[1];


__global__ void cuda_do_math(float* input_device, float* result_device, float* device_pivot){

	int k = blockIdx.x*MAXTHREAD + threadIdx.x;
	int col_dim = DIM;
	int d_i = device_i[0];
	int d_j = device_j[0];
	
	if(k<DIM)
	{
		if(d_i==d_j)
		{
			device_pivot[0] = input_device[d_i*col_dim+d_j];
			result_device[d_i*col_dim+k] /= device_pivot[0];
			input_device[d_i*col_dim+k] /= device_pivot[0];
		}else{
			device_pivot[0] = input_device[d_i*col_dim+d_j]/input_device[d_j*col_dim+d_j];
			input_device [d_i*col_dim+k] -= (device_pivot[0] * input_device [d_j*col_dim+k]);
			result_device[d_i*col_dim+k] -= (device_pivot[0] * result_device[d_j*col_dim+k]);
		}
	}
	
}


__global__ void elim(float* input_device, float* result_device, float* device_pivot)
{
	int i = threadIdx.y + blockDim.y * blockIdx.y;
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int col_dim = DIM;
	int p = device_pivot[0];
	float pivot_cu = input_device[i + col_dim * p];
	__syncthreads();
	
	if(i != device_pivot[0])
	{
		result_device[i + col_dim * j] -= pivot_cu * result_device[p + col_dim * j];
		input_device[i + col_dim * j] -= pivot_cu *input_device[p + col_dim * j];
	}
}

__global__ void normalize(float* input_device, float* result_device, float* device_pivot, int* temp_index){
	int k = blockIdx.x*MAXTHREAD + threadIdx.x;
	int col_dim = DIM;
	int p = device_pivot[0];
	temp_index[k] = input_device[ (col_dim+1)*p ];
	__syncthreads();
	
	input_device[ (p + col_dim*k) ] /= temp_index[k];
	result_device[ (p + col_dim*k) ] /= temp_index[k];
}

__global__ void reduce_max(float* input_device, float* result_device, int* temp_index, int* pindex_device){
	int k = blockIdx.x*MAXTHREAD + threadIdx.x;
	int col_dim = DIM;
	int range = DIM;
	int d_j = device_j[0];
	
	if( k>=d_j && k<col_dim && input_device[k*DIM+d_j]!=0 ){
		pindex_device[0] = k;
	}
}


__global__ void copy_row(float* input_device, float* result_device, float* temp_row, int* pindex_device){

	int k = blockIdx.x*MAXTHREAD + threadIdx.x;
	int col_dim = DIM;
	int d_i = pindex_device[0];
	int d_j = device_j[0];
	float tmp = 0;
	
	if(k<DIM)
	{
		//temp_row[k] = input_device[d_j*col_dim+k];
		input_device[d_j*col_dim+k] += input_device[d_i*col_dim+k];
		//input_device[d_i*col_dim+k] = temp_row[k];
		//
		//temp_row[k] = result_device[d_j*col_dim+k];
		result_device[d_j*col_dim+k] += result_device[d_i*col_dim+k];
		//result_device[d_i*col_dim+k] = temp_row[k];
	}
	
}

/** matrix inverse */
void inv(float* input, int row_dim, int col_dim, float* output)
{	
	int size = sizeof(float)*row_dim*col_dim;
	// check square matrix
	if(col_dim == row_dim)
	{
		cudaMalloc(&temp_row, col_dim*sizeof( float ));
		cudaMalloc(&temp_index, col_dim*sizeof( float ));
		cudaMalloc(&device_pivot, sizeof( float ));
		cudaMalloc(&pindex_device, sizeof( int ));
	
		cudaMemcpy(input_device, input, size, cudaMemcpyHostToDevice);
		cudaMemcpy(result_device, output, size, cudaMemcpyHostToDevice);
	
		for(int j = 0;j < col_dim; j++)
		{
			//find max magnitude
			int p = -1;
			/*
			float tmp = 0;
			
			for(int i = j; i < row_dim; i++)
			{
				if(abs(input[i*col_dim+j]) > tmp) 
				{
					tmp = abs(input[i*col_dim+j]);
					p = i;
				}
			}
			*/
			cudaMemcpyToSymbol(device_j, &j, sizeof( int ));
			reduce_max<<<col_dim/MAXTHREAD+1, MAXTHREAD>>>(input_device, result_device, temp_index, pindex_device);
			if (cudaGetLastError() != cudaSuccess) {cout<< "error"<<endl;}
			
			
			//int tt[1];
			//tt[0]=-1;
			//cudaMemcpyFromSymbol(&tt, device_pivot, sizeof( int ));
			//cudaMemcpy(&p, pindex_device, sizeof( int ), cudaMemcpyDeviceToHost);
			//cout<<p<<"p "<<tt[0]<<endl;
			
			// have zero row
			/*
			if(p == -1)
			{
				cout << "it's singular";
				return;
			}
			*/
			
			//cudaMemcpyToSymbol(device_i, &p, sizeof( int ));  //Actually is p
			//cudaMemcpyToSymbol(device_i, pindex_device, sizeof( int ));
			copy_row<<<col_dim/MAXTHREAD+1, MAXTHREAD>>>(input_device, result_device, temp_row, pindex_device);

			
			normalize<<<col_dim/MAXTHREAD+1, MAXTHREAD>>>(input_device, result_device, device_pivot, temp_index);
			elim<<<row_dim,col_dim>>>(input_device, result_device, device_pivot);
			
			/*

			//row operation
			for (int i = 0; i < row_dim; i++)
			{
				cudaMemcpyToSymbol(device_i, &i, sizeof( int ));
				
				if (cudaGetLastError() != cudaSuccess) {cout<< "error"<<endl;}
				
				cuda_do_math<<<col_dim/MAXTHREAD+1, MAXTHREAD>>>(input_device, result_device, device_pivot);			
				if (cudaGetLastError() != cudaSuccess) {cout<< "error"<<endl;}
				//print(output, row_dim, col_dim);
				//cout << "----------------------\n";
			}
			
			*/
		}
		
		cudaMemcpy(input, input_device, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(output, result_device, size, cudaMemcpyDeviceToHost);
				
		cudaFree(temp_row);
		cudaFree(temp_index);
		cudaFree(pindex_device);
		cudaFree(device_pivot);
		
		
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
		for(int j=0; j < row_dim; j++)
		{
			cout << matrix[i*col_dim+j]<<" ";
		}
		cout<<";"<<endl;
	}
}

/** matrix save */
void fprint(float* matrix, int row_dim, int col_dim)
{
	fstream  file; 
	file.open("inMatrix.txt",ios::out);   
	
	for(int i=0; i < row_dim; i++)
	{
		for(int j=0; j < row_dim; j++)
		{
			file << matrix[i*col_dim+j]<<" ";
		}
		file<<""<<endl;
	}
	file.close();
}


int main ()
{

	float* input;
	float* result;

	//random seed
	srand(0);
	
	//set dimention
	int row_dim = DIM;
	int col_dim = DIM;
	
	/* CUDA */
	int size = sizeof(float)*row_dim*col_dim;
	

	//initial array
	input = new float [size];
	result = new float [size];
	
    for(int i = 0; i < row_dim; i++)
    {
        for(int j = 0;j < col_dim; j++)
        {
            input[i*col_dim+j] = (float)(rand()%9);
            result[i*col_dim+j] = (i == j)?1.0f:0.0f;
        }
    }
	
	//fprint(input, row_dim, col_dim);
	
	/* CUDA */
	cudaMalloc(&input_device, size);
	cudaMalloc(&result_device, size);
	
    //check input
    
	fprint(input, row_dim, col_dim);
    
    cout << "----------------------\n";
    
    //test inverse
    inv(input, row_dim, col_dim, result);
    
    //check result
	//print(result, row_dim, col_dim);
    //print(input, row_dim, col_dim);
	
	/* CUDA */
	cudaFree(input_device);
	cudaFree(result_device);
	
	delete input;
	delete result;
    
	return 0;
}
