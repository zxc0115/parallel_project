#include <iostream>
#include <fstream>
#include <cstdlib>
using namespace std;


#define TILE_WIDTH 16  
 
__global__ void MatrixMulKernle(int m, int n, int k, float *A,float  *B, float *C)
{
	 //å®??share memoryï¼Œå??¨æ–¼æ¯å€‹blockä¸?
	__shared__ float ds_A[TILE_WIDTH][TILE_WIDTH]; 
	__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
 
	//ç°¡å??æ?è¨˜æ?,?ºç¾ä¸‹é¢6?‹è¡¨ç¤ºç??°æ–¹å°±æ˜¯å¹³è??„åœ°?¹ã€?
	int bx = blockIdx.x;		int by = blockIdx.y;
	int tx = threadIdx.x;		int ty = threadIdx.y;
 
	//ç¢ºå?çµæ??©é™£ä¸­ç?è¡Œå???
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
 
	//?¨æ?è®Šæ•¸
	float Cvalue = 0;
 
	//å¾ªç’°è®€?¥A,B tileï¼Œè?ç®—ç??œçŸ©????†é?æ®µé€²è?è¨ˆç?
	for (int t=0; t<(n-1)/TILE_WIDTH+1; ++t)
	{
		//å°†A,B?©é™£?¦ç??–ç?çµæ??¾å…¥shared memoryä¸­ï?æ¯å€‹threadè®€?–ç›¸?‰æ–¼C?ƒç??„A/B?©é™£?ƒç?
		if (Row < m && t * TILE_WIDTH + tx < n)		//è¶Šç??•ç?ï¼Œæ»¿è¶³ä»»?å¤§å°ç??©é™£?¸ä?ï¼ˆå¯?¸ï?
			//ds_A[tx][ty] = A[t*TILE_WIDTH + tx][Row];
		    ds_A[tx][ty] = A[Row*n+t*TILE_WIDTH+tx];//ä»¥å?ä½µç??¹å?è¼‰å…¥tile
		else
			ds_A[tx][ty] = 0.0;
 
		if (t * TILE_WIDTH + ty < n && Col < k)
			//ds_B[tx][ty] = B[Col][t*TILE_WIDTH + ty];
            ds_B[tx][ty] = B[(t*TILE_WIDTH + ty)*k+Col];
		else
			ds_B[tx][ty] = 0.0;	
 
		//ä¿è?tileä¸­æ??‰ç??ƒç?è¢«è???
		__syncthreads();
		
		for (int i = 0; i < TILE_WIDTH; ++i)
            Cvalue += ds_A[i][ty] * ds_B[tx][i];//å¾shared memoryä¸­å???
 
		//ç¢ºä??€?‰threadå®Œæ?è¨ˆç?å¾Œï??²è?ä¸‹ä??‹é?æ®µç?è¨ˆç?
		__syncthreads();
 
		if(Row < m && Col < k)
			C[Row*k+Col]=Cvalue;		
	}
}



class Matrix{
public:
	int row;
	int col;
	float *element;
public:
	Matrix(){
		row = 0;
		col = 0;
		element = NULL;
	}
	
	Matrix(int r, int c, char mode='z'){
		row = r;
		col = c;
		element = new float [r*c];
		if(mode=='r'){
			for(int i=0;i<r*c;i++){
				element[i] = (float)(rand()%1000)/1000-0.5f;
			}
		}else if(mode=='z'){
			for(int i=0;i<r*c;i++){
				element[i] = 0.0f;
			}
		}
	}
	
	~Matrix(){
		delete element;
	}
	
	void print(){
		for(int i=0;i<row;i++){
			for(int j=0;j<col;j++){
				cout<<element[i*col+j]<<" ";
			}
			cout<<endl;
		}
	}
	
	Matrix(const Matrix &src){
		row = src.row;
		col = src.col;
		delete element;
		element = new float [src.row*src.col];
		for(int i=0;i<src.row*src.col;i++){
			element[i] = src.element[i];
		}
	}
	
	void operator=(const Matrix &src){
		row = src.row;
		col = src.col;
		delete element;
		element = new float [src.row*src.col];
		for(int i=0;i<src.row*src.col;i++){
			element[i] = src.element[i];
		}
	}
	
public:
	Matrix operator+(const Matrix &src){
		
		Matrix des(row,col,' ');
		for(int i=0;i<row*col;i++){
			des.element[i] = element[i] + src.element[i];
		}
		return des;
	}
	
	Matrix operator-(const Matrix &src){
		
		Matrix des(row,col,' ');
		for(int i=0;i<row*col;i++){
			des.element[i] = element[i] - src.element[i];
		}
		return des;
	}
	
	Matrix operator*(const Matrix &src){
		
		if(row!=src.row || col!=src.col){ cerr<<"Matrix dim error!!"<<endl; throw; }
		
		Matrix des(row,col,' ');
		for(int i=0;i<row*col;i++){
			des.element[i] = element[i]*src.element[i];
		}
		return des;
	}
	
	Matrix dot(const Matrix &src){
		
		if(col!=src.row){ cerr<<"Matrix dim error!!"<<endl; throw; }
		
		Matrix des(row,src.col,' ');
		for(int i=0;i<row;i++){
			for(int j=0;j<src.col;j++){
				int current = i*src.col+j;
				des.element[current] = 0;
				for(int k=0;k<col;k++){
					des.element[current] += element[i*col+k]*src.element[k*src.col+j];
				}
			}
		}
		return des;
	}
	
	Matrix multiply(const float value){
		
		Matrix des(row,col,' ');
		for(int i=0;i<row*col;i++){
			des.element[i] = element[i]*value;
		}
		return des;
	}
	
	Matrix transpose(){
		Matrix des(col,row,' ');
		for(int i=0;i<row;i++){
			for(int j=0;j<col;j++){
				des.element[j*row+i] = element[i*col+j];
			}
		}
		return des;
	}
	
public:
	Matrix ReLU(){
		Matrix des(row,col,' ');
		for(int i=0;i<row*col;i++){
			des.element[i] = (element[i]>0)?element[i]:0;
		}
		return des;
	}
	
	Matrix D_ReLU(){
		Matrix des(row,col,' ');
		for(int i=0;i<row*col;i++){
			des.element[i] = (element[i]>0)?1:0;
		}
		return des;
	}	
	
	Matrix addBias(){
		Matrix des(row,col+1,' ');
		for(int i=0;i<row;i++){
			for(int j=0;j<col;j++){
				des.element[i*(col+1)+j] = element[i*col+j];
			}
			des.element[i*(col+1)+col] = 1;
		}
		return des;
	}
	
	Matrix removeBias(){
		Matrix des(row,col-1,' ');
		for(int i=0;i<row;i++){
			for(int j=0;j<col-1;j++){
				des.element[i*(col-1)+j] = element[i*col+j];
			}
		}
		return des;
	}
	
public:
	void readMatrixFile(const char* fname){
		fstream file; 
		file.open(fname,ios::in);
		file>>row;
		file>>col;
		delete element;
		element = new float [row*col];
		for(int i=0;i<row*col;i++){
			file>>element[i];
		}
		file.close();
	}
	
	void writeMatrixFile(const char* fname){
		fstream file; 
		file.open(fname,ios::out);
		file<< row << " " << col <<endl;
		for(int i=0;i<row*col;i++){
			file<< element[i] << " ";
		}
		file.close();
	}

public:
	Matrix cuda_dot(const Matrix &src){
		
		if(col!=src.row){ cerr<<"Matrix dim error!!"<<endl; throw; }
		
		Matrix des(row,src.col,' ');
		
		int m=row,n=col,k=src.col;
 
		/** CUDA ->-> */
		int size = sizeof(float);
		float *d_a;
		float *d_b;
		float *d_c;
		cudaMalloc((void**)&d_a,m*n*size);
		cudaMalloc((void**)&d_b,n*k*size);
		cudaMalloc((void**)&d_c,m*k*size);
		
		cudaMemcpy(d_a, element, size*m*n, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, src.element, size*n*k, cudaMemcpyHostToDevice);
		
		dim3 dimGrid((k-1)/TILE_WIDTH+1,(m-1)/TILE_WIDTH+1,1);	//?‘ä??–æ•´
		dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);
		MatrixMulKernle<<<dimGrid,dimBlock>>>(m,n,k,d_a,d_b,d_c);
		
		cudaMemcpy(des.element, d_c, size*m*k, cudaMemcpyDeviceToHost);
		
		
		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);
		/** CUDA <-<- */
		
		return des;
	}
};



int main(){
	srand(0);
	
	float eta = 0.005;	//learning rate
	
	Matrix designMtx(20,2,'r');
	designMtx.readMatrixFile("x.txt");
	
	Matrix target(20,1,'r');
	target.readMatrixFile("y.txt");

	int input_node = 2;
	int layers1_node = 1024;
	int layers2_node = 1024;
	int output_node = 1;
	
	Matrix w1(input_node,layers1_node,'r');
	Matrix w2(layers1_node+1,layers2_node,'r');
	Matrix w3(layers2_node+1,output_node,'r');
	
	for(int it=0;it<100;it++)
	{
		//FP
		Matrix l1_a;
		Matrix l1_z;
		l1_a = designMtx.cuda_dot(w1);
		l1_z = l1_a.ReLU();
		l1_z = l1_z.addBias();
		
		Matrix l2_a;
		Matrix l2_z;
		l2_a = l1_z.cuda_dot(w2);
		l2_z = l2_a.ReLU();
		l2_z = l2_z.addBias();
		
		Matrix l3_o;
		l3_o = l2_z.cuda_dot(w3);
		
		//BP	
		Matrix diffO;
		diffO = l3_o - target;
		Matrix gradient_w3;
		gradient_w3 = l2_z.transpose().cuda_dot(diffO);
		w3 = w3 - gradient_w3.multiply(eta);
		
		Matrix diff2;
		diff2 = diffO.cuda_dot(w3.transpose());
		Matrix d_h2;
		d_h2 = l2_a.D_ReLU();
		diff2 = d_h2*diff2.removeBias();
		Matrix gradient_w2;
		gradient_w2 = l1_z.transpose().cuda_dot(diff2);
		w2 = w2 - gradient_w2.multiply(eta);
		
		Matrix diff1;
		diff1 = diff2.cuda_dot(w2.transpose());
		
		Matrix d_h1;
		d_h1 = l1_a.D_ReLU();
		diff1 = d_h1*diff1.removeBias();
		Matrix gradient_w1;
		gradient_w1 = designMtx.transpose().cuda_dot(diff1);
		w1 = w1 - gradient_w1.multiply(eta);
		
	}
	
	return 0;
} 
