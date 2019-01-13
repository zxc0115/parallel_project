#include <iostream>
#include <fstream>
#include <cstdlib>
using namespace std;

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
		l1_a = designMtx.dot(w1);
		l1_z = l1_a.ReLU();
		l1_z = l1_z.addBias();
		
		Matrix l2_a;
		Matrix l2_z;
		l2_a = l1_z.dot(w2);
		l2_z = l2_a.ReLU();
		l2_z = l2_z.addBias();
		
		Matrix l3_o;
		l3_o = l2_z.dot(w3);
		
		//BP	
		Matrix diffO;
		diffO = l3_o - target;
		Matrix gradient_w3;
		gradient_w3 = l2_z.transpose().dot(diffO);
		w3 = w3 - gradient_w3.multiply(eta);
		
		Matrix diff2;
		diff2 = diffO.dot(w3.transpose());
		Matrix d_h2;
		d_h2 = l2_a.D_ReLU();
		diff2 = d_h2*diff2.removeBias();
		Matrix gradient_w2;
		gradient_w2 = l1_z.transpose().dot(diff2);
		w2 = w2 - gradient_w2.multiply(eta);
		
		Matrix diff1;
		diff1 = diff2.dot(w2.transpose());
		
		Matrix d_h1;
		d_h1 = l1_a.D_ReLU();
		diff1 = d_h1*diff1.removeBias();
		Matrix gradient_w1;
		gradient_w1 = designMtx.transpose().dot(diff1);
		w1 = w1 - gradient_w1.multiply(eta);
		
	}
	
	return 0;
} 
