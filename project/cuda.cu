nclude <iostream>
#include <cmath>
#include <algorithm>
using namespace std;
__global__ void RowOperation(float** matrix_cu,int rank_cu, float pivot_cu)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i == rank_cu)
	{
                inverse_cu[i][j] /= pivot_cu;
                matrix_cu[i][j] /= pivot_cu;
	}
	else
	{
		inverse_cu[i][j] -= pivot_cu * inverse_cu[rank_cu][j] / inverse[rank_cu][rank_cu];
		matrix_cu[i][j] -= pivot_cu * matrix_cu[rank_cu][j] / matrix_cu[rank_cu][rank_cu];
	}
}


/** matrix inverse */
void inv(float** matrix, int row_dim, int col_dim,float** inverse)
{
	// check square matrix
	if(col_dim == row_dim)
	{
		for(int j = 0;j < col_dim; j++)
		{
			//find max magnitude
			float tmp = 0;
			int p = -1;
			for(int i = j; i < row_dim; i++)
			{
				if(abs(matrix[i][j]) > tmp) 
				{
					tmp = abs(matrix[i][j]);
					p = i;
				}
			}
			
			// have zero row
			if(p == -1)
			{
				cout << "it's singular";
				return;
			}
			
			if( j!=p )
			{
				swap(matrix[j],matrix[p]);
				swap(inverse[j],inverse[p]);
			}
			
			//row operation
			
		}
	}
	else
	{
		cout << "it isn't sqare matrix";
		return;
	}
}

/** matrix print */
void print(float** matrix, int row_dim, int col_dim)
{
	for(int i=0; i < row_dim; i++)
	{
		for(int j=0; j < row_dim; j++)
		{
			cout << matrix[i][j]<<" ";
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
	float** inverse = new float* [row_dim];
	float** result = new float* [row_dim];
    for(int i = 0; i < row_dim; i++)
    {
        inverse[i] = new float [col_dim];
        result[i] = new float [col_dim];
        for(int j = 0;j < col_dim; j++)
        {
            inverse[i][j] = float(rand()%9);
            result[i][j] = (i == j)?1.0f:0.0f;
        }
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
