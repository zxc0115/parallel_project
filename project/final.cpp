#include <iostream>
#include <cmath>
#include <algorithm>
using namespace std;

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
			for (int i = 0; i < row_dim; i++)
			{
				if (i == j) 
				{
					float pivot = matrix[i][j];
					for (int k =0;k < col_dim; k++)
                       			{
						inverse[i][k] /= pivot;
						matrix[i][k] /= pivot;
					}
					matrix[i][j]=1; 
				}
				else 
				{
					float pivot = matrix[i][j]/matrix[j][j];
					for (int k = 0;k < col_dim; k++)
					{
						matrix[i][k] -= (pivot * matrix[j][k]);
						inverse[i][k] -= (pivot * inverse[j][k]);
					}
					matrix[i][j]=0;
				}
			}
		}
	}
	else
	{
		cout << "it isn't sqare matrix";
		return;
	}
}
void print(float** matrix, int row_dim, int col_dim)
{
	for(int i=0; i < row_dim; i++)
	{
		for(int j=0; j < row_dim; j++)
			cout << matrix[i][j];
		cout<<endl;
	}
}

int main ()
{
	//random seed
	srand(0);
	
	//set dimention
	int row_dim = 2;
	int col_dim = 2;
	
	//initial array
	float** inverse = new float* [row_dim];
	float** result = new float* [row_dim];
    for(int i = 0; i < row_dim; i++)
    {
        inverse[i] = new float [col_dim];
        result[i] = new float [col_dim];
        for(int j = 0;j < col_dim; j++)
        {
            inverse[i][j] = float(rand()%5);
            result[i][j] = (i == j)?1.0f:0.0f;
        }
    }
    print(inverse, row_dim, col_dim);
    cout << "----------------------\n";
    //test inverse
    inv(inverse, row_dim, col_dim, result);
    print(result, row_dim, col_dim);
    //check result 
	return 0;
}
