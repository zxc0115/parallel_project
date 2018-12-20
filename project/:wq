#include <iostream>
#include <cmath>
#include <algorithm>

using namespace std;
void inv(float ** matrix, int row_dim, int col_dim,float** inverse)
{
	if(col_dim == row_dim)
	{
		for(int j = 0;j < col_dim; j++)
		{
			float tmp = 0;
			int p = 0;
			for(int i = j; i < row_dim; i++)
				if(abs(matrix[i][j]) > tmp) 
				{
					tmp = abs(matrix[i][j]);
					p = i;
				}
			// have zero row
			if(p == 0)
			{ 
				cout << "it's singular";
				return NULL;
			}
			if( j!=p )
			{
				swap(matrix[j],matrix[p])
				swap(inverse[j],inverse[p])
			}
			//row operation
			for (int i = 0; i < row_dim; i++)
			{
				if (i == j) 
				{
					for (int k = j + 1;k < col_dim; k++)
                       			{
						matrix[i][k] /= matrix[i][j];
						inverse[i][k] /= matrix[i][j];
					}
					matrix[i][j]=1; 
				}
				else 
				{
					float pivot = matrix[i][j]/matrix[j][j];
					matrix[i][j]=0;
					for (int k = j + 1;k < col_dim; k++)
					{
						matrix[i][k] -= (pivot * matrix[j][k]);
						inverse[i][k] -= (pivot * matrix[j][k]);
					}
				}
			}
		}
	}
	else
	{
		cout << "it isn't sqare matrix";
		return NULL;
	}
}
int main ()
{
	float** inverse = new float* [row_dim];
                for(int i = 0; i < row_dim; i++)
                {
                        inverse[i] = new float [col_dim];
                        for(int j = 0;j < col_dim; j++)
                                if(i == j) inverse[i][j]=1;
                                else inverse[i][j]=0;
                }
	
	return 0;
}
