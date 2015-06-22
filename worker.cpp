#include "worker.h"
#define ARMA_DONT_USE_CXX11
#include <armadillo>
#include <vector>
#include <cstdlib>
#include <cstdio>

#define ERR(e) {fprintf(stderr, "Error: %s\n", e); return-1;}

using std::vector;

/*	* Read a matrix of float type in armadillo library by row into float array,
	* The array is malloced in the funciton and should be freed by the user. 
	* 
	* Parameters
	*	matrix:	a armadillo float type matrix
	*	rows: 	number of rows of the matrix
	*	cols:	number of columns of the matrix
	*
	* Return value
	*	Pointer of the float array is returned. NULL is returned in error. User should
	*	free the pointer afterwards. 
	*
*/

float* matrixToArray(arma::fmat matrix, int rows, int cols){
	std::cout<<"reach here"<<std::endl;
	float* ret=(float*)malloc(sizeof(float)*rows*cols);
	for(int i=0;i<rows;i++){
		for(int j=0;j<cols;j++){
			ret[i*cols+j]=matrix(i, j);
		}
	}
	return ret;
}

/*	* Process an image(matrix) with a kernel(matrix). The parameters pRet, ret_row, ret_col
	* are modified to show the output matrix.
	* 
	* Parameters
	*	pImage: 	array of a image in row vector form.
	*	img_row:	number of rows for the input image.
	*	img_col:	number of columns for the input image.
	*	pKernel:	array of a kernel in row vector form.
	*	ker_row:	number of rows for the kernel matrix.
	*	ker_col:	number of columns for the kernel matrix.
	*	pRet: 		array of output matrix in row vector form. Must be NULL in input. Set to NULL
					if the function is in error.
	*	ret_row:	number of rows of the output matrix.
	*	ret_col: 	number of columns of the output matrix.
	*	mode: 		different ways to produce the result matrix.
	*					1, the image matrix is the uppest one. 
	*					2, the image matrix is the middle one. 
	*					3, the image matrix is the bottom one. 
	*					4, the image matrix is not seperated(used when only one job is running)
	*
	* Return value
	*	-1 for fail, 0 for success.
	* 	If success, pRet would be an allocated result array in row vector form, ret_row/ret_col
	*	would be the size of the result matrix. User should free pRet manually. 
*/
int process(float* pImage, int img_row, int img_col, float* pKernel, int ker_row, int ker_col,
						float*& pRet, int& ret_row, int& ret_col, int mode){
	//check for inputs
	if(ker_col==0||ker_row==0)
		ERR("invalid kernel size 0!");
	if(img_col==0||img_row==0)
		ERR("invalid image size 0!");
	if(img_col<ker_col||img_row<ker_row)
		ERR("invalid image size, smaller than kernel!");
	if(1!=ker_col%2||1!=ker_row%2)
		ERR("kernel height or width is not odd!");
	if(pRet!=NULL)
		ERR("input matrix pointer is not NULL!");

	//initialize armadillo matrix from input row vector array
	arma::fmat m_img(pImage, img_col, img_row);
	m_img=m_img.t();
	arma::fmat m_ker(pKernel, ker_col, ker_row);
	m_ker=m_ker.t();

	//check for different modes
	if(mode==1){		//for uppest submatrix of an image
		ret_col=img_col;
		ret_row=img_row-ker_row/2;
		arma::fmat m_ret(ret_row, ret_col);
		m_ret.fill(0);
		for(int i=0;i<ker_row;i++){
			for(int j=0;j<ker_col;j++){		//loop for every value in the kernel
				int x=j+1-(ker_col/2+1);
				int y=i+1-(ker_row/2+1);
				int img_x, img_y, ret_x, ret_y, h, w;
				if(x<=0 && y<=0){
					img_x=img_y=0;
					ret_x=-x;
					ret_y=-y;
					h=ret_row+y;
					w=ret_col+x;
				}else if(x<=0 && y >0){
					img_x=0;
					img_y=y;
					ret_x=-x;
					ret_y=0;
					h=ret_row;
					w=ret_col+x;
				}else if(x>0 && y<=0){
					img_x=x;
					img_y=0;
					ret_x=0;
					ret_y=-y;
					h=ret_row+y;
					w=ret_col-x;
				}else{
					img_x=x;
					img_y=y;
					ret_x=ret_y=0;
					h=ret_row;
					w=ret_col-x;
				}
				//for each value in kernel matrix, do a submatrix element-wise addition
				m_ret.submat(ret_y, ret_x, ret_y+h-1, ret_x+w-1)+=m_img.submat(img_y, img_x, img_y+h-1, img_x+w-1)*m_ker(i,j);
			}
		}
		//convert armadillo matrix to a float array
		pRet=matrixToArray(m_ret, ret_row, ret_col);
	}else if(mode==2){		//for middle submatrix of an image
		ret_col=img_col;
		ret_row=img_row-ker_row/2*2;
		arma::fmat m_ret(ret_row, ret_col);
		m_ret.fill(0);
		for(int i=0;i<ker_row;i++){
			for(int j=0;j<ker_col;j++){
				int x=j+1-(ker_col/2+1);
				int y=i+1-(ker_row/2+1);
				int img_x, img_y, ret_x, ret_y, h, w;
				if(x<=0 && y<=0){
					img_x=0;
					img_y=ker_row/2+y;
					ret_x=-x;
					ret_y=0;
					h=ret_row;
					w=ret_col+x;
				}else if(x<=0 && y >0){
					img_x=0;
					img_y=ker_row/2+y;
					ret_x=-x;
					ret_y=0;
					h=ret_row;
					w=ret_col+x;
				}else if(x>0 && y<=0){
					img_x=x;
					img_y=ker_row/2+y;
					ret_x=0;
					ret_y=0;
					h=ret_row;
					w=ret_col-x;
				}else{
					img_x=x;
					img_y=ker_row/2+y;
					ret_x=ret_y=0;
					h=ret_row;
					w=ret_col-x;
				}
				m_ret.submat(ret_y, ret_x, ret_y+h-1, ret_x+w-1)+=m_img.submat(img_y, img_x, img_y+h-1, img_x+w-1)*m_ker(i,j);
			}
		}
		pRet=matrixToArray(m_ret, ret_row, ret_col);
	}else if(mode==3){		//for bottom submatrix of an image
		ret_col=img_col;
		ret_row=img_row-ker_row/2;
		arma::fmat m_ret(ret_row, ret_col);
		m_ret.fill(0);
		for(int i=0;i<ker_row;i++){
			for(int j=0;j<ker_col;j++){
				int x=j+1-(ker_col/2+1);
				int y=i+1-(ker_row/2+1);
				int img_x, img_y, ret_x, ret_y, h, w;
				if(x<=0 && y<=0){
					img_x=0;
					img_y=ker_row/2+y;
					ret_x=-x;
					ret_y=0;
					h=ret_row;
					w=ret_col+x;
				}else if(x<=0 && y >0){
					img_x=0;
					img_y=ker_row/2+y;
					ret_x=-x;
					ret_y=0;
					h=ret_row-y;
					w=ret_col+x;
				}else if(x>0 && y<=0){
					img_x=x;
					img_y=ker_row/2+y;
					ret_x=0;
					ret_y=0;
					h=ret_row;
					w=ret_col-x;
				}else{
					img_x=x;
					img_y=ker_row/2+y;
					ret_x=ret_y=0;
					h=ret_row-y;
					w=ret_col-x;
				}
				m_ret.submat(ret_y, ret_x, ret_y+h-1, ret_x+w-1)+=m_img.submat(img_y, img_x, img_y+h-1, img_x+w-1)*m_ker(i,j);
			}
		}
		pRet=matrixToArray(m_ret, ret_row, ret_col);

	}else if(mode==4){		//for a full image, used when only one job is running
		ret_col=img_col;
		ret_row=img_row;
		arma::fmat m_ret(ret_row, ret_col);
		m_ret.fill(0);
		for(int i=0;i<ker_row;i++){
			for(int j=0;j<ker_col;j++){
				int x=j+1-(ker_col/2+1);
				int y=i+1-(ker_row/2+1);
				int img_x, img_y, ret_x, ret_y, h, w;

				if(x<=0){
					img_x=0;
					ret_x=-x;
					w=ret_col+x;
				}else{
					img_x=x;
					ret_x=0;
					w=ret_col-x;
				}
				if(y<=0){
					img_y=0;
					ret_y=-y;
					h=ret_row+y;
				}else{
					img_y=y;
					ret_y=0;
					h=ret_row-y;
				}
				m_ret.submat(ret_y, ret_x, ret_y+h-1, ret_x+w-1)+=m_img.submat(img_y, img_x, img_y+h-1, img_x+w-1)*m_ker(i,j);
			}
		}
		pRet=matrixToArray(m_ret, ret_row, ret_col);
	}
	else{
		ERR("invalid mode, should be 1(uppest), 2(middle), or 3(downnest).");
	}
	return 0;
}

//the main() is for debug
/*int main(){
	float image[]={1.5,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
	float kernel[]={-1,-1,-1,-1,9,-1,-1,-1,-1};
	float* ret=NULL;
	int ret_col, ret_row;
	process(image, 4, 4, kernel, 3, 3, ret, ret_row, ret_col, 1);
	std::cout<<"Size is: "<<ret_row<<", "<<ret_col<<std::endl;
	for(int i=0;i<ret_row*ret_col;i++){
		std::cout<<ret[i]<<" ";
	}
	std::cout<<std::endl;
}*/