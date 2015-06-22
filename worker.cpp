#include "worker.h"
#define ARMA_DONT_USE_CXX11
#include <armadillo>
#include <vector>
#include <cstdlib>
#include <cstdio>

#define ERR(e) {fprintf(stderr, "Error: %s\n", e); return-1;}

using std::vector;

/*	* process an image(matrix) with a kernel(matrix), input are image in a row vector form, the row/col size of image,
	* kernel in a row vector form, the row/col size of kernel, and the mode. 
	* mode = 1 means the uppest subimage, mode = 2 means the middle subimages, mode = 3 means the subimage in bottom
	*
	* return value:
	*	-1 for fail, 0 for success
	* if success, the input vector ret would be the row vector of result matrix, ret_row/ret_col would be the size of
	* result matrix. 
*/

int process(vector<float>image, int img_row, int img_col, vector<float>kernel, int ker_row, int ker_col,
						vector<float> & ret, int& ret_row, int& ret_col, int mode){
	if(ker_col==0||ker_row==0)
		ERR("invalid kernel size 0!");
	if(img_col==0||img_row==0)
		ERR("invalid image size 0!");
	if(img_col<ker_col||img_row<ker_row)
		ERR("invalid image size, smaller than kernel!");
	if(1!=ker_col%2||1!=ker_row%2)
		ERR("kernel height or width is not odd!");
	if(image.size()!=img_row*img_col)
		ERR("image vector size is not correct!");
	if(kernel.size()!=ker_row*ker_col)
		ERR("kernel vector size is not correct!");

	arma::fmat m_img(image);
	m_img.reshape(img_col, img_row);
	m_img=m_img.t();
	arma::fmat m_ker(kernel);
	m_ker.reshape(ker_col, ker_row);
	m_ker=m_ker.t();

	if(mode==1){
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
				//printf("ret_x=%d, ret_y=%d, w=%d, h=%d, img_x=%d, img_y=%d\n", ret_x, ret_y, w, h, img_x, img_y);
				m_ret.submat(ret_y, ret_x, ret_y+h-1, ret_x+w-1)+=m_img.submat(img_y, img_x, img_y+h-1, img_x+w-1)*m_ker(i,j);
				//std::cout<<"success"<<std::endl;	
			}
		}
		m_ret=m_ret.t();
		m_ret.reshape(1,ret_row*ret_col);
		ret=arma::conv_to<vector<float> >::from(m_ret);
	}else if(mode==2){		
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
		m_ret=m_ret.t();
		m_ret.reshape(1,ret_row*ret_col);
		ret=arma::conv_to<vector<float> >::from(m_ret);

	}else if(mode==3){
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
		m_ret=m_ret.t();
		m_ret.reshape(1,ret_row*ret_col);
		ret=arma::conv_to<vector<float> >::from(m_ret); 

	}else{
		ERR("invalid mode, should be 1(uppest), 2(middle), or 3(downnest).");
	}
	return 0;
}

//the main() is for debug
/*int main(){
	vector<float> image={1.5,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
	vector<float> kernel={-1,-1,-1,-1,9,-1,-1,-1,-1};
	vector<float> ret;
	int ret_col, ret_row;
	process(image, 4, 4, kernel, 3, 3, ret, ret_row, ret_col, 1);
	std::cout<<"Size is: "<<ret_row<<", "<<ret_col<<std::endl;
	for(int i=0;i<ret.size();i++){
		std::cout<<ret[i]<<" ";
	}
	std::cout<<std::endl;
}*/