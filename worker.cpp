#include "worker.h"
#include <armadillo>
#include <vector>
#include <stdlib>
#include <stdio>

#define ERR(e) {fprintf(stderr, "Error: %s\n", e); return-1;}

using std::vector;

int process(vector<float>image, int img_row, int img_col, vector<float>kernel, int ker_row, int ker_col,
						vector<float> & ret, int& ret_row, int& ret_col, int mode)){
	if(ker_col==0||ker_row)
		ERR("invalid kernel size 0!");
	if(img_col==0||img_row==0)
		ERR("invalid image size 0!");
	if(img_col<ker_col||img_row<ker_row)
		ERR("invalid image size, smaller than kernel!");
	if(1!=ker_col%2||1!=ker_row%2)
		ERR("kernel height or width is not odd!");

	fmat m_img(image);
	m_img.reshape(img_col, img_row);
	m_img=m_img.t();
	fmat m_ker(kernel);
	m_ker.reshape(ker_col, ker_row);
	m_ker=m_ker.t();

	if(mode==1){
		ret_col=img_col;
		ret_row=img_row-ker_row/2;
		fmat m_ret(ret_row, ret_col);
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
				m_ret.submat(ret_x, ret_y, ret_x+w-1, ret_y+h-1)+=m_img(img_x, img_y, img_x+w-1, img_y+h-1)*m_ker(i,j);
			}
		}
		m_ret=m_ret.t();
		m_ret.reshape(1,ret_row*ret_col);
		ret=conv_to<vector<float> >::from(m_ret);
	}else if(mode==2){		
		ret_col=img_col;
		ret_row=img_row-ker_row/2*2;
		fmat m_ret(ret_row, ret_col);
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
					h=y;
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
				m_ret.submat(ret_x, ret_y, ret_x+w-1, ret_y+h-1)+=m_img(img_x, img_y, img_x+w-1, img_y+h-1)*m_ker(i,j);
			}
		}
		m_ret=m_ret.t();
		m_ret.reshape(1,ret_row*ret_col);
		ret=conv_to<vector<float> >::from(m_ret);

	}else if(mode==3){
		ret_col=img_col;
		ret_row=img_row-ker_row/2;
		fmat m_ret(ret_row, ret_col);
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
					h=y;
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
				m_ret.submat(ret_x, ret_y, ret_x+w-1, ret_y+h-1)+=m_img(img_x, img_y, img_x+w-1, img_y+h-1)*m_ker(i,j);
			}
		}
		m_ret=m_ret.t();
		m_ret.reshape(1,ret_row*ret_col);
		ret=conv_to<vector<float> >::from(m_ret);

	}else{
		ERR("invalid mode, should be 1(uppest), 2(middle), or 3(downnest).");
	}
	return 0;
}

int main(){
	vector<float> image={1,2,3,4,5,6,7,8,9};
	vector<float> kernel={-1,-1,-1,-1,9,-1,-1,-1,-1};
	vector<float> ret;
	int ret_col, ret_row;
	process(image, 3, 3, kernel, 3, 3, ret, ret_row, ret_col, 1);
	cout<<"Size is: "<<ret.size()<<std::endl;
	for(int i=0;i<ret.size();i++){
		cout<<ret[i]<<" ";
	}
	cout<<std::endl;
}