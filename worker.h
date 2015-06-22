#ifndef WORKER_H
#define WORKER_H

#include <vector>
using std::vector;

int process(float* pImage, int img_row, int img_col, float* pKernel, int ker_row, int ker_col,
						float*& pRet, int& ret_row, int& ret_col, int mode);


#endif