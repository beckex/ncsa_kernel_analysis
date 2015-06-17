#ifndef WORKER_H
#define WORKER_H

int process(vector<float>image, int img_row, int img_col, vector<float>kernel, int ker_row, int ker_col,
						vector<float> & ret, int& ret_row, int& ret_col, int mode);


#endif