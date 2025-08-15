#include <fstream>
#include <iostream>
#include <mfem.hpp>
#include <sstream>
#include <string>
#include <vector>



void sortCSRRows(int m, int nnz, int *csrRowPtr, int *csrColInd,
                 double *csrVal);

int readMat(int *nrows, int *nnz, std::vector<int> &row_ptr,
            std::vector<int> &col_index, std::vector<double> &values);

int generateMat(int *nrows, int *nnz, std::vector<int> &row_ptr,
                std::vector<int> &col_index, std::vector<double> &values);
int generateMatMFEM(int *nrows, int *nnz, std::vector<int> &row_ptr,
                    std::vector<int> &col_index, std::vector<double> &values,
                    int meshsize);

void matDecompose2LM(int *nrows, int *nnz, std::vector<int> &row_ptr,
                     std::vector<int> &col_index, std::vector<double> &values);
