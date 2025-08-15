
#include "matMFEM.hh"
#include "metis.h"
#include <cmath>
#include <map>

int main() {

  int nrows, nnz;
  std::vector<double> values(1);
  std::vector<int> col_indices(1);
  std::vector<int> row_ptr(1);
  int meshsize = 50;
  generateMatMFEM(&nrows, &nnz, row_ptr, col_indices, values, meshsize);
  std::cout << "Matrix generated..." << std::endl;

  int ncon = 1;
  int objval;
  int nprocs = 100;
  int options[METIS_NOPTIONS];
  std::vector<int> part(nrows);
  options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
  options[METIS_OPTION_NCUTS] = 1;
  METIS_PartGraphKway(&nrows, &ncon, row_ptr.data(), col_indices.data(), NULL,
                      NULL, NULL, &nprocs, NULL, NULL, NULL, &objval,
                      part.data());
  std::cout << "METIS partitioning done with objective value " << objval
            << std::endl;

  // 存放每个分区的索引编号
  std::vector<std::vector<int>> partition_nodes(nprocs);
  for (int i = 0; i < nrows; i++) {
    partition_nodes[part[i]].push_back(i);
  }

  // 提取每个分区的子矩阵
  std::vector<std::vector<double>> sub_values(nprocs);
  std::vector<std::vector<int>> sub_col_indices(nprocs);
  std::vector<std::vector<int>> sub_row_ptr(nprocs);

  for (int p = 0; p < nprocs; p++) {
    int sub_nrows = partition_nodes[p].size();
    if (sub_nrows == 0)
      continue;

    // 创建全局到局部的索引映射
    std::map<int, int> global_to_local;
    for (int i = 0; i < sub_nrows; i++) {
      global_to_local[partition_nodes[p][i]] = i;
    }

    sub_row_ptr[p].resize(sub_nrows + 1);
    sub_row_ptr[p][0] = 0;

    int local_nnz = 0;

    // 第一遍：计算每行的非零元个数
    for (int i = 0; i < sub_nrows; i++) {
      int global_row = partition_nodes[p][i];
      int start = row_ptr[global_row];
      int end = row_ptr[global_row + 1];

      int local_row_nnz = 0;
      for (int j = start; j < end; j++) {
        int global_col = col_indices[j];
        // 只保留在同一分区内的连接
        if (global_to_local.find(global_col) != global_to_local.end()) {
          local_row_nnz++;
        }
      }
      sub_row_ptr[p][i + 1] = sub_row_ptr[p][i] + local_row_nnz;
      local_nnz += local_row_nnz;
    }

    // 分配空间
    sub_col_indices[p].resize(local_nnz);
    sub_values[p].resize(local_nnz);

    // 第二遍：填充实际数据并修改对角线元素
    int current_index = 0;
    for (int i = 0; i < sub_nrows; i++) {
      int global_row = partition_nodes[p][i];
      int start = row_ptr[global_row];
      int end = row_ptr[global_row + 1];

      double row_sum = 0.0;                // 计算该行非对角线元素之和
      int diagonal_index = -1;             // 记录对角线元素的位置
      int row_start_index = current_index; // 记录该行在子矩阵中的起始位置

      // 先填充所有元素，同时计算非对角线元素之和
      for (int j = start; j < end; j++) {
        int global_col = col_indices[j];
        if (global_to_local.find(global_col) != global_to_local.end()) {
          int local_col = global_to_local[global_col];
          sub_col_indices[p][current_index] = local_col;
          sub_values[p][current_index] = values[j];

          if (local_col == i) {
            // 这是对角线元素，记录位置但暂不计入row_sum
            diagonal_index = current_index;
          } else {
            // 非对角线元素，计入row_sum
            row_sum += std::abs(values[j]); // 使用绝对值
          }
          current_index++;
        }
      }

      sub_values[p][diagonal_index] = row_sum;
    }

    std::cout << "Partition " << p << ": " << sub_nrows << " rows, "
              << local_nnz << " non-zeros" << std::endl;
  }

  // 可选：输出子矩阵信息用于验证
  // for (int p = 0; p < std::min(5, nprocs); p++) { //
  // 只显示前5个分区的详细信息
  //   if (partition_nodes[p].size() == 0)
  //     continue;

  //   std::cout << "\nPartition " << p << " matrix structure:" << std::endl;
  //   std::cout << "Rows: " << partition_nodes[p].size() << std::endl;
  //   std::cout << "Non-zeros: " << sub_values[p].size() << std::endl;

  //   // 显示前几行的结构
  //   int show_rows = std::min(3, (int)partition_nodes[p].size());
  //   for (int i = 0; i < show_rows; i++) {
  //     std::cout << "Row " << i << ": ";
  //     for (int j = sub_row_ptr[p][i]; j < sub_row_ptr[p][i + 1]; j++) {
  //       std::cout << "(" << sub_col_indices[p][j] << "," << sub_values[p][j]
  //                 << ") ";
  //     }
  //     std::cout << std::endl;
  //   }
  // }

  return 0;
}