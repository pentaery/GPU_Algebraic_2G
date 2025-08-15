// #include "fem/coefficient.hpp"
// #include "fem/gridfunc.hpp"
#include "matMFEM.hh"
#include "fem/datacollection.hpp"
#include "fem/gridfunc.hpp"
#include <cmath>

double coefficient_func(const mfem::Vector &x) {
  int dim = x.Size();               // 获取维度（2 或 3）
  int partition = 40, contrast = 6; // 分区数
  int count = 0;
  for (int i = 0; i < dim; ++i) {
    if (std::fmod(x(i), (double)(2.0 / partition)) <
        ((double)(1.0 / partition))) {
      count++;
    }
  }

  if (count >= dim - 1) {
    return pow(10, contrast); // 如果在中心区域，则返回 100
  } else {
    return 1.0;
  }
}

void ComputeTranspose(const mfem::SparseMatrix &A, mfem::SparseMatrix &At) {
  // 获取原始矩阵的维度
  int num_rows = A.Height();
  int num_cols = A.Width();

  // 创建转置矩阵 (列数为原始矩阵的行数，行数为原始矩阵的列数)
  At = mfem::SparseMatrix(num_cols, num_rows);

  // 获取原始矩阵的 CSR 数据
  const int *I = A.GetI();          // 行指针
  const int *J = A.GetJ();          // 列索引
  const double *Data = A.GetData(); // 非零值

  // 遍历原始矩阵的非零元素
  for (int i = 0; i < num_rows; ++i) {
    for (int j_ptr = I[i]; j_ptr < I[i + 1]; ++j_ptr) {
      int j = J[j_ptr];
      double value = Data[j_ptr];
      At.Add(j, i, value); // 交换行和列
    }
  }

  // 完成转置矩阵的构造
  At.Finalize();
}

void sortCSRRows(int m, int nnz, int *csrRowPtr, int *csrColInd,
                 double *csrVal) {
  // 遍历每一行
  for (int row = 0; row < m; ++row) {
    // 获取当前行的起始和结束位置
    int start = csrRowPtr[row];
    int end = csrRowPtr[row + 1];
    int row_nnz = end - start; // 当前行的非零元素个数

    if (row_nnz <= 1) {
      // 如果行内元素少于 2 个，无需排序
      continue;
    }

    // 将列索引和值绑定为 pair 进行排序
    std::vector<std::pair<int, double>> row_pairs(row_nnz);
    for (int i = start; i < end; ++i) {
      row_pairs[i - start] = std::make_pair(csrColInd[i], csrVal[i]);
    }

    // 按列索引升序排序
    std::sort(row_pairs.begin(), row_pairs.end(),
              [](const std::pair<int, float> &a,
                 const std::pair<int, float> &b) { return a.first < b.first; });

    // 将排序后的结果写回原始数组
    for (int i = start; i < end; ++i) {
      csrColInd[i] = row_pairs[i - start].first;
      csrVal[i] = row_pairs[i - start].second;
    }
  }
}

int generateMatMFEM(int *nrows, int *nnz, std::vector<int> &row_ptr,
                    std::vector<int> &col_index, std::vector<double> &values,
                    int meshsize) {

  // 1. Parse command-line options.
  // const char *mesh_file = "../../data/structured3d.mesh";
  int order = 0;
  const char *device_config = "cpu";
  // 2. Enable hardware devices such as GPUs, and programming models such as
  //    CUDA, OCCA, RAJA and OpenMP based on command line options.

  mfem::Device device(device_config);
  device.Print();

  // 3. Read the mesh from the given mesh file. We can handle triangular,
  //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
  //    the same code.
  // mfem::Mesh *mesh = new mfem::Mesh(mesh_file, 1, 0);
  int dim = 3;
  double sx = 1.0, sy = 1.0, sz = 1.0;
  mfem::Mesh *mesh = new mfem::Mesh(
      mfem::Mesh::MakeCartesian3D(meshsize, meshsize, meshsize,
                                  mfem::Element::HEXAHEDRON, sx, sy, sz, true));
  std::ofstream mesh_ofs("mesh.vtk");
  if (mesh_ofs) {
    mesh->PrintVTK(mesh_ofs);
  } else {
    std::cerr << "Failed to open mesh.vtk for writing." << std::endl;
  }

  // int dim = mesh->Dimension();

  // 5. Define a finite element space on the mesh. Here we use the
  //    Raviart-Thomas finite elements of the specified order.
  mfem::FiniteElementCollection *hdiv_coll(
      new mfem::RT_FECollection(order, dim));
  mfem::FiniteElementCollection *l2_coll(new mfem::L2_FECollection(order, dim));

  mfem::FiniteElementSpace *R_space =
      new mfem::FiniteElementSpace(mesh, hdiv_coll);
  mfem::FiniteElementSpace *W_space =
      new mfem::FiniteElementSpace(mesh, l2_coll);

  mfem::Array<int> boundary_dofs;
  R_space->GetBoundaryTrueDofs(boundary_dofs);

  std::cout << "***********************************************************\n";
  std::cout << "dim(R) = " << R_space->GetVSize() << "\n";
  std::cout << "dim(W) = " << W_space->GetVSize() << "\n";
  std::cout << "***********************************************************\n";

  // 7. Define the coefficients of the PDE.
  // mfem::FunctionCoefficient k_coeff(coefficient_func);
  mfem::ConstantCoefficient k_coeff(1.0);
  // PermeabilityCoefficient k_coeff; // 使用自定义的渗透率系数

  mfem::GridFunction k_gridfunc(W_space);
  k_gridfunc.ProjectCoefficient(k_coeff);
  mfem::VisItDataCollection data_coll("Permeability", mesh);
  data_coll.RegisterField("Permeability", &k_gridfunc);
  data_coll.Save();

  // 9. Assemble the finite element matrices for the Darcy operator
  //
  //                            D = [ M  B^T ]
  //                                [ B   0  ]
  //     where:
  //
  //     M = \int_\Omega k u_h \cdot v_h d\Omega   u_h, v_h \in R_h
  //     B   = -\int_\Omega \div u_h q_h d\Omega   u_h \in R_h, q_h \in W_h
  mfem::BilinearForm *mVarf(new mfem::BilinearForm(R_space));
  mfem::MixedBilinearForm *bVarf(new mfem::MixedBilinearForm(R_space, W_space));

  mVarf->AddDomainIntegrator(new DiagonalMassIntegrator(k_coeff));
  mVarf->Assemble();
  mVarf->Finalize();

  // bVarf->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
  bVarf->AddDomainIntegrator(
      new CustomRT0P0Integrator(mesh)); // 自定义的 RT0-P0 积分器
  bVarf->Assemble();
  bVarf->Finalize();

  mfem::SparseMatrix &M(mVarf->SpMat());
  mfem::Vector diag;
  M.GetDiag(diag);

  mfem::SparseMatrix &B(bVarf->SpMat());
  for (int i = 0; i < R_space->GetVSize(); i++) {
    M(i, i) = 1.0 / M(i, i); // 将 M 的对角线元素取倒数
  }
  for (int i = 0; i < boundary_dofs.Size(); i++) {
    M(boundary_dofs[i], boundary_dofs[i]) = 0.0;
  }

  mfem::SparseMatrix BT;
  ComputeTranspose(B, BT);
  mfem::SparseMatrix *C = Mult(B, M);
  mfem::SparseMatrix *A = Mult(*C, BT);
  // A->Print(std::cout);

  const int *i = A->GetI();            // row pointers
  const int *j = A->GetJ();            // column indices
  const double *a_data = A->GetData(); // values
  *nnz = A->NumNonZeroElems();         // number of non-zero elements
  *nrows = A->Height();
  row_ptr.resize(*nrows + 1);
  col_index.resize(*nnz);
  values.resize(*nnz);
  std::copy(i, i + *nrows + 1, row_ptr.begin());
  std::copy(j, j + *nnz, col_index.begin());
  std::copy(a_data, a_data + *nnz, values.begin());

  sortCSRRows(*nrows, *nnz, row_ptr.data(), col_index.data(), values.data());

  // 17. Free the used memory.
  delete mVarf;
  delete bVarf;
  delete W_space;
  delete R_space;
  delete l2_coll;
  delete hdiv_coll;
  delete mesh;
  delete C;
  delete A;
  // delete mesh;

  return 0;
}
