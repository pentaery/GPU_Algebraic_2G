#include <fstream>
#include <iostream>
#include <mfem.hpp>
#include <sstream>
#include <string>
#include <vector>


class DiagonalMassIntegrator : public mfem::BilinearFormIntegrator {
private:
  mfem::Coefficient &k; // 空间依赖的系数（可能是一个 GridFunction）
                        // 存储每个元素的 k 值（分片常数）

public:
  DiagonalMassIntegrator(mfem::Coefficient &k_) : k(k_) {}

  virtual void AssembleElementMatrix(const mfem::FiniteElement &el,
                                     mfem::ElementTransformation &Trans,
                                     mfem::DenseMatrix &elmat) override {
    int dof = el.GetDof(); // RT0: dof = 边数（2D 三角形为 3）
    int dim = el.GetDim();
    if (dof <= 0 || dim <= 0) {
      throw std::runtime_error("Invalid element dimensions or DOFs");
    }

    elmat.SetSize(dof, dof);
    elmat = 0.0;

    // 为每个自由度（边）计算 k 的倒数平均值
    mfem::Vector k_avg(dof); // 存储每个自由度的 k_avg
    k_avg = 0.0;
    for (int j = 0; j < dof; j++) {
      double k_current = k.Eval(Trans, mfem::IntegrationPoint());
      k_avg(j) = 1.0 / (2 * k_current); // 转换为 k_avg
    }

    // 使用默认积分规则（RT0 基函数为常数，order=0 足够）
    const mfem::IntegrationRule *ir = IntRule;
    if (!ir) {
      int order = 0; // RT0 基函数是常数
      ir = &mfem::IntRules.Get(el.GetGeomType(), order);
    }

    double area_or_volume = 0.0;

    for (int i = 0; i < ir->GetNPoints(); i++) {
      const mfem::IntegrationPoint &ip = ir->IntPoint(i);
      Trans.SetIntPoint(&ip);
      area_or_volume += ip.weight * Trans.Weight();
    }
    mfem::DenseMatrix vshape(dof, dim);
    mfem::Vector vshape_sq(dof);
    for (int j = 0; j < dof; ++j) {
      elmat(j, j) += area_or_volume * k_avg(j);
    }
  }
};

class CustomRT0P0Integrator : public mfem::BilinearFormIntegrator {
private:
  mfem::Mesh *mesh;

public:
  CustomRT0P0Integrator(mfem::Mesh *mesh_) : mesh(mesh_) {}

  virtual void AssembleElementMatrix2(const mfem::FiniteElement &trial_fe,
                                      const mfem::FiniteElement &test_fe,
                                      mfem::ElementTransformation &Trans,
                                      mfem::DenseMatrix &elmat) override {
    int trial_dof = trial_fe.GetDof(); // RT0 的自由度数
    int test_dof = test_fe.GetDof();   // P0 的自由度数 (通常为 1)
    int elem_idx = Trans.ElementNo;

    mfem::Array<int> edges, orientations;
    mesh->GetElementEdges(elem_idx, edges, orientations);

    // 初始化元素矩阵
    elmat.SetSize(test_dof, trial_dof);
    elmat = 0.0;

    // 获取网格维度
    int dim = mesh->Dimension();

    // 存储边长（2D）或面积（3D）
    mfem::Vector measure(edges.Size());

    if (dim == 2) {
      // 2D: Compute edge lengths
      mfem::Array<int> edges, orientations;
      mesh->GetElementEdges(elem_idx, edges, orientations);

      const mfem::IntegrationRule *ir = &mfem::IntRules.Get(
          mfem::Geometry::SEGMENT, 1); // 1D integration for edges

      for (int i = 0; i < edges.Size(); i++) {
        mfem::ElementTransformation *edge_trans =
            mesh->GetEdgeTransformation(edges[i]);
        double meas = 0.0;
        for (int j = 0; j < ir->GetNPoints(); j++) {
          const mfem::IntegrationPoint &ip = ir->IntPoint(j);
          edge_trans->SetIntPoint(&ip);
          meas += ip.weight * edge_trans->Weight();
        }
        measure(i) = meas;
      }
    } else if (dim == 3) {
      // 3D: Compute face areas
      mfem::Array<int> faces, orientations;
      mesh->GetElementFaces(elem_idx, faces, orientations);

      for (int i = 0; i < faces.Size(); i++) {
        // Get face geometry and transformation
        int face_idx = faces[i];
        mfem::Geometry::Type face_geom = mesh->GetFaceGeometry(face_idx);
        mfem::ElementTransformation *face_trans =
            mesh->GetFaceTransformation(face_idx);

        // Select integration rule based on face geometry
        const mfem::IntegrationRule *ir =
            &mfem::IntRules.Get(face_geom, 1); // 2D integration for faces

        double meas = 0.0;
        for (int j = 0; j < ir->GetNPoints(); j++) {
          const mfem::IntegrationPoint &ip = ir->IntPoint(j);
          face_trans->SetIntPoint(&ip);
          meas += ip.weight * face_trans->Weight();
        }
        measure(i) = meas;
      }
    }

    for (int i = 0; i < trial_dof; i++) {
      for (int j = 0; j < test_dof; j++) {
        elmat(j, i) += measure(i);
      }
    }
  }
};

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
