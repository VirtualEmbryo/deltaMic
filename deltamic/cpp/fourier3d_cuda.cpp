#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

torch::Tensor fourier3d_cuda_forward(
    torch::Tensor mesh_vertices,
    torch::Tensor mesh_faces, 
    torch::Tensor faces_coeff,
    torch::Tensor xi_list);

torch::Tensor fourier3d_cuda_backward(
    torch::Tensor mesh_vertices,
    torch::Tensor mesh_faces, 
    torch::Tensor faces_coeff,
    torch::Tensor grad_output_list,
    torch::Tensor ftmesh_list,
    torch::Tensor xi_list);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor fourier3d_forward(
    torch::Tensor mesh_vertices,
    torch::Tensor mesh_faces, 
    torch::Tensor faces_coeff,
    torch::Tensor xi_list) {
  CHECK_INPUT(mesh_vertices);
  CHECK_INPUT(mesh_faces);
  CHECK_INPUT(faces_coeff);
  CHECK_INPUT(xi_list);

  return fourier3d_cuda_forward(
      mesh_vertices,
      mesh_faces,
      faces_coeff,
      xi_list);
}

torch::Tensor fourier3d_backward(
    torch::Tensor mesh_vertices,
    torch::Tensor mesh_faces, 
    torch::Tensor faces_coeff,
    torch::Tensor grad_output_list, 
    torch::Tensor ftmesh_list,
    torch::Tensor xi_list) {
  CHECK_INPUT(mesh_vertices);
  CHECK_INPUT(mesh_faces);
  CHECK_INPUT(faces_coeff);
  CHECK_INPUT(grad_output_list);
  CHECK_INPUT(ftmesh_list);
  CHECK_INPUT(xi_list);

  return fourier3d_cuda_backward(
      mesh_vertices,
      mesh_faces,
      faces_coeff,
      grad_output_list,
      ftmesh_list,
      xi_list);
}

/*

torch::Tensor fourier3d_forward(
    torch::Tensor mesh_vertices,
    torch::Tensor mesh_faces, 
    torch::Tensor faces_coeff,
    torch::Tensor xi_list) {
  CHECK_INPUT(mesh_vertices);
  CHECK_INPUT(mesh_faces);
  CHECK_INPUT(faces_coeff);
  CHECK_INPUT(xi_list);

  return fourier3d_cuda_forward(
      mesh_vertices,
      mesh_faces,
      faces_coeff,
      xi_list);
}

torch::Tensor fourier3d_backward(
    torch::Tensor mesh_vertices,
    torch::Tensor mesh_faces, 
    torch::Tensor faces_coeff,
    torch::Tensor grad_output_list, 
    torch::Tensor ftmesh_list,
    torch::Tensor xi_list) {
  CHECK_INPUT(mesh_vertices);
  CHECK_INPUT(mesh_faces);
  CHECK_INPUT(faces_coeff);
  CHECK_INPUT(grad_output_list);
  CHECK_INPUT(ftmesh_list);
  CHECK_INPUT(xi_list);

  return fourier3d_cuda_backward(
      mesh_vertices,
      mesh_faces,
      faces_coeff,
      grad_output_list,
      ftmesh_list,
      xi_list);
}

*/


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_fourier3d_cpp", &fourier3d_forward, "Fourier3d forward");
  m.def("backward_fourier3d_cpp", &fourier3d_backward, "Fourier3d backward");
}
