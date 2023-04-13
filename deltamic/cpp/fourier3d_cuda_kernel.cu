#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


__forceinline__ torch::Tensor get_nonunit_triangle_normal(torch::Tensor mesh_vertices,torch::Tensor mesh_faces, size_t k){
  return (mesh_vertices[mesh_faces[k][1]]-mesh_vertices[mesh_faces[k][0]]).cross(
      mesh_vertices[mesh_faces[k][2]]-mesh_vertices[mesh_faces[k][0]]);
}



torch::Tensor get_internal_triangle_and_mesh_areas_and_edgenormals( torch::Tensor trisar, torch::Tensor edgenor, torch::Tensor gradmeshar, torch::Tensor mesh_vertices, torch::Tensor mesh_faces, torch::Tensor faces_coeff) 
  {
    size_t nt=mesh_faces.size(0);
    torch::Tensor trinor = torch::zeros_like(mesh_vertices[0]);

    for (size_t kt=0; kt<nt;kt++){
        trinor = get_nonunit_triangle_normal(mesh_vertices,mesh_faces, kt);
        trisar[kt] = trinor.norm();
        trinor/=trisar[kt]; //unit triangle normal
        trisar[kt]*=0.5*faces_coeff[kt];
        edgenor[kt][0]=(mesh_vertices[mesh_faces[kt][2]]-mesh_vertices[mesh_faces[kt][1]]).cross(trinor);
        edgenor[kt][1]=(mesh_vertices[mesh_faces[kt][0]]-mesh_vertices[mesh_faces[kt][2]]).cross(trinor);
        edgenor[kt][2]=(mesh_vertices[mesh_faces[kt][1]]-mesh_vertices[mesh_faces[kt][0]]).cross(trinor);
        edgenor[kt]*=faces_coeff[kt];
        gradmeshar[mesh_faces[kt][0]]-=edgenor[kt][0]/2;
        gradmeshar[mesh_faces[kt][1]]-=edgenor[kt][1]/2;
        gradmeshar[mesh_faces[kt][2]]-=edgenor[kt][2]/2;
    }
    torch::Tensor meshar = trisar.sum();
    return meshar;
  }

__forceinline__ torch::Tensor get_area_triangle(torch::Tensor mesh_vertices,torch::Tensor mesh_faces, size_t k){
    return 0.5*(get_nonunit_triangle_normal(mesh_vertices,mesh_faces,k)).norm();
}

torch::Tensor get_internal_triangle_and_mesh_areas(torch::Tensor trisar, torch::Tensor mesh_vertices, torch::Tensor mesh_faces, torch::Tensor faces_coeff) 
  {
    size_t nt=mesh_faces.size(0);
    for (size_t k=0;k<nt;k++)
      {
	trisar[k]+=get_area_triangle(mesh_vertices, mesh_faces, k)*faces_coeff[k];
      }
    torch::Tensor meshar = trisar.sum();
    return meshar;
  }

  template <typename scalar_t>
  __device__ __forceinline__ scalar_t dot_product(scalar_t a, scalar_t b, scalar_t c, scalar_t x, scalar_t y, scalar_t z) {
    return a*x + b*y + c*z;
}



template <typename scalar_t>
__device__ __forceinline__ c10::complex<scalar_t> f_expmit_f(scalar_t z) {
  c10::complex<scalar_t> i_unit(0.0,1.0);
  return exp(-i_unit*z);
}


template <typename scalar_t>
__device__ __forceinline__ scalar_t square(scalar_t z) {
  return z*z;
}


template <typename scalar_t>
__device__ __forceinline__ c10::complex<scalar_t> grad_fourier_xi_tri_a_f(scalar_t xia,scalar_t xib,scalar_t xic,c10::complex<scalar_t> emixia,c10::complex<scalar_t> emixib,c10::complex<scalar_t> emixic,scalar_t Seps)
{
  scalar_t xiab=xib-xia;
  scalar_t xibc=xic-xib;
  scalar_t xica=xia-xic;
  scalar_t xid;
  c10::complex<scalar_t> ftxi;
  c10::complex<scalar_t> i_unit(0.0,1.0);
  c10::complex<scalar_t>  emixid(0.0,1.0);
  c10::complex<scalar_t> mi_unit(0.0,-1.0);
  c10::complex<scalar_t> gftxid(0.0,1.0);
  c10::complex<scalar_t> gradftxia(0.0,1.0);


  if ((abs(xiab)<=Seps&&abs(xibc)<=Seps)||(abs(xibc)<=Seps&&abs(xica)<=Seps)||(abs(xica)<=Seps&&abs(xiab)<=Seps)) {    
    xid=(xia+xib+xic)/(scalar_t)3.0;
    ftxi=f_expmit_f(xid)/(scalar_t)2.0;
    gftxid=mi_unit*(ftxi)/(scalar_t)3.0;
    gradftxia=gftxid;
  }
  else if (abs(xiab)<=Seps) {
    xid=(xia+xib)/(scalar_t)2.0;
    emixid=f_expmit_f(xid);
    xid-=xic;
    ftxi=(i_unit*(emixid)+(emixid-emixic)/xid)/xid;
    gftxid=(emixid/(scalar_t)2.0-ftxi)/xid;
    gradftxia=gftxid;
  }
  else if (abs(xibc)<=Seps) {
    xid=(xib+xic)/(scalar_t)2.0;
    emixid=f_expmit_f(xid);
    xid-=xia;
    ftxi=(i_unit*(emixid)+(emixid-emixia)/xid)/xid;
    gftxid=(emixid/(scalar_t)2.0-ftxi)/xid;
    gradftxia=mi_unit*(ftxi)-(scalar_t)2.0*gftxid;
  }
  else if (abs(xica)<=Seps) {
    xid=(xic+xia)/(scalar_t)2.0;
    emixid=f_expmit_f(xid);
    xid-=xib;
    ftxi=(i_unit*(emixid)+(emixid-emixib)/xid)/xid;
    gftxid=(emixid/(scalar_t)2.0-ftxi)/xid;
    gradftxia=gftxid;
  }
  else {
    xiab=(scalar_t)1.0/xiab;
    xibc=(scalar_t)1.0/xibc;
    xica=(scalar_t)1.0/xica;
    gradftxia= (xiab-xica-i_unit)*((xica*xiab)*emixia)+xiab*((xiab*xibc)*emixib)-xica*((xibc*xica)*emixic);
  }
  return gradftxia;
}


template <typename scalar_t>
__device__ __forceinline__ c10::complex<scalar_t> grad_fourier_xi_tri_b_f(scalar_t xia,scalar_t xib,scalar_t xic,c10::complex<scalar_t> emixia,c10::complex<scalar_t> emixib,c10::complex<scalar_t> emixic,scalar_t Seps)
{
  c10::complex<scalar_t> ftxi;
  scalar_t xiab=xib-xia;
  scalar_t xibc=xic-xib;
  scalar_t xica=xia-xic;
  scalar_t xid;
  c10::complex<scalar_t> i_unit(0.0,1.0);
  c10::complex<scalar_t>  emixid(0.0,1.0);
  c10::complex<scalar_t> mi_unit(0.0,-1.0);
  c10::complex<scalar_t> gftxid(0.0,1.0);
  c10::complex<scalar_t> gradftxib(0.0,1.0);

  if ((abs(xiab)<=Seps&&abs(xibc)<=Seps)||(abs(xibc)<=Seps&&abs(xica)<=Seps)||(abs(xica)<=Seps&&abs(xiab)<=Seps)) {    
    xid=(xia+xib+xic)/(scalar_t)3.0;
    ftxi=f_expmit_f(xid)/(scalar_t)2.0;
    gftxid=mi_unit*(ftxi)/(scalar_t)3.0;
    gradftxib=gftxid;
  }
  else if (abs(xiab)<=Seps) {
    xid=(xia+xib)/(scalar_t)2.0;
    emixid=f_expmit_f(xid);
    xid-=xic;
    ftxi=(i_unit*(emixid)+(emixid-emixic)/xid)/xid;
    gftxid=(emixid/(scalar_t)2.0-ftxi)/xid;
    gradftxib=gftxid;
  }
  else if (abs(xibc)<=Seps) {
    xid=(xib+xic)/(scalar_t)2.0;
    emixid=f_expmit_f(xid);
    xid-=xia;
    ftxi=(i_unit*(emixid)+(emixid-emixia)/xid)/xid;
    gftxid=(emixid/(scalar_t)2.0-ftxi)/xid;
    gradftxib=gftxid;
  }
  else if (abs(xica)<=Seps) {
    xid=(xic+xia)/(scalar_t)2.0;
    emixid=f_expmit_f(xid);
    xid-=xib;
    ftxi=(i_unit*(emixid)+(emixid-emixib)/xid)/xid;
    gftxid=(emixid/(scalar_t)2.0-ftxi)/xid;
    gradftxib=mi_unit*(ftxi)-(scalar_t)2.0*gftxid;
  }
  else {
    xiab=(scalar_t)1.0/xiab;
    xibc=(scalar_t)1.0/xibc;
    xica=(scalar_t)1.0/xica;
    gradftxib= (xibc-xiab-i_unit)*((xiab*xibc)*emixib)+xibc*((xibc*xica)*emixic)-xiab*((xica*xiab)*emixia);
  }
  return gradftxib;
}


template <typename scalar_t>
__device__ __forceinline__ c10::complex<scalar_t> grad_fourier_xi_tri_c_f(scalar_t xia,scalar_t xib,scalar_t xic,c10::complex<scalar_t> emixia,c10::complex<scalar_t> emixib,c10::complex<scalar_t> emixic,scalar_t Seps)
{
  c10::complex<scalar_t> ftxi;
  scalar_t xiab=xib-xia;
  scalar_t xibc=xic-xib;
  scalar_t xica=xia-xic;
  scalar_t xid;
  c10::complex<scalar_t> i_unit(0.0,1.0);
  c10::complex<scalar_t>  emixid(0.0,1.0);
  c10::complex<scalar_t> mi_unit(0.0,-1.0);
  c10::complex<scalar_t> gftxid(0.0,1.0);
  c10::complex<scalar_t> gradftxic(0.0,1.0);

  if ((abs(xiab)<=Seps&&abs(xibc)<=Seps)||(abs(xibc)<=Seps&&abs(xica)<=Seps)||(abs(xica)<=Seps&&abs(xiab)<=Seps)) {
    xid=(xia+xib+xic)/(scalar_t)3.0;
    ftxi=f_expmit_f(xid)/(scalar_t)2.0;
    gftxid=mi_unit*(ftxi)/(scalar_t)3.0;
    gradftxic=gftxid;
  }
  else if (abs(xiab)<=Seps) {
    xid=(xia+xib)/(scalar_t)2.0;
    emixid=f_expmit_f(xid);
    xid-=xic;
    ftxi=(i_unit*(emixid)+(emixid-emixic)/xid)/xid;
    gftxid=(emixid/(scalar_t)2.0-ftxi)/xid;
    gradftxic=mi_unit*(ftxi)-(scalar_t)2.0*gftxid;
  }
  else if (abs(xibc)<=Seps) {
    xid=(xib+xic)/(scalar_t)2.0;
    emixid=f_expmit_f(xid);
    xid-=xia;
    ftxi=(i_unit*(emixid)+(emixid-emixia)/xid)/xid;
    gftxid=(emixid/(scalar_t)2.0-ftxi)/xid;
    gradftxic=gftxid;
  }
  else if (abs(xica)<=Seps) {
    xid=(xic+xia)/(scalar_t)2.0;
    emixid=f_expmit_f(xid);
    xid-=xib;
    ftxi=(i_unit*(emixid)+(emixid-emixib)/xid)/xid;
    gftxid=(emixid/(scalar_t)2.0-ftxi)/xid;
    gradftxic=gftxid;
  }
  else {
    xiab=(scalar_t)1.0/xiab;
    xibc=(scalar_t)1.0/xibc;
    xica=(scalar_t)1.0/xica;
    gradftxic= (xica-xibc-i_unit)*((xibc*xica)*emixic)+xica*((xica*xiab)*emixia)-xibc*((xiab*xibc)*emixib);
  }
  return gradftxic;
}




template <typename scalar_t>
__device__ __forceinline__ c10::complex<scalar_t> fourier_xi_tri_f(scalar_t xia,scalar_t xib,scalar_t xic,c10::complex<scalar_t> emixia,c10::complex<scalar_t> emixib,c10::complex<scalar_t> emixic,scalar_t Seps)
{
  c10::complex<scalar_t> ftxi;
  scalar_t xiab=xib-xia;
  scalar_t xibc=xic-xib;
  scalar_t xica=xia-xic;
  scalar_t xid;
  c10::complex<scalar_t> i_unit(0.0,1.0);
  c10::complex<scalar_t>  emixid(0.0,1.0);

  
  if ((abs(xiab)<=Seps&&abs(xibc)<=Seps)||(abs(xibc)<=Seps&&abs(xica)<=Seps)||(abs(xica)<=Seps&&abs(xiab)<=Seps)) {
      xid=(xia+xib+xic)/(scalar_t)3.0;
      ftxi=f_expmit_f(xid)/(scalar_t)2.0;
    }
    else if (abs(xiab)<=Seps) {
      xid=(xia+xib)/(scalar_t)2.0;
      emixid=f_expmit_f(xid);
      xid-=xic;
      ftxi=(i_unit*(emixid)+(emixid-emixic)/xid)/xid;
    }
    else if (abs(xibc)<=Seps) {
      xid=(xib+xic)/(scalar_t)2.0;
      emixid=f_expmit_f(xid);
      xid-=xia;
      ftxi=(i_unit*(emixid)+(emixid-emixia)/xid)/xid;
    }
    else if (abs(xica)<=Seps) {
      xid=(xic+xia)/(scalar_t)2.0;
      emixid=f_expmit_f(xid);
      xid-=xib;
      ftxi=(i_unit*(emixid)+(emixid-emixib)/xid)/xid;
    }
    else {

      ftxi=emixia/(xiab*xica)+emixib/(xibc*xiab)+emixic/(xica*xibc);
    }

    return ftxi;
};


template <typename scalar_t>
__global__ void Mesh_NUFT_kernel(size_t nv, size_t nt, long prodn,
     torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> xi_list,
     torch::PackedTensorAccessor<long,2,torch::RestrictPtrTraits,size_t> tris,
     torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> nodes, 
     torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> trisar, 
     torch::PackedTensorAccessor<c10::complex<scalar_t>,1,torch::RestrictPtrTraits,size_t> ftmesh_list,
     scalar_t Seps){

    long index = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    for (long m = index;  m< prodn; m +=stride){
       
        scalar_t xic0 = xi_list[m][0];
        scalar_t xic1 = xi_list[m][1];
        scalar_t xic2 = xi_list[m][2];

        scalar_t xix0;
        scalar_t xix1;
        scalar_t xix2;
        
        //loop on the triangles
        for (long kt=0;kt<nt;kt++){
            xix0 = dot_product(nodes[tris[kt][0]][0],nodes[tris[kt][0]][1],nodes[tris[kt][0]][2],xic0,xic1,xic2);
            xix1 = dot_product(nodes[tris[kt][1]][0],nodes[tris[kt][1]][1],nodes[tris[kt][1]][2],xic0,xic1,xic2);
            xix2 = dot_product(nodes[tris[kt][2]][0],nodes[tris[kt][2]][1],nodes[tris[kt][2]][2],xic0,xic1,xic2);
            ftmesh_list[m]+=trisar[kt]*fourier_xi_tri_f(xix0,xix1,xix2,f_expmit_f(xix0),f_expmit_f(xix1),f_expmit_f(xix2),Seps);
        }
    }
}


template <typename scalar_t>
__global__ void Grad_Mesh_NUFT_kernel(size_t nv, size_t nt, long prodn,long l,
     torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> xi_list,
     torch::PackedTensorAccessor<long,2,torch::RestrictPtrTraits,size_t> tris,
     torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> nodes, 
     torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> trisar, 
     torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> w, 
     torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> gradmeshar, 
     torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> gradenergy, 
     torch::PackedTensorAccessor<c10::complex<scalar_t>,1,torch::RestrictPtrTraits,size_t> grad_output_list, 
     torch::PackedTensorAccessor<c10::complex<scalar_t>,1,torch::RestrictPtrTraits,size_t> ftmesh_list,
     torch::PackedTensorAccessor<long,2,torch::RestrictPtrTraits,size_t> triangles_belonging_to_vertex_map, 
     torch::PackedTensorAccessor<long,2,torch::RestrictPtrTraits,size_t> index_of_vertex_in_triangles, 
     torch::PackedTensorAccessor<long,1,torch::RestrictPtrTraits,size_t> number_of_elements_inserted, 
     scalar_t Seps)
{
    long index = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;
    
    for (long n = index;  n< l*nv; n +=stride){
      long kv = n/l;
      long r = n%l;

      for (long m = (long) ((scalar_t)r*(scalar_t)prodn/(scalar_t)l); m< (long) (((scalar_t)r+(scalar_t)1)*(scalar_t)prodn/(scalar_t)l); m++){
          scalar_t xic0 = xi_list[m][0];
          scalar_t xic1 = xi_list[m][1];
          scalar_t xic2 = xi_list[m][2];

          c10::complex<scalar_t> gradftmesh_0(0.0,0.0);
          c10::complex<scalar_t> gradftmesh_1(0.0,0.0);
          c10::complex<scalar_t> gradftmesh_2(0.0,0.0);

          scalar_t xix0;
          scalar_t xix1;
          scalar_t xix2;

          c10::complex<scalar_t> gradftt;
          c10::complex<scalar_t> ft;
          for (long kt = 0; kt< number_of_elements_inserted[kv]; kt++){
              //COMPUTE THE RELEVANT VALUES OF GRADFTMESH

              long idx_tri = triangles_belonging_to_vertex_map[kv][kt];

              xix0 = dot_product(nodes[tris[triangles_belonging_to_vertex_map[kv][kt]][0]][0],nodes[tris[triangles_belonging_to_vertex_map[kv][kt]][0]][1],nodes[tris[triangles_belonging_to_vertex_map[kv][kt]][0]][2],xic0,xic1,xic2);
              xix1 = dot_product(nodes[tris[triangles_belonging_to_vertex_map[kv][kt]][1]][0],nodes[tris[triangles_belonging_to_vertex_map[kv][kt]][1]][1],nodes[tris[triangles_belonging_to_vertex_map[kv][kt]][1]][2],xic0,xic1,xic2);
              xix2 = dot_product(nodes[tris[triangles_belonging_to_vertex_map[kv][kt]][2]][0],nodes[tris[triangles_belonging_to_vertex_map[kv][kt]][2]][1],nodes[tris[triangles_belonging_to_vertex_map[kv][kt]][2]][2],xic0,xic1,xic2);

              if (index_of_vertex_in_triangles[kv][kt]==0){
                  gradftt = grad_fourier_xi_tri_a_f(xix0,xix1,xix2,f_expmit_f(xix0),f_expmit_f(xix1),f_expmit_f(xix2),Seps);
                  ft = fourier_xi_tri_f(xix0,xix1,xix2,f_expmit_f(xix0),f_expmit_f(xix1),f_expmit_f(xix2),Seps);
                  
                  gradftmesh_0+=gradftt*trisar[idx_tri]*xic0-ft*w[idx_tri][0][0];
                  gradftmesh_1+=gradftt*trisar[idx_tri]*xic1-ft*w[idx_tri][0][1];
                  gradftmesh_2+=gradftt*trisar[idx_tri]*xic2-ft*w[idx_tri][0][2];
              }
              
              if (index_of_vertex_in_triangles[kv][kt]==1){
                  gradftt = grad_fourier_xi_tri_b_f(xix0,xix1,xix2,f_expmit_f(xix0),f_expmit_f(xix1),f_expmit_f(xix2),Seps);
                  ft = fourier_xi_tri_f(xix0,xix1,xix2,f_expmit_f(xix0),f_expmit_f(xix1),f_expmit_f(xix2),Seps);

                  gradftmesh_0+=gradftt*trisar[idx_tri]*xic0-ft*w[idx_tri][1][0];
                  gradftmesh_1+=gradftt*trisar[idx_tri]*xic1-ft*w[idx_tri][1][1];
                  gradftmesh_2+=gradftt*trisar[idx_tri]*xic2-ft*w[idx_tri][1][2];
              }
              if (index_of_vertex_in_triangles[kv][kt]==2){
                  gradftt = grad_fourier_xi_tri_c_f(xix0,xix1,xix2,f_expmit_f(xix0),f_expmit_f(xix1),f_expmit_f(xix2),Seps);
                  ft = fourier_xi_tri_f(xix0,xix1,xix2,f_expmit_f(xix0),f_expmit_f(xix1),f_expmit_f(xix2),Seps);

                  gradftmesh_0+=gradftt*trisar[idx_tri]*xic0-ft*w[idx_tri][2][0];
                  gradftmesh_1+=gradftt*trisar[idx_tri]*xic1-ft*w[idx_tri][2][1];
                  gradftmesh_2+=gradftt*trisar[idx_tri]*xic2-ft*w[idx_tri][2][2];
              }
          }

          gradftmesh_0 -= ftmesh_list[m]*(gradmeshar[kv][0]);
          gradftmesh_1 -= ftmesh_list[m]*(gradmeshar[kv][1]);
          gradftmesh_2 -= ftmesh_list[m]*(gradmeshar[kv][2]);
          
          gradenergy[kv][r][0]+=gradftmesh_0.real()*grad_output_list[m].real() + gradftmesh_0.imag()*grad_output_list[m].imag();
          gradenergy[kv][r][1]+=gradftmesh_1.real()*grad_output_list[m].real() + gradftmesh_1.imag()*grad_output_list[m].imag();
          gradenergy[kv][r][2]+=gradftmesh_2.real()*grad_output_list[m].real() + gradftmesh_2.imag()*grad_output_list[m].imag();
        }      
    }
}  

template <typename scalar_t>
__global__ void Additions_kernel(
  long nv, 
  torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> gradenergy_big, 
  torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> gradenergy, 
  long l){

  long index = blockIdx.x * blockDim.x + threadIdx.x;
  long stride = blockDim.x * gridDim.x;
  
  for (long n = index;  n<nv; n +=stride){
    for(long r = 0; r<l; r++){
      gradenergy[n][0]+=gradenergy_big[n][r][0];
      gradenergy[n][1]+=gradenergy_big[n][r][1];
      gradenergy[n][2]+=gradenergy_big[n][r][2];
    }
  }
}




torch::Tensor fourier3d_cuda_forward(torch::Tensor mesh_vertices, torch::Tensor mesh_faces, torch::Tensor faces_coeff, torch::Tensor xi_list) 
  {
    size_t nv=mesh_vertices.size(0);
    size_t nt=mesh_faces.size(0);
    
    torch::Tensor trisar = torch::zeros({mesh_faces.size(0)},torch::TensorOptions().dtype(mesh_vertices.scalar_type()).device(mesh_vertices.device()));

    torch::Tensor ftmesh_list;
    if (mesh_vertices.dtype()==torch::kFloat32){
      ftmesh_list = torch::zeros({xi_list.size(0)},torch::TensorOptions().dtype(torch::kComplexFloat).device(mesh_vertices.device()));
    }
    else if (mesh_vertices.dtype()==torch::kFloat64){
      ftmesh_list = torch::zeros({xi_list.size(0)},torch::TensorOptions().dtype(torch::kComplexDouble).device(mesh_vertices.device()));
    }
    else {
      
    }
    torch::Tensor meshar = get_internal_triangle_and_mesh_areas(trisar, mesh_vertices, mesh_faces,faces_coeff);
    trisar*=2/meshar;

    long prodn = xi_list.size(0);
    
    //RUN KERNEL
    int blockSize = 512;
    int numBlocks = (prodn + blockSize - 1) / blockSize;
    
    AT_DISPATCH_FLOATING_TYPES(xi_list.type(), "fourier3d_forward_cuda", ([&] {
      Mesh_NUFT_kernel<scalar_t><<<numBlocks,blockSize>>>(nv,nt,prodn,
      xi_list.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
      mesh_faces.packed_accessor<long,2,torch::RestrictPtrTraits,size_t>(),
      mesh_vertices.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
      trisar.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
      ftmesh_list.packed_accessor<c10::complex<scalar_t>,1,torch::RestrictPtrTraits,size_t>(),
      sqrt(std::numeric_limits<scalar_t>::epsilon()));
    }));

    

    cudaDeviceSynchronize();
    
  return ftmesh_list;
}



torch::Tensor fourier3d_cuda_backward(torch::Tensor mesh_vertices, torch::Tensor mesh_faces, torch::Tensor faces_coeff, torch::Tensor grad_output_list, torch::Tensor ftmesh_list, torch::Tensor xi_list) {
  long nv = mesh_vertices.size(0);
  long nt = mesh_faces.size(0);
  long prodn = xi_list.size(0);
  long l = 256;

  
  torch::Tensor gradenergy_big = torch::zeros({nv,l,3},torch::TensorOptions().dtype(mesh_vertices.dtype()).device(mesh_vertices.device()));
  torch::Tensor gradenergy = torch::zeros({nv,3},torch::TensorOptions().dtype(mesh_vertices.dtype()).device(mesh_vertices.device()));
  torch::Tensor trisar = torch::zeros({mesh_faces.size(0)},torch::TensorOptions().dtype(mesh_vertices.dtype()).device(mesh_vertices.device()));// list of internal triangle areas
  torch::Tensor w = torch::zeros({mesh_faces.size(0),3,3},torch::TensorOptions().dtype(mesh_vertices.dtype()).device(mesh_vertices.device()));// list of internal edge normals (per triangle)
  torch::Tensor gradmeshar = torch::zeros({mesh_vertices.size(0),3},torch::TensorOptions().dtype(mesh_vertices.dtype()).device(mesh_vertices.device()));
  
  
  
  torch::Tensor meshar =get_internal_triangle_and_mesh_areas_and_edgenormals(trisar, w,gradmeshar, mesh_vertices, mesh_faces,faces_coeff);

  trisar*=2/meshar;
  gradmeshar*=1/meshar;
  w*=1/meshar;

  torch::Tensor triangles_belonging_to_vertex_map = torch::zeros({nv,30}, torch::TensorOptions().dtype(torch::kLong).device(mesh_vertices.device()));//  Means that we are only accepting up to 30 triangles connected to one vertex.    //More than sufficient
  torch::Tensor index_of_vertex_in_triangles = torch::zeros({nv,30}, torch::TensorOptions().dtype(torch::kLong).device(mesh_vertices.device()));
  torch::Tensor number_of_elements_inserted = torch::zeros({nv}, torch::TensorOptions().dtype(torch::kLong).device(mesh_vertices.device()));

  torch::Tensor vertex;
  torch::Tensor nelm;
  for (int i =0; i<nt; i++){
      for (int j =0; j<3; j++){
          vertex = mesh_faces[i][j];
          nelm = number_of_elements_inserted[vertex];
          triangles_belonging_to_vertex_map[vertex][nelm]=i;
          index_of_vertex_in_triangles[vertex][nelm]=j;
          number_of_elements_inserted[vertex]+=1;
      }
  }
  
  //RUN Kernel :
  int blockSize = 32;
  int numBlocks = (l*nv + blockSize - 1) / blockSize;


  AT_DISPATCH_FLOATING_TYPES(xi_list.type(), "fourier3d_backward_cuda", ([&] {
    Grad_Mesh_NUFT_kernel<scalar_t><<<numBlocks,blockSize>>>(nv,nt,prodn,l,
    xi_list.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
    mesh_faces.packed_accessor<long,2,torch::RestrictPtrTraits,size_t>(),
    mesh_vertices.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
    trisar.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
    w.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
    gradmeshar.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
    gradenergy_big.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
    grad_output_list.packed_accessor<c10::complex<scalar_t>,1,torch::RestrictPtrTraits,size_t>(),
    ftmesh_list.packed_accessor<c10::complex<scalar_t>,1,torch::RestrictPtrTraits,size_t>(),
    triangles_belonging_to_vertex_map.packed_accessor<long,2,torch::RestrictPtrTraits,size_t>(),
    index_of_vertex_in_triangles.packed_accessor<long,2,torch::RestrictPtrTraits,size_t>(),
    number_of_elements_inserted.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
    sqrt(std::numeric_limits<scalar_t>::epsilon()));
  }));
  cudaDeviceSynchronize();


  //Mix different results with each other
  blockSize = 256;
  numBlocks = (3*nv + blockSize - 1) / blockSize;
  AT_DISPATCH_FLOATING_TYPES(xi_list.type(), "fourier3d_backward_cuda", ([&] {
    Additions_kernel<scalar_t><<<numBlocks,blockSize>>>(nv,
    gradenergy_big.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
    gradenergy.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
    l);
  }));
  cudaDeviceSynchronize();
  
  return gradenergy ;
}




