import math
from torch import nn
from torch.autograd import Function
import torch

from .fourier3d_cpp import forward_fourier3d_cpp,backward_fourier3d_cpp

torch.manual_seed(42)

class MeshFTCUDA(Function):
    
    @staticmethod
    def forward(ctx, Verts, Faces, Faces_coeff, xi0,xi1,xi2, Filter, narrowband_thresh):

        assert Verts.dtype in (torch.float, torch.double) , "Mesh FT not implemented for " + str(Verts.dtype)

        Gridxi0, Gridxi1, Gridxi2 = torch.meshgrid(xi0,xi1,xi2,indexing = 'ij')
        bool_filter = (Filter>narrowband_thresh).contiguous()
        xi_list = torch.vstack((Gridxi0[bool_filter],Gridxi1[bool_filter],Gridxi2[bool_filter])).transpose(0,1).contiguous()
        ftmesh_list = forward_fourier3d_cpp(Verts, Faces, Faces_coeff, xi_list).contiguous()
        ctx.save_for_backward(Verts, Faces,Faces_coeff, xi_list,ftmesh_list, bool_filter)
        ftmesh = torch.complex(torch.zeros_like(Filter),torch.zeros_like(Filter))
        ftmesh[bool_filter]=ftmesh_list
        return ftmesh

    @staticmethod
    def backward(ctx, grad_output):
        Verts, Faces,Faces_coeff, xi_list,ftmesh_list, bool_filter=ctx.saved_tensors
        grad_output_list = grad_output[bool_filter].contiguous()
        gradVerts = backward_fourier3d_cpp(Verts, Faces, Faces_coeff, grad_output_list, ftmesh_list, xi_list)
        return gradVerts, None, None, None, None, None, None, None
