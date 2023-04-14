import torch
import numpy as np
from torch import nn

def normalize_tensor(t,contrast_variable = 0):
    return(((t-t.min())/(t.max()-t.min())  + contrast_variable )/(1+contrast_variable))


def compute_box_size(verts,offset=0.1):
    #Compute automatically the box according to the position of the vertices. 
    #An offset of 0.1 means 10% of excess size. 
    box_size = np.zeros((3,2))
    extent = verts.max(axis=0)-verts.min(axis=0)
    box_size[:,1]=verts.max(axis=0)+extent*offset/2
    box_size[:,0]=verts.min(axis=0)-extent*offset/2
    box_size[:,0]=box_size[:,0].min(axis=0)
    box_size[:,1]=box_size[:,1].max(axis=0)
    return(box_size)

def get_internal_triangle_and_mesh_areas_with_coeff(Verts,Faces,Faces_coeff):
    
    Pos = Verts[Faces]
    Sides = Pos-Pos[:,[2,0,1]]

    Lengths_sides =torch.norm(Sides,dim=2)
    Half_perimeters = torch.sum(Lengths_sides,dim=1)/2
    Diffs = torch.zeros_like(Lengths_sides)
    Diffs[:,0] = Half_perimeters - Lengths_sides[:,0]
    Diffs[:,1] = Half_perimeters - Lengths_sides[:,1]
    Diffs[:,2] = Half_perimeters - Lengths_sides[:,2]
    trisar = Faces_coeff* (Half_perimeters*Diffs[:,0]*Diffs[:,1]*Diffs[:,2])**(0.5)
    meshar = torch.sum(trisar)
    return(trisar,meshar)

def MeshFTPy(Verts,Faces,Faces_coeff,xi0, xi1, xi2, OTF, narrowband_thresh):
    #This is the real surrogate of our torch function in python.
    
    trisar,meshar = get_internal_triangle_and_mesh_areas_with_coeff(Verts,Faces,Faces_coeff)

    tom = 2/meshar
    trisar*=tom

    v0 = Faces[:,0]
    v1 = Faces[:,1]
    v2 = Faces[:,2]

    boolean_grid = torch.abs(OTF)>narrowband_thresh

    Gridxi0, Gridxi1, Gridxi2 = torch.meshgrid(xi0,xi1,xi2,indexing = 'ij')
    Gridxi0 = Gridxi0.to(xi0.device)
    Gridxi1.to(xi0.device)
    Gridxi2.to(xi1.device)
    
    N_Gridxi0 = Gridxi0[boolean_grid]
    N_Gridxi1 = Gridxi1[boolean_grid]
    N_Gridxi2 = Gridxi2[boolean_grid]
    
    xiv =  (N_Gridxi0.reshape(-1,1)@(Verts[:,0].reshape(1,-1)) +
              N_Gridxi1.reshape(-1,1)@(Verts[:,1].reshape(1,-1)) + 
              N_Gridxi2.reshape(-1,1)@(Verts[:,2].reshape(1,-1)))

    emxiv = torch.exp(-1j* xiv)

    eps = 1.19209e-07 #For double:2.22045e-16     For float: 1.19209e-07
    Seps = np.sqrt(eps)

    xia = xiv[:,v0]
    xib = xiv[:,v1]
    xic = xiv[:,v2]
    emixia = emxiv[:,v0]
    emixib = emxiv[:,v1]
    emixic = emxiv[:,v2]
    xiab = xib-xia
    xibc = xic-xib
    xica = xia-xic

    C1 = ((torch.abs(xiab)<Seps).type(torch.long) * (torch.abs(xibc)<Seps).type(torch.long) + 
    (torch.abs(xibc)<Seps).type(torch.long) * (torch.abs(xica)<Seps).type(torch.long) +
    (torch.abs(xica)<Seps).type(torch.long) * (torch.abs(xiab)<Seps).type(torch.long)).clip(0,1)
    C2 = (torch.abs(xiab)<=Seps).type(torch.long)
    C3 = (torch.abs(xibc)<=Seps).type(torch.long)
    C4 = (torch.abs(xica)<=Seps).type(torch.long)

    R1 = C1
    R2 = C2 * (1-R1)
    R3 = C3 * (1-R2)*(1-R1)
    R4 = C4 * (1-R3)*(1-R2)*(1-R1)
    R5 = torch.ones_like(xiab).type(torch.long) * (1-R4)*(1-R3)*(1-R2)*(1-R1)

    R1=R1.type(torch.bool)
    R2=R2.type(torch.bool)
    R3=R3.type(torch.bool)
    R4=R4.type(torch.bool)
    R5=R5.type(torch.bool)


    Grid_results = torch.zeros_like(emixia)

    Grid_results[R1] = torch.exp(-1j*((xia[R1]+xib[R1]+xic[R1])/3))/2
    Grid_results[R2] = (1j*torch.exp(-1j*((xia[R2]+xib[R2])/2))+(torch.exp(-1j*((xia[R2]+xib[R2])/2))-emixic[R2])/((xia[R2]+xib[R2])/2-xic[R2]))/((xia[R2]+xib[R2])/2-xic[R2])
    Grid_results[R3] = (1j*torch.exp(-1j*((xib[R3]+xic[R3])/2))+(torch.exp(-1j*((xib[R3]+xic[R3])/2))-emixia[R3])/((xib[R3]+xic[R3])/2-xia[R3]))/((xib[R3]+xic[R3])/2-xia[R3])
    Grid_results[R4] = (1j*torch.exp(-1j*((xic[R4]+xia[R4])/2))+(torch.exp(-1j*((xic[R4]+xia[R4])/2))-emixib[R4])/((xic[R4]+xia[R4])/2-xib[R4]))/((xic[R4]+xia[R4])/2-xib[R4])
    Grid_results[R5] = emixia[R5]/(xiab[R5]*xica[R5])+emixib[R5]/(xibc[R5]*xiab[R5])+emixic[R5]/(xica[R5]*xibc[R5])


    ftmesh = torch.complex(torch.zeros_like(Gridxi0),torch.zeros_like(Gridxi0))
    trisar_complex = torch.complex(trisar,torch.zeros_like(trisar))
    ftmesh[torch.abs(OTF)>narrowband_thresh] = Grid_results@(trisar_complex)
    return(ftmesh)


class Fourier3dMesh(nn.Module):
    
    """
    Module for the meshFT layer. Takes in a triangle mesh and returns a fourier transform.
    """
    def __init__(self, box_size,box_shape,device = 'cpu', dtype = torch.float,OTF = None, narrowband_thresh = 0.01):
        """
        box_size: [[x_min,xmax],[y_min,y_max],[z_min,z_max]] Size of the box (in the spatial dimensions of the mesh)
        box_shape: [x_res,y_res,z_res] Size of the fourier box (in voxels)
        """

        super().__init__()
        self.box_size = torch.tensor(box_size,device = device, dtype = dtype)
        self.box_shape = list(box_shape)
        self.dtype = dtype
        self.device = device
        self.xi0,self.xi1,self.xi2 = self._compute_spatial_frequency_grid()
        
        if OTF == None: 
            self.OTF = torch.ones(self.box_shape, device = device, dtype = dtype)
        else: 
            self.OTF = OTF
        self.narrowband_thresh = narrowband_thresh
        
    def forward(self, Verts,Faces, Faces_coeff):
        """
        Verts: vertex tensor. float tensor of shape (n_vertex, 3)
        Faces: faces tensor. int tensor of shape (n_faces, 3)
                  if j cols, triangulate/tetrahedronize interior first.
        return meshFT: complex fourier transform of the mesh of shape self.box_shape
        """
        if self.device=='cpu': 
            ftmesh = MeshFTPy(Verts-self.box_size[:,0],Faces, Faces_coeff, self.xi0, self.xi1, self.xi2,self.OTF,self.narrowband_thresh)
        else : 
            from .cuda_class import MeshFTCUDA
            ftmesh = MeshFTCUDA.apply(Verts-self.box_size[:,0],Faces, Faces_coeff, self.xi0, self.xi1, self.xi2,self.OTF,self.narrowband_thresh)
            
        return ftmesh
    
    def _compute_spatial_frequency_grid(self): 
        n0,n1,n2 = self.box_shape
        nn0,nn1,nn2 = n0-1,n1-1,n2-1

        xi0 = torch.zeros(n0,dtype = self.dtype,device = self.device)
        xi1 = torch.zeros(n1,dtype = self.dtype,device = self.device)
        xi2 = torch.zeros(n2,dtype = self.dtype,device = self.device)

        s0,s1,s2 = np.pi/(self.box_size[0,1]-self.box_size[0,0]),np.pi/(self.box_size[1,1]-self.box_size[1,0]),np.pi/(self.box_size[2,1]-self.box_size[2,0])

        K0 = torch.arange(n0,dtype = self.dtype,device = self.device)
        Kn0 = 2*K0-nn0
        xi0 = Kn0*s0
        K1 = torch.arange(n1,dtype = self.dtype,device = self.device)
        Kn1 = 2*K1-nn1
        xi1 = Kn1*s1
        K2 = torch.arange(n2,dtype = self.dtype,device = self.device)
        Kn2 = 2*K2-nn2
        xi2 = Kn2*s2
        return(xi0,xi1,xi2)
    













def render_image_from_ftmesh(ftmesh,OTF,box_shape):
    ftexp = ftmesh*OTF
    image_ft = torch.abs(torch.fft.ifftn(ftexp, list(box_shape)))
    return(image_ft)











def get_internal_triangle_and_mesh_areas(Verts,Faces):
    nv = len(Verts)
    nt = len(Faces)
    
    Pos = Verts[Faces]
    Sides = Pos-Pos[:,[2,0,1]]

    Lengths_sides =torch.norm(Sides,dim=2)
    Half_perimeters = torch.sum(Lengths_sides,dim=1)/2
    Diffs = torch.zeros(Lengths_sides.shape)
    Diffs[:,0] = Half_perimeters - Lengths_sides[:,0]
    Diffs[:,1] = Half_perimeters - Lengths_sides[:,1]
    Diffs[:,2] = Half_perimeters - Lengths_sides[:,2]
    trisar = (Half_perimeters*Diffs[:,0]*Diffs[:,1]*Diffs[:,2])**(0.5)
    meshar = torch.sum(trisar)
    return(trisar,meshar)

def get_internal_triangle_and_mesh_areas_and_normals(Verts,Faces):
    nv = len(Verts)
    nt = len(Faces)
    
    Pos = Verts[Faces]
    Sides = Pos-Pos[:,[2,0,1]]

    Lengths_sides =torch.norm(Sides,dim=2)
    Half_perimeters = torch.sum(Lengths_sides,dim=1)/2
    Diffs = torch.zeros_like(Lengths_sides)
    Diffs[:,0] = Half_perimeters - Lengths_sides[:,0]
    Diffs[:,1] = Half_perimeters - Lengths_sides[:,1]
    Diffs[:,2] = Half_perimeters - Lengths_sides[:,2]
    trisar = (Half_perimeters*Diffs[:,0]*Diffs[:,1]*Diffs[:,2])**(0.5)
    meshar = torch.sum(trisar)
    nor = torch.cross(Pos[:,0],Pos[:,1],dim=1) + torch.cross(Pos[:,1],Pos[:,2],dim=1) + torch.cross(Pos[:,2],Pos[:,0],dim=1)
    return(trisar,meshar,nor)

def compute_volume_manifold(Verts,Faces):
    Coords = Verts[Faces]
    cross_prods = torch.cross(Coords[:,1],Coords[:,2],dim=1)
    dots = torch.sum(cross_prods*Coords[:,0],dim=1)
    Vol = -torch.sum(dots)/6
    return(Vol)

def fourier_from_mesh_narrowband_autofluorescence(OTF, narrowband_thresh,regint, xi0, xi1, xi2, Verts, Faces):
    trisar,meshar,nor = get_internal_triangle_and_mesh_areas_and_normals(Verts,Faces)

    regvol = compute_volume_manifold(Verts,Faces)

    nor2=nor*regint/regvol
    

    tom = 2/meshar
    trisar*=tom

    v0 = Faces[:,0]
    v1 = Faces[:,1]
    v2 = Faces[:,2]

    boolean_grid = OTF>narrowband_thresh

    Gridxi0, Gridxi1, Gridxi2 = torch.meshgrid(xi0,xi1,xi2,indexing = 'ij')
    Gridxi0 = Gridxi0.to(xi0.device)
    Gridxi1.to(xi0.device)
    Gridxi2.to(xi1.device)
    

    N_Gridxi0 = Gridxi0[boolean_grid]
    N_Gridxi1 = Gridxi1[boolean_grid]
    N_Gridxi2 = Gridxi2[boolean_grid]
    

    dpGrid = N_Gridxi0**2 + N_Gridxi1**2 + N_Gridxi2**2
    xiDiv0 = N_Gridxi0/dpGrid
    xiDiv1 = N_Gridxi1/dpGrid
    xiDiv2 = N_Gridxi2/dpGrid

    xiv =  (N_Gridxi0.reshape(-1,1)@(Verts[:,0].reshape(1,-1)) +
              N_Gridxi1.reshape(-1,1)@(Verts[:,1].reshape(1,-1)) + 
              N_Gridxi2.reshape(-1,1)@(Verts[:,2].reshape(1,-1)))

    emxiv = torch.exp(-1j* xiv)

    eps = 1.19209e-07 #For double:2.22045e-16     For float: 1.19209e-07
    Seps = np.sqrt(eps)

    xia = xiv[:,v0]
    xib = xiv[:,v1]
    xic = xiv[:,v2]
    emixia = emxiv[:,v0]
    emixib = emxiv[:,v1]
    emixic = emxiv[:,v2]
    xiab = xib-xia
    xibc = xic-xib
    xica = xia-xic

    C1 = ((torch.abs(xiab)<Seps).type(torch.long) * (torch.abs(xibc)<Seps).type(torch.long) + 
    (torch.abs(xibc)<Seps).type(torch.long) * (torch.abs(xica)<Seps).type(torch.long) +
    (torch.abs(xica)<Seps).type(torch.long) * (torch.abs(xiab)<Seps).type(torch.long)).clip(0,1)
    C2 = (torch.abs(xiab)<=Seps).type(torch.long)
    C3 = (torch.abs(xibc)<=Seps).type(torch.long)
    C4 = (torch.abs(xica)<=Seps).type(torch.long)

    R1 = C1
    R2 = C2 * (1-R1)
    R3 = C3 * (1-R2)*(1-R1)
    R4 = C4 * (1-R3)*(1-R2)*(1-R1)
    R5 = torch.ones_like(xiab).type(torch.long) * (1-R4)*(1-R3)*(1-R2)*(1-R1)

    R1=R1.type(torch.bool)
    R2=R2.type(torch.bool)
    R3=R3.type(torch.bool)
    R4=R4.type(torch.bool)
    R5=R5.type(torch.bool)

    dpGrid = N_Gridxi0**2 + N_Gridxi1**2 + N_Gridxi2**2
    xiDiv0 = N_Gridxi0/dpGrid
    xiDiv1 = N_Gridxi1/dpGrid
    xiDiv2 = N_Gridxi2/dpGrid

    Imagpart =(xiDiv0.reshape(-1,1)@(nor2[:,0].reshape(1,-1)) + 
     xiDiv1.reshape(-1,1)@(nor2[:,1].reshape(1,-1)) + 
     xiDiv2.reshape(-1,1)@(nor2[:,2].reshape(1,-1))) 
    trisar_matrix = trisar.expand_as(Imagpart)
    Phase_matrix = trisar_matrix + 1j*Imagpart

    Grid_results = torch.zeros_like(emixia)

    Grid_results[R1] = torch.exp(-1j*((xia[R1]+xib[R1]+xic[R1])/3))/2
    Grid_results[R2] = (1j*torch.exp(-1j*((xia[R2]+xib[R2])/2))+(torch.exp(-1j*((xia[R2]+xib[R2])/2))-emixic[R2])/((xia[R2]+xib[R2])/2-xic[R2]))/((xia[R2]+xib[R2])/2-xic[R2])
    Grid_results[R3] = (1j*torch.exp(-1j*((xib[R3]+xic[R3])/2))+(torch.exp(-1j*((xib[R3]+xic[R3])/2))-emixia[R3])/((xib[R3]+xic[R3])/2-xia[R3]))/((xib[R3]+xic[R3])/2-xia[R3])
    Grid_results[R4] = (1j*torch.exp(-1j*((xic[R4]+xia[R4])/2))+(torch.exp(-1j*((xic[R4]+xia[R4])/2))-emixib[R4])/((xic[R4]+xia[R4])/2-xib[R4]))/((xic[R4]+xia[R4])/2-xib[R4])
    Grid_results[R5] = emixia[R5]/(xiab[R5]*xica[R5])+emixib[R5]/(xibc[R5]*xiab[R5])+emixic[R5]/(xica[R5]*xibc[R5])
    
    ftmesh = torch.complex(torch.zeros_like(Gridxi0),torch.zeros_like(Gridxi0))
    ftmesh[OTF>narrowband_thresh] = torch.sum(Grid_results*Phase_matrix,axis=1)
    return(ftmesh)
def compute_spatial_frequency_grid(box_shape,box_size,device = 'cpu',dtype = torch.float): 
    n0,n1,n2 = box_shape
    nn0,nn1,nn2 = n0-1,n1-1,n2-1

    xi0 = torch.zeros(n0,dtype = dtype,device = device)
    xi1 = torch.zeros(n1,dtype = dtype,device = device)
    xi2 = torch.zeros(n2,dtype = dtype,device = device)
    
    s0,s1,s2 = np.pi/box_size[0],np.pi/box_size[1],np.pi/box_size[2]
    
    K0 = torch.arange(n0,dtype = dtype,device = device)
    Kn0 = 2*K0-nn0
    xi0 = Kn0*s0
    K1 = torch.arange(n1,dtype = dtype,device = device)
    Kn1 = 2*K1-nn1
    xi1 = Kn1*s1
    K2 = torch.arange(n2,dtype = dtype,device = device)
    Kn2 = 2*K2-nn2
    xi2 = Kn2*s2
    return(xi0,xi1,xi2)


def center_verts_in_box(verts, box_size, offset= 0.4):
    box_size = np.array(box_size)

    extent_verts = verts.max(axis=0) - verts.min(axis=0)
    extent_box = box_size[:,1]-box_size[:,0]
    verts*=min(extent_box/extent_verts)
    verts*=1-offset

    extent_verts = verts.max(axis=0) - verts.min(axis=0)

    displacement = 0.5 * (extent_box-extent_verts)
    verts+= box_size[:,1]-verts.max(axis=0)-displacement
    verts.max(axis=0),verts.min(axis=0)
    return(verts)
