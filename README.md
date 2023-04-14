# deltaMic
**deltaMic** is a PyTorch-based python library that provides a differentiable renderer of microscopy images. We compute Fourier transform of triangle meshes in a given box, and multiply it with a given Point-Spread-Function in the Fourier space. We support operations both on CPU and GPU. We provide **C++/CUDA** bindings to compute efficiently the forward and backward passes, to enable differentiable rasterization of triangulated data at scale. 


Our main contribution, that allows meshFT to compute transforms in tractable times, relies on a tunable narrow-band filter in the frequency space that avoid computing high frequencies of the Fourier transform. (see API). We use our pipeline to do inverse rendering on 3D microscopy images. 


### Installation

`pip install requirements.txt && python3 setup.py install`


### Example 

Load a mesh and compute its Fourier transform

```shell
pip install meshFT

```

```py
from deltamic import normalize_tensor,render_image_from_ftmesh,Fourier3dMesh,compute_box_size,generate_gaussian_psf
import trimesh
import numpy as np
import torch

# Creation of initial configurations: 
device = 'cuda:0'
box_shape = np.array([200]*3)
filename = "data/spot.obj"

Mesh_gt = trimesh.load(filename)
faces = np.array(Mesh_gt.faces)
verts = np.array(Mesh_gt.vertices)
box_size = torch.tensor(compute_box_size(verts, offset = 0.2))

Verts = torch.tensor(verts, dtype = torch.float, device = device,requires_grad = True)
Faces = torch.tensor(faces, dtype = torch.long, device = device)
Faces_coeff = torch.ones(len(Faces),dtype = torch.float, device = device)

print("Vertices shape: ",verts.shape,"Faces shape: ", faces.shape)
print("Min/Max x/y/z position of vertices: ",verts.min(axis=0),verts.max(axis=0))
print("box_size: ",box_size, "box_shape",box_shape)

narrowband_thresh = torch.tensor(1e-5,dtype = torch.float, device = device)
meshFT = Fourier3dMesh(box_size,box_shape,device=device, dtype = torch.float32)
sigma_matrix = (1e-3*torch.eye(3,device=device))
OTF = generate_gaussian_psf(sigma_matrix,meshFT.xi0,meshFT.xi1,meshFT.xi2).to(device)

# Image creation:
ftmesh = meshFT(Verts,Faces, Faces_coeff)
image_ft=normalize_tensor(render_image_from_ftmesh(ftmesh, OTF, box_shape))

# Backward pass
loss = torch.sum(image_ft)
loss.backward()
print(Verts.grad)

# Visualize image
import napari
v = napari.view_image(image_ft.detach().cpu().numpy())
```

---
### API and Documentation

#### Fourier_transform
First we need to build our function 
- `class Fourier3dMesh(self, box_size,box_shape,device = 'cpu', dtype = torch.float,OTF=None narrowband_thresh = 0.01)`: 
    - `box_shape: [x_res,y_res,z_res]` Size of the Fourier transform box (in voxels)
    - `box_size:[[x_min,xmax],[y_min,y_max],[z_min,z_max]]` Dimensions of the box (in the spatial dimensions of the mesh)
    - `narrowband_thresh` threshold under which frequencies are not computed
    - `OTF` Optical transfer function, i.e Fourier transform of the PSF. Used to do the narrow-band computation.
    - `return meshFT`, a PyTorch autodiff function that compute the Fourier transform of a triangle mesh in the given box.
    
#### PSF design

We provide differentiable implementations of PSFs. Parameters such as PSF width, refraction indices, optical aberrations (i.e Zernike polynomial coefficients), can be learned during the optimization. See example notebooks for real use-cases. 
- `Gaussian_psf(self, box_shape,box_size,sigma=1, device = 'cpu')`: Implements a differentiable gaussian PSF
- `Gibson_Lanni_psf(self,box_shape, device = 'cpu', **kwargs)`: Implements a differentiable Gibson-Lanni PSF
- `Hanser_psf(self,box_shape,xi0,xi1,xi2, device = 'cpu', **kwargs)`: Implements a differentiable single-objective PSF by computing the pupil function
Differentiable parameters of PSFs models can be conveniently obtained with the method `model.parameters()`


#### Rendering
Our rendering function takes the Fourier transform of the mesh `ftmesh` and the optical transfer function `OTF` and returns an image of size `box_shape`

- `render_image_from_ftmesh(ftmesh,OTF,box_shape)`:
    - `return image`, a 3D image of size `box_shape`

---

### Credits, contact, citations
If you use this tool, please cite 

```
@misc{ichbiah2023differentiable,
      title={Differentiable Rendering for 3D Fluorescence Microscopy}, 
      author={Sacha Ichbiah and Fabrice Delbary and Hervé Turlier},
      year={2023},
      eprint={2303.10440},
      archivePrefix={arXiv},
      primaryClass={physics.bio-ph}
}
```