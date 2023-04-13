import torch
from largesteps.geometry import compute_matrix
import robust_laplacian
import numpy as np 

def cot_laplacian(
    verts: torch.Tensor, faces: torch.Tensor, eps: float = 1e-12):
    """
    Returns the Laplacian matrix with cotangent weights and the inverse of the
    face areas.

    Args:
        verts: tensor of shape (V, 3) containing the vertices of the graph
        faces: tensor of shape (F, 3) containing the vertex indices of each face
    Returns:
        2-element tuple containing
        - **L**: Sparse FloatTensor of shape (V,V) for the Laplacian matrix.
           Here, L[i, j] = cot a_ij + cot b_ij iff (i, j) is an edge in meshes.
           See the description above for more clarity.
        - **inv_areas**: FloatTensor of shape (V,) containing the inverse of sum of
           face areas containing each vertex
    """
    V, F = verts.shape[0], faces.shape[0]

    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Side lengths of each triangle, of shape (sum(F_n),)
    # A is the side opposite v1, B is opposite v2, and C is opposite v3
    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    # Area of each triangle (with Heron's formula); shape is (sum(F_n),)
    s = 0.5 * (A + B + C)
    # note that the area can be negative (close to 0) causing nans after sqrt()
    # we clip it to a small positive value
    # pyre-fixme[16]: `float` has no attribute `clamp_`.
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=eps).sqrt()

    # Compute cotangents of angles, of shape (sum(F_n), 3)
    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=1)
    cot /= 4.0

    # Construct a sparse matrix by basically doing:
    # L[v1, v2] = cota
    # L[v2, v0] = cotb
    # L[v0, v1] = cotc
    ii = faces[:, [1, 2, 0]]
    jj = faces[:, [2, 0, 1]]
    idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
    # pyre-fixme[16]: Module `sparse` has no attribute `FloatTensor`.
    L = torch.sparse.FloatTensor(idx, cot.view(-1), (V, V))

    # Make it symmetric; this means we are also setting
    # L[v2, v1] = cota
    # L[v0, v2] = cotb
    # L[v1, v0] = cotc
    L += L.t()

    # For each vertex, compute the sum of areas for triangles containing it.
    idx = faces.view(-1)
    inv_areas = torch.zeros(V, dtype=torch.float32, device=verts.device)
    val = torch.stack([area] * 3, dim=1).view(-1)
    inv_areas.scatter_add_(0, idx, val)
    idx = inv_areas > 0
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    inv_areas[idx] = 1.0 / inv_areas[idx]
    inv_areas = inv_areas.view(-1, 1)

    return L, inv_areas

def mesh_laplacian_smoothing(verts,faces):
    r"""
    Computes the laplacian smoothing objective for a batch of meshes.
    This function supports three variants of Laplacian smoothing,
    namely with uniform weights("uniform"), with cotangent weights ("cot"),
    and cotangent curvature ("cotcurv").For more details read [1, 2].

    Args:
        meshes: Meshes object with a batch of meshes.
        method: str specifying the method for the laplacian.
    Returns:
        loss: Average laplacian smoothing loss across the batch.
        Returns 0 if meshes contains no meshes or all empty meshes.

    Consider a mesh M = (V, F), with verts of shape Nx3 and faces of shape Mx3.
    The Laplacian matrix L is a NxN tensor such that LV gives a tensor of vectors:
    for a uniform Laplacian, LuV[i] points to the centroid of its neighboring
    vertices, a cotangent Laplacian LcV[i] is known to be an approximation of
    the surface normal, while the curvature variant LckV[i] scales the normals
    by the discrete mean curvature. For vertex i, assume S[i] is the set of
    neighboring vertices to i, a_ij and b_ij are the "outside" angles in the
    two triangles connecting vertex v_i and its neighboring vertex v_j
    for j in S[i], as seen in the diagram below.

    .. code-block:: python

               a_ij
                /\
               /  \
              /    \
             /      \
        v_i /________\ v_j
            \        /
             \      /
              \    /
               \  /
                \/
               b_ij

        The definition of the Laplacian is LV[i] = sum_j w_ij (v_j - v_i)
        For the uniform variant,    w_ij = 1 / |S[i]|
        For the cotangent variant,
            w_ij = (cot a_ij + cot b_ij) / (sum_k cot a_ik + cot b_ik)
        For the cotangent curvature, w_ij = (cot a_ij + cot b_ij) / (4 A[i])
        where A[i] is the sum of the areas of all triangles containing vertex v_i.

    There is a nice trigonometry identity to compute cotangents. Consider a triangle
    with side lengths A, B, C and angles a, b, c.

    .. code-block:: python

               c
              /|\
             / | \
            /  |  \
         B /  H|   \ A
          /    |    \
         /     |     \
        /a_____|_____b\
               C

        Then cot a = (B^2 + C^2 - A^2) / 4 * area
        We know that area = CH/2, and by the law of cosines we have

        A^2 = B^2 + C^2 - 2BC cos a => B^2 + C^2 - A^2 = 2BC cos a

        Putting these together, we get:

        B^2 + C^2 - A^2     2BC cos a
        _______________  =  _________ = (B/H) cos a = cos a / sin a = cot a
           4 * area            2CH


    [1] Desbrun et al, "Implicit fairing of irregular meshes using diffusion
    and curvature flow", SIGGRAPH 1999.

    [2] Nealan et al, "Laplacian Mesh Optimization", Graphite 2006.
    """

    # We don't want to backprop through the computation of the Laplacian;
    # just treat it as a magic constant matrix that is used to transform
    # verts into normals
    with torch.no_grad():
        L, inv_areas = robust_laplacian_torch(verts, faces)
        norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
        idx = norm_w > 0
        # pyre-fixme[58]: `/` is not supported for operand types `float` and
        #  `Tensor`.
        norm_w[idx] = 1.0 / norm_w[idx]
        loss = L.mm(verts) * norm_w - verts
    loss = loss.norm(dim=1)

    return loss.mean()

def robust_laplacian_torch(verts,faces):
    with torch.no_grad():
        L, M = robust_laplacian.mesh_laplacian(verts.detach().cpu().numpy(), faces.cpu().numpy())
        coo=L.tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        i = torch.LongTensor(indices).to(verts.device)
        v = torch.FloatTensor(values).to(verts.device)
        shape = coo.shape

        coom=M.tocoo()
        valuesm = coom.data
        indicesm = np.vstack((coom.row, coom.col))
        i = torch.LongTensor(indicesm).to(verts.device)
        v = torch.FloatTensor(valuesm).to(verts.device)
        shapem = coom.shape

        L = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        M = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return(L,M)
    
def compute_area_loss(Verts,Faces):
    Pos = Verts[Faces]
    Sides = Pos-Pos[:,[2,0,1]]

    Lengths_sides =torch.norm(Sides,dim=2)
    Half_perimeters = torch.sum(Lengths_sides,axis=1)/2
    Diffs = torch.zeros_like(Lengths_sides)
    Diffs[:,0] = Half_perimeters - Lengths_sides[:,0]
    Diffs[:,1] = Half_perimeters - Lengths_sides[:,1]
    Diffs[:,2] = Half_perimeters - Lengths_sides[:,2]
    Areas = (Half_perimeters*Diffs[:,0]*Diffs[:,1]*Diffs[:,2])**(0.5)
    return(torch.sum(Areas))


# 3D version of gradient preconditionning for triangle meshes, adapted from https://github.com/rgl-epfl/large-steps-pytorch
def compute_precondition_matrix(Verts, Faces, lambda_):
    #We use p = 2
    M = compute_matrix(Verts, Faces, lambda_) 
    inv_m = torch.inverse(M.to_dense())
    precondition_matrix = inv_m @ inv_m
    return(precondition_matrix)





class VectorAdam(torch.optim.Optimizer):
    #from https://github.com/iszihan/VectorAdam
    def __init__(self, params, lr=0.1, betas=(0.9, 0.999), eps=1e-8, axis=-1):
        defaults = dict(lr=lr, betas=betas, eps=eps, axis=axis)
        super(VectorAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(VectorAdam, self).__setstate__(state)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            eps = group['eps']
            axis = group['axis']
            for p in group["params"]:
                state = self.state[p]
                # Lazy initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["g1"] = torch.zeros_like(p.data)
                    state["g2"] = torch.zeros_like(p.data)

                g1 = state["g1"]
                g2 = state["g2"]
                state["step"] += 1
                grad = p.grad.data

                g1.mul_(b1).add_(grad, alpha=1-b1)
                if axis is not None:
                    dim = grad.shape[axis]
                    grad_norm = torch.norm(grad, dim=axis).unsqueeze(axis).repeat_interleave(dim, dim=axis)
                    grad_sq = grad_norm * grad_norm
                    g2.mul_(b2).add_(grad_sq, alpha=1-b2)
                else:
                    g2.mul_(b2).add_(grad.square(), alpha=1-b2)

                m1 = g1 / (1-(b1**state["step"]))
                m2 = g2 / (1-(b2**state["step"]))
                gr = m1 / (eps + m2.sqrt())
                p.data.sub_(gr, alpha=lr)
                

def Green_area(Verts, Edges):
    p0 = Verts[Edges[:,0]]
    p1 = Verts[Edges[:,1]]
    dx = p1[:,0]-p0[:,0]
    dy = p1[:,1]-p0[:,1]
    a= torch.sum(p0[:,1]*dx-p0[:,0]*dy)*2
    return a

