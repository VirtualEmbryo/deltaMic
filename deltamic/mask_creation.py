import numpy as np 
import trimesh
from trimesh import remesh
from scipy.spatial import KDTree
from trimesh.geometry import faces_to_edges
import trimesh.grouping as grouping
from scipy.spatial import KDTree

def subdivide_mesh(Verts,Faces,max_edge): 

    mesh = trimesh.Trimesh(vertices=Verts,
                           faces=Faces)


    longest_edge = np.linalg.norm(mesh.vertices[mesh.edges[:, 0]] -
                                  mesh.vertices[mesh.edges[:, 1]],
                                  axis=1).max()
    max_iter = max(int(np.ceil(np.log2(longest_edge / max_edge))), 0)*2+1
    # get the same mesh sudivided so every edge is shorter
    # than a factor of our pitch
    Verts,Faces = remesh.subdivide_to_size(mesh.vertices,
                                    mesh.faces,
                                    max_edge=max_edge,
                                    max_iter=max_iter)
    return(Verts,Faces)


def compute_max_size_edges(Verts,Faces): 
    V = Faces[:,[0,1,2]]
    P = Verts[V]
    E1 = P[:,1]-P[:,0]
    E2 = P[:,2]-P[:,1]
    E3 = P[:,2]-P[:,1]
    Edges = np.hstack((E1,E2,E3))
    Edges_norm = np.linalg.norm(Edges,axis=1)
    return(np.amax(Edges_norm))

def subdivide_triangle_mesh_with_materials(verts, faces, threshold,fast = True):
    
    longest_edge = compute_max_size_edges(verts,faces)
    if fast : 
        max_iter = max(int(np.ceil(np.log2(longest_edge / threshold)))-2, 0) 
    else :
        max_iter = max(int(np.ceil(np.log2(longest_edge / threshold))), 0) 
    print("Number of remeshing iterations:", max_iter)
    # get the same mesh sudivided so every edge is shorter
    # than a factor of our pitch
    new_verts,new_faces = verts.copy(),faces.copy()
    for i in range(max_iter):
        new_verts,new_faces = subdivide_with_materials(new_verts,new_faces)
        #print(compute_max_size_edges(new_verts,new_faces),threshold)
        if i==2 : break
    return(new_verts,new_faces)

def reorder_faces(Faces):
    for i,face in enumerate(Faces) : 
        if face[3]<face[4]:
            Faces[i]=Faces[i][[0,2,1,4,3]]
    return(Faces)


def Sample_verts_keeping_faces_normals(Faces,Verts):

    V = Faces[:,[0,1,2]].astype(int)
    P = Verts[V]
    print(P.shape)
    Centroids = np.mean(P, axis = 1)
    Sampled_verts = np.hstack((Centroids,Faces[:,[3,4,5,6,7]]))
    return(Sampled_verts)

def subdivide_with_materials(vertices,
              Faces,
              face_index=None,
              vertex_attributes=None,
              return_index=False):
    """
    Subdivide a mesh into smaller triangles.
    Note that if `face_index` is passed, only those
    faces will be subdivided and their neighbors won't
    be modified making the mesh no longer "watertight."
    Parameters
    ------------
    vertices : (n, 3) float
      Vertices in space
    faces : (m, 4) int
      Indexes of vertices which make up triangular faces
    face_index : faces to subdivide.
      if None: all faces of mesh will be subdivided
      if (n,) int array of indices: only specified faces
    vertex_attributes : dict
      Contains (n, d) attribute data
    return_index : bool
      If True, return index of original face for new faces
    Returns
    ----------
    new_vertices : (q, 3) float
      Vertices in space
    new_faces : (p, 5) int
      Remeshed faces
    index_dict : dict
      Only returned if `return_index`, {index of
      original face : index of new faces}.
    """
    faces = Faces[:,:3]
    if face_index is None:
        face_mask = np.ones(len(faces), dtype=bool)
    else:
        face_mask = np.zeros(len(faces), dtype=bool)
        face_mask[face_index] = True

    # the (c, 3) int array of vertex indices
    
    faces_subset = faces[face_mask]

    # find the unique edges of our faces subset
    edges = np.sort(faces_to_edges(faces_subset), axis=1)
    unique, inverse = grouping.unique_rows(edges)
    # then only produce one midpoint per unique edge
    mid = vertices[edges[unique]].mean(axis=1)
    mid_idx = inverse.reshape((-1, 3)) + len(vertices)

    # the new faces_subset with correct winding
    f = np.column_stack([faces_subset[:, 0],
                         mid_idx[:, 0],
                         mid_idx[:, 2],
                         mid_idx[:, 0],
                         faces_subset[:, 1],
                         mid_idx[:, 1],
                         mid_idx[:, 2],
                         mid_idx[:, 1],
                         faces_subset[:, 2],
                         mid_idx[:, 0],
                         mid_idx[:, 1],
                         mid_idx[:, 2]]).reshape((-1, 3))

    # add the 3 new faces_subset per old face all on the end
    # by putting all the new faces after all the old faces
    # it makes it easier to understand the indexes
    new_faces = np.vstack((faces[~face_mask], f))
    # stack the new midpoint vertices on the end
    new_vertices = np.vstack((vertices, mid))

    if vertex_attributes is not None:
        new_attributes = {}
        for key, values in vertex_attributes.items():
            attr_tris = values[faces_subset]
            attr_mid = np.vstack([
                attr_tris[:, g, :].mean(axis=1)
                for g in [[0, 1],
                          [1, 2],
                          [2, 0]]])
            attr_mid = attr_mid[unique]
            new_attributes[key] = np.vstack((
                values, attr_mid))
        return new_vertices, new_faces, new_attributes

    if return_index:
        # turn the mask back into integer indexes
        nonzero = np.nonzero(face_mask)[0]
        # new faces start past the original faces
        # but we've removed all the faces in face_mask
        start = len(faces) - len(nonzero)
        # indexes are just offset from start
        stack = np.arange(
            start, start + len(f) * 4).reshape((-1, 4))
        # reformat into a slightly silly dict for some reason
        index_dict = {k: v for k, v in zip(nonzero, stack)}

        return new_vertices, new_faces, index_dict
    
    #f
    new_faces_materials = np.array([Faces[:,3:].transpose()]*4).transpose(2,0,1).reshape(-1,2)
    
    new_faces_mm = np.vstack((new_faces.transpose(),new_faces_materials.transpose())).transpose()
    return new_vertices, new_faces_mm

def create_voxel_centroids(nx,ny,nz):
    XV = np.linspace(0.5,nx-0.5,nx-1)
    YV = np.linspace(0.5,ny-0.5,ny-1)
    ZV = np.linspace(0.5,nz-0.5,nz-1)
    xvv, yvv, zvv = np.meshgrid(XV,YV,ZV)
    xvv=np.transpose(xvv,(1,0,2)).flatten()
    yvv=np.transpose(yvv,(1,0,2)).flatten()
    zvv=zvv.flatten()
    Points=np.vstack(([xvv,yvv,zvv])).transpose()#.astype(int)
    return(Points)

def compute_box_size(verts,offset=0.1):
    #Compute automatically the box according to the position of the vertices. 
    #An offset of 0.1 means 10% of excess size. 
    box_size = np.zeros((3,2))
    extent = verts.max(axis=0)-verts.min(axis=0)
    box_size[:,1]=verts.max(axis=0)+extent*offset/2
    box_size[:,0]=verts.min(axis=0)-extent*offset/2
    #box_size[:,0]=box_size[:,0].min(axis=0)
    #box_size[:,1]=box_size[:,1].max(axis=0)
    return(box_size)


def create_coords(box_shape, box_size):
    nx,ny,nz = box_shape
    XV = np.linspace(box_size[0,0],box_size[0,1],nx)
    YV = np.linspace(box_size[1,0],box_size[1,1],ny)
    ZV = np.linspace(box_size[2,0],box_size[2,1],nz)
    xvv, yvv, zvv = np.meshgrid(XV,YV,ZV)
    xvv=np.transpose(xvv,(1,0,2)).flatten()
    yvv=np.transpose(yvv,(1,0,2)).flatten()
    zvv=zvv.flatten()
    Points=np.vstack(([xvv,yvv,zvv])).transpose()
    return(Points)


def create_semantic_segmentation_mask(verts,faces,box_shape, box_size):
    voxel_size = (box_size[:,1]-box_size[:,0])/box_shape
    dmax = np.amin(voxel_size)
    Verts, Faces = verts.copy(),faces.copy()[:,:3]
    Verts,Faces = subdivide_mesh(Verts,Faces,max_edge = dmax)

    nx,ny,nz = box_shape
    Points = create_coords(box_shape, box_size)
    Tree = KDTree(Points)
    Distances = Tree.query(Verts)
    dist,idx = Distances
    membrane = np.zeros(nx*ny*nz)
    membrane[idx]=1
    membrane=membrane.reshape(nx,ny,nz)
    return(membrane)

def create_instance_segmentation_mask(verts,faces,box_shape,box_size):
    Faces = faces.copy()
    Verts = verts.copy()
    assert Faces.shape[1]==5

    Faces = reorder_faces(Faces)

    voxel_size = (box_size[:,1]-box_size[:,0])/box_shape
    dmax = np.amin(voxel_size)/1

    Verts,Faces=subdivide_triangle_mesh_with_materials(Verts,Faces,dmax)

    topology = Faces[:,:3]

    coordinates = create_coords(box_shape, box_size)

    P = Verts[topology]
    T1 = P[:,1]-P[:,0]
    T2 = P[:,2]-P[:,1]
    Normal = np.cross(T1,T2,axis=1)
    Norms = np.linalg.norm(Normal,axis=1)
    Normal[:,0]/=Norms
    Normal[:,1]/=Norms
    Normal[:,2]/=Norms
    Faces_with_normal = np.hstack((Faces,Normal))
    Sampled_Verts = Sample_verts_keeping_faces_normals(Faces_with_normal, Verts)
    Sample_Tree = KDTree(Sampled_Verts[:,[0,1,2]])
    _,indices =  Sample_Tree.query(coordinates)
    Coords_ind = Sampled_Verts[:,[0,1,2]][indices]
    Verts_query_normals = (Sampled_Verts[:,[5,6,7]][indices])
    Vectors = Coords_ind-coordinates
    Dot_product_normals = np.sum(np.multiply(Vectors,Verts_query_normals),axis=1)
    Dot_sign = np.sign(Dot_product_normals)
    Normals_num = ((-Dot_sign+1)/2).astype(int)
    Line = np.arange(Verts_query_normals.shape[0])
    Indexes=np.stack((Line,Normals_num)).transpose()
    Faces_index=Sampled_Verts[indices][:,[3,4]][Indexes[:,0],Indexes[:,1]]
    IM_Mask = (Faces_index.reshape(*box_shape))
    IM_Mask +=1
    return(IM_Mask)

