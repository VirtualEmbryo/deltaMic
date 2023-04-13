# Utility to generate meshes (with cubify and other), and to write/read them.
import scipy.spatial
import trimesh
import numpy as np

def combine_all_verts_faces_mm(All_verts,All_faces): 
        sum_val = 0
        for i in range(1,len(All_faces)):
            sum_val+=len(All_verts[i-1])
            All_faces[i][:,:3]+=sum_val
        V = np.vstack(All_verts)
        F = np.vstack(All_faces).astype(np.int32)
        return(V,F)

def combine_all_verts_faces(All_verts,All_faces): 
    sum_val = 0
    for i in range(1,len(All_faces)):
        sum_val+=len(All_verts[i-1])
        All_faces[i]+=sum_val
    V = np.vstack(All_verts)
    F = np.vstack(All_faces)
    return(V,F)




def create_initial_mesh_from_points_mm(Points,n_subdivisions = 3,threshold_dist = 0.9):

    distance_matrix = scipy.spatial.distance.cdist(Points,Points)
    distances = distance_matrix[distance_matrix!=0]
    min_dist = np.amin(distances)*threshold_dist
    All_verts = []
    All_faces = []
    print("Minimal distance:",min_dist)
    for i in range(len(Points)):
        mesh = trimesh.primitives.Sphere(center = Points[i],radius=min_dist/2,subdivisions = n_subdivisions)
        Verts,Faces = np.array(mesh.vertices), np.array(mesh.faces)
        Faces_label_1 = np.zeros(len(Faces))
        Faces_label_2 = np.ones(len(Faces))*(i+1)
        Faces_mm = np.vstack((Faces.transpose(),Faces_label_1.reshape(1,-1),Faces_label_2.reshape(1,-1))).transpose()#,axis=1)
        
        All_verts.append(Verts.copy())
        All_faces.append(Faces_mm.copy())#aces.copy())

    V,F = combine_all_verts_faces_mm(All_verts,All_faces)
    return(V,F)


#I/O
