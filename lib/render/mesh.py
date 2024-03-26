# Ref: https://github.com/NVlabs/nvdiffrecmc/blob/ea12ed9e03a9edba4eb6a351b5dcfe653d5c1ec5/render/mesh.py

import os
import numpy as np
import torch

from . import obj
from . import util

######################################################################################
# Base mesh class
######################################################################################
class Mesh:
    def __init__(self, v_pos=None, t_pos_idx=None, v_nrm=None, t_nrm_idx=None, v_tex=None, t_tex_idx=None, 
        v_tng=None, t_tng_idx=None, material=None, base=None):
        self.v_pos = v_pos
        self.v_nrm = v_nrm
        self.v_tex = v_tex
        self.v_tng = v_tng
        self.t_pos_idx = t_pos_idx
        self.t_nrm_idx = t_nrm_idx
        self.t_tex_idx = t_tex_idx
        self.t_tng_idx = t_tng_idx
        self.material = material

        if base is not None:
            self.copy_none(base)

    def copy_none(self, other):
        if self.v_pos is None:
            self.v_pos = other.v_pos
        if self.t_pos_idx is None:
            self.t_pos_idx = other.t_pos_idx
        if self.v_nrm is None:
            self.v_nrm = other.v_nrm
        if self.t_nrm_idx is None:
            self.t_nrm_idx = other.t_nrm_idx
        if self.v_tex is None:
            self.v_tex = other.v_tex
        if self.t_tex_idx is None:
            self.t_tex_idx = other.t_tex_idx
        if self.v_tng is None:
            self.v_tng = other.v_tng
        if self.t_tng_idx is None:
            self.t_tng_idx = other.t_tng_idx
        if self.material is None:
            self.material = other.material

    def clone(self):
        out = Mesh(base=self)
        if out.v_pos is not None:
            out.v_pos = out.v_pos.clone().detach()
        if out.t_pos_idx is not None:
            out.t_pos_idx = out.t_pos_idx.clone().detach()
        if out.v_nrm is not None:
            out.v_nrm = out.v_nrm.clone().detach()
        if out.t_nrm_idx is not None:
            out.t_nrm_idx = out.t_nrm_idx.clone().detach()
        if out.v_tex is not None:
            out.v_tex = out.v_tex.clone().detach()
        if out.t_tex_idx is not None:
            out.t_tex_idx = out.t_tex_idx.clone().detach()
        if out.v_tng is not None:
            out.v_tng = out.v_tng.clone().detach()
        if out.t_tng_idx is not None:
            out.t_tng_idx = out.t_tng_idx.clone().detach()
        return out

######################################################################################
# Mesh loading helper
######################################################################################

def load_mesh(filename, mtl_override=None):
    name, ext = os.path.splitext(filename)
    if ext == ".obj":
        return obj.load_obj(filename, clear_ks=True, mtl_override=mtl_override)
    assert False, "Invalid mesh file extension"

######################################################################################
# Simple smooth vertex normal computation
######################################################################################
def auto_normals(imesh):

    i0 = imesh.t_pos_idx[:, 0]
    i1 = imesh.t_pos_idx[:, 1]
    i2 = imesh.t_pos_idx[:, 2]

    v0 = imesh.v_pos[i0, :]
    v1 = imesh.v_pos[i1, :]
    v2 = imesh.v_pos[i2, :]

    face_normals = torch.cross(v1 - v0, v2 - v0)

    # Splat face normals to vertices
    v_nrm = torch.zeros_like(imesh.v_pos)
    v_nrm.scatter_add_(0, i0[:, None].repeat(1,3), face_normals)
    v_nrm.scatter_add_(0, i1[:, None].repeat(1,3), face_normals)
    v_nrm.scatter_add_(0, i2[:, None].repeat(1,3), face_normals)

    # Normalize, replace zero (degenerated) normals with some default value
    v_nrm = torch.where(util.dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda'))
    v_nrm = util.safe_normalize(v_nrm)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(v_nrm))

    return Mesh(v_nrm=v_nrm, t_nrm_idx=imesh.t_pos_idx, base=imesh)

######################################################################################
# Compute tangent space from texture map coordinates
# Follows http://www.mikktspace.com/ conventions
######################################################################################
def compute_tangents(imesh):
    vn_idx = [None] * 3
    pos = [None] * 3
    tex = [None] * 3
    for i in range(0,3):
        pos[i] = imesh.v_pos[imesh.t_pos_idx[:, i]]
        tex[i] = imesh.v_tex[imesh.t_tex_idx[:, i]]
        vn_idx[i] = imesh.t_nrm_idx[:, i]

    tangents = torch.zeros_like(imesh.v_nrm)
    tansum   = torch.zeros_like(imesh.v_nrm)

    # Compute tangent space for each triangle
    uve1 = tex[1] - tex[0]
    uve2 = tex[2] - tex[0]
    pe1  = pos[1] - pos[0]
    pe2  = pos[2] - pos[0]
    
    nom   = (pe1 * uve2[..., 1:2] - pe2 * uve1[..., 1:2])
    denom = (uve1[..., 0:1] * uve2[..., 1:2] - uve1[..., 1:2] * uve2[..., 0:1])
    
    # Avoid division by zero for degenerated texture coordinates
    tang = nom / torch.where(denom > 0.0, torch.clamp(denom, min=1e-6), torch.clamp(denom, max=-1e-6))

    # Update all 3 vertices
    for i in range(0,3):
        idx = vn_idx[i][:, None].repeat(1,3)
        tangents.scatter_add_(0, idx, tang)                # tangents[n_i] = tangents[n_i] + tang
        tansum.scatter_add_(0, idx, torch.ones_like(tang)) # tansum[n_i] = tansum[n_i] + 1
    tangents = tangents / tansum

    # Normalize and make sure tangent is perpendicular to normal
    tangents = util.safe_normalize(tangents)
    tangents = util.safe_normalize(tangents - util.dot(tangents, imesh.v_nrm) * imesh.v_nrm)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(tangents))

    return Mesh(v_tng=tangents, t_tng_idx=imesh.t_nrm_idx, base=imesh)

######################################################################################
# Subdivide mesh
# Ref: https://github.com/philgras/neural-head-avatars/blob/0048afe9c9034157c63838801e9a2dd3126f806e/nha/util/meshes.py#L17
######################################################################################

def append_edge(edge_map_, edges_, idx_a, idx_b):
    if idx_b < idx_a:
        idx_a, idx_b = idx_b, idx_a

    if not (idx_a, idx_b) in edge_map_:
        e_id = len(edges_)
        edges_.append([idx_a, idx_b])
        edge_map_[(idx_a, idx_b)] = e_id
        edge_map_[(idx_b, idx_a)] = e_id


def edge_subdivide(vertices, uvs, faces, uvfaces):
    """
    subdivides mesh based on edge midpoints. every triangle is subdivided into 4 child triangles.
    old faces are kept in array
    :param vertices: V x 3 ... vertex coordinates
    :param uvs: T x 2 ... uv coordinates
    :param faces: F x 3 face vertex idx array
    :param uvfaces: F x 3 face uv idx array
    :return:
        - vertices ... np.array of vertex coordinates with shape V + n_edges x 3
        - uvs ... np.array of uv coordinates with shape T + n_edges x 2
        - faces ... np.array of face vertex idcs with shape F + 4*F x 3
        - uv_faces ... np.array of face uv idcs with shape F + 4*F x 3
        - edges ... np.array of shape n_edges x 2 giving the indices of the vertices of each edge
        - uv_edges ... np.array of shape n_edges x 2 giving the indices of the uv_coords of each edge

        all returns are a concatenation like np.concatenate((array_old, array_new), axis=0) so that
        order of old entries is not changed and so that also old faces are still present.
    """
    n_faces = faces.shape[0]
    n_vertices = vertices.shape[0]
    n_uvs = uvs.shape[0]

    # if self.edges is None:
    # if True:
    # compute edges
    edges = []
    edge_map = dict()
    for i in range(0, n_faces):
        append_edge(edge_map, edges, faces[i, 0], faces[i, 1])
        append_edge(edge_map, edges, faces[i, 1], faces[i, 2])
        append_edge(edge_map, edges, faces[i, 2], faces[i, 0])
    n_edges = len(edges)
    edges = np.array(edges).astype(int)

    # compute edges uv space
    uv_edges = []
    uv_edge_map = dict()
    for i in range(0, n_faces):
        append_edge(uv_edge_map, uv_edges, uvfaces[i, 0], uvfaces[i, 1])
        append_edge(uv_edge_map, uv_edges, uvfaces[i, 1], uvfaces[i, 2])
        append_edge(uv_edge_map, uv_edges, uvfaces[i, 2], uvfaces[i, 0])
    uv_n_edges = len(uv_edges)
    uv_edges = np.array(uv_edges).astype(int)

    #    print('edges:', edges.shape)
    #    print('self.edge_map :', len(edge_map ))
    #
    #    print('uv_edges:', uv_edges.shape)
    #    print('self.uv_edge_map :', len(uv_edge_map ))
    #
    #    print('vertices:', vertices.shape)
    #    print('normals:', normals.shape)
    #    print('uvs:', uvs.shape)
    #    print('faces:', faces.shape)
    #    print('uvfaces:', uvfaces.shape)

    ############
    # vertices
    v = np.zeros((n_vertices + n_edges, 3))
    # copy original vertices
    v[:n_vertices, :] = vertices
    # compute edge midpoints
    vertices_edges = vertices[edges]
    v[n_vertices:, :] = 0.5 * (vertices_edges[:, 0] + vertices_edges[:, 1])

    # uvs
    f_uvs = np.zeros((n_uvs + uv_n_edges, 2))
    # copy original uvs
    f_uvs[:n_uvs, :] = uvs
    # compute edge midpoints
    uvs_edges = uvs[uv_edges]
    f_uvs[n_uvs:, :] = 0.5 * (uvs_edges[:, 0] + uvs_edges[:, 1])

    # new topology
    f = np.concatenate((faces, np.zeros((4 * n_faces, 3))), axis=0)
    f_uv_id = np.concatenate((uvfaces, np.zeros((4 * n_faces, 3))), axis=0)
    # f_uv = np.zeros((4*n_faces*3, 2))
    for i in range(0, n_faces):
        # vertex ids
        a = int(faces[i, 0])
        b = int(faces[i, 1])
        c = int(faces[i, 2])
        ab = n_vertices + edge_map[(a, b)]
        bc = n_vertices + edge_map[(b, c)]
        ca = n_vertices + edge_map[(c, a)]
        # uvs
        a_uv = int(uvfaces[i, 0])
        b_uv = int(uvfaces[i, 1])
        c_uv = int(uvfaces[i, 2])
        ab_uv = n_uvs + uv_edge_map[(a_uv, b_uv)]
        bc_uv = n_uvs + uv_edge_map[(b_uv, c_uv)]
        ca_uv = n_uvs + uv_edge_map[(c_uv, a_uv)]

        ## triangle 1
        f[n_faces + 4 * i, 0] = a
        f[n_faces + 4 * i, 1] = ab
        f[n_faces + 4 * i, 2] = ca
        f_uv_id[n_faces + 4 * i, 0] = a_uv
        f_uv_id[n_faces + 4 * i, 1] = ab_uv
        f_uv_id[n_faces + 4 * i, 2] = ca_uv

        ## triangle 2
        f[n_faces + 4 * i + 1, 0] = ab
        f[n_faces + 4 * i + 1, 1] = b
        f[n_faces + 4 * i + 1, 2] = bc
        f_uv_id[n_faces + 4 * i + 1, 0] = ab_uv
        f_uv_id[n_faces + 4 * i + 1, 1] = b_uv
        f_uv_id[n_faces + 4 * i + 1, 2] = bc_uv

        ## triangle 3
        f[n_faces + 4 * i + 2, 0] = ca
        f[n_faces + 4 * i + 2, 1] = ab
        f[n_faces + 4 * i + 2, 2] = bc
        f_uv_id[n_faces + 4 * i + 2, 0] = ca_uv
        f_uv_id[n_faces + 4 * i + 2, 1] = ab_uv
        f_uv_id[n_faces + 4 * i + 2, 2] = bc_uv

        ## triangle 4
        f[n_faces + 4 * i + 3, 0] = ca
        f[n_faces + 4 * i + 3, 1] = bc
        f[n_faces + 4 * i + 3, 2] = c
        f_uv_id[n_faces + 4 * i + 3, 0] = ca_uv
        f_uv_id[n_faces + 4 * i + 3, 1] = bc_uv
        f_uv_id[n_faces + 4 * i + 3, 2] = c_uv

    return v, f_uvs, f, f_uv_id, edges, uv_edges
