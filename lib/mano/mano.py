import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from typing import NewType, Optional
from dataclasses import dataclass, fields

from .. import utils

logger = utils.get_logger(__name__)

Tensor = NewType('Tensor', torch.Tensor)

def vertices2joints(J_regressor: Tensor, vertices: Tensor) -> Tensor:
    ''' Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    '''

    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])

def blend_shapes(betas: Tensor, shape_disps: Tensor) -> Tensor:
    ''' Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    '''

    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape

def batch_rodrigues(
    rot_vecs: Tensor,
    epsilon: float = 1e-8,
) -> Tensor:
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def transform_mat(R: Tensor, t: Tensor) -> Tensor:
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)

def batch_rigid_transform(
    rot_mats: Tensor,
    joints: Tensor,
    parents: Tensor,
    dtype=torch.float32
) -> Tensor:
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms

def vertex_normals(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert vertices.ndimension() == 3
    assert faces.ndimension() == 3
    assert vertices.shape[0] == faces.shape[0]
    assert vertices.shape[2] == 3
    assert faces.shape[2] == 3
    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.view(-1, 3)
    vertices_faces = vertices_faces.view(-1, 3, 3)

    normals.index_add_(
        0,
        faces[:, 1].long(),
        torch.cross(
            vertices_faces[:, 2] - vertices_faces[:, 1],
            vertices_faces[:, 0] - vertices_faces[:, 1],
        ),
    )
    normals.index_add_(
        0,
        faces[:, 2].long(),
        torch.cross(
            vertices_faces[:, 0] - vertices_faces[:, 2],
            vertices_faces[:, 1] - vertices_faces[:, 2],
        ),
    )
    normals.index_add_(
        0,
        faces[:, 0].long(),
        torch.cross(
            vertices_faces[:, 1] - vertices_faces[:, 0],
            vertices_faces[:, 2] - vertices_faces[:, 0],
        ),
    )

    normals = torch.nn.functional.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals

def get_normal_coord_system(normals):
    """
    returns tensor of basis vectors of coordinate system that moves with normals:

    e_x = always points horizontally
    e_y = always pointing up
    e_z = normal vector

    returns tensor of shape N x 3 x 3

    :param normals: tensor of shape N x 3
    :return:
    """
    device = normals.device
    dtype = normals.dtype
    N = len(normals)

    assert len(normals.shape) == 2

    normals = normals.detach()
    e_y = torch.tensor([0., 1., 0.], device=device, dtype=dtype)
    e_x = torch.tensor([0., 0., 1.], device=device, dtype=dtype)

    basis = torch.zeros(len(normals), 3, 3, dtype=dtype, device=device)
    # e_z' = e_n
    basis[:, 2] = torch.nn.functional.normalize(normals, p=2, dim=-1)

    # e_x' = e_n x e_y except e_n || e_y then e_x' = e_x
    normal_parallel_ey_mask = ((basis[:, 2] * e_y[None]).sum(dim=-1).abs() == 1)
    basis[:, 0] = torch.cross(e_y.expand(N, 3), basis[:, 2], dim=-1)
    basis[normal_parallel_ey_mask][:, 0] = e_x[None]
    basis[:, 0] = torch.nn.functional.normalize(basis[:, 0], p=2, dim=-1)
    basis[normal_parallel_ey_mask][:, 0] = e_x[None]
    basis[:, 0] = torch.nn.functional.normalize(basis[:, 0], p=2, dim=-1)

    # e_y' = e_z' x e_x'
    basis[:, 1] = torch.cross(basis[:, 2], basis[:, 0], dim=-1)
    #basis[:, 1] = torch.nn.functional.normalize(basis[:, 1], p=2, dim=-1)

    assert torch.all(torch.norm(basis, dim=-1, p=2) > .99)
    return basis

# Ref: https://github.com/philgras/neural-head-avatars/blob/0048afe9c9034157c63838801e9a2dd3126f806e/nha/util/meshes.py
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

@dataclass
class ModelOutput:
    vertices: Optional[Tensor] = None
    joints: Optional[Tensor] = None
    full_pose: Optional[Tensor] = None
    global_orient: Optional[Tensor] = None
    transl: Optional[Tensor] = None
    v_shaped: Optional[Tensor] = None

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __iter__(self):
        return self.keys()

    def keys(self):
        keys = [t.name for t in fields(self)]
        return iter(keys)

    def values(self):
        values = [getattr(self, t.name) for t in fields(self)]
        return iter(values)

    def items(self):
        data = [(t.name, getattr(self, t.name)) for t in fields(self)]
        return iter(data)

@dataclass
class MANOOutput(ModelOutput):
    betas: Optional[Tensor] = None
    hand_pose: Optional[Tensor] = None

class Mano(torch.nn.Module):
    # Ref: https://github.com/vchoutas/smplx/blob/566532a4636d9336403073884dbdd9722833d425/smplx/body_models.py
    def __init__(self, cfg_mano):
        super().__init__()
        
        with open(cfg_mano.PKL_PATH, "rb") as f:
            mano_data = pickle.load(f, encoding="latin1")

        shapedirs = np.array(mano_data["shapedirs"], dtype=np.float32)  # (|v|, 3, 10)
        self.num_betas = shapedirs.shape[-1]
        self.register_buffer("shapedirs", torch.from_numpy(shapedirs))

        faces = np.array(mano_data["f"], dtype=np.int64) # (|F|, 3)  |F| = 1538
        self.register_buffer("faces", torch.from_numpy(faces))

        v_template = np.array(mano_data["v_template"], dtype=np.float32)    # (|v|, 3)  |v| = 778
        self.register_buffer("v_template", torch.from_numpy(v_template))

        J_regressor = np.array(mano_data["J_regressor"].todense(), dtype=np.float32)    # (16, |v|)
        self.register_buffer("J_regressor", torch.from_numpy(J_regressor))
        
        # pose blend shape basis
        posedirs = np.array(mano_data["posedirs"], dtype=np.float32)    # (|v|, 3, 135)
        num_pose_basis = posedirs.shape[-1]
        # reshape to (135, 778*3)
        posedirs = np.reshape(posedirs, [-1, num_pose_basis]).T
        self.register_buffer("posedirs", torch.from_numpy(posedirs))

        # indices of parents for each joints
        parents = np.array(mano_data["kintree_table"][0], dtype=np.int64)   # (16,)
        parents[0] = -1
        self.register_buffer("parents", torch.from_numpy(parents))

        lbs_weights = np.array(mano_data["weights"], dtype=np.float32)  # (|v|, 16)
        self.register_buffer("lbs_weights", torch.from_numpy(lbs_weights))

        # create array for mean pose
        # if flat_hand is false, then use the mean that is given in data, rather than the flat open hand
        hand_mean = torch.from_numpy(np.array(mano_data["hands_mean"], dtype=np.float32)) # (45,)
        self.register_buffer("hand_mean", hand_mean)
        global_orient_mean = torch.zeros([3], dtype=torch.float32)
        pose_mean = torch.cat([global_orient_mean, self.hand_mean], dim=0)
        self.register_buffer("pose_mean", pose_mean)

        # add uvs
        with open(cfg_mano.HTML_UV, "rb") as f:
            html_uv = pickle.load(f)
        verts_uvs = np.array(html_uv["verts_uvs"], dtype=np.float32)
        verts_uvs[:, 1] = 1 - verts_uvs[:, 1]   # Y-origin in HTML texture is top, whereas in our convention it is in bottom
        self.register_buffer("verts_uvs", torch.from_numpy(verts_uvs))
        faces_uvs = np.array(html_uv["faces_uvs"], dtype=np.int64)
        self.register_buffer("faces_uvs", torch.from_numpy(faces_uvs))

        for id_subdiv in range(cfg_mano.SUBDIVIDE):
            self.subdivide()
        
    def subdivide(self):
        n_verts = self.v_template.shape[0]
        # Ref: https://github.com/SeanChenxy/HandAvatar/blob/3b1c70b9d8d829bfcea1255743daea6dd8ed0b1d/smplx/manohd/subdivide.py#L99
        verts, verts_uvs, faces, faces_uvs, edges, uv_edges = edge_subdivide(self.v_template.numpy(), self.verts_uvs.numpy(), self.faces.numpy(), self.faces_uvs.numpy())

        new_shapedirs = self.shapedirs[edges]
        new_shapedirs = new_shapedirs.mean(dim=1)  # n_edges x 3 x 10
        shapedirs = torch.cat((self.shapedirs, new_shapedirs), dim=0)

        new_posedirs = self.posedirs.permute(1, 0).view(n_verts, 3, 135)  # V x 3 x 135
        new_posedirs = new_posedirs[edges]  # n_edges x 2 x 3 x 135
        new_posedirs = new_posedirs.mean(dim=1)  # n_edges x 3 x 135
        new_posedirs = new_posedirs.view(len(edges) * 3, 135).permute(1, 0)
        posedirs = torch.cat((self.posedirs, new_posedirs), dim=1)

        new_J_regressor = torch.zeros(16, len(edges)).to(self.J_regressor.dtype).to(self.J_regressor.device)
        J_regressor = torch.cat((self.J_regressor, new_J_regressor), dim=1)

        new_lbs_weights = self.lbs_weights[edges]  # n_edges x 2 x 16
        new_lbs_weights = new_lbs_weights.mean(dim=1)  # n_edges x 16
        lbs_weights = torch.cat((self.lbs_weights, new_lbs_weights), dim=0)

        self.v_template = torch.from_numpy(verts).float()
        self.verts_uvs = torch.from_numpy(verts_uvs).float()
        self.faces = torch.from_numpy(faces).long()
        self.faces_uvs = torch.from_numpy(faces_uvs).long()
        self.posedirs = posedirs
        self.shapedirs = shapedirs
        self.J_regressor = J_regressor
        self.lbs_weights = lbs_weights


    def forward(self, betas, global_orient, hand_pose, transl, offsets=None, flat_hand_mean=False):
        """
        betas: torch.tensor, shape BxN_b
        global_orient: torch.tensor, shape Bx3
        hand_pose: torch.tensor, BxP
        transl: torch.tensor, shape Bx3
        """
        full_pose = torch.cat([global_orient, hand_pose], dim=1)
        if not flat_hand_mean:
            full_pose = full_pose + self.pose_mean
        
        vertices, joints = self.lbs(betas, full_pose, offsets)
        
        joints = joints + transl[:, None, :]
        vertices = vertices + transl[:, None, :]

        distal_joints = vertices[:, [333, 444, 672, 555, 744]]
        joints = torch.cat([joints, distal_joints], 1)

        output = MANOOutput(vertices=vertices,
                            joints=joints,
                            betas=betas,
                            global_orient=global_orient,
                            hand_pose=hand_pose,
                            full_pose=full_pose)

        return output
    
    def lbs(self, betas, pose, offsets):
        batch_size = max(betas.shape[0], pose.shape[0])
        device, dtype = betas.device, betas.dtype

        # add shape contribution
        v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)

        # get the joints
        J = vertices2joints(self.J_regressor, v_shaped)

        # add pose blend shapes
        # N x J x 3 x 3
        ident = torch.eye(3, dtype=dtype, device=device)
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view([batch_size, -1, 3, 3])
        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P x V*3) -> N x V x 3
        pose_offsets = torch.matmul(pose_feature, self.posedirs).view(batch_size, -1, 3)
        v_posed = pose_offsets + v_shaped

        if offsets is not None:
            normals = vertex_normals(v_posed, self.faces[None].expand(v_posed.shape[0], -1, -1))
            B, V, _3 = normals.shape
            normal_coord_sys = get_normal_coord_system(normals.view(-1, 3)).view(B, V, 3, 3)
            offsets = torch.matmul(normal_coord_sys.permute(0, 1, 3, 2), offsets.unsqueeze(-1)).squeeze(-1)
            v_posed = v_posed + offsets
        
        # get global joint location
        J_transformed, A = batch_rigid_transform(rot_mats, J, self.parents, dtype=dtype)

        # do skinning
        # W is N x V x (J+1)
        W = self.lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
        # (N x V x (J+1)) x (N x (J+1) x 16)
        num_joints = self.J_regressor.shape[0]
        T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)
        homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1], dtype=dtype, device=device)
        v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

        verts = v_homo[:, :, :3, 0]

        return verts, J_transformed
    


        