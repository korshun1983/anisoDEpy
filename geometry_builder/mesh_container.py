import numpy as np


class MeshContainer:
    """
    Lightweight storage for a 6-node triangular mesh (Tri6, Gmsh type 9).
    """
    def __init__(self, node_tags, coord, elem_types, elem_tags, elem_node_tags):
        # ---------- nodal data ----------
        self.node_tags = node_tags
        self.coord     = coord.reshape(-1, 3)[:, :2]      # 2-D slice

        # ---------- elemental data ----------
        self.elem_types       = elem_types
        self.elem_tags        = elem_tags
        self.elem_node_tags   = elem_node_tags

        # ---------- extract only Tri6 (type 9) ----------
        tri6_mask = (elem_types == 9)
        if not np.any(tri6_mask):
            raise ValueError("No Tri6 elements (type 9) found in the mesh.")

        idx           = np.where(tri6_mask)[0][0]          # first block of type 9
        nnode_per_el  = 6
        flat          = elem_node_tags[idx]                # flat connectivity
        if flat.size % nnode_per_el:
            raise RuntimeError("Tri6 connectivity size not divisible by 6")

        self.tri6 = flat.reshape(-1, nnode_per_el) - 1     # convert to 0-based

    # -------------------- handy shortcuts --------------------
    @property
    def nnod(self):
        return self.coord.shape[0]

    @property
    def nelem(self):
        return self.tri6.shape[0]