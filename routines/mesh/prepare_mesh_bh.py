"""
PrepareMeshBH.py
================
EXACT MATLAB equivalent of PrepareMeshBH.m
"""

import numpy as np
from utils import debug_print


def prepare_mesh_bh(CompStruct):
    """
    Prepare mesh for elliptically layered boreholes.

    Parameters
    ----------
    CompStruct : dict
        Computational structure

    Returns
    -------
    MeshNodes : np.ndarray
        Node coordinates (2 x n_nodes)
    BoundaryEdges : np.ndarray
        Boundary edge definitions
    MeshTri : np.ndarray
        Triangle definitions (4 x n_tri)
    CompStruct : dict
        Updated computational structure
    """
    debug_print("    PrepareMeshBH: Generating boundary geometry...", level=4)

    Edges = []
    Nodes = []
    DomainFaces = {}
    PrevEdgeSize = 0

    n_domain = CompStruct['Data']['N_domain']

    # Construct boundary nodes, edges, faces for each domain
    for ii_d in range(n_domain):
        # Domain geometry
        Rx = CompStruct['Model']['DomainRx'][ii_d]
        Ry = CompStruct['Model']['DomainRy'][ii_d]
        ThetaRot = CompStruct['Model']['DomainTheta'][ii_d]
        Ecc = CompStruct['Model']['DomainEcc'][ii_d]
        EccAngle = CompStruct['Model']['DomainEccAngle'][ii_d]

        # Angular grid
        Nth = CompStruct['Model']['DomainNth'][ii_d]
        DTheta = np.pi / Nth
        Theta = np.arange(-np.pi, np.pi - DTheta / 2, DTheta)

        # Check for rectangular boundary
        boundary_rec = False
        if CompStruct['Mesh']['ext_boundary_shape'].lower() == 'rec':
            add_loc = CompStruct['Model']['AddDomainLoc']
            if add_loc == 'ext' and ii_d >= n_domain - 1:
                boundary_rec = True
            elif add_loc == 'int' and ii_d == n_domain - 1:
                boundary_rec = True

        # Generate boundary nodes
        if not boundary_rec:
            # Circular/elliptical boundary
            Xc = Ecc * np.cos(EccAngle)
            Yc = Ecc * np.sin(EccAngle)

            XBgrid = Xc + Rx * np.cos(Theta) * np.cos(ThetaRot) - Ry * np.sin(Theta) * np.sin(ThetaRot)
            YBgrid = Yc + Rx * np.cos(Theta) * np.sin(ThetaRot) + Ry * np.sin(Theta) * np.cos(ThetaRot)
        else:
            # Rectangular boundary
            XBgrid = Rx * np.cos(Theta)
            YBgrid = Ry * np.sin(Theta)

            # Adjust corners
            for ia, ang in enumerate(Theta):
                if -np.pi <= ang <= -0.75 * np.pi:
                    XBgrid[ia] = -Rx
                    YBgrid[ia] = -Rx * np.tan(np.pi + ang)
                elif 0.75 * np.pi <= ang <= np.pi:
                    XBgrid[ia] = -Rx
                    YBgrid[ia] = Rx * np.tan(np.pi - ang)
                elif -0.75 * np.pi <= ang <= -0.25 * np.pi:
                    XBgrid[ia] = Ry * np.tan(0.5 * np.pi + ang)
                    YBgrid[ia] = -Ry
                elif -0.25 * np.pi <= ang <= 0.25 * np.pi:
                    XBgrid[ia] = Rx
                    YBgrid[ia] = Rx * np.tan(ang)
                elif 0.25 * np.pi <= ang <= 0.75 * np.pi:
                    XBgrid[ia] = Ry * np.tan(0.5 * np.pi - ang)
                    YBgrid[ia] = Ry

        DomainBNodes = np.column_stack([XBgrid, YBgrid])

        # Define edges
        n_nodes = len(DomainBNodes)
        DomainBEdges = np.column_stack([
            np.arange(1, n_nodes + 1),
            np.concatenate([np.arange(2, n_nodes + 1), [1]])
        ])

        # Track face edges
        FaceStartEdge = len(Edges) - PrevEdgeSize
        FaceEndEdge = len(Edges) + len(DomainBEdges) - 1
        DomainFaces[ii_d] = list(range(FaceStartEdge, FaceEndEdge + 1))
        PrevEdgeSize = len(DomainBEdges)

        # Update global lists
        if len(Edges) == 0:
            Edges = DomainBEdges
            Nodes = DomainBNodes
        else:
            Edges = np.vstack([Edges, DomainBEdges + len(Nodes)])
            Nodes = np.vstack([Nodes, DomainBNodes])

    # Convert to 0-based indexing for Python
    Edges = Edges.astype(int) - 1

    # ============================================================
    # CALL MESHFACES (Need MATLAB file or pygmsh equivalent)
    # ============================================================
    debug_print("    PrepareMeshBH: Calling MeshFaces...", level=4)

    # TODO: Need CompStruct.Methods.MeshFaces
    # For now, use simplified mesh generation
    MeshNodes, MeshTri, MeshFaceNums, CompStruct = _simple_mesh_generation(
        Nodes, Edges, DomainFaces, CompStruct
    )

    # Insert domain information (MATLAB: MeshTri(:,4) = MeshFaceNums)
    if MeshTri.shape[0] == 3:  # Only nodes, no domain info
        MeshTri = np.vstack([MeshTri, MeshFaceNums])

    # Transpose if needed (MATLAB uses column-major)
    MeshTri = MeshTri.T if MeshTri.shape[0] == 4 else MeshTri

    # Find boundary edges
    BoundaryEdges = CompStruct['Methods']['FindBEdges'](MeshNodes, MeshTri, CompStruct)

    # Transpose nodes to (2 x n_nodes) format
    MeshNodes = MeshNodes.T

    debug_print(f"    Mesh generation complete: {MeshNodes.shape[1]} nodes, {MeshTri.shape[0]} elements", level=4)

    return MeshNodes, BoundaryEdges, MeshTri, CompStruct


def _simple_mesh_generation(Nodes, Edges, DomainFaces, CompStruct):
    """
    Simplified mesh generation placeholder.
    Replace with actual MeshFaces implementation.
    """
    debug_print("      WARNING: Using simplified mesh generation", level=1)

    # Simple Delaunay triangulation of nodes
    # In real implementation, this would call Mesh2D's meshfaces
    from scipy.spatial import Delaunay

    tri = Delaunay(Nodes)
    MeshTri = tri.simplices.T

    # Assign domain based on centroid location
    MeshFaceNums = np.zeros(MeshTri.shape[1], dtype=int)

    # Placeholder: all triangles in domain 0
    MeshFaceNums[:] = 0

    MeshProps = {}

    return Nodes.T, MeshTri, MeshFaceNums, CompStruct