import gmsh
import numpy as np
from .mesh_container import MeshContainer


def build_cylindrical(model_dict):
    """
    Build a 2-D cylindrical (annular) mesh with Tri6 elements using Gmsh.
    Returns a MeshContainer object.
    """
    gmsh.initialize()
    gmsh.model.add("WaveGuide")

    m = model_dict['Model']
    layers = len(m['DomainType'])
    surf_tags = []

    # ---------- create each layer ----------
    for i in range(layers):
        rx  = m['DomainRx'][i]
        ry  = m['DomainRy'][i]
        ecc = m['DomainEcc'][i]
        ang = m['DomainEccAngle'][i]
        rot = m['DomainTheta'][i]

        tag = gmsh.model.occ.addDisk(0, 0, 0, rx, ry)

        # optional eccentric shift
        if ecc != 0:
            gmsh.model.occ.translate([(2, tag)], ecc * np.cos(ang), ecc * np.sin(ang))

        # optional rotation
        if rot != 0:
            gmsh.model.occ.rotate([(2, tag)], 0, 0, 0, 0, 0, 1, rot)

        surf_tags.append(tag)

    # ---------- boolean cut to obtain nested rings ----------
    for i in range(layers - 1, 0, -1):
        gmsh.model.occ.cut([(2, surf_tags[i])], [(2, surf_tags[i - 1])],
                           removeObject=True, removeTool=False)

    gmsh.model.occ.synchronize()

    # ---------- physical groups for domains ----------
    for i, tag in enumerate(surf_tags):
        gmsh.model.addPhysicalGroup(2, [tag], i + 1)
        gmsh.model.setPhysicalName(2, i + 1, f"layer_{i + 1}")

    # ---------- mark boundary edges ----------
    b_edges = gmsh.model.getBoundary([(2, t) for t in surf_tags],
                                     oriented=False, recursive=True)
    inner, outer = [], []
    for e in b_edges:
        com = gmsh.model.occ.getCenterOfMass(e[0], e[1])
        r = np.hypot(com[0], com[1])
        if abs(r - m['DomainRx'][0]) < 1e-6:
            inner.append(e[1])
        else:
            outer.append(e[1])

    gmsh.model.addPhysicalGroup(1, inner, 1001)
    gmsh.model.addPhysicalGroup(1, outer, 1002)
    gmsh.model.setPhysicalName(1, 1001, "inner_edges")
    gmsh.model.setPhysicalName(1, 1002, "outer_edges")

    # ---------- meshing ----------
    if 'hmax' in m:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", m['hmax'])

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(3)          # promote to 6-node triangle

    node_tags, coord, _ = gmsh.model.mesh.getNodes()
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements()

    gmsh.finalize()
    return MeshContainer(node_tags, coord, elem_types,
                         elem_tags, elem_node_tags)