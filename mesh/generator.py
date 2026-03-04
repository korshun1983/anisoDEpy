"""
mesh/generator.py
Генерация треугольной сетки с помощью Gmsh.
Используются полные окружности как границы слоёв.
"""

import gmsh
import numpy as np


def generate_mesh(geometry, mesh_size=0.2, order=1, filename=None, visualize=False):
    gmsh.initialize()
    gmsh.model.add("waveguide")

    radii = sorted(set(geometry.radii_list()))

    # Создаём окружности через OCC
    circle_tags = []
    for r in radii:
        circ = gmsh.model.occ.addCircle(0, 0, 0, r)
        circle_tags.append(circ)

    gmsh.model.occ.synchronize()

    # Создаём петли и поверхности в geo
    loops = []
    for i, tag in enumerate(circle_tags):
        if i == len(circle_tags)-1:
            # Внешняя граница (самый большой радиус) — против часовой
            loop = gmsh.model.geo.addCurveLoop([tag])
        else:
            # Внутренние границы — по часовой (отрицательный тег)
            loop = gmsh.model.geo.addCurveLoop([-tag])
        loops.append(loop)

    gmsh.model.geo.synchronize()

    surfaces = []
    for i in range(len(loops)):
        if i == 0:
            surf = gmsh.model.geo.addPlaneSurface([loops[0]])
        else:
            surf = gmsh.model.geo.addPlaneSurface([loops[i], loops[i-1]])
        surfaces.append(surf)

    gmsh.model.geo.synchronize()

    # Физические группы
    phys_tags = {}
    for i, surf in enumerate(surfaces):
        phys = gmsh.model.addPhysicalGroup(2, [surf])
        gmsh.model.setPhysicalName(2, phys, f"Layer_{i}")
        phys_tags[i] = phys

    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    gmsh.option.setNumber("Mesh.ElementOrder", order)
    gmsh.model.mesh.generate(2)

    if visualize:
        gmsh.fltk.run()
    if filename:
        gmsh.write(filename)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    nodes = np.array(node_coords).reshape((-1, 3))[:, :2]   # только x,y

    elements_by_layer = {}
    for layer_idx, phys in phys_tags.items():
        elem_types, elem_tags, node_tags_elem = gmsh.model.mesh.getElements(dim=2, tag=phys)
        target_type = 2 if order == 1 else 9
        for typ, conn in zip(elem_types, node_tags_elem):
            if typ == target_type:
                npe = 3 if order == 1 else 6
                elems = np.array(conn).reshape((-1, npe))
                elems -= 1   # 0-базовая индексация
                elements_by_layer[layer_idx] = elems
                break

    gmsh.finalize()
    return nodes, elements_by_layer, phys_tags