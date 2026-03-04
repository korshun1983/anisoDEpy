import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from models.materials import Fluid, IsotropicSolid
from models.geometry import Layer, Geometry
from mesh.generator import generate_mesh


def main():
    # Материалы
    fluid = Fluid(rho=1000.0, c=1500.0)
    solid = IsotropicSolid(rho=2230.0, E=20e9, nu=0.25)

    # Геометрия
    R_inner = 0.1
    R_outer = 2.0
    n_points = 12   # количество точек на каждой границе

    layer_fluid = Layer(outer_radius=R_inner, material=fluid, n_points=n_points)
    layer_solid = Layer(outer_radius=R_outer, material=solid,
                        inner_radius=R_inner, n_points=n_points)

    geometry = Geometry([layer_fluid, layer_solid])

    # Генерация сетки
    mesh_size = 0.6   # глобальный размер элемента
    order = 1          # линейные треугольники
    nodes, elements_by_layer, phys_tags = generate_mesh(
        geometry, mesh_size=mesh_size, order=order, visualize=False
    )

    # Визуализация
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ['lightblue', 'lightgreen']

    for layer_idx, elems in elements_by_layer.items():
        # elems может содержать 3 или 6 узлов; для триангуляции matplotlib нужны только 3 угловых
        if elems.shape[1] == 6:
            tri_elems = elems[:, [0, 1, 2]]   # квадратичный -> берём только угловые
        else:
            tri_elems = elems
        triang = tri.Triangulation(nodes[:, 0], nodes[:, 1], tri_elems)
        ax.triplot(triang, color=colors[layer_idx % len(colors)], lw=0.5, alpha=0.7)

    # Отметим границы
    for r in geometry.radii_list():
        circle = plt.Circle((0, 0), r, fill=False, edgecolor='red', linestyle='--', linewidth=1)
        ax.add_patch(circle)

    ax.set_aspect('equal')
    ax.set_xlabel('x, м')
    ax.set_ylabel('y, м')
    ax.set_title('Сетка волновода (Gmsh)')
    ax.grid(True, linestyle=':', alpha=0.5)

    margin = 0.1 * R_outer
    ax.set_xlim(-R_outer - margin, R_outer + margin)
    ax.set_ylim(-R_outer - margin, R_outer + margin)
    plt.tight_layout()
    plt.show()

    # Статистика
    print(f"Всего узлов: {len(nodes)}")
    for idx, elems in elements_by_layer.items():
        name = 'жидкость' if idx == 0 else 'твёрдое тело'
        print(f"Слой {idx} ({name}): {len(elems)} элементов")


if __name__ == '__main__':
    main()