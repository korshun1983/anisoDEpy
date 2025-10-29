import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import gmsh

# Основная функция для работы с сеткой
def create_mesh(CompStruct):
    """
    Создает сетку на основе заданных параметров CompStruct
    с элементами 3-го порядка и граничными индексами

    Аргументы:
        CompStruct: объект с параметрами модели

    Вывод:
        tuple: (nodes, elements, triangulation, boundary_elements) - 
               узлы, элементы, триангуляция и граничные элементы
    """
    # Получаем параметры областей
    radii = np.array([CompStruct.Model.DomainRx[i] for i in range(CompStruct.Data.N_domain)])
    centers_x = np.array([CompStruct.Model.DomainEcc[i] * math.cos(CompStruct.Model.DomainEccAngle[i])
                          for i in range(CompStruct.Data.N_domain)])
    centers_y = np.array([CompStruct.Model.DomainEcc[i] * math.sin(CompStruct.Model.DomainEccAngle[i])
                          for i in range(CompStruct.Data.N_domain)])

    # Устанавливаем размеры сетки
    mesh_size_min = radii[0] * 0.05
    mesh_size_max = radii[-1] * 0.1

    # Инициализируем GMSH с OpenCASCADE
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("concentric_domains")

    # Используем OpenCASCADE для точного задания геометрии
    gmsh.model.occ.synchronize()

    # Создаем физические группы для каждой поверхности
    surfaces = []
    physical_groups = []

    # Домен 0: внутренний круг
    disk0 = gmsh.model.occ.addDisk(centers_x[0], centers_y[0], 0, radii[0], radii[0])
    surfaces.append((2, disk0))
    physical_groups.append((2, [disk0], 1))
    print(f"Домен 0: внутренний круг, тег: {disk0}")

    # Домен 1: промежуточное кольцо (нужно создать новые диски для операции вычитания)
    if CompStruct.Data.N_domain >= 2:
        # Создаем новые диски для операции вычитания
        disk1_outer = gmsh.model.occ.addDisk(centers_x[1], centers_y[1], 0, radii[1], radii[1])
        disk1_inner = gmsh.model.occ.addDisk(centers_x[0], centers_y[0], 0, radii[0], radii[0])

        # Вычитаем внутренний диск из внешнего
        obj1 = gmsh.model.occ.cut([(2, disk1_outer)], [(2, disk1_inner)])
        if obj1[0]:  # Если есть результат вычитания
            ring1_tag = obj1[0][0][1]
            surfaces.append((2, ring1_tag))
            physical_groups.append((2, [ring1_tag], 2))
            print(f"Домен 1: промежуточное кольцо, тег: {ring1_tag}")

    # Домен 2: внешнее кольцо (также создаем новые диски)
    if CompStruct.Data.N_domain >= 3:
        # Создаем новые диски для операции вычитания
        disk2_outer = gmsh.model.occ.addDisk(centers_x[2], centers_y[2], 0, radii[2], radii[2])
        disk2_inner = gmsh.model.occ.addDisk(centers_x[1], centers_y[1], 0, radii[1], radii[1])

        # Вычитаем внутренний диск из внешнего
        obj2 = gmsh.model.occ.cut([(2, disk2_outer)], [(2, disk2_inner)])
        if obj2[0]:  # Если есть результат вычитания
            ring2_tag = obj2[0][0][1]
            surfaces.append((2, ring2_tag))
            physical_groups.append((2, [ring2_tag], 3))
            print(f"Домен 2: внешнее кольцо, тег: {ring2_tag}")

    # Синхронизируем геометрию
    gmsh.model.occ.synchronize()

    # Создаем физические группы для каждой поверхности
    for dim, tags, physical_tag in physical_groups:
        gmsh.model.addPhysicalGroup(dim, tags, physical_tag)
        print(f"Создана физическая группа {physical_tag} для домена")

    # Получаем граничные кривые и создаем для них физические группы
    boundary_curves = {}
    for i, (dim, tag) in enumerate(surfaces):
        boundaries = gmsh.model.getBoundary([(dim, tag)])
        boundary_tags = [b[1] for b in boundaries]

        # Создаем физическую группу для границы
        boundary_physical_tag = 100 + i  # 101, 102, 103 и т.д.
        gmsh.model.addPhysicalGroup(1, boundary_tags, boundary_physical_tag)
        boundary_curves[i] = boundary_tags
        print(f"Границы домена {i}: {boundary_tags}")

    # Устанавливаем параметры сетки
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size_min)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size_max)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Delaunay

    # Устанавливаем порядок элементов = 3 (кубические)
    gmsh.option.setNumber("Mesh.ElementOrder", 3)

    # Генерируем сетку 2D
    gmsh.model.mesh.generate(2)

    # Получаем узлы сетки
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    nodes = node_coords.reshape(-1, 3)
    x = nodes[:, 0]
    y = nodes[:, 1]

    # Получаем элементы (кубические треугольники - тип 21 в GMSH)
    element_types, element_tags, node_tags_elements = gmsh.model.mesh.getElements(2)

    triangles = []
    for i, elem_type in enumerate(element_types):
        if elem_type == 21:  # 21 - кубический треугольник (10 узлов)
            tri_node_tags = node_tags_elements[i]
            triangles = tri_node_tags.reshape(-1, 10) - 1  # 0-based indexing

    # Получаем граничные элементы для каждой области
    boundary_elements = {}
    for i in range(len(surfaces)):  # Для каждого домена
        boundary_physical_tag = 100 + i
        try:
            # Получаем 1D элементы (граничные) для этой физической группы
            boundary_element_types, boundary_element_tags, boundary_node_tags = gmsh.model.mesh.getElements(1,
                                                                                                            boundary_physical_tag)

            boundary_elements_list = []
            for j, elem_type in enumerate(boundary_element_types):
                if elem_type == 26:  # 26 - кубическая линия (4 узла)
                    boundary_nodes = boundary_node_tags[j].reshape(-1, 4) - 1  # 0-based indexing
                    boundary_elements_list.extend(boundary_nodes.tolist())

            boundary_elements[i] = boundary_elements_list
            print(f"Граничные элементы домена {i}: {len(boundary_elements_list)} элементов")
        except Exception as e:
            # Если не удалось получить граничные элементы для этой области
            boundary_elements[i] = []
            print(f"Ошибка получения граничных элементов для домена {i}: {e}")

    # Создаем триангуляцию для визуализации (используем только угловые узлы)
    corner_triangles = triangles[:, [0, 1, 2]] if len(triangles) > 0 else np.array([])
    triangulation = tri.Triangulation(x, y, corner_triangles) if len(corner_triangles) > 0 else None

    # Завершаем работу с GMSH
    gmsh.finalize()

    return nodes, triangles, triangulation, boundary_elements


## ================= Функции рисования ============================================================================
def plot_mesh(triangulation, nodes, boundary_elements=None, title='2D Mesh for Concentric Domains'):
    """
    Отображает сетку с элементами 3-го порядка

    Args:
        triangulation: объект триангуляции matplotlib
        nodes: узлы сетки
        boundary_elements: граничные элементы для выделения
        title: заголовок графика
    """
    x, y = nodes[:, 0], nodes[:, 1]

    plt.figure(figsize=(14, 10))

    # Рисуем триангуляцию (только ребра между угловыми узлами)
    if triangulation is not None:
        plt.triplot(triangulation, 'k-', linewidth=0.5, alpha=0.6)

    # Рисуем все узлы
    plt.plot(x, y, 'o', markersize=1.5, alpha=0.7, color='blue', label='All nodes')

    # Выделяем граничные элементы, если они заданы
    if boundary_elements is not None:
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        for domain_idx, boundary_nodes_list in boundary_elements.items():
            if boundary_nodes_list:
                # Преобразуем в массив и удаляем дубликаты
                all_boundary_nodes = np.unique(np.array(boundary_nodes_list).flatten())
                boundary_x = x[all_boundary_nodes]
                boundary_y = y[all_boundary_nodes]

                color = colors[domain_idx % len(colors)]
                plt.plot(boundary_x, boundary_y, 's', markersize=4,
                         color=color, label=f'Domain {domain_idx} boundary')

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('X coordinate', fontsize=12)
    plt.ylabel('Y coordinate', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()


def plot_mesh_detailed(nodes, elements, boundary_elements=None):
    """
    Детальное отображение сетки с элементами 3-го порядка

    Args:
        nodes: узлы сетки
        elements: элементы (кубические треугольники)
        boundary_elements: граничные элементы
    """
    x, y = nodes[:, 0], nodes[:, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # 1. Общий вид сетки
    if len(elements) > 0:
        # Используем только угловые узлы для триангуляции
        corner_triangles = elements[:, [0, 1, 2]]
        triangulation = tri.Triangulation(x, y, corner_triangles)
        ax1.triplot(triangulation, 'k-', linewidth=0.5, alpha=0.6)

    ax1.plot(x, y, 'o', markersize=1.5, alpha=0.7, color='blue')
    ax1.set_title('Mesh Overview', fontsize=12)
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # 2. Детальный вид с выделением границ
    if len(elements) > 0:
        corner_triangles = elements[:, [0, 1, 2]]
        triangulation = tri.Triangulation(x, y, corner_triangles)
        ax2.triplot(triangulation, 'k-', linewidth=0.3, alpha=0.4)

    # Рисуем все узлы разными цветами в зависимости от типа
    if len(elements) > 0:
        # Угловые узлы
        corner_nodes = np.unique(elements[:, :3].flatten())
        # Узлы на ребрах (порядок 2 и 3)
        edge_nodes = np.unique(elements[:, 3:9].flatten())
        # Центральные узлы (порядок 3)
        center_nodes = np.unique(elements[:, 9:].flatten())

        ax2.plot(x[corner_nodes], y[corner_nodes], 'o', markersize=3,
                 color='blue', label='Corner nodes (order 1)')
        ax2.plot(x[edge_nodes], y[edge_nodes], 's', markersize=2,
                 color='green', label='Edge nodes (order 2-3)')
        ax2.plot(x[center_nodes], y[center_nodes], '^', markersize=2,
                 color='red', label='Center nodes (order 3)')

    # Выделяем граничные элементы
    if boundary_elements is not None:
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        for domain_idx, boundary_nodes_list in boundary_elements.items():
            if boundary_nodes_list:
                all_boundary_nodes = np.unique(np.array(boundary_nodes_list).flatten())
                boundary_x = x[all_boundary_nodes]
                boundary_y = y[all_boundary_nodes]

                color = colors[domain_idx % len(colors)]
                ax2.plot(boundary_x, boundary_y, 'o', markersize=4,
                         markeredgecolor='black', markeredgewidth=0.5,
                         color=color, label=f'Domain {domain_idx} boundary')

    ax2.set_title('Detailed View with Boundary Highlight', fontsize=12)
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    plt.tight_layout()
    plt.show()


def plot_domains_separately(nodes, elements, boundary_elements):
    """
    Отображает каждый домен отдельно для проверки

    Args:
        nodes: узлы сетки
        elements: элементы (кубические треугольники)
        boundary_elements: граничные элементы
    """
    x, y = nodes[:, 0], nodes[:, 1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Цвета для разных доменов
    domain_colors = ['red', 'green', 'blue']
    domain_names = ['Domain 0 (Inner Circle)', 'Domain 1 (Middle Ring)', 'Domain 2 (Outer Ring)']

    for domain_idx in range(3):
        ax = axes[domain_idx]

        # Отображаем все элементы прозрачно
        if len(elements) > 0:
            corner_triangles = elements[:, [0, 1, 2]]
            triangulation = tri.Triangulation(x, y, corner_triangles)
            ax.triplot(triangulation, 'k-', linewidth=0.3, alpha=0.2)

        # Выделяем границы текущего домена
        if domain_idx in boundary_elements and boundary_elements[domain_idx]:
            boundary_nodes = np.unique(np.array(boundary_elements[domain_idx]).flatten())
            boundary_x = x[boundary_nodes]
            boundary_y = y[boundary_nodes]

            color = domain_colors[domain_idx]
            ax.plot(boundary_x, boundary_y, 'o', markersize=4,
                    color=color, label=f'Domain {domain_idx} boundary')

        ax.set_title(domain_names[domain_idx], fontsize=12)
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_domain_fill(nodes, elements, boundary_elements):
    """
    Отображает заполнение каждого домена цветом

    Args:
        nodes: узлы сетки
        elements: элементы (кубические треугольники)
        boundary_elements: граничные элементы
    """
    x, y = nodes[:, 0], nodes[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Цвета для разных доменов
    domain_colors = ['lightblue', 'lightgreen', 'lightcoral']

    # Для простоты отображаем только угловые узлы
    if len(elements) > 0:
        corner_triangles = elements[:, [0, 1, 2]]
        triangulation = tri.Triangulation(x, y, corner_triangles)

        # Визуализируем триангуляцию с цветовой заливкой
        # Для демонстрации просто покажем сетку
        ax.triplot(triangulation, 'k-', linewidth=0.5, alpha=0.6)

    # Выделяем границы доменов
    if boundary_elements is not None:
        colors = ['red', 'green', 'blue']
        for domain_idx, boundary_nodes_list in boundary_elements.items():
            if boundary_nodes_list:
                all_boundary_nodes = np.unique(np.array(boundary_nodes_list).flatten())
                boundary_x = x[all_boundary_nodes]
                boundary_y = y[all_boundary_nodes]

                color = colors[domain_idx % len(colors)]
                ax.plot(boundary_x, boundary_y, 'o', markersize=3,
                        color=color, label=f'Domain {domain_idx} boundary')

    ax.set_title('Mesh with Domain Boundaries', fontsize=14)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    plt.show()