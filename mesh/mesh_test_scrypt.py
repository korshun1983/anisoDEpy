import numpy as np
from structures import BigCompStruct
from mesh_generator import create_mesh, plot_mesh, plot_mesh_detailed, plot_domains_separately, plot_domain_fill


# Создаем и настраиваем структуру параметров
CompStruct = BigCompStruct()

# Устанавливаем параметры областей
CompStruct.Model.DomainRx = [0.1000, 8.7796, 13.1195]
CompStruct.Model.DomainRy = [0.1000, 8.7796, 13.1195]
CompStruct.Data.N_domain = 3
CompStruct.Model.DomainTheta = [0, 0, 0]
CompStruct.Model.AddDomainLoc = 'ext'
CompStruct.Model.DomainEcc = [0, 0, 0]
CompStruct.Model.DomainEccAngle = [0, 0, 0]

print("Создание сетки с тремя доменами и элементами 3-го порядка...")
print(f"Радиусы доменов: {CompStruct.Model.DomainRx}")

# Создаем сетку
nodes, elements, triangulation, boundary_elements = create_mesh(CompStruct)

print(f"Сетка создана: {len(nodes)} узлов, {len(elements)} элементов")
print("\nГраничные элементы по областям:")
for domain_idx, boundary_nodes in boundary_elements.items():
    print(f"  Область {domain_idx}: {len(boundary_nodes)} граничных элементов")

# Отображаем общую сетку
plot_mesh(triangulation, nodes, boundary_elements,
          '2D Mesh with Three Domains and Cubic Elements (Order 3)')

# Детальное отображение
plot_mesh_detailed(nodes, elements, boundary_elements)

# Отдельное отображение каждого домена
plot_domains_separately(nodes, elements, boundary_elements)

# Отображение с заполнением
plot_domain_fill(nodes, elements, boundary_elements)

# Дополнительная информация об элементах
if len(elements) > 0:
    print(f"\nИнформация об элементах:")
    print(f"  Узлов на элемент: {elements.shape[1]}")
    print(f"  Угловые узлы: индексы 0-2")
    print(f"  Узлы на ребрах: индексы 3-8")
    print(f"  Центральные узлы: индексы 9")
    print(f"  Всего узлов разных типов:")
    all_nodes = elements.flatten()
    unique_nodes = np.unique(all_nodes)
    print(f"    Уникальных узлов: {len(unique_nodes)}")