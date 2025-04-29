import gmsh
import meshio
import matplotlib.pyplot as plt
import numpy as np

gmsh.initialize()
gmsh.model.add("triangle_mesh")

lc = 1e-1  # characteristic length

# Задание геометрии
p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
p2 = gmsh.model.geo.addPoint(1, 0, 0, lc)
p3 = gmsh.model.geo.addPoint(0.5, 0.866, 0, lc)

l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p2, p3)
l3 = gmsh.model.geo.addLine(p3, p1)

cl = gmsh.model.geo.addCurveLoop([l1, l2, l3])
s = gmsh.model.geo.addPlaneSurface([cl])

# Определение порядка (3 — третий порядок)
gmsh.model.mesh.setOrder(3)

# Опционально: вставка дополнительной точки (например, в центре треугольника)
gmsh.model.geo.synchronize()
gmsh.model.mesh.field.add("Ball", 1)
gmsh.model.mesh.field.setNumber(1, "XCenter", 0.5)
gmsh.model.mesh.field.setNumber(1, "YCenter", 0.288)
gmsh.model.mesh.field.setNumber(1, "Radius", 0.1)
gmsh.model.mesh.field.setNumber(1, "VIn", lc / 5)
gmsh.model.mesh.field.setNumber(1, "VOut", lc)
gmsh.model.mesh.field.setAsBackgroundMesh(1)

# Генерация сетки
gmsh.model.mesh.generate(2)

# Сохранение
gmsh.write("triangle_mesh.msh")

gmsh.finalize()



# Загрузка .msh-файла
mesh = meshio.read("triangle_mesh.msh")

# Получаем координаты узлов
points = mesh.points[:, :2]  # Только x, y координаты

# Получаем элементы: ищем "triangle" (в 2D)  — может быть "triangle6" для 2-го порядка
cells = None
for cell_block in mesh.cells:
    if cell_block.type in ["triangle", "triangle6", "triangle10"]:
        cells = cell_block.data
        break

if cells is None:
    raise ValueError("Нет треугольных элементов в .msh")

# Рисуем сетку
plt.figure(figsize=(8, 8))
for tri in cells:
    polygon = points[tri]
    # замыкаем треугольник
    polygon = np.vstack([polygon, polygon[0]])
    plt.plot(polygon[:, 0], polygon[:, 1], "k-")

plt.gca().set_aspect("equal")
plt.title("2D треугольная сетка из .msh")
plt.xlabel("x")
plt.ylabel("y")
plt.show()