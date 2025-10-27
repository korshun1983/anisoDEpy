import numpy as np
from scipy.sparse import csr_matrix, find, block_diag
from structures import BigCompStruct

def ICMatrices_fluid_HTTI_SAFE_cubic(CompStruct: BigCompStruct, BasicMatrices, FEMatrices, ii_int, ii_d1, ii_d2):

    # Инициализация
    D1NodesNum = len(FEMatrices['DNodes'][ii_d1])
    D1FEVarNum = CompStruct['Data']['DVarNum'][ii_d1] * D1NodesNum
    D2NodesNum = len(FEMatrices['DNodes'][ii_d2])
    D2FEVarNum = CompStruct['Data']['DVarNum'][ii_d2] * D2NodesNum

    # Создание нулевых разреженных матриц
    FEMatrices['ZeroD12'] = csr_matrix((D1FEVarNum, D2FEVarNum))
    FEMatrices['ZeroD21'] = csr_matrix((D2FEVarNum, D1FEVarNum))

    # Определение типа доменов
    if CompStruct['Model']['DomainType'][ii_d1] == 'fluid':
        ii_df = ii_d1
        ii_ds = ii_d2
    elif CompStruct['Model']['DomainType'][ii_d1] == 'HTTI':
        ii_df = ii_d2
        ii_ds = ii_d1

    # Граничные элементы в 2D формулировке - это ребра
    BoundaryEdges = FEMatrices['BoundaryEdges']
    boundary_mask = BoundaryEdges[2, :] == ii_int
    BElements = BoundaryEdges[0:1, boundary_mask]
    NBElements = BElements.shape[1]

    BNodesFull = []

    DfNodesNum = len(FEMatrices['DNodes'][ii_df])
    DfFEVarNum = DfNodesNum
    DsNodesNum = len(FEMatrices['DNodes'][ii_ds])
    DsFEVarNum = CompStruct['Data']['DVarNum'][ii_ds] * DsNodesNum
    BFEVarNumFluid = DfFEVarNum
    BFEVarNumSolid = DsFEVarNum

    # Создание нулевой разреженной матрицы для граничных условий
    BMatrixDfs = csr_matrix((BFEVarNumFluid, BFEVarNumSolid))

    # Обработка всех ребер, принадлежащих границе
    for ii_ed in range(NBElements):
        # Поиск начального и конечного узлов ребра
        EdgeNodes = []
        if CompStruct['Model']['DomainType'][ii_d1] == 'fluid':
            EdgeNodes = [BElements[0, ii_ed], BElements[1, ii_ed]]
        elif CompStruct['Model']['DomainType'][ii_d1] == 'HTTI':
            EdgeNodes = [BElements[1, ii_ed], BElements[0, ii_ed]]

        # Поиск элементов, содержащих эти узлы
        FNodes_df = np.isin(FEMatrices['DElements'][ii_df][0:10, :], EdgeNodes[0:2])
        NNodes_df = np.sum(FNodes_df, axis=0)
        DfEdgeEl = np.argmax(NNodes_df)

        FNodes_ds = np.isin(FEMatrices['DElements'][ii_ds][0:10, :], EdgeNodes[0:2])
        NNodes_ds = np.sum(FNodes_ds, axis=0)
        DsEdgeEl = np.argmax(NNodes_ds)

        # Чтение узлов, принадлежащих прилегающим элементам
        TriNodes_f = FEMatrices['DElements'][ii_df][0:10, DfEdgeEl]
        TriNodes_s = FEMatrices['DElements'][ii_ds][0:10, DsEdgeEl]

        # Поиск промежуточных узлов для жидкостного домена
        EdgeNodesPos_Df = [
            np.where(TriNodes_f == EdgeNodes[0])[0][0],
            np.where(TriNodes_f == EdgeNodes[1])[0][0]
        ]

        # Поиск внутренних узлов элемента ребра
        if EdgeNodesPos_Df[1] > (EdgeNodesPos_Df[0] % 3):
            EdgeNodesPos_Df.extend([(EdgeNodesPos_Df[0] + 1) * 2, (EdgeNodesPos_Df[0] + 1) * 2 + 1])
        else:
            EdgeNodesPos_Df.extend([(EdgeNodesPos_Df[1] + 1) * 2 + 1, (EdgeNodesPos_Df[1] + 1) * 2])

        EdgeNodes.extend([TriNodes_f[EdgeNodesPos_Df[2]], TriNodes_f[EdgeNodesPos_Df[3]]])

        # Поиск позиций узлов в твердом домене
        EdgeNodesPos_Ds = [
            np.where(TriNodes_s == EdgeNodes[0])[0][0],
            np.where(TriNodes_s == EdgeNodes[1])[0][0],
            np.where(TriNodes_s == EdgeNodes[2])[0][0],
            np.where(TriNodes_s == EdgeNodes[3])[0][0]
        ]

        BNodesFull.extend(EdgeNodes)

        # Поиск координат граничного элемента (ребра)
        EdgeNodesX = FEMatrices['MeshNodes'][0, EdgeNodes[0:2]]  # x координаты
        EdgeNodesY = FEMatrices['MeshNodes'][1, EdgeNodes[0:2]]  # y координаты

        # Чтение физических свойств элемента
        ElPhysProps = {}
        ElPhysProps['DfEl'] = CompStruct['Methods']['getPhysProps'][ii_df](
            CompStruct, BasicMatrices, FEMatrices, ii_df, DfEdgeEl)
        ElPhysProps['DsEl'] = CompStruct['Methods']['getPhysProps'][ii_ds](
            CompStruct, BasicMatrices, FEMatrices, ii_ds, DsEdgeEl)

        # Поиск компонент вектора ребра
        EdgeProps = {}
        EdgeProps['DxEdge'] = EdgeNodesX[1] - EdgeNodesX[0]
        EdgeProps['DyEdge'] = EdgeNodesY[1] - EdgeNodesY[0]
        EdgeProps['Dl'] = np.sqrt(EdgeProps['DxEdge'] ** 2 + EdgeProps['DyEdge'] ** 2)

        # Определение ориентированной нормали к граничному ребру
        edge_vector = np.array([EdgeProps['DxEdge'] / EdgeProps['Dl'], EdgeProps['DyEdge'] / EdgeProps['Dl'], 0])
        z_axis = np.array([0, 0, -1])
        normal = np.cross(edge_vector, z_axis)
        EdgeProps['Normal'] = normal

        # Вычисление вкладов элемента в матрицы жесткости и масс
        BMatrixDfs_edge, BMatrixDsf_edge = CompStruct['Methods']['IC_matrix'][ii_int](
            BasicMatrices, CompStruct, ii_int, ii_df, ii_ds, ElPhysProps, EdgeProps,
            {'Df': EdgeNodesPos_Df, 'Ds': EdgeNodesPos_Ds})

        # Идентификация позиций узлов элемента и переменных в соответствующих матрицах
        VarRowArr = []
        VarColArr = []

        for ii_n in range(4):
            DfNodePos = np.where(FEMatrices['DNodes'][ii_df] == EdgeNodes[ii_n])[0]
            DsNodePos = np.where(FEMatrices['DNodes'][ii_ds] == EdgeNodes[ii_n])[0]

            if len(DfNodePos) > 0 and len(DsNodePos) > 0:
                DfNodePos = DfNodePos[0]
                DsNodePos = DsNodePos[0]

                VarRowArr.extend(range(
                    CompStruct['Data']['DVarNum'][ii_df] * DfNodePos,
                    CompStruct['Data']['DVarNum'][ii_df] * (DfNodePos + 1)
                ))
                VarColArr.extend(range(
                    CompStruct['Data']['DVarNum'][ii_ds] * DsNodePos,
                    CompStruct['Data']['DVarNum'][ii_ds] * (DsNodePos + 1)
                ))

        # Подготовка разреженных матриц для вставки в матрицы домена
        VarRows = np.tile(VarRowArr, CompStruct['Data']['DVarNum'][ii_ds] * 4)
        VarCols = np.repeat(VarColArr, CompStruct['Data']['DVarNum'][ii_df] * 4)

        # Создание векторов элементов для вставки
        strBMatrixDfs_el = BMatrixDfs_edge.flatten()

        # Создание разреженных матриц
        spBMatrixDfs_el = csr_matrix(
            (strBMatrixDfs_el, (VarRows, VarCols)),
            shape=(BFEVarNumFluid, BFEVarNumSolid)
        )

        # Вставка матриц элемента в матрицы домена
        BMatrixDfs += spBMatrixDfs_el

    # Формирование полного списка узлов, принадлежащих интерфейсу
    FEMatrices['BNodesFull'][ii_int] = np.sort(np.unique(BNodesFull))

    # Вычисление матрицы граничных условий твердое-жидкость
    BMatrixDsf = BMatrixDfs.T

    # Назначение прямоугольных блоков матриц, ответственных за граничные условия
    if CompStruct['Model']['DomainType'][ii_d1] == 'fluid':
        FEMatrices['PMatrixD12'][ii_int] = BMatrixDfs
        FEMatrices['PMatrixD21'][ii_int] = BMatrixDsf
    elif CompStruct['Model']['DomainType'][ii_d1] == 'HTTI':
        FEMatrices['PMatrixD12'][ii_int] = -BMatrixDsf
        FEMatrices['PMatrixD21'][ii_int] = -BMatrixDfs

    return FEMatrices