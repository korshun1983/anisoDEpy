import numpy as np
from scipy.sparse import csr_matrix, find, block_diag
from structures import BigCompStruct

def ICMatrices_ff_ss_SAFE_cubic(CompStruct: BigCompStruct, BasicMatrices, FEMatrices, ii_int, ii_d1, ii_d2):

    # Инициализация
    D1NodesNum = len(FEMatrices['DNodes'][ii_d1])
    D1FEVarNum = CompStruct['Data']['DVarNum'][ii_d1] * D1NodesNum
    D2NodesNum = len(FEMatrices['DNodes'][ii_d2])
    D2FEVarNum = CompStruct['Data']['DVarNum'][ii_d2] * D2NodesNum

    # Создание нулевых разреженных матриц
    FEMatrices['ZeroD12'] = csr_matrix((D1FEVarNum, D2FEVarNum))
    FEMatrices['ZeroD21'] = csr_matrix((D1FEVarNum, D2FEVarNum))

    # Подготовка интерфейсных элементов
    BoundaryEdges = FEMatrices['BoundaryEdges']
    boundary_mask = BoundaryEdges[2, :] == ii_int
    BElements = BoundaryEdges[0:2, boundary_mask]
    NBElements = BElements.shape[1]

    BNodesFull = []

    # Обработка всех ребер, принадлежащих границе
    for ii_ed in range(NBElements):
        # Поиск начального и конечного узлов ребра
        EdgeNodes = [BElements[0, ii_ed], BElements[1, ii_ed]]

        # Поиск элементов, содержащих эти узлы
        FNodes = np.isin(FEMatrices['DElements'][ii_d1][0:9, :], EdgeNodes)
        NNodes = np.sum(FNodes, axis=0)
        D1EdgeEl = np.argmax(NNodes)

        # Чтение узлов, принадлежащих прилегающим элементам
        TriNodes = FEMatrices['DElements'][ii_d1][0:9, D1EdgeEl]

        # Поиск промежуточных узлов
        EdgeNodesPos_D1 = [
            np.where(TriNodes == EdgeNodes[0])[0][0],
            np.where(TriNodes == EdgeNodes[1])[0][0]
        ]

        # Поиск внутренних узлов элемента ребра
        if EdgeNodesPos_D1[1] > (EdgeNodesPos_D1[0] % 3):
            EdgeNodesPos_D1.extend([(EdgeNodesPos_D1[0] + 1) * 2, (EdgeNodesPos_D1[0] + 1) * 2 + 1])
        else:
            EdgeNodesPos_D1.extend([(EdgeNodesPos_D1[1] + 1) * 2 + 1, (EdgeNodesPos_D1[1] + 1) * 2])

        EdgeNodes.extend([TriNodes[EdgeNodesPos_D1[2]], TriNodes[EdgeNodesPos_D1[3]]])
        BNodesFull.extend(EdgeNodes)

    # Формирование полного списка узлов, принадлежащих интерфейсу
    FEMatrices['BNodesFull'][ii_int] = np.sort(np.unique(BNodesFull))

    return FEMatrices
