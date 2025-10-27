import numpy as np
from scipy.sparse import csr_matrix, find, block_diag
from structures import BigCompStruct


def AssembleFullMatrices_rigid_SAFE_cubic(CompStruct: BigCompStruct, BasicMatrices, FEMatrices, FullMatrices, ii_int, ii_d1, ii_d2):


    # Поиск граничных элементов
    BoundaryEdges = FEMatrices['BoundaryEdges']
    boundary_mask = BoundaryEdges[2, :] == ii_int
    BElements = BoundaryEdges[0:2, boundary_mask]
    NBElements = BElements.shape[1]

    BNodesFull = []

    for ii_ed in range(NBElements):
        # Поиск начального и конечного узлов ребра
        EdgeNodes = [BElements[0, ii_ed], BElements[1, ii_ed]]

        # Поиск элементов, содержащих эти узлы
        FNodes = np.isin(FEMatrices['DElements'][ii_d1][0:10, :], EdgeNodes)
        NNodes = np.sum(FNodes, axis=0)
        D1EdgeEl = np.argmax(NNodes)

        # Чтение узлов, принадлежащих прилегающим элементам
        TriNodes = FEMatrices['DElements'][ii_d1][0:10, D1EdgeEl]

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

    # Идентификация узлов для удаления
    FEMatrices['BNodesFull'][ii_int] = np.sort(np.unique(BNodesFull))
    FEMatrices['DNodesRem'][ii_d1] = np.unique(np.concatenate([FEMatrices['DNodesRem'][ii_d1], BNodesFull]))
    RemoveNodesPos = np.isin(FEMatrices['DNodes'][ii_d1], FEMatrices['DNodesRem'][ii_d1])
    FEMatrices['DNodesComp'][ii_d1] = FEMatrices['DNodes'][ii_d1][~RemoveNodesPos]

    # Поиск позиций переменных для удаления
    BNodesFullSize = len(FEMatrices['BNodesFull'][ii_int])
    RemoveVarPos = []

    for ii_n in range(BNodesFullSize):
        D1NodePos = np.where(FEMatrices['DNodes'][ii_d1] == FEMatrices['BNodesFull'][ii_int][ii_n])[0]

        if len(D1NodePos) > 0:
            D1NodePos = D1NodePos[0]
            RemoveVarPos_range = np.arange(
                BasicMatrices['Pos'][ii_d1][0] - 1 + CompStruct['Data']['DVarNum'][ii_d1] * D1NodePos,
                BasicMatrices['Pos'][ii_d1][0] - 1 + CompStruct['Data']['DVarNum'][ii_d1] * (D1NodePos + 1)
            )
            RemoveVarPos.extend(RemoveVarPos_range)

    # Обновление позиций переменных для обнуления
    FEMatrices['DZeroVarPos'][ii_d1] = np.unique(np.concatenate([FEMatrices['DZeroVarPos'][ii_d1], RemoveVarPos]))

    return FEMatrices, FullMatrices