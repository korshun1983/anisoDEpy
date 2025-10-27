import numpy as np
from scipy.sparse import csr_matrix, find, block_diag
from structures import BigCompStruct
from matrix_functions import basicMatrices

def AssembleFullMatrices_ff_ss_SAFE_cubic(CompStruct: BigCompStruct, BasicMatrices: basicMatrices, FEMatrices, FullMatrices, ii_int, ii_d1, ii_d2):
    """
    Собирает полные матрицы для интерфейсов жидкость-жидкость и твердое-твердое

    Parameters:
    -----------
    CompStruct : dict
        Структура с параметрами модели
    BasicMatrices : dict
        Базовые матрицы
    FEMatrices : dict
        Матрицы МКЭ
    FullMatrices : dict
        Полные матрицы системы
    ii_int : int
        Номер интерфейса
    ii_d1 : int
        Номер первого домена
    ii_d2 : int
        Номер второго домена

    Returns:
    --------
    tuple
        (FEMatrices, FullMatrices) - обновленные матрицы
    """

    # Добавление матриц следующего блока
    FullMatrices['K1Matrix'] = block_diag([FullMatrices['K1Matrix'], FEMatrices['K1Matrix_d'][ii_d2]])
    FullMatrices['K2Matrix'] = block_diag([FullMatrices['K2Matrix'], FEMatrices['K2Matrix_d'][ii_d2]])
    FullMatrices['K3Matrix'] = block_diag([FullMatrices['K3Matrix'], FEMatrices['K3Matrix_d'][ii_d2]])
    FullMatrices['MMatrix'] = block_diag([FullMatrices['MMatrix'], FEMatrices['MMatrix_d'][ii_d2]])
    FullMatrices['PMatrix'] = block_diag([FullMatrices['PMatrix'], FEMatrices['PMatrix_d'][ii_d2]])

    # Обновление списка узлов для удаления
    FEMatrices['DNodesRem'][ii_d2] = np.unique(
        np.concatenate([FEMatrices['DNodesRem'][ii_d2], FEMatrices['BNodesFull'][ii_int]]))
    RemoveNodesPos = np.isin(FEMatrices['DNodes'][ii_d2], FEMatrices['DNodesRem'][ii_d2])
    FEMatrices['DNodesComp'][ii_d2] = FEMatrices['DNodes'][ii_d2][~RemoveNodesPos]

    # Поиск позиций переменных для сложения и удаления
    BNodesFullSize = len(FEMatrices['BNodesFull'][ii_int])
    AddVarPos = []
    RemoveVarPos = []

    for ii_n in range(BNodesFullSize):
        D1NodePos = np.where(FEMatrices['DNodes'][ii_d1] == FEMatrices['BNodesFull'][ii_int][ii_n])[0]
        D2NodePos = np.where(FEMatrices['DNodes'][ii_d2] == FEMatrices['BNodesFull'][ii_int][ii_n])[0]

        if len(D1NodePos) > 0 and len(D2NodePos) > 0:
            D1NodePos = D1NodePos[0]
            D2NodePos = D2NodePos[0]

            AddVarPos_range = np.arange(
                BasicMatrices['Pos'][ii_d1][0] - 1 + CompStruct['Data']['DVarNum'][ii_d1] * D1NodePos,
                BasicMatrices['Pos'][ii_d1][0] - 1 + CompStruct['Data']['DVarNum'][ii_d1] * (D1NodePos + 1)
            )
            RemoveVarPos_range = np.arange(
                BasicMatrices['Pos'][ii_d2][0] - 1 + CompStruct['Data']['DVarNum'][ii_d2] * D2NodePos,
                BasicMatrices['Pos'][ii_d2][0] - 1 + CompStruct['Data']['DVarNum'][ii_d2] * (D2NodePos + 1)
            )

            AddVarPos.extend(AddVarPos_range)
            RemoveVarPos.extend(RemoveVarPos_range)

    AddVarPos = np.array(AddVarPos)
    RemoveVarPos = np.array(RemoveVarPos)

    # Обновление позиций переменных
    FEMatrices['DTakeFromVarPos'][ii_d2] = np.unique(np.concatenate([FEMatrices['DTakeFromVarPos'][ii_d2], AddVarPos]))
    FEMatrices['DPutToVarPos'][ii_d2] = np.unique(np.concatenate([FEMatrices['DPutToVarPos'][ii_d2], RemoveVarPos]))

    # Суммирование строк и столбцов для совпадающих узлов
    FullMatrices['K1Matrix'][AddVarPos, :] += FullMatrices['K1Matrix'][RemoveVarPos, :]
    FullMatrices['K1Matrix'][:, AddVarPos] += FullMatrices['K1Matrix'][:, RemoveVarPos]

    FullMatrices['K2Matrix'][AddVarPos, :] += FullMatrices['K2Matrix'][RemoveVarPos, :]
    FullMatrices['K2Matrix'][:, AddVarPos] += FullMatrices['K2Matrix'][:, RemoveVarPos]

    FullMatrices['K3Matrix'][AddVarPos, :] += FullMatrices['K3Matrix'][RemoveVarPos, :]
    FullMatrices['K3Matrix'][:, AddVarPos] += FullMatrices['K3Matrix'][:, RemoveVarPos]

    FullMatrices['MMatrix'][AddVarPos, :] += FullMatrices['MMatrix'][RemoveVarPos, :]
    FullMatrices['MMatrix'][:, AddVarPos] += FullMatrices['MMatrix'][:, RemoveVarPos]

    FullMatrices['PMatrix'][AddVarPos, :] += FullMatrices['PMatrix'][RemoveVarPos, :]
    FullMatrices['PMatrix'][:, AddVarPos] += FullMatrices['PMatrix'][:, RemoveVarPos]

    # Обнуление элементов, которые будут удалены
    FullMatrices['K1Matrix'][RemoveVarPos, :] = 0
    FullMatrices['K1Matrix'][:, RemoveVarPos] = 0

    FullMatrices['K2Matrix'][RemoveVarPos, :] = 0
    FullMatrices['K2Matrix'][:, RemoveVarPos] = 0

    FullMatrices['K3Matrix'][RemoveVarPos, :] = 0
    FullMatrices['K3Matrix'][:, RemoveVarPos] = 0

    FullMatrices['MMatrix'][RemoveVarPos, :] = 0
    FullMatrices['MMatrix'][:, RemoveVarPos] = 0

    FullMatrices['PMatrix'][RemoveVarPos, :] = 0
    FullMatrices['PMatrix'][:, RemoveVarPos] = 0

    return FEMatrices, FullMatrices