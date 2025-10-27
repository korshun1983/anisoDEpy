import numpy as np
from scipy.sparse import csr_matrix, find, block_diag
from structures import BigCompStruct
from matrix_functions import basicMatrices


def AssembleFullMatrices_fs_SAFE_cubic(CompStruct: BigCompStruct, BasicMatrices: basicMatrices, FEMatrices, FullMatrices, ii_int, ii_d1, ii_d2):


    # Добавление матриц следующего блока
    FullMatrices['K1Matrix'] = block_diag([FullMatrices['K1Matrix'], FEMatrices['K1Matrix_d'][ii_d2]])
    FullMatrices['K2Matrix'] = block_diag([FullMatrices['K2Matrix'], FEMatrices['K2Matrix_d'][ii_d2]])
    FullMatrices['K3Matrix'] = block_diag([FullMatrices['K3Matrix'], FEMatrices['K3Matrix_d'][ii_d2]])
    FullMatrices['MMatrix'] = block_diag([FullMatrices['MMatrix'], FEMatrices['MMatrix_d'][ii_d2]])

    # Специальная обработка PMatrix с интерфейсными матрицами
    curFullMatSize = FullMatrices['PMatrix'].shape
    curMatD12Size = FEMatrices['PMatrixD12'][ii_int].shape

    curZeroMat12Size = (curFullMatSize[0] - curMatD12Size[0], curMatD12Size[1])
    ZeroMatrix12 = csr_matrix(curZeroMat12Size)
    InsertPMatrixD12 = np.vstack([ZeroMatrix12, FEMatrices['PMatrixD12'][ii_int]])

    curMatD21Size = FEMatrices['PMatrixD21'][ii_int].shape
    curZeroMat21Size = (curMatD21Size[0], curFullMatSize[1] - curMatD21Size[1])
    ZeroMatrix21 = csr_matrix(curZeroMat21Size)
    InsertPMatrixD21 = np.hstack([ZeroMatrix21, FEMatrices['PMatrixD21'][ii_int]])

    FullMatrices['PMatrix'] = np.block([
        [FullMatrices['PMatrix'], InsertPMatrixD12],
        [InsertPMatrixD21, FEMatrices['PMatrix_d'][ii_d2]]
    ])

    # Обновление списка узлов для удаления
    FEMatrices['DNodesRem'][ii_d2] = np.unique(FEMatrices['DNodesRem'][ii_d2])
    RemoveNodesPos = np.isin(FEMatrices['DNodes'][ii_d2], FEMatrices['DNodesRem'][ii_d2])
    FEMatrices['DNodesComp'][ii_d2] = FEMatrices['DNodes'][ii_d2][~RemoveNodesPos]

    return FEMatrices, FullMatrices
