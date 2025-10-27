import numpy as np
from structures import BigCompStruct


def IC_matrix_FS(BasicMatrices, CompStruct: BigCompStruct, ii_int, ii_df, ii_ds, ElPhysProps, EdgeProps, EdgeNodesPos):

    # Инициализация
    NEdge_nodes = CompStruct['Advanced']['NEdge_nodes']
    Dl = EdgeProps['Dl']

    # Выделение памяти для матриц
    BMatrixDfs_edge = np.zeros((CompStruct['Data']['DVarNum'][ii_df] * NEdge_nodes,
                                CompStruct['Data']['DVarNum'][ii_ds] * NEdge_nodes))
    BMatrixDsf_edge = np.zeros((CompStruct['Data']['DVarNum'][ii_ds] * NEdge_nodes,
                                CompStruct['Data']['DVarNum'][ii_df] * NEdge_nodes))

    # Вычисление матриц элемента интерфейса
    ElMatrices = CompStruct['Methods']['IC_el_matrix'][ii_int](
        BasicMatrices, CompStruct, ElPhysProps, EdgeProps, EdgeNodesPos)

    # Формирование матриц интерфейсных условий
    BMatrixDfs_edge = Dl * ElMatrices['NfltRhonNMatrix']
    BMatrixDsf_edge = Dl * ElMatrices['NfltRhonNMatrix'].T

    return BMatrixDfs_edge, BMatrixDsf_edge