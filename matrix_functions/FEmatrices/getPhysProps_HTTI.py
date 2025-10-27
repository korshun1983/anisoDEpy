import numpy as np
from structures import BigCompStruct

def getPhysProps_HTTI(CompStruct: BigCompStruct, BasicMatrices, FEMatrices, ii_d, ii_el):


    # Инициализация
    N_nodes = CompStruct['Advanced']['N_nodes']

    # Выделение памяти для матриц
    ElPhysProps = {}
    ElPhysProps['CijMatrix'] = np.zeros((N_nodes, 6, 6))
    ElPhysProps['RhoVec'] = np.zeros(N_nodes)

    # Получение упругих модулей и плотности для домена
    Cij_6x6 = FEMatrices['PhysProp'][ii_d]['c_ij'] * 1.e9
    RhoValue = FEMatrices['PhysProp'][ii_d]['rho'] * 1.e3

    # Обработка всех узлов элемента
    for ii in range(N_nodes):
        ElPhysProps['RhoVec'][ii] = RhoValue
        ElPhysProps['CijMatrix'][ii, :, :] = Cij_6x6

    return ElPhysProps