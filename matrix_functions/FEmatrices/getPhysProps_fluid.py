import numpy as np
from scipy.sparse import csr_matrix, find, block_diag
from structures import BigCompStruct

def getPhysProps_fluid(CompStruct: BigCompStruct, BasicMatrices, FEMatrices, ii_d, ii_el):

    # Инициализация
    N_nodes = CompStruct['Advanced']['N_nodes']

    # Выделение памяти для матриц
    ElPhysProps = {}
    ElPhysProps['LambdaVec'] = np.zeros(N_nodes)
    ElPhysProps['RhoVec'] = np.zeros(N_nodes)
    ElPhysProps['Rho2LambdaVec'] = np.zeros(N_nodes)

    # Получение упругих модулей и плотности для домена
    LambdaValue = FEMatrices['PhysProp'][ii_d]['lambda'] * 1.e9
    RhoValue = FEMatrices['PhysProp'][ii_d]['rho'] * 1.e3
    Rho2LambdaValue = RhoValue ** 2 / LambdaValue

    # Обработка всех узлов элемента
    for ii in range(N_nodes):
        ElPhysProps['RhoVec'][ii] = RhoValue
        ElPhysProps['LambdaVec'][ii] = LambdaValue
        ElPhysProps['Rho2LambdaVec'][ii] = Rho2LambdaValue

    return ElPhysProps