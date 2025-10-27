import numpy as np
from scipy.sparse import csr_matrix, find, block_diag
from structures import BigCompStruct

def RemoveRedundantVariables_SAFE(CompStruct: BigCompStruct, BasicMatrices, FEMatrices, FullMatrices):

    # Инициализация
    FullRemoveNodesArr = []
    FullRemoveVarArr = []
    FullVarArr = np.arange(1, BasicMatrices['Pos'][CompStruct['Data']['N_domain']][1] + 1)

    # Формирование массива переменных, которые должны быть удалены из задачи
    for ii_d in range(CompStruct['Data']['N_domain']):
        if len(FEMatrices['DNodesRem'][ii_d]) > 0:
            VarArr = np.arange(1, CompStruct['Data']['DVarNum'][ii_d] + 1)
            RemoveNodesPos = np.where(np.isin(FEMatrices['DNodes'][ii_d], FEMatrices['DNodesRem'][ii_d]))[0]

            if len(RemoveNodesPos) > 0:
                LargeVarArr = np.tile(VarArr, (len(RemoveNodesPos), 1)).T
                DRemoveVarPos = CompStruct['Data']['DVarNum'][ii_d] * RemoveNodesPos
                DRemoveVarPos = np.tile(DRemoveVarPos, (CompStruct['Data']['DVarNum'][ii_d], 1))
                DRemoveVarPos = DRemoveVarPos + LargeVarArr
                DRemoveVarPos = BasicMatrices['Pos'][ii_d][0] - 1 + DRemoveVarPos.flatten()

                FullRemoveVarArr.extend(DRemoveVarPos)

    FullKeepVarArr = np.where(~np.isin(FullVarArr, FullRemoveVarArr))[0]

    # Обновление полных матриц с учетом удаленных переменных
    FullMatrices['K1Matrix'] = FullMatrices['K1Matrix'][FullKeepVarArr, :][:, FullKeepVarArr]
    FullMatrices['K2Matrix'] = FullMatrices['K2Matrix'][FullKeepVarArr, :][:, FullKeepVarArr]
    FullMatrices['K3Matrix'] = FullMatrices['K3Matrix'][FullKeepVarArr, :][:, FullKeepVarArr]
    FullMatrices['MMatrix'] = FullMatrices['MMatrix'][FullKeepVarArr, :][:, FullKeepVarArr]
    FullMatrices['PMatrix'] = FullMatrices['PMatrix'][FullKeepVarArr, :][:, FullKeepVarArr]

    return FEMatrices, FullMatrices