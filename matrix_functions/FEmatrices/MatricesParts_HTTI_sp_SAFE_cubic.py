import numpy as np
from scipy.sparse import csr_matrix, find, block_diag
from structures import BigCompStruct

def MatricesParts_HTTI_sp_SAFE_cubic(CompStruct: BigCompStruct, BasicMatrices, FEMatrices, ii_d):


    # Инициализация
    NDomainElements = FEMatrices['DElements'][ii_d].shape[1]
    DNodesNum = len(FEMatrices['DNodes'][ii_d])
    DFEVarNum = CompStruct['Data']['DVarNum'][ii_d] * DNodesNum
    VarVecArr = np.zeros(CompStruct['Data']['DVarNum'][ii_d] * CompStruct['Advanced']['N_nodes'], dtype=int)

    # Создание нулевых разреженных матриц
    FEMatrices['K1Matrix_d'][ii_d] = csr_matrix((DFEVarNum, DFEVarNum))
    FEMatrices['K2Matrix_d'][ii_d] = csr_matrix((DFEVarNum, DFEVarNum))
    FEMatrices['B1tCB2Matrix_d'][ii_d] = csr_matrix((DFEVarNum, DFEVarNum))
    FEMatrices['B2tCB1Matrix_d'][ii_d] = csr_matrix((DFEVarNum, DFEVarNum))
    FEMatrices['K3Matrix_d'][ii_d] = csr_matrix((DFEVarNum, DFEVarNum))
    FEMatrices['MMatrix_d'][ii_d] = csr_matrix((DFEVarNum, DFEVarNum))
    FEMatrices['PMatrix_d'][ii_d] = csr_matrix((DFEVarNum, DFEVarNum))

    # Обработка всех элементов, принадлежащих домену
    for ii_el in range(NDomainElements):
        # Чтение номеров узлов, принадлежащих элементу
        TriNodes = FEMatrices['DElements'][ii_d][0:9, ii_el]

        # Идентификация позиций узлов элемента и переменных в соответствующих матрицах
        for ii_n in range(10):
            DNodePos = np.where(FEMatrices['DNodes'][ii_d] == TriNodes[ii_n])[0]
            if len(DNodePos) > 0:
                DNodePos = DNodePos[0]
                start_idx = CompStruct['Data']['DVarNum'][ii_d] * ii_n
                end_idx = CompStruct['Data']['DVarNum'][ii_d] * (ii_n + 1)
                VarVecArr[start_idx:end_idx] = range(
                    CompStruct['Data']['DVarNum'][ii_d] * DNodePos,
                    CompStruct['Data']['DVarNum'][ii_d] * (DNodePos + 1)
                )

        # Чтение физических свойств элемента
        ElPhysProps = CompStruct['Methods']['getPhysProps'][ii_d](
            CompStruct, BasicMatrices, FEMatrices, ii_d, ii_el)

        # Чтение свойств элемента
        TriProps = {}
        TriProps['a'] = FEMatrices['DEMeshProps'][ii_d]['a'][:, ii_el]
        TriProps['b'] = FEMatrices['DEMeshProps'][ii_d]['b'][:, ii_el]
        TriProps['c'] = FEMatrices['DEMeshProps'][ii_d]['c'][:, ii_el]
        TriProps['delta'] = FEMatrices['DEMeshProps'][ii_d]['delta'][:, ii_el]

        # Вычисление векторов, входящих в выражения для матриц производных функций формы
        TriProps['dxL'] = 0.5 * TriProps['b']
        TriProps['dyL'] = 0.5 * TriProps['c']

        # Вычисление вкладов элемента в матрицы жесткости и масс
        ElMatrices = CompStruct['Methods']['KM_matrix'][ii_d](
            BasicMatrices, CompStruct, FEMatrices, ii_d, ii_el, ElPhysProps, TriProps)

        # Подготовка разреженных матриц для вставки в матрицы домена
        VarRows = np.tile(VarVecArr, 30)
        VarCols = np.repeat(VarVecArr, 30)

        # Создание векторов элементов для вставки
        strK1Matrix_el = ElMatrices['K1Matrix'].flatten()
        strK2Matrix_el = ElMatrices['K2Matrix'].flatten()
        strB1tCB2Matrix_el = ElMatrices['B1tCB2Matrix'].flatten()
        strB2tCB1Matrix_el = ElMatrices['B2tCB1Matrix'].flatten()
        strK3Matrix_el = ElMatrices['K3Matrix'].flatten()
        strMMatrix_el = ElMatrices['MMatrix'].flatten()

        # Создание разреженных матриц
        spK1Matrix_el = csr_matrix((strK1Matrix_el, (VarRows, VarCols)), shape=(DFEVarNum, DFEVarNum))
        spK2Matrix_el = csr_matrix((strK2Matrix_el, (VarRows, VarCols)), shape=(DFEVarNum, DFEVarNum))
        spB1tCB2Matrix_el = csr_matrix((strB1tCB2Matrix_el, (VarRows, VarCols)), shape=(DFEVarNum, DFEVarNum))
        spB2tCB1Matrix_el = csr_matrix((strB2tCB1Matrix_el, (VarRows, VarCols)), shape=(DFEVarNum, DFEVarNum))
        spK3Matrix_el = csr_matrix((strK3Matrix_el, (VarRows, VarCols)), shape=(DFEVarNum, DFEVarNum))
        spMMatrix_el = csr_matrix((strMMatrix_el, (VarRows, VarCols)), shape=(DFEVarNum, DFEVarNum))

        # Вставка матриц элемента в матрицы домена
        FEMatrices['K1Matrix_d'][ii_d] += spK1Matrix_el
        FEMatrices['K2Matrix_d'][ii_d] += spK2Matrix_el
        FEMatrices['B1tCB2Matrix_d'][ii_d] += spB1tCB2Matrix_el
        FEMatrices['B2tCB1Matrix_d'][ii_d] += spB2tCB1Matrix_el
        FEMatrices['K3Matrix_d'][ii_d] += spK3Matrix_el
        FEMatrices['MMatrix_d'][ii_d] += spMMatrix_el

    return FEMatrices
