from scipy.sparse import csr_matrix, find, block_diag
from structures import BigCompStruct
from matrix_functions import basicMatrices



def AssembleFullFEMatrices_sp_SAFE(CompStruct: BigCompStruct, BasicMatrices: basicMatrices, FEMatrices):

    FullMatrices = {}
    N_domain = 1  # предполагаем, должен быть определен из CompStruct

    # Инициализация разреженных матриц
    DVarNum = CompStruct['Data']['DVarNum'][N_domain][1]  # размер матриц
    FullMatrices['K1Matrix'] = csr_matrix((DVarNum, DVarNum))
    FullMatrices['K2Matrix'] = csr_matrix((DVarNum, DVarNum))
    FullMatrices['K3Matrix'] = csr_matrix((DVarNum, DVarNum))
    FullMatrices['MMatrix'] = csr_matrix((DVarNum, DVarNum))

    BCMatrices = {}
    BCMatrices['ICBCMatrix'] = csr_matrix((DVarNum, DVarNum))

    # Временные переменные для матриц доменов (должны быть определены)
    K1Matrix_d = FEMatrices.get('K1Matrix_d', {})
    K2Matrix_d = FEMatrices.get('K2Matrix_d', {})
    K3Matrix_d = FEMatrices.get('K3Matrix_d', {})
    MMatrix_d = FEMatrices.get('MMatrix_d', {})

    for ii_d in range(1, CompStruct['Data']['N_domains'] + 1):
        # Вставка блоков матриц в полные матрицы
        if ii_d in K1Matrix_d:
            DVecRepRow, DVecRepCol, DVecRepV = find(K1Matrix_d[ii_d])
            DVecRepRow += CompStruct['Data']['DVarNum'][ii_d][0] - 1
            DVecRepCol += CompStruct['Data']['DVarNum'][ii_d][0] - 1
            DSize = K1Matrix_d[ii_d].shape
            FullMatrices['K1Matrix'] += csr_matrix((DVecRepV, (DVecRepRow, DVecRepCol)), shape=DSize)

        if ii_d in K2Matrix_d:
            DVecRepRow, DVecRepCol, DVecRepV = find(K2Matrix_d[ii_d])
            DVecRepRow += CompStruct['Data']['DVarNum'][ii_d][0] - 1
            DVecRepCol += CompStruct['Data']['DVarNum'][ii_d][0] - 1
            DSize = K2Matrix_d[ii_d].shape
            FullMatrices['K2Matrix'] += csr_matrix((DVecRepV, (DVecRepRow, DVecRepCol)), shape=DSize)

        if ii_d in K3Matrix_d:
            DVecRepRow, DVecRepCol, DVecRepV = find(K3Matrix_d[ii_d])
            DVecRepRow += CompStruct['Data']['DVarNum'][ii_d][0] - 1
            DVecRepCol += CompStruct['Data']['DVarNum'][ii_d][0] - 1
            DSize = K3Matrix_d[ii_d].shape
            FullMatrices['K3Matrix'] += csr_matrix((DVecRepV, (DVecRepRow, DVecRepCol)), shape=DSize)

        if ii_d in MMatrix_d:
            DVecRepRow, DVecRepCol, DVecRepV = find(MMatrix_d[ii_d])
            DVecRepRow += CompStruct['Data']['DVarNum'][ii_d][0] - 1
            DVecRepCol += CompStruct['Data']['DVarNum'][ii_d][0] - 1
            DSize = MMatrix_d[ii_d].shape
            FullMatrices['MMatrix'] += csr_matrix((DVecRepV, (DVecRepRow, DVecRepCol)), shape=DSize)

    # Вставка граничных условий
    if 'Methods' in CompStruct and 'InsertBC' in CompStruct['Methods']:
        FullMatrices = CompStruct['Methods']['InsertBC'](FullMatrices, BasicMatrices, CompStruct)

    # Обработка интерфейсных условий
    for ii_int in range(1, CompStruct['Data']['N_domains']):
        ii_D1 = ii_int
        ii_D2 = ii_D1 + 1
        if 'Methods' in CompStruct and 'InsertIC' in CompStruct['Methods']:
            FullMatrices = CompStruct['Methods']['InsertIC'](FullMatrices, BasicMatrices, CompStruct, ii_D1, ii_D2,
                                                             ii_int)

    return FullMatrices