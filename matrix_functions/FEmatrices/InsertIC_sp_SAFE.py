from scipy.sparse import csr_matrix, find, block_diag


def InsertIC_sp_SAFE(FullMatrices, BasicMatrices, CompStruct, ii_l1, ii_l2, ii_BC):

    # Временная реализация - должна быть дополнена
    # В оригинальном MATLAB коде много условий, которые требуют детальной реализации

    # Пример обработки интерфейса жидкость-твердое тело
    if (CompStruct['Model']['LayerType'][ii_l1] == 'fluid' and
            CompStruct['Model']['LayerType'][ii_l2] == 'HTTI'):

        # Эмуляция обработки интерфейсных матриц
        BMatrixD1 = {}  # Должны быть определены
        BMatrixD2 = {}  # Должны быть определены

        if ii_l1 in BMatrixD1 and ii_l2 in BMatrixD2:
            D1VecRepRow, D1VecRepCol, D1VecRepV = find(BMatrixD1[ii_BC])
            D2VecRepRow, D2VecRepCol, D2VecRepV = find(BMatrixD2[ii_BC])

            D1VecRepRow += CompStruct['Data']['DVarNum'][ii_l2][0] - 1
            D1VecRepCol += CompStruct['Data']['DVarNum'][ii_l1][0] - 1
            D2VecRepRow += CompStruct['Data']['DVarNum'][ii_l1][0] - 1
            D2VecRepCol += CompStruct['Data']['DVarNum'][ii_l2][0] - 1

            N_domain = 1  # Должен быть определен из CompStruct
            DSize = CompStruct['Data']['DVarNum'][N_domain][1]

            FullMatrices['PMatrix'] += csr_matrix(
                (D1VecRepV, (D1VecRepRow, D1VecRepCol)), shape=(DSize, DSize)
            ) + csr_matrix(
                (D2VecRepV, (D2VecRepRow, D2VecRepCol)), shape=(DSize, DSize)
            )

    return FullMatrices