from scipy.sparse import csr_matrix, find, block_diag
from structures import BigCompStruct

def InsertBC_sp_SAFE(MMatrix, SMOp, SOp, DOp, BasicMatrices, CompStruct: BigCompStruct):

    # Вставка внутренних граничных условий
    ii_l = 0  # Python использует 0-based индексацию
    BCType = CompStruct['Model']['BCType'][ii_l]
    LayerType = CompStruct['Model']['LayerType'][ii_l]

    if BCType == 'free':
        if LayerType == 'fluid':
            for ii_n in range(CompStruct['Data']['N_harmonics']):
                ii_vL = 0
                CondPos = BasicMatrices['Pos'][ii_l][ii_n][ii_vL][0]  # начальная радиальная позиция
                MMatrix[CondPos, CondPos] = 0
                for ii_kd in range(3):
                    SMOp[ii_kd][CondPos, :] = SOp[ii_kd][CondPos, :]
        elif LayerType == 'HTTI':
            for ii_n in range(CompStruct['Data']['N_harmonics']):
                for ii_v in range(CompStruct['Data']['LayerVarNum'][ii_l]):
                    CondPos = BasicMatrices['Pos'][ii_l][ii_n][ii_v][0]  # начальная радиальная позиция
                    MMatrix[CondPos, CondPos] = 0
                    for ii_kd in range(3):
                        SMOp[ii_kd][CondPos, :] = SOp[ii_kd][CondPos, :]

    elif BCType == 'rigid':
        if LayerType == 'fluid':
            for ii_n in range(CompStruct['Data']['N_harmonics']):
                for ii_v in range(CompStruct['Data']['LayerVarNum'][ii_l]):
                    CondPos = BasicMatrices['Pos'][ii_l][ii_n][ii_v][0]  # начальная радиальная позиция
                    MMatrix[CondPos, CondPos] = 0
                    for ii_kd in range(3):
                        SMOp[ii_kd][CondPos, :] = DOp['ur'][ii_kd][CondPos, :]
        elif LayerType == 'HTTI':
            for ii_n in range(CompStruct['Data']['N_harmonics']):
                ii_vL = 0
                CondPos = BasicMatrices['Pos'][ii_l][ii_n][ii_vL][0]  # начальная радиальная позиция
                MMatrix[CondPos, CondPos] = 0
                for ii_kd in range(3):
                    SMOp[ii_kd][CondPos, :] = DOp['ur'][ii_kd][CondPos, :]

                ii_vL = 1
                CondPos = BasicMatrices['Pos'][ii_l][ii_n][ii_vL][0]  # начальная радиальная позиция
                MMatrix[CondPos, CondPos] = 0
                for ii_kd in range(3):
                    SMOp[ii_kd][CondPos, :] = DOp['uth'][ii_kd][CondPos, :]

                ii_vL = 2
                CondPos = BasicMatrices['Pos'][ii_l][ii_n][ii_vL][0]  # начальная радиальная позиция
                MMatrix[CondPos, CondPos] = 0
                for ii_kd in range(3):
                    SMOp[ii_kd][CondPos, :] = DOp['uz'][ii_kd][CondPos, :]

    # Вставка внешних граничных условий
    ii_l = CompStruct['Data']['N_layers'] - 1  # 0-based индексация
    BCType = CompStruct['Model']['BCType'][ii_l + 1]
    LayerType = CompStruct['Model']['LayerType'][ii_l]

    if BCType == 'free':
        if LayerType == 'fluid':
            for ii_n in range(CompStruct['Data']['N_harmonics']):
                for ii_v in range(CompStruct['Data']['LayerVarNum'][ii_l]):
                    CondPos = BasicMatrices['Pos'][ii_l][ii_n][ii_v][1]  # конечная радиальная позиция
                    MMatrix[CondPos, CondPos] = 0
                    for ii_kd in range(3):
                        SMOp[ii_kd][CondPos, :] = SOp[ii_kd][CondPos, :]
        elif LayerType == 'HTTI':
            for ii_n in range(CompStruct['Data']['N_harmonics']):
                for ii_v in range(CompStruct['Data']['LayerVarNum'][ii_l]):
                    CondPos = BasicMatrices['Pos'][ii_l][ii_n][ii_v][1]  # конечная радиальная позиция
                    MMatrix[CondPos, CondPos] = 0
                    for ii_kd in range(3):
                        SMOp[ii_kd][CondPos, :] = SOp[ii_kd][CondPos, :]

    elif BCType == 'rigid':
        if LayerType == 'fluid':
            for ii_n in range(CompStruct['Data']['N_harmonics']):
                for ii_v in range(CompStruct['Data']['LayerVarNum'][ii_l]):
                    CondPos = BasicMatrices['Pos'][ii_l][ii_n][ii_v][1]  # конечная радиальная позиция
                    MMatrix[CondPos, CondPos] = 0
                    for ii_kd in range(3):
                        SMOp[ii_kd][CondPos, :] = DOp['ur'][ii_kd][CondPos, :]
        elif LayerType == 'HTTI':
            for ii_n in range(CompStruct['Data']['N_harmonics']):
                ii_vL = 0
                CondPos = BasicMatrices['Pos'][ii_l][ii_n][ii_vL][1]  # конечная радиальная позиция
                MMatrix[CondPos, CondPos] = 0
                for ii_kd in range(3):
                    SMOp[ii_kd][CondPos, :] = DOp['ur'][ii_kd][CondPos, :]

                ii_vL = 1
                CondPos = BasicMatrices['Pos'][ii_l][ii_n][ii_vL][1]  # конечная радиальная позиция
                MMatrix[CondPos, CondPos] = 0
                for ii_kd in range(3):
                    SMOp[ii_kd][CondPos, :] = DOp['uth'][ii_kd][CondPos, :]

                ii_vL = 2
                CondPos = BasicMatrices['Pos'][ii_l][ii_n][ii_vL][1]  # конечная радиальная позиция
                MMatrix[CondPos, CondPos] = 0
                for ii_kd in range(3):
                    SMOp[ii_kd][CondPos, :] = DOp['uz'][ii_kd][CondPos, :]

    return MMatrix, SMOp