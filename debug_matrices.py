#!/usr/bin/env python3
# debug_matrices.py
# Сравнение K, M перед solve_safe на 1 кГц (аналог MATLAB)
import sys
from pathlib import Path
import numpy as np
from private.St1_SetModel import St1_SetModel
from private.St2_PrepareModel import St2_PrepareModel
from private.St3_PrepareBasicMatrices import St3_PrepareBasicMatrices


def debug_1kHz():
    models_dir = Path(__file__).with_name('models')
    json_file = models_dir / 'Bakken-B' / 'BakkenB-00.json'

    InputParam = St1_SetModel(json_file)
    InputParam['Model']['f_array'] = [1]      # 1 kHz only
    InputParam['Model']['N_disp'] = 1

    CompStruct = St2_PrepareModel(InputParam)
    CompStruct['if_grid'] = 0
    f_khz = 1.0

    CompStruct, _, FEMatrices, _ = St3_PrepareBasicMatrices(CompStruct, InputParam)
    K = FEMatrices['K']
    M = FEMatrices['M']
    omega = 2 * np.pi * f_khz * 1e3

    # ---- вывод ----
    print(f'\nDEBUG 1 kHz:')
    print(f'  K shape: {K.shape}, nnz: {K.nnz}, dtype: {K.dtype}')
    print(f'  M shape: {M.shape}, nnz: {M.nnz}, dtype: {M.dtype}')
    print(f'  omega: {omega:.3f} rad/s')

    # первые 5×5 блоки
    print('\n  K[0:5,0:5] (complex):')
    print(K[0:5, 0:5].toarray())
    print('\n  M[0:5,0:5] (complex):')
    print(M[0:5, 0:5].toarray())

    # нормы
    print(f'\n  Frobenius norm K: {np.linalg.norm(K.toarray(), ord="fro"):.6f}')
    print(f'  Frobenius norm M: {np.linalg.norm(M.toarray(), ord="fro"):.6f}')

    # сохраняем для сравнения с MATLAB
    np.savez('debug_K_M_1kHz.npz', K=K, M=M, omega=omega)
    print('\n  Saved debug_K_M_1kHz.npz for comparison with MATLAB')