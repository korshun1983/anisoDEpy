import numpy as np
from scipy.sparse import csr_matrix, find as sparse_find


def insertIC_sp_safe(mmatrix, smop, sop, dop, basic_matrices, comp_struct, ii_l1, ii_l2, ii_bc):
    """
    Insert interface conditions into spectral method matrices.
    [T.Zharnikov, D.Syresin, SMR v0.12_12.2012]
    Converted to Python 2025
    """

    layer_type_1 = comp_struct['Model']['LayerType'][ii_l1]
    layer_type_2 = comp_struct['Model']['LayerType'][ii_l2]

    if layer_type_1 == 'fluid' and layer_type_2 == 'HTTI':
        # Fluid-solid contact
        for ii_n in range(comp_struct['Data']['N_harmonics']):
            # s_rr continuity
            ii_vL1 = 0
            cond_pos_l1 = basic_matrices['Pos'][ii_l1][ii_n][ii_vL1][1]  # Terminal
            ii_vL2 = 0
            cond_pos_l2 = basic_matrices['Pos'][ii_l2][ii_n][ii_vL2][0]  # Initial

            mmatrix[cond_pos_l1, cond_pos_l1] = 0
            mmatrix[cond_pos_l2, cond_pos_l2] = 0

            for ii_kd in range(3):
                # s_rr continuity
                smop[ii_kd][cond_pos_l1, :] = sop[ii_kd][cond_pos_l1, :] - sop[ii_kd][cond_pos_l2, :]
                # u_r continuity
                smop[ii_kd][cond_pos_l2, :] = dop['ur'][ii_kd][cond_pos_l1, :] - dop['ur'][ii_kd][cond_pos_l2, :]

            # s_rth = 0
            ii_vL1 = 1
            cond_pos_l1 = basic_matrices['Pos'][ii_l1][ii_n][ii_vL1][1]
            mmatrix[cond_pos_l1, cond_pos_l1] = 0
            for ii_kd in range(3):
                smop[ii_kd][cond_pos_l1, :] = sop[ii_kd][cond_pos_l1, :]

            # s_rz = 0
            ii_vL1 = 2
            cond_pos_l1 = basic_matrices['Pos'][ii_l1][ii_n][ii_vL1][1]
            mmatrix[cond_pos_l1, cond_pos_l1] = 0
            for ii_kd in range(3):
                smop[ii_kd][cond_pos_l1, :] = sop[ii_kd][cond_pos_l1, :]

    elif layer_type_1 == 'HTTI' and layer_type_2 == 'HTTI':
        # Solid-solid contact
        bc_type = comp_struct['Model']['BCType'][ii_bc]

        if bc_type == 'SSstiff':  # Stiff contact
            for ii_n in range(comp_struct['Data']['N_harmonics']):
                for var_idx in range(3):  # s_rr, s_rth, s_rz
                    ii_vL1 = var_idx
                    ii_vL2 = var_idx
                    cond_pos_l1 = basic_matrices['Pos'][ii_l1][ii_n][ii_vL1][1]
                    cond_pos_l2 = basic_matrices['Pos'][ii_l2][ii_n][ii_vL2][0]

                    mmatrix[cond_pos_l1, cond_pos_l1] = 0
                    mmatrix[cond_pos_l2, cond_pos_l2] = 0

                    for ii_kd in range(3):
                        # Stress continuity
                        smop[ii_kd][cond_pos_l1, :] = sop[ii_kd][cond_pos_l1, :] - sop[ii_kd][cond_pos_l2, :]
                        # Displacement continuity
                        if var_idx == 0:  # u_r
                            smop[ii_kd][cond_pos_l2, :] = dop['ur'][ii_kd][cond_pos_l1, :] - dop['ur'][ii_kd][
                                                                                             cond_pos_l2, :]
                        elif var_idx == 1:  # u_th
                            smop[ii_kd][cond_pos_l2, :] = dop['uth'][ii_kd][cond_pos_l1, :] - dop['uth'][ii_kd][
                                                                                              cond_pos_l2, :]
                        elif var_idx == 2:  # u_z
                            smop[ii_kd][cond_pos_l2, :] = dop['uz'][ii_kd][cond_pos_l1, :] - dop['uz'][ii_kd][
                                                                                             cond_pos_l2, :]

        elif bc_type == 'SSslip':  # Slip contact
            for ii_n in range(comp_struct['Data']['N_harmonics']):
                # s_rr continuity, u_r continuity
                ii_vL1 = 0
                ii_vL2 = 0
                cond_pos_l1 = basic_matrices['Pos'][ii_l1][ii_n][ii_vL1][1]
                cond_pos_l2 = basic_matrices['Pos'][ii_l2][ii_n][ii_vL2][0]

                mmatrix[cond_pos_l1, cond_pos_l1] = 0
                mmatrix[cond_pos_l2, cond_pos_l2] = 0

                for ii_kd in range(3):
                    smop[ii_kd][cond_pos_l1, :] = sop[ii_kd][cond_pos_l1, :] - sop[ii_kd][cond_pos_l2, :]
                    smop[ii_kd][cond_pos_l2, :] = dop['ur'][ii_kd][cond_pos_l1, :] - dop['ur'][ii_kd][cond_pos_l2, :]

                # s_rth = 0 on both sides (no shear continuity)
                ii_vL1 = 1
                cond_pos_l1 = basic_matrices['Pos'][ii_l1][ii_n][ii_vL1][1]
                ii_vL2 = 1
                cond_pos_l2 = basic_matrices['Pos'][ii_l2][ii_n][ii_vL2][0]

                mmatrix[cond_pos_l1, cond_pos_l1] = 0
                mmatrix[cond_pos_l2, cond_pos_l2] = 0

                for ii_kd in range(3):
                    smop[ii_kd][cond_pos_l1, :] = sop[ii_kd][cond_pos_l1, :]
                    smop[ii_kd][cond_pos_l2, :] = sop[ii_kd][cond_pos_l2, :]

                # s_rz = 0 on both sides (no shear continuity)
                ii_vL1 = 2
                cond_pos_l1 = basic_matrices['Pos'][ii_l1][ii_n][ii_vL1][1]
                ii_vL2 = 2
                cond_pos_l2 = basic_matrices['Pos'][ii_l2][ii_n][ii_vL2][0]

                mmatrix[cond_pos_l1, cond_pos_l1] = 0
                mmatrix[cond_pos_l2, cond_pos_l2] = 0

                for ii_kd in range(3):
                    smop[ii_kd][cond_pos_l1, :] = sop[ii_kd][cond_pos_l1, :]
                    smop[ii_kd][cond_pos_l2, :] = sop[ii_kd][cond_pos_l2, :]

    return mmatrix, smop