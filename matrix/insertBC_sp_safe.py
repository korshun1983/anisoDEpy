def insertBC_sp_safe(mmatrix, smop, sop, dop, basic_matrices, comp_struct):
    """
    Insert boundary conditions into spectral method matrices.
    [T.Zharnikov, D.Syresin, SMR v0.12_12.2012]
    Converted to Python 2025
    """

    # Setting inner boundary conditions
    ii_l = 0  # 0-indexed (MATLAB: ii_l = 1)
    bc_type = comp_struct['Model']['BCType'][ii_l]
    layer_type = comp_struct['Model']['LayerType'][ii_l]

    if bc_type == 'free':
        if layer_type == 'fluid':
            for ii_n in range(comp_struct['Data']['N_harmonics']):
                ii_vL = 0
                cond_pos = basic_matrices['Pos'][ii_l][ii_n][ii_vL][0]  # Initial radial position
                mmatrix[cond_pos, cond_pos] = 0
                for ii_kd in range(3):
                    smop[ii_kd][cond_pos, :] = sop[ii_kd][cond_pos, :]
        elif layer_type == 'HTTI':
            for ii_n in range(comp_struct['Data']['N_harmonics']):
                for ii_v in range(comp_struct['Data']['LayerVarNum'][ii_l]):
                    cond_pos = basic_matrices['Pos'][ii_l][ii_n][ii_v][0]
                    mmatrix[cond_pos, cond_pos] = 0
                    for ii_kd in range(3):
                        smop[ii_kd][cond_pos, :] = sop[ii_kd][cond_pos, :]

    elif bc_type == 'rigid':
        if layer_type == 'fluid':
            for ii_n in range(comp_struct['Data']['N_harmonics']):
                for ii_v in range(comp_struct['Data']['LayerVarNum'][ii_l]):
                    cond_pos = basic_matrices['Pos'][ii_l][ii_n][ii_v][0]
                    mmatrix[cond_pos, cond_pos] = 0
                    for ii_kd in range(3):
                        smop[ii_kd][cond_pos, :] = dop['ur'][ii_kd][cond_pos, :]
        elif layer_type == 'HTTI':
            for ii_n in range(comp_struct['Data']['N_harmonics']):
                # u_r
                ii_vL = 0
                cond_pos = basic_matrices['Pos'][ii_l][ii_n][ii_vL][0]
                mmatrix[cond_pos, cond_pos] = 0
                for ii_kd in range(3):
                    smop[ii_kd][cond_pos, :] = dop['ur'][ii_kd][cond_pos, :]

                # u_th
                ii_vL = 1
                cond_pos = basic_matrices['Pos'][ii_l][ii_n][ii_vL][0]
                mmatrix[cond_pos, cond_pos] = 0
                for ii_kd in range(3):
                    smop[ii_kd][cond_pos, :] = dop['uth'][ii_kd][cond_pos, :]

                # u_z
                ii_vL = 2
                cond_pos = basic_matrices['Pos'][ii_l][ii_n][ii_vL][0]
                mmatrix[cond_pos, cond_pos] = 0
                for ii_kd in range(3):
                    smop[ii_kd][cond_pos, :] = dop['uz'][ii_kd][cond_pos, :]

    # Setting outer boundary conditions
    ii_l = comp_struct['Data']['N_layers'] - 1  # 0-indexed (MATLAB: ii_l = N_layers)
    bc_type = comp_struct['Model']['BCType'][ii_l + 1]
    layer_type = comp_struct['Model']['LayerType'][ii_l]

    if bc_type == 'free':
        if layer_type == 'fluid':
            for ii_n in range(comp_struct['Data']['N_harmonics']):
                for ii_v in range(comp_struct['Data']['LayerVarNum'][ii_l]):
                    cond_pos = basic_matrices['Pos'][ii_l][ii_n][ii_v][1]  # Terminal radial position
                    mmatrix[cond_pos, cond_pos] = 0
                    for ii_kd in range(3):
                        smop[ii_kd][cond_pos, :] = sop[ii_kd][cond_pos, :]
        elif layer_type == 'HTTI':
            for ii_n in range(comp_struct['Data']['N_harmonics']):
                for ii_v in range(comp_struct['Data']['LayerVarNum'][ii_l]):
                    cond_pos = basic_matrices['Pos'][ii_l][ii_n][ii_v][1]
                    mmatrix[cond_pos, cond_pos] = 0
                    for ii_kd in range(3):
                        smop[ii_kd][cond_pos, :] = sop[ii_kd][cond_pos, :]

    elif bc_type == 'rigid':
        if layer_type == 'fluid':
            for ii_n in range(comp_struct['Data']['N_harmonics']):
                for ii_v in range(comp_struct['Data']['LayerVarNum'][ii_l]):
                    cond_pos = basic_matrices['Pos'][ii_l][ii_n][ii_v][1]
                    mmatrix[cond_pos, cond_pos] = 0
                    for ii_kd in range(3):
                        smop[ii_kd][cond_pos, :] = dop['ur'][ii_kd][cond_pos, :]
        elif layer_type == 'HTTI':
            for ii_n in range(comp_struct['Data']['N_harmonics']):
                # u_r
                ii_vL = 0
                cond_pos = basic_matrices['Pos'][ii_l][ii_n][ii_vL][1]
                mmatrix[cond_pos, cond_pos] = 0
                for ii_kd in range(3):
                    smop[ii_kd][cond_pos, :] = dop['ur'][ii_kd][cond_pos, :]

                # u_th
                ii_vL = 1
                cond_pos = basic_matrices['Pos'][ii_l][ii_n][ii_vL][1]
                mmatrix[cond_pos, cond_pos] = 0
                for ii_kd in range(3):
                    smop[ii_kd][cond_pos, :] = dop['uth'][ii_kd][cond_pos, :]

                # u_z
                ii_vL = 2
                cond_pos = basic_matrices['Pos'][ii_l][ii_n][ii_vL][1]
                mmatrix[cond_pos, cond_pos] = 0
                for ii_kd in range(3):
                    smop[ii_kd][cond_pos, :] = dop['uz'][ii_kd][cond_pos, :]

    return mmatrix, smop
