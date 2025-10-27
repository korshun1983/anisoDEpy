import numpy as np
import scipy

class ElMatrices:
    def __init__(self):
        ElMatrices.NfltRhonNMatrix = np.array([])
        ElMatrices.B1tCB1Matrix = np.array([])
        ElMatrices.B1tCB2Matrix = np.array([])
        ElMatrices.B2tCB1Matrix = np.array([])
        ElMatrices.B2tCB2Matrix = np.array([])

class basicMatrices:
    def __init__(self):
        basicMatrices.Lx = np.zeros((6, 3))
        basicMatrices.Ly = np.zeros((6, 3))
        basicMatrices.Lz = np.zeros((6, 3))
        basicMatrices.Lx_fluid = np.array([1], [0], [0])
        basicMatrices.Ly_fluid = np.array([0], [1], [0])
        basicMatrices.Lz_fluid = np.array([0], [0], [1])
        basicMatrices.E3 = np.eye(3)
        basicMatrices.LEdgeIntMatrix9 = np.zeros((10, 10))
        basicMatrices.LIntMatrix9 = np.zeros((10,10,10))
        basicMatrices.NLMatrix = np.zeroes((10,4,4,4))
        basicMatrices.NodeLCoord = np.zeros((10,3))
        basicMatrices.dNLMatrix = np.zeros((3,10,4,4,4))
        basicMatrices.dNLMatrix_val = np.zeros((3,10,10))
        basicMatrices.NNNConvMatrixInt = np.zeros((10,10,10))
        basicMatrices.dNNNConvMatrixInt = np.zeros((3,10,10,10))
        basicMatrices.NNdNConvMatrixInt = np.zeros((3,10,10,10))
        basicMatrices.dNNdNConvMatrixInt = np.zeros((3,3,10,10,10))
        basicMatrices.dNNdNConvMatrixIntLarge = {np.zeros((3,3,10,10,10)),np.zeros((3,3,10,10,10)),np.zeros((3,3,10,10,10))}
        basicMatrices.dNNNConvMatrixIntLarge = {np.zeros((3,10,10,10)),np.zeros((3,10,10,10)),np.zeros((3,10,10,10))}
        basicMatrices.NNdNConvMatrixIntLarge = {np.zeros((3,10,10,10)),np.zeros((3,10,10,10)),np.zeros((3,10,10,10))}
        basicMatrices.NNNConvMatrixIntLarge = {np.zeros((10,10,10)),np.zeros((10,10,10)),np.zeros((10,10,10))}
        basicMatrices.NLEdgeMatrix = np.zeros(4,4,4)