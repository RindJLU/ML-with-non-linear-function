# coding: utf-8

import numpy as np
import projectq
from projectq.ops import Y, Ry, Rz, Measure, H, X
from projectq.meta import Control
import matplotlib.pyplot as plt


class NonLinear(object):
    def __init__(self):
        self.eng = projectq.MainEngine()
        self.qureg = self.eng.allocate_qureg(3)
        self.eng.flush()
        self.res = []

    def cal_loss(self, phi):
        input_wavefun = np.dot(Ry(2*phi).matrix, np.array([[1], [0]]))
        wavefun = np.kron(np.kron(np.array([[1], [0]]), np.array([[1], [0]])), input_wavefun)
        self.eng.backend.set_wavefunction(wavefun, self.qureg)

        with Control(self.eng, self.qureg[0]):
            Ry(2*phi) | self.qureg[1]
        with Control(self.eng, self.qureg[1]):
            H | self.qureg[2]
        Rz(-np.pi / 2) | self.qureg[1]
        with Control(self.eng, self.qureg[0]):
            Ry(-2*phi) | self.qureg[1]

        self.eng.flush()

        self.eng.backend.collapse_wavefunction([self.qureg[1]], [0])
        result = self.eng.backend.get_probability('0', [self.qureg[2]])

        Measure | self.qureg
        return np.arccos(np.sqrt(result))


class MatrixCal(object):
    """define the matrix calculator"""
    def __init__(self, theta):
        """define gates and calculate"""
        self.P1 = np.array([[1, 0], [0, 0]])
        self.P2 = np.array([[0, 0], [0, 1]])
        self.Ry = np.array(Ry(2*theta).matrix)
        self.Rz = np.array(Rz(-np.pi/2).matrix)
        self.Y = np.array(Y.matrix)
        self.I = np.array([[1, 0], [0, 1]])

        mat1 = np.kron(np.kron(self.P1, self.I)+np.kron(self.P2, self.Ry), self.I)
        mat2 = np.kron(self.I, np.kron(self.P1, self.I)+np.kron(self.P2, self.Y))
        mat3 = np.kron(self.I, np.kron(self.Rz, self.I))
        mat4 = np.kron(np.kron(self.P1, self.I)+np.kron(self.P2, self.Ry), self.I)
        mat = np.dot(np.dot(np.dot(mat4, mat3), mat2), mat1)  # the calculating order is different in the represent

        print(mat)


if __name__ == '__main__':
    non_linear_cal = NonLinear()
    for phi in np.arange(0, np.pi/2, 0.01):
        non_linear_cal.res.append(non_linear_cal.cal_loss(phi))
    plt.plot(non_linear_cal.res)
    plt.show()

    # phi = 0.3*np.pi
    # MatrixCal(phi)

