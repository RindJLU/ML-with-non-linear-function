# coding: utf-8

import numpy as np
import projectq
from projectq.ops import Y, Ry, Rz, Measure, H, X
from projectq.meta import Control
import matplotlib.pyplot as plt


class NonLinear(object):
    def __init__(self):
        self.eng = projectq.MainEngine()
        self.qureg = self.eng.allocate_qureg(2)
        self.eng.flush()
        self.res = []

    def cal_loss(self, phi, n):
        wavefun_shape = np.zeros(2**(n + 1))
        wavefun_shape[0] = 1
        init_wavefun = wavefun_shape
        self.eng.backend.set_wavefunction(init_wavefun, self.qureg)
        Ry(2 * phi) | self.qureg[0]
        with Control(self.eng, self.qureg[0]):
            Y | self.qureg[1]
        Rz(-np.pi / 2) | self.qureg[0]
        Ry(-2 * phi) | self.qureg[0]
        self.eng.flush()

        self.eng.backend.collapse_wavefunction([self.qureg[0]], [0])
        result = self.eng.backend.get_probability('0', [self.qureg[1]])
        print(result)
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


# define a function that could construct the non-linear function with desired curve
eng = projectq.MainEngine()

def act_fun(x, n, eng):
    """
    :param x: input
    :param n: a parameter determine the function shape
    :return: non_liear function output
    """


if __name__ == '__main__':
    non_linear_cal = NonLinear()
    x = np.arange(-np.pi/2, np.pi/2, 0.05)
    for phi in x:
        non_linear_cal.res.append(non_linear_cal.cal_loss(phi, 1))
    plt.scatter(x, non_linear_cal.res, label='Quantum non-linear result')
    plt.legend()
    y_compare = np.arctan((np.tan(x))**2)
    plt.plot(x, y_compare, label='non-linear function')
    plt.legend()
    # put some labels
    plt.xlabel('input')
    plt.ylabel('output')
    plt.title('Comparison between real function and quantum result')
    plt.show()

    # phi = 0.3*np.pi
    # MatrixCal(phi)


