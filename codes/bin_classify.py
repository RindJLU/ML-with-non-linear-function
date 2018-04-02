# coding: utf-8

import numpy as np
import projectq
from projectq.ops import H, X, Y, Measure, Ry, Rz
from projectq.meta import Control
import matplotlib.pyplot as plt


class QtmClassifier(object):
    """only process two dimensional data"""
    def __init__(self):
        # self.x = [[np.pi / 2, 0], [np.pi / 2, np.pi], [0, np.pi / 2], [np.pi, np.pi / 2]]
        self.x = [[[0, 1], [0, 0]], [[0, 1], [1, 0]], [[0, 0], [0, 1]], [[1, 0], [0, 1]]]
        self.y = [0, 0, 1, 1]
        self.nqubits = len(self.x[0])*len(self.x[0][0]) + 2
        self.alpha = np.pi*np.random.rand(len(self.x[0]) + 1)  # the last one is the bios
        self.init_wavefun = np.zeros(2 ** self.nqubits)
        self.init_wavefun[0] = 1

        self.eng = projectq.MainEngine()
        self.qureg = self.eng.allocate_qureg(self.nqubits)
        self.eng.flush()
        self.loss = []

    def data2theta(self):
        self.x = [[np.pi/2, 0], [np.pi/2, np.pi], [0, np.pi/2], [np.pi, np.pi/2]]

    def cal_loss(self, a_list, b_list):
        loss = 0
        for i in range(len(a_list)):
            loss += (a_list[i] - b_list[i])**2
        return loss

    def active_fun(self):
        y_sim = []
        for data in self.x:
            self.eng.backend.set_wavefunction(self.init_wavefun, self.qureg)
            for w in range(len(data)): # represent the data with same weight
                for i in range(len(data[0])):
                    if data[w][i]:
                        X | self.qureg[2 * w + i]
                    with Control(self.eng, self.qureg[2 * w + i]):
                        Ry(2 * self.alpha[w] / 2**i) | self.qureg[-2]

            # add bios
            Ry(2 * self.alpha[-1]) | self.qureg[-2]

            with Control(self.eng, self.qureg[-2]):
                Y | self.qureg[-1]

            Rz(-np.pi / 2) | self.qureg[2]

            Ry(-2 * self.alpha[-1]) | self.qureg[-2]

            for w in range(len(data)): # represent the data with same weight
                for i in range(len(data[0])):
                    with Control(self.eng, self.qureg[2 * w + i]):
                        Ry(-2 * self.alpha[w] / 2**i) | self.qureg[-2]

            self.eng.flush()
            self.eng.backend.collapse_wavefunction([self.qureg[-2]], [0])
            result = self.eng.backend.get_probability('0', [self.qureg[-1]])

            Measure | self.qureg
            y_sim.append(np.arccos(np.sqrt(result))/(np.pi/2))
            # y_sim.append(result)
        print(y_sim)
        loss = self.cal_loss(self.y, y_sim)
        return loss

    def run(self):
        delta_theta = 0.001
        update_theta = 0.05
        for iter in range(200):
            for j in range(len(self.alpha)):
                self.alpha[j] += delta_theta
                loss_plus = self.active_fun()
                self.alpha[j] += -2*delta_theta
                loss_min = self.active_fun()

                # update
                self.alpha[j] += delta_theta - update_theta*(loss_plus - loss_min)/(2*delta_theta)
                self.loss.append(loss_plus)

    def test(self):
        self.x = [[[0, 1], [0, 1]]]
        loss = self.active_fun()
        return loss


class Classifier(object):
    def __init__(self):
        self.x = [[np.pi / 2, 0], [np.pi / 2, np.pi], [0, np.pi / 2], [np.pi, np.pi / 2]]
        self.y = [0, 0, 1, 1]
        self.alpha = np.random.rand(3)
        self.loss = []

    def cal_loss(self, a_list, b_list):
        loss = 0
        for i in range(len(a_list)):
            loss += (a_list[i] - b_list[i]) ** 2
        return loss

    def active_fun(self):
        y_sim = []
        for x in self.x:
            z = 0
            for i in range(len(x)):
                z += x[i] * self.alpha[i]
            z += self.alpha[-1]
            # y_sim.append(np.arctan((np.tan(z))**2)/(np.pi/2))
            a = np.cos(z)**2+((np.cos(z))**4)*(np.sin(z))**2
            b = (np.cos(z))**2+((np.cos(z))**4)*(np.sin(z))**2+np.sin(z)**6
            y_sim.append(np.arccos(np.sqrt(a/b))/(np.pi/2))
        print(y_sim)
        loss = self.cal_loss(self.y, y_sim)
        return loss

    def run(self):
        delta_theta = 0.0001
        update_theta = 0.005
        for iter in range(50):
            for j in range(len(self.alpha)):
                self.alpha[j] += delta_theta
                loss_plus = self.active_fun()
                self.alpha[j] += -2 * delta_theta
                loss_min = self.active_fun()

                # update
                self.alpha[j] += delta_theta - update_theta * (loss_plus - loss_min) / (2 * delta_theta)
                self.loss.append(loss_plus)


if __name__ == '__main__':
    A = QtmClassifier()
    A.run()
    plt.plot(A.loss)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.show()
    print(A.alpha)

# solotions

# 900 iterations: [2.22887464 0.91296014 3.20649227]
# corresponding y: [0.13345016159439585, 0.1068248414476267, 0.7863151870220224, 0.897513508358606] average 16% error
#  rates

# classical
# [ 0.16048769  0.90978973 -0.13233032]
# [0.11377565 0.84801917 0.07284581]
# [0.02490078 0.85803413 0.21530903] [0.04298708701503266, 0.023939031029136443, 0.9999622802711221, 0.9968225628033159]

# first success:
# [0.46479099 2.83095792 3.05828674]
# [0.014110351851995885, 0.01717800843828657, 0.9619506244678849, 0.9667178975425114]

