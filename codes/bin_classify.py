# coding: utf-8

import numpy as np
import projectq
from projectq.ops import H, Measure, Ry, Rz
from projectq.meta import Control
import matplotlib.pyplot as plt


class QtmClassifier(object):
    """only process two dimensional data"""
    def __init__(self):
        # self.x = [
        #     [np.array([[1, 1], [0, 0]])],
        #     [np.array([[0, 0], [1, 1]])],
        #     [np.array([[1, 0], [1, 0]])],
        #     [np.array([[0, 1], [0, 1]])]
        # ]
        self.x = [[np.pi / 2, 0], [np.pi / 2, np.pi], [0, np.pi / 2], [np.pi, np.pi / 2]]
        self.y = [0, 0, 1, 1]
        self.alpha = 3*np.random.rand(3)

        self.eng = projectq.MainEngine()
        self.qureg = self.eng.allocate_qureg(4)
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
        for i in self.x:
            self.eng.backend.set_wavefunction(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), self.qureg)
            Ry(i[0]) | self.qureg[0]
            Ry(i[1]) | self.qureg[1]

            with Control(self.eng, self.qureg[0]):
                Ry(2*i[0]*self.alpha[0]) | self.qureg[2]
            with Control(self.eng, self.qureg[1]):
                Ry(2*i[1]*self.alpha[1]) | self.qureg[2]
            with Control(self.eng, self.qureg[2]):
                H | self.qureg[3]
            Rz(-np.pi / 2) | self.qureg[2]
            with Control(self.eng, self.qureg[1]):
                Ry(-2*i[1]*self.alpha[1]) | self.qureg[2]
            with Control(self.eng, self.qureg[0]):
                Ry(-2*i[0]*self.alpha[0]) | self.qureg[2]

            # add bios
            Ry(2*self.alpha[2]) | self.qureg[3]

            self.eng.flush()
            self.eng.backend.collapse_wavefunction([self.qureg[2]], [0])

            result = self.eng.backend.get_probability('0', [self.qureg[3]])

            Measure | self.qureg
            y_sim.append(np.arccos(np.sqrt(result))*np.sqrt(2))
        print(y_sim)
        loss = self.cal_loss(self.y, y_sim)
        return loss

    def run(self):
        delta_theta = 0.001
        update_theta = 0.002
        for iter in range(300):
            for j in range(len(self.alpha)):
                self.alpha[j] += delta_theta
                loss_plus = self.active_fun()
                self.alpha[j] += -2*delta_theta
                loss_min = self.active_fun()

                # update
                self.alpha[j] += delta_theta - update_theta*(loss_plus - loss_min)/(2*delta_theta)
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
