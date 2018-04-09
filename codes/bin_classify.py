# coding: utf-8

import numpy as np
import projectq
from projectq.ops import H, X, Y, Measure, Ry, Rz
import pickle
from projectq.meta import Control
import matplotlib.pyplot as plt


class QtmClassifier(object):
    """only process two dimensional data"""
    def __init__(self, x, y, n_qubits, epoch):
        """
        Initialization of the class
        :param x: (ndarray, shape(num_data, 2)), two elements represents HR and VR
        :param y: (ndarray, shape(num_data, 1)), label of corresponding data
        :param n_qubits: the number of qubits that represent one element of a vector in x
        :param epoch: the maximum iteration of the optimization
        """
        n = len(x)
        self.train_x = x[0:(n - round(n / 6))]
        self.train_y = y[0:(n - round(n / 6))]
        self.test_x = x[(n - round(n / 6)):]
        self.test_y = y[(n - round(n / 6)):]
        self.epoch = epoch
        self.n_qubits1data = n_qubits
        self.n_qubits = len(self.train_x[0]) * n_qubits + 2  # two qubits for one ancilla qubit and one output qubit
        self.alpha = np.pi*np.random.rand(len(self.train_x[0]) + 1) - np.pi/2  # the last one is the bios
        self.init_wavefun = np.zeros(len(self.train_x[0]) ** self.n_qubits)
        self.init_wavefun[0] = 1  # set the wave_function to |0>

        self.eng = projectq.MainEngine()
        self.qureg = self.eng.allocate_qureg(self.n_qubits)
        self.eng.flush()

        self.data2theta(self.train_x)  # transform the data into binary representation
        self.data2theta(self.test_x)
        self.loss = []

    def data2theta(self, data):
        """
        this function mainly normalize the given data, so that the data could be stored in the wavefunction.
        :return: None
        """
        # first normalize
        encoded_data = []  # shape -- len(data), (n_qubits1data + 1） for every element: [[0 0 1 1 1], [1 0 0 1 1], norm]
        for i in range(len(data[:])):
            norm_temp = np.linalg.norm(data[i])
            data[i] *= 1 / norm_temp
            # encoded_vec = [(self.n_qubits1data * [0])] + [(self.n_qubits1data * [0])]  # manually choose two elements
            encoded_vec = np.zeros([2, self.n_qubits1data]).tolist()
            # here cannot just use len(self.data[0]) * [(self.n_qubits1data * [0])], since the two would be the same all
            # the time like the function "copy"
            encoded_vec.append(norm_temp)
            for vec_index in range(len(data[i])):
                # the following operation
                vec_ele = bin(int(round(data[i][vec_index] * (2**(self.n_qubits1data*3)))))[2:]
                if len(vec_ele) > 2 * self.n_qubits1data:
                    cuted_bin_str = vec_ele[0:(len(vec_ele) - 2 * self.n_qubits1data)]
                    for j in range(len(cuted_bin_str)):
                        encoded_vec[vec_index][self.n_qubits1data - len(cuted_bin_str) + j] = int(cuted_bin_str[j])
            encoded_data.append(encoded_vec)
        if data.all() == self.train_x.all():
            self.train_x = np.array(encoded_data)
        elif data.all() == self.test_x.all():
            self.test_x = np.array(encoded_data)

    def cal_loss(self, a_list, b_list):
        """
        calcu
        :param a_list:
        :param b_list:
        :return:
        """
        loss = 0
        for i in range(len(a_list)):
            loss += (a_list[i] - b_list[i])**2
        return loss

    def active_fun(self, train_or_test):
        """

        :param train_or_test: bool, True or False
        :return: loss
        """
        if train_or_test == 'train':
            x_epoch = self.train_x
            y_epoch = self.train_y
            return_test_label = False
        elif train_or_test == 'test':
            x_epoch = self.test_x
            y_epoch = self.test_y
            return_test_label = True
        else:
            print('error!!!')

        y_sim = []
        for encoded_data in x_epoch:
            self.eng.backend.set_wavefunction(self.init_wavefun, self.qureg)
            for w in range(2):  # represent the data with same weight
                for i in range(len(encoded_data[0])):
                    if encoded_data[w][i]:
                        X | self.qureg[self.n_qubits1data * w + i]
                    with Control(self.eng, self.qureg[self.n_qubits1data * w + i]):
                        Ry(2 * encoded_data[-1] * self.alpha[w] / 2**i) | self.qureg[-2]
            # add bios
            Ry(2 * self.alpha[-1]) | self.qureg[-2]

            with Control(self.eng, self.qureg[-2]):
                Y | self.qureg[-1]

            Rz(-np.pi / 2) | self.qureg[2]

            Ry(-2 * self.alpha[-1]) | self.qureg[-2]

            for w in range(2):  # represent the data with same weight
                for i in range(len(encoded_data[0])):
                    with Control(self.eng, self.qureg[self.n_qubits1data * w + i]):
                        Ry(-2 * encoded_data[-1] * self.alpha[w] / 2**i) | self.qureg[-2]

            self.eng.flush()
            # self.eng.backend.collapse_wavefunction([self.qureg[-2]], [0])
            result = self.eng.backend.get_probability('0', [self.qureg[-1]])

            Measure | self.qureg
            y_sim.append(np.arccos(np.sqrt(result))/(np.pi/2))
            # y_sim.append(result)
        loss = self.cal_loss(y_epoch, y_sim)
        if return_test_label:
            return loss, y_sim
        else:
            return loss

    def run(self):
        delta_theta = 0.0001
        update_theta = 0.0005
        for iter in range(self.epoch):  # print epochs
            if (iter+1) % 5 == 0:
                print('This is the {}th epoch'.format(iter + 1)), print('the loss is {}'.format(self.loss[-1]))

            for j in range(len(self.alpha)):
                self.alpha[j] += delta_theta
                loss_plus = self.active_fun('train')
                self.alpha[j] += -2*delta_theta
                loss_min = self.active_fun('train')

                # update
                print(update_theta*(loss_plus - loss_min)/(2*delta_theta), self.alpha)
                self.alpha[j] += delta_theta - update_theta*(loss_plus - loss_min)/(2*delta_theta)
            self.loss.append(loss_plus)

    def test(self):
        print('this is the test')
        num_test = len(self.test_y)
        loss, y_sim = self.active_fun('test')
        loss = loss / num_test
        print('the test loss is {}'.format(loss))
        print(y_sim, self.test_y)

    def plot(self, train_or_test, plt, input_x):
        """only put one legend for every type of numbers"""
        if train_or_test == 'train':
            plot_data_x = self.train_x
            plot_data_y = self.train_y
            mark = 'o'
            pos_index = 0
            tag = 'train set'
        elif train_or_test == 'test':
            plot_data_x = self.test_x
            plot_data_y = self.test_y
            mark = '*'
            pos_index = 25
            tag = 'test set'
        else:
            print('error!!!')

        col = ['r', 'g']
        count_6 = 0; count_9 = 1
        for p in range(len(plot_data_x)):
            k = p + pos_index
            if count_6 == int(plot_data_y[p]):
                plt.scatter(input_x[k, 0] * plot_data_x[p, 2], input_x[k, 1] * plot_data_x[p, 2],
                            color=col[int(plot_data_y[p])], marker=mark, label=tag + ' 6')
                count_6 = -1
                plt.legend()
            elif count_9 == int(A.train_y[p]):
                plt.scatter(input_x[k, 0] * plot_data_x[p, 2], input_x[k, 1] * plot_data_x[p, 2],
                            color=col[int(plot_data_y[p])], marker=mark, label=tag + ' 9')
                count_9 = -1
                plt.legend()
            else:
                plt.scatter(input_x[k, 0] * plot_data_x[p, 2], input_x[k, 1] * plot_data_x[p, 2],
                            color=col[int(plot_data_y[p])], marker=mark)


class Classifier(object):
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
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
        loss = self.cal_loss(self.y, y_sim)
        return loss

    def run(self):
        delta_theta = 0.0001
        update_theta = 0.0004
        for iter in range(300):
            for j in range(len(self.alpha)):
                self.alpha[j] += delta_theta
                loss_plus = self.active_fun()
                self.alpha[j] += -2 * delta_theta
                loss_min = self.active_fun()

                # update
                self.alpha[j] += delta_theta - update_theta * (loss_plus - loss_min) / (2 * delta_theta)
            self.loss.append(loss_plus)


if __name__ == '__main__':
    # import data
    pkl_file = open('/home/yufeng/PycharmProjects/undergraduate thesis/non-linear/vector_like_info_OCR_10_100_2.pkl', 'rb')
    data = np.array(pickle.load(pkl_file))
    pkl_file.close()
    mixed_data = np.zeros([200, 3])
    mixed_data[:100, 0:2] = data[6, :, :]
    mixed_data[100:, 0:2] = data[9, :, :]
    mixed_data[100:, 2] = np.ones(100)  # represent 6 as 0, 9 as 1
    np.random.shuffle(mixed_data)

    input_x = mixed_data[0:30, 0:2]
    input_y = mixed_data[0:30, 2]
    A = QtmClassifier(input_x, input_y, 5, 20)
    # A = Classifier(mixed_data[0:90, 0:2], mixed_data[0:90, 2])
    # print(A.train_y)
    A.run()
    print(A.alpha)
    A.test()

    # plot the picture
    plt.subplot(121)
    plt.plot(A.loss)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('Loss and iterations')

    plt.subplot(122)

    A.plot('train', plt, input_x)
    A.plot('test', plt, input_x)

    x_range = np.arange(0, 6, 0.01)
    for y_index in range(10):
        y_temp = ((-A.alpha[0]*x_range - A.alpha[2] - 9*np.pi/4 + y_index*np.pi/2)/A.alpha[1])
        plt.plot(x_range, y_temp)
    plt.xlabel('Horizontal Ratio')
    plt.ylabel('Vertical Ratio')
    plt.title('Decision boundary')
    plt.axis([0, 5, 0, 4])

    fig = plt.gcf()
    fig.set_size_inches(10.5, 4.5)
    fig.savefig('Qtm_result.png', dpi=100)
    plt.show()

# solutions

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

# the test loss is 0.1331819874405067 for 10 test data
# [0.5137614471671018, 0.4189086854090739, 0.7833234142633081, 0.459393964363378, 0.7833234142633081,
# 0.8323764390889591, 0.6047199264988363, 0.4506091869163517, 0.08591327708361007, 0.8253921289916818]
# [0. 1. 1. 0. 1. 1. 1. 0. 0. 1.]

# OCR
# [0.9894346786671705, 0.07737197228271495, 0.1914173493323471, 0.17387539966295704, 0.07013300270042955]
#  [1. 0. 0. 0. 0.]

# [-0.06199423  0.33994924  0.01287681]
# [0.8383418738557524, 0.042007335902489645, 0.020572833606634353, 0.01712816562027808, 0.12656422465978862]
# [1. 0. 0. 0. 0.]

# [-0.14857458  0.26655323  0.40453079]
# the test loss is 0.0387647128583178
# [0.7953108021194065, 0.0297510803144226, 0.34096794135435604, 0.11297331639439545, 0.8516131322045738]
# [1. 0. 0. 0. 1.]
