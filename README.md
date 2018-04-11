### This is yufeng's test codes about non-linear active function in quantum machine learning. 
reference "Quantum Neuron: an elementary building block for machine learning on quantum computers"

#### 1. Introduction
In Machine Learning, active function is an very important term, because it can add non-linearity to the optimization precess.
Quantum Neural Network(QNN) is quiet popular these days, and the main difficulty is that the quantum system is linear since 
all the transformation is unitary. However, by implement "repeat-until-success" trick, we could construct such a non-linear
function in quantum system, typically using quantum circuits for demo.

#### 2. Math about the non-linear function
Our target is to construct the function :
<img src="http://chart.googleapis.com/chart?cht=tx&chl=\Large f(\theta) = \tan^{-1}\left(\tan^{2}(\theta)\right) " style="border:none;"> 
the function like this:

In our quantum circuits, there is one input register and another quantum register contains one ancilla qubits and one output
qubits. By perform several controlled-y-rotation operation, the ancilla qubits could contain

<img src="http://chart.googleapis.com/chart?cht=tx&chl=\Large \theta = \sum_{i}x_{i}\omega_{i} " style="border:none;"> 
    
the detail of this preparation will be discussed later.

$$
  \begin{pmatrix}
   1 & 2 & 3 \\
   4 & 5 & 6 \\
   7 & 8 & 9
  \end{pmatrix} \tag{1}
$$

    
The main part of the non-linear functon is constructed as followes:\
First, implement a controlled-Y operation, after which, part of the wavefunction information is passed to thr output qubits,
and more importantly, entanglement is created between ancilla qubits and output qubits. After perform reverse operation
of that in the preparation step, the final wavefunction would like this:


It is clear that we are only interested  in the amplitude of 
<img src="http://chart.googleapis.com/chart?cht=tx&chl=\Large |00\rangle " style="border:none;">  and 
<img src="http://chart.googleapis.com/chart?cht=tx&chl=\Large |01\rangle} " style="border:none;">. 
These two computational basis can be rewrite as:
<img src="http://chart.googleapis.com/chart?cht=tx&chl=\Large |0\rangle \otimes (\sin^{2}(\theta) |0\rangle + \cos^{2}(\theta)|1\rangle)" style="border:none;"> 
which means if collapse the wave function of ancilla qubits to 
<img src="http://chart.googleapis.com/chart?cht=tx&chl=\Large |0\rangle " style="border:none;">, by measuring output qubits,
we could get the desired non-linear function.

The quantum circuit is as follows:\
![circuits](https://github.com/RindJLU/ML-with-non-linear-function/blob/master/pictures/FireShot%20Capture%203%20-%20%20-%20https___arxiv.org_pdf_1711.11240.pdf.png)

the quantum circuit result and the theoretical result:\
![non-linear_fun](https://github.com/RindJLU/ML-with-non-linear-function/blob/master/pictures/non-linear%20function.png)

#### 3. Test the non-linearity of the quantum circuits

circuit\
![circuits](https://github.com/RindJLU/ML-with-non-linear-function/blob/master/pictures/circuit.png)

detail\
![detail](https://github.com/RindJLU/ML-with-non-linear-function/blob/master/pictures/circuits%20detail.png)

To test the non-linearity of the circuits, let's consider a simple classification problem, horizontal stripe or vertical 
stripe. For 2*2 block, there are four possibilities:

![2by2block](https://github.com/RindJLU/ML-with-non-linear-function/blob/master/pictures/FireShot%20Capture%205%20-%20down.php%20(1068%C3%971205)_%20-%20https___mails.jlu.edu.cn_down.php.png)

here we encoded the blocks as follows:
(pi/4, 0), (pi/4, pi/2), (0, pi.4), (pi/2, pi/4)

![2 by 2](https://github.com/RindJLU/ML-with-non-linear-function/blob/master/pictures/2%20by%202%20block%20encoded%20data.png)

result(with 200 epoch):

loss 

![loss_stripe_classify](https://github.com/RindJLU/ML-with-non-linear-function/blob/master/pictures/first%20success.png)

theta: [0.46479099 2.83095792 3.05828674]\
result: [0.014110351851995885, 0.01717800843828657, 0.9619506244678849, 0.9667178975425114] vs [0, 0, 1, 1] 

#### 4. Generalization 
We first test the hand-write data in MNIST, we encoded every figure by two parameters, horizontal ratio(number of left half
pixels divided by number of right half pixels). However, because the data in MNIST is 28 by 28, so the features are not very 
distinct. In light of this, I choose another datasets: xxxxxxxxxxxxx, whose hand-write figures are sized 128*128.
the encoded data are plotted as follows:

![029](https://github.com/RindJLU/ML-with-non-linear-function/blob/master/pictures/Dist_029.png)
![6and9](https://github.com/RindJLU/ML-with-non-linear-function/blob/master/pictures/DIST_6_9.png)

results:

![result1](https://github.com/RindJLU/ML-with-non-linear-function/blob/master/pictures/Qtm_result_30train_sets_50epoch_6.png)
![result2](https://github.com/RindJLU/ML-with-non-linear-function/blob/master/pictures/Qtm_result_30train_sets_30epoch_5.png)
