import numpy as np


class Linear:

    def __init__(self, in_features, out_features, linear_weight_init_fn = None, bias_init_fn = None, debug=False):
        """
        Initialize the weights and biases with zeros
        Checkout np.zeros function.
        Read the writeup to identify the right shapes for all.
        """
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros((out_features, 1))

        if linear_weight_init_fn is not None:
            self.W = linear_weight_init_fn(out_features, in_features)

        if bias_init_fn is not None:
            self.b = bias_init_fn(out_features)

        self.debug = debug

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        Read the writeup for implementation details
        """
        self.A = A  # TODO
        self.N = len(A)  # TODO store the batch size of input
        # Think how will self.Ones helps in the calculations and uncomment below
        self.Ones = np.ones((self.N,1)) #对每行A[i]（1，C0）--   A[i] * W + b.T (1, C1) 
        Z = np.dot(A, self.W.T) + np.dot(self.Ones, self.b.T) #shape(N, C1)
        
        return Z

    def backward(self, dLdZ):

        dZdA = self.W.T
        dZdW = self.A
        dZdb = self.Ones

        dLdA = np.dot(dLdZ, dZdA.T)
        dLdW = np.dot(dLdZ.T, dZdW) 
        dLdb = np.dot(dLdZ.T, dZdb)
        self.dLdW = dLdW / self.N
        self.dLdb = dLdb / self.N

        if self.debug:

            self.dZdA = dZdA
            self.dZdW = dZdW
            self.dZdb = dZdb
            self.dLdA = dLdA

        return dLdA
