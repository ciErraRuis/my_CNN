# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        batch_size = A.shape[0]
        in_channels = self.in_channels
        input_size = A.shape[2]
        output_size = input_size - self.kernel_size + 1 #1个convol结果每个channel的长度
        Z = np.zeros((batch_size, self.out_channels, output_size))
        # W.shape:[out_channels, in_channels, kernel_size]
        for i in range(output_size):
            Z[:,:,i] = np.tensordot(A[:,:,i:i+self.kernel_size], self.W, axes = ([1,2], [1,2]))

        for i in range(self.out_channels):
            Z[:,i,:] += self.b[i]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        input_size = self.A.shape[2]
        kernel_size1 = dLdZ.shape[2]
        output_size = input_size - kernel_size1 + 1
        for i in range(output_size):
            #A.shape:[N, Cin, input_size], dLdZ.shape:[N, Cout, output_size]
            #--> dLdW.shape: [Cout, Cin, kernel_size]
            self.dLdW[:,:,i] = np.tensordot(dLdZ, 
                                            self.A[:,:,i:i+kernel_size1], axes = ([0,2],[0,2]))
 
        self.dLdb = np.sum(dLdZ, axis = (0,2))

        #padding
        kernel_size2 = self.W.shape[2]
        dLdZ_padded = np.pad(dLdZ,((0,0),(0,0),(kernel_size2 - 1,kernel_size2 - 1)),
                             'constant', constant_values = (0,0)) 
        W_flipped = self.W[:,:,::-1] #(out_channels, input_channels, kernel_size)

        # convol and get dLdA
        dLdA = np.zeros_like(self.A)
        for i in range(self.A.shape[2]):
            #dLdA.shape:[N, Cin, input_size], dLdZ_padded:[N, Cout, output_size], W:[Cout, Cin, kernel]
            dLdA[:,:,i] = np.tensordot(dLdZ_padded[:,:,i:i+kernel_size2], W_flipped,
                                       axes = ([1,2],[0,2]))
        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size,
                 weight_init_fn, bias_init_fn)
    
        self.downsample1d = Downsample1d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Call Conv1d_stride1
        temp = self.conv1d_stride1.forward(A)

        # downsample
        Z = self.downsample1d.forward(temp)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        temp = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(temp)

        return dLdA
