import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.N, self.C_in, self.h_in, self.w_in = A.shape
        C_out = self.C_in 
        self.h_out = self.h_in - self.kernel + 1
        self.w_out = self.w_in - self.kernel + 1

        Z = np.zeros((self.N, C_out, self.h_out, self.w_out))
        self.maxind = np.zeros((self.N, self.C_in, self.h_out, self.w_out), dtype = int)

        for n in range(self.N):
            for c in range(C_out):
                for i in range(self.h_out):
                    for j in range(self.w_out):
                        self.maxind[n, c, i, j] = np.argmax(A[n, c, i:i+self.kernel, j:j+self.kernel])
                        Z[n, c, i, j] = np.amax(A[n, c, i:i+self.kernel, j:j+self.kernel])

        return Z


    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        dLdA = np.zeros((self.N, self.C_in, self.h_in, self.w_in))
        for n in range(self.N):
            for c in range(self.C_in):
                for i in range(self.h_out):
                    for j in range(self.w_out):
                        h = i + self.maxind[n, c, i, j] // self.kernel
                        w = j + self.maxind[n, c, i, j] % self.kernel
                        dLdA[n, c, h, w] += dLdZ[n, c, i, j]
        
        return dLdA

class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_weight)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_weight)
        """
        self.N, self.C_in, self.h_in, self.w_in = A.shape

        self.h_out, self.w_out = self.h_in - self.kernel + 1, self.w_in - self.kernel + 1

        Z = np.zeros((self.N, self.C_in, self.h_out, self.w_out))

        for n in range(self.N):
            for c in range(self.C_in):
                for i in range(self.h_out):
                    for j in range(self.w_out):
                        Z[n, c, i, j] =  np.sum(A[n, c, i:i+self.kernel, j:j+self.kernel]) / \
                            (self.kernel * self.kernel) 
            
        return Z
        

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros((self.N, self.C_in, self.h_in, self.w_in))

        for n in range(self.N):
            for c in range(self.C_in):
                for i in range(self.h_out):
                    for j in range(self.w_out):
                        dLdA[n, c, i:i+self.kernel, j:j+self.kernel] +=  dLdZ[n, c, i, j]  / (self.kernel * self.kernel)

        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        #pool
        temp = self.maxpool2d_stride1.forward(A)

        #downsampling
        Z = self.downsample2d.forward(temp)

        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        #upsampling
        temp = self.downsample2d.backward(dLdZ)

        #backprop
        dLdA = self.maxpool2d_stride1.backward(temp)

        return dLdA
    

class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        #pool
        temp = self.meanpool2d_stride1.forward(A)

        #downsample
        Z = self.downsample2d.forward(temp)

        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        #upsample
        temp = self.downsample2d.backward(dLdZ)

        #backprop
        dLdA = self.meanpool2d_stride1.backward(temp)

        return dLdA