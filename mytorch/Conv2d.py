import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)  #shape:[Cout, 1]
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape) #shape:[Cout, Cin, K, K]
        self.dLdb = np.zeros(self.b.shape) #shape:[Cout, 1]

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A

        #input para
        [N, C_in, H_in, W_in]  = A.shape

        #output para
        C_out = self.out_channels
        H_out = H_in - self.kernel_size + 1
        W_out = W_in - self.kernel_size + 1

        #initialize output
        Z = np.zeros((N, C_out, H_out, W_out))

        #convol
        for i in range(H_out):
            for j in range(W_out):
                # A.shape:[N, C_in, H_in, W_in] , W.shape:[C_out, C_in, kernel, kernel]
                Z[:, :, i, j] = \
                    np.tensordot(A[:,:,i:i+self.kernel_size,j:j+self.kernel_size],\
                                self.W, ([1,2,3],[1,2,3]))
                
        for i in range(C_out):
            Z[:,i,:,:] += self.b[i]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        #dLdW -> convol A by dLdZ
        kernel_h1, kernel_w1 = dLdZ.shape[2], dLdZ.shape[3]
        h_out = self.A.shape[2] - kernel_h1 + 1
        w_out = self.A.shape[3] - kernel_w1 + 1      
        
        self.dLdW = np.zeros((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        for i in range(h_out):
            for j in range (w_out):
                self.dLdW[:,:, i, j] = np.tensordot(dLdZ, self.A[:,:,i:i+kernel_h1,j:j+kernel_w1], axes = ([0,2,3],[0,2,3]))
    
        #dLdb -> dLdZ sum
        self.dLdb = np.sum(dLdZ, axis = (0,2,3))  # TODO

        #dLdA (N, Cin, H) -> convol dLdZ_padded (N, Cout, Hout, Wout) by W_flipped (Cout, Cin, K, K)
        k = self.kernel_size - 1
        dLdZ_padded = np.pad(dLdZ, ((0,0),(0,0),(k,k),(k,k)), 'constant', constant_values = 0)
        W_flipped = self.W[:, :, ::-1, ::-1]

        dLdA = np.zeros_like(self.A)
        for i in range(self.A.shape[2]):
            for j in range(self.A.shape[3]):
                dLdA[:,:,i,j] = np.tensordot(dLdZ_padded[:,:,i:i+self.kernel_size,j:j+self.kernel_size],\
                                    W_flipped, axes = ([1,2,3],[0,2,3]))
        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        # Call Conv2d_stride1
        temp = self.conv2d_stride1.forward(A)

        # downsample
        Z = self.downsample2d.forward(temp)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample1d backward
        temp = self.downsample2d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(temp)

        return dLdA
