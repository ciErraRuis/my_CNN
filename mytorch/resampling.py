import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        W_input = A.shape[2]
        W_output = self.upsampling_factor * (W_input - 1) + 1
        
        Z = np.zeros((A.shape[0], A.shape[1], W_output))
        
        Z[:, :, ::self.upsampling_factor] = A  # 将第downsampling * n位置的0变成A对应的值

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        dLdA = dLdZ[:, :, ::self.upsampling_factor]  # 每隔downsampling_factor项取一个dLdZ的值

        return dLdA


class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        [self.N, self.in_channels, self.W] = A.shape

        Z = A[:, :, ::self.downsampling_factor]  # 每隔downsampling_factor项取一个A的值

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        dLdA = np.zeros((self.N, self.in_channels, self.W))
        dLdA[:, :, ::self.downsampling_factor] = dLdZ  #将第downsampling * n位置的0变成dLdZ的值

        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        input_height, input_width = A.shape[2], A.shape[3]
        output_height, output_width = self.upsampling_factor * (input_height - 1) + 1, self.upsampling_factor * (input_width - 1) + 1 
        Z = np.zeros((A.shape[0], A.shape[1], output_height, output_width))  # Z(batch_size, inchannels, output_height, out_put width)
        Z[:, :, ::self.upsampling_factor, ::self.upsampling_factor] = A
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        dLdA = dLdZ[:, :, ::self.upsampling_factor, ::self.upsampling_factor]

        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        [self.N, self.in_channels, self.W1, self.W2] = A.shape
        Z = A[:, :, ::self.downsampling_factor, ::self.downsampling_factor]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        dLdA = np.zeros((self.N, self.in_channels, self.W1, self.W2))
        dLdA[:, :, ::self.downsampling_factor, ::self.downsampling_factor] = dLdZ  
        
        return dLdA