#@author M. Schultheiss

from keras.models import Model
from keras.layers import *

class UNet:
    """UNet Convolutional Neural Network.

	Literature:
        U-Net: Convolutional Networks for Biomedical Image Segmentation. Ronneberger et al. 2015
    """
    def __init__(self, slice_shape, filter_count=64, layer_depth=4, kernel_size_down=(3,3), kernel_size_pool=(3,3), dropout=True, activation="sigmoid", is_flat=False, dilation_rates=[0,0,0,0,0,0,0,0,0], output_channels=1):
        """
        Args:
            slice_shape: The shape of the input slices.
            filter_count: The number of filters used on the first UNet layer.
            layer_depth: The depth of the UNet. Default is 4.

        Returns:
            Nothing. Use get_keras_model to obtain keras model.
        """
        if len(slice_shape) != 3:
            raise ValueError("Slice shape must have 3 dimensions (X,Y, channel), e.g. [256,256,1] for a single CT slide.")

        self.slice_shape = slice_shape
        self.layer_depth = layer_depth
        self.dilation_rates = dilation_rates
        self.kernel_size_down = kernel_size_down
        self.kernel_size_pool = kernel_size_pool
        self.filter_count = filter_count
        self.dropout = dropout
        self.activation = activation
        self.is_flat= is_flat
        self.output_channels = output_channels

    def visualize(self, filename):
        from keras.utils import plot_model
        plot_model(self.get_keras_model(), to_file=filename)

    def get_keras_model(self):
        """
            Returns a keras model class representing a U-Net convolutional neural network.
        """
        input_layer = Input(shape=self.slice_shape)
        lastLayer = input_layer

        layers = {}
        for i in range(0, self.layer_depth):
            print(i)
            filterCount = self.filter_count*(2**i)

            if self.dilation_rates[i] != 0:
                lastLayer = Conv2D(filters=filterCount, dilation_rate= self.dilation_rates[i], kernel_size=self.kernel_size_down, activation='relu', padding='same', name="conv_down_"+str(i+1))(lastLayer)

            else:
                lastLayer = Conv2D(filters=filterCount,  kernel_size=self.kernel_size_down, activation='relu', padding='same', name="conv_down_"+str(i+1))(lastLayer)


            if not self.is_flat:
                lastLayer = Conv2D(filters=filterCount, kernel_size=self.kernel_size_down, activation='relu',
                                   padding='same', name="conv_down_" + str(i + 1)+"b")(lastLayer)

            layers[i] = lastLayer
            lastLayer = MaxPool2D(pool_size=self.kernel_size_pool, name="pool_layer_"+str(i+1))(lastLayer) # Downsample by factor 2

        layersUp = {}
        for i in range(self.layer_depth-1, -1, -1): # Iterate layers from bottom but do not pass the top layer
            print(i)
            lastLayer  = UpSampling2D(size=self.kernel_size_pool, name="upsampling_layer_"+str(i+1))(lastLayer) # Sample bottom layer up
            filterCount = self.filter_count*(2**i)
            # ...and concatenate with next layer:
            lastLayer = concatenate([lastLayer, layers[i]], axis=-1)

            # Apply Convolutional Layer on Concatenation
            lastLayer = Conv2D(filterCount, kernel_size=self.kernel_size_down, activation='relu', padding='same', name="conv_up_"+str(i+1))(lastLayer)
            if not self.is_flat:
                lastLayer = Conv2D(filterCount, kernel_size=self.kernel_size_down, activation='relu', padding='same',
                                   name="conv_up_" + str(i + 1)+"b")(lastLayer)

            layersUp[i] = lastLayer

        l = Conv2D(filters=64, kernel_size=(1,1), activation='relu')(lastLayer)
        if self.dropout and isinstance(self.dropout, (bool)):
            l = Dropout(0.4)(l)
        elif self.dropout != False and isinstance(self.dropout, (float)):
            l = Dropout(self.dropout)(l)
        elif self.dropout == False and isinstance(self.dropout, (bool)):
            pass
        else:
            raise ValueError("dropout parameter must be either bool or float.")

        output_layer = Conv2D(filters=self.output_channels, kernel_size=(1,1), activation=self.activation)(l)
        model = Model(input_layer, output_layer)

        return model


if __name__ == "__main__":
    from keras.utils import plot_model
    u = UNet((512, 512, 64), 64)

