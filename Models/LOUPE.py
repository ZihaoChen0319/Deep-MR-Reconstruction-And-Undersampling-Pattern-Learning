"""
    Based on https://github.com/cagladbahadir/LOUPE

    For more details, please read:

    Bahadir, Cagla Deniz, Adrian V. Dalca, and Mert R. Sabuncu.
    "Learning-based Optimization of the Under-sampling Pattern in MRI."
    IPMI 2019. https://arxiv.org/abs/1901.01960.
"""

# import
import tensorflow as tf
import keras.models
from keras import backend as K
from keras.layers import Layer, Input, LeakyReLU, Conv2D, Conv2DTranspose, UpSampling2D, Concatenate, Add, \
    BatchNormalization
from tensorflow_addons.layers import InstanceNormalization


class RescaleProbMap(Layer):
    """
    Rescale Probability Map

    given a prob map x, rescales it so that it obtains the desired sparsity

    if mean(x) > sparsity, then rescaling is easy: x' = x * sparsity / mean(x)
    if mean(x) < sparsity, one can basically do the same thing by rescaling
                            (1-x) appropriately, then taking 1 minus the result.
    """

    def __init__(self, sparsity, **kwargs):
        self.sparsity = sparsity
        super(RescaleProbMap, self).__init__(**kwargs)

    def build(self, input_shape):
        super(RescaleProbMap, self).build(input_shape)

    def call(self, x):
        xbar = K.mean(x)
        r = self.sparsity / xbar
        beta = (1 - self.sparsity) / (1 - xbar)

        # compute adjucement
        le = tf.cast(tf.less_equal(r, 1), tf.float32)
        return le * x * r + (1 - le) * (1 - (1 - x) * beta)

    def compute_output_shape(self, input_shape):
        return input_shape


class ProbMask(Layer):
    """
    Probability mask layer
    Contains a layer of weights, that is then passed through a sigmoid.

    Modified from Local Linear Layer code in https://github.com/adalca/neuron
    """

    def __init__(self, slope=10,
                 initializer=None,
                 **kwargs):
        """
        note that in v1 the initial initializer was uniform in [-A, +A] where A is some scalar.
        e.g. was RandomUniform(minval=-2.0, maxval=2.0, seed=None),
        But this is uniform *in the logit space* (since we take sigmoid of this), so probabilities
        were concentrated a lot in the edges, which led to very slow convergence, I think.

        IN v2, the default initializer is a logit of the uniform [0, 1] distribution,
        which fixes this issue
        """

        if initializer == None:
            self.initializer = self._logit_slope_random_uniform
        else:
            self.initializer = initializer

        # higher slope means a more step-function-like logistic function
        # note: slope is converted to a tensor so that we can update it
        #   during training if necessary
        self.slope = tf.Variable(slope, dtype=tf.float32)
        super(ProbMask, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        takes as input the input data, which is [N x ... x 2]
        """
        # Create a trainable weight variable for this layer.
        lst = list(input_shape)
        lst[-1] = 1
        input_shape_h = tuple(lst)

        self.mult = self.add_weight(name='logit_weights',
                                    shape=input_shape_h[1:],
                                    initializer=self.initializer,
                                    trainable=True)

        super(ProbMask, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        logit_weights = 0 * x[..., 0:1] + self.mult
        return tf.sigmoid(self.slope * logit_weights)

    def compute_output_shape(self, input_shape):
        lst = list(input_shape)
        lst[-1] = 1
        return tuple(lst)

    def _logit_slope_random_uniform(self, shape, dtype=None, eps=0.01):
        # eps could be very small, or somethinkg like eps = 1e-6
        #   the idea is how far from the tails to have your initialization.
        x = K.random_uniform(shape, dtype=dtype, minval=eps, maxval=1.0 - eps)  # [0, 1]

        # logit with slope factor
        return - tf.math.log(1. / x - 1.) / self.slope


class ThresholdRandomMask(Layer):
    """
    Local thresholding layer

    Takes as input the input to be thresholded, and the threshold

    Modified from Local Linear Layer code in https://github.com/adalca/neuron
    """

    def __init__(self, slope=12, **kwargs):
        """
        if slope is None, it will be a hard threshold.
        """
        # higher slope means a more step-function-like logistic function
        # note: slope is converted to a tensor so that we can update it
        #   during training if necessary
        self.slope = None
        if slope is not None:
            self.slope = tf.Variable(slope, dtype=tf.float32)
        super(ThresholdRandomMask, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ThresholdRandomMask, self).build(input_shape)

    def call(self, x):
        inputs = x[0]
        thresh = x[1]
        if self.slope is not None:
            return tf.sigmoid(self.slope * (inputs - thresh))
        else:
            return thresh < inputs

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class RandomMask(Layer):
    """
    Create a random binary mask of the same size as the input shape
    """

    def __init__(self, **kwargs):
        super(RandomMask, self).__init__(**kwargs)

    def build(self, input_shape):
        super(RandomMask, self).build(input_shape)

    def call(self, x):
        input_shape = tf.shape(x)
        threshs = K.random_uniform(input_shape, minval=0.0, maxval=1.0, dtype='float32')
        return (0 * x) + threshs

    def compute_output_shape(self, input_shape):
        return input_shape


class ComplexAbs(Layer):
    """
    Complex Absolute

    Inputs: [kspace, mask]
    """

    def __init__(self, **kwargs):
        super(ComplexAbs, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ComplexAbs, self).build(input_shape)

    def call(self, inputs):
        two_channel = tf.complex(inputs[..., 0], inputs[..., 1])
        two_channel = tf.expand_dims(two_channel, -1)

        two_channel = tf.abs(two_channel)
        two_channel = tf.cast(two_channel, tf.float32)
        return two_channel

    def compute_output_shape(self, input_shape):
        list_input_shape = list(input_shape)
        list_input_shape[-1] = 1
        return tuple(list_input_shape)


class UnderSample(Layer):
    """
    Under-sampling by multiplication of k-space with the mask

    Inputs: [kspace (2-channel), mask (single-channel)]
    """

    def __init__(self, **kwargs):
        super(UnderSample, self).__init__(**kwargs)

    def build(self, input_shape):
        super(UnderSample, self).build(input_shape)

    def call(self, inputs):
        k_space_r = tf.multiply(inputs[0][..., 0], inputs[1][..., 0])
        k_space_i = tf.multiply(inputs[0][..., 1], inputs[1][..., 0])

        k_space = tf.stack([k_space_r, k_space_i], axis=-1)
        k_space = tf.cast(k_space, tf.float32)
        return k_space

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class ConcatenateZero(Layer):
    """
    Concatenate input with a zero'ed version of itself

    Input: tf.float32 of size [batch_size, ..., n]
    Output: tf.float32 of size [batch_size, ..., n*2]
    """

    def __init__(self, **kwargs):
        super(ConcatenateZero, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ConcatenateZero, self).build(input_shape)

    def call(self, inputx):
        return tf.concat([inputx, inputx * 0], -1)

    def compute_output_shape(self, input_shape):
        input_shape_list = list(input_shape)
        input_shape_list[-1] *= 2
        return tuple(input_shape_list)


class FFT(Layer):
    """
    fft layer, assuming the real/imag are input/output via two features

    Input: tf.float32 of size [batch_size, ..., 2]
    Output: tf.float32 of size [batch_size, ..., 2]
    """

    def __init__(self, **kwargs):
        super(FFT, self).__init__(**kwargs)

    def build(self, input_shape):
        # some input checking
        assert input_shape[-1] == 2, 'input has to have two features'
        self.ndims = len(input_shape) - 2
        assert self.ndims in [1, 2, 3], 'only 1D, 2D or 3D supported'

        # super
        super(FFT, self).build(input_shape)

    def call(self, inputx):
        assert inputx.shape.as_list()[-1] == 2, 'input has to have two features'

        # get the right fft
        if self.ndims == 1:
            fft = tf.signal.fft
        elif self.ndims == 2:
            fft = tf.signal.fft2d
        else:
            fft = tf.signal.fft3d

        # get fft complex image
        fft_im = fft(tf.complex(inputx[..., 0], inputx[..., 1]))
        fft_im = tf.signal.fftshift(fft_im, axes=(1, 2))

        # go back to two-feature representation
        fft_im = tf.stack([tf.math.real(fft_im), tf.math.imag(fft_im)], axis=-1)
        return tf.cast(fft_im, tf.float32)

    def compute_output_shape(self, input_shape):
        return input_shape


class IFFT(Layer):
    """
    ifft layer, assuming the real/imag are input/output via two features

    Input: tf.float32 of size [batch_size, ..., 2]
    Output: tf.float32 of size [batch_size, ..., 2]
    """

    def __init__(self, **kwargs):
        super(IFFT, self).__init__(**kwargs)

    def build(self, input_shape):
        # some input checking
        assert input_shape[-1] == 2, 'input has to have two features'
        self.ndims = len(input_shape) - 2
        assert self.ndims in [1, 2, 3], 'only 1D, 2D or 3D supported'

        # super
        super(IFFT, self).build(input_shape)

    def call(self, inputx):
        assert inputx.shape.as_list()[-1] == 2, 'input has to have two features'

        # get the right fft
        if self.ndims == 1:
            ifft = tf.signal.ifft
        elif self.ndims == 2:
            ifft = tf.signal.ifft2d
        else:
            ifft = tf.signal.ifft3d

        # get ifft complex image
        ifft_im = tf.complex(inputx[..., 0], inputx[..., 1])
        ifft_im = ifft(tf.signal.ifftshift(ifft_im, axes=(1, 2)))

        # go back to two-feature representation
        ifft_im = tf.stack([tf.math.real(ifft_im), tf.math.imag(ifft_im)], axis=-1)
        return tf.cast(ifft_im, tf.float32)

    def compute_output_shape(self, input_shape):
        return input_shape


def loupe_model(input_shape=(256, 256, 1),
                n_class=1,
                filt=32,
                kern=3,
                sparsity=None,
                pmask_slope=5,
                pmask_init=None,
                sample_slope=12,
                model_type='v2',
                mask_type=None):
    """
    loupe_model

    Parameters:
        input_shape: input shape
        filt: number of base filters
        kern: kernel size
        model_type: 'unet', 'v1', 'v2'
        sparsity: desired sparsity (only for model type 'v2')
        pmask_slope: slope of logistic parameter in probability mask
        acti=None: activation

    Returns:
        keras model

    UNet leaky two channel
    """

    if model_type == 'unet':  # Creates a single U-Net
        inputs = Input(shape=input_shape, name='input_layer')
        last_tensor = inputs

    elif model_type == 'loupe_mask_input':

        # inputs
        inputs = Input(shape=input_shape, name='input_layer')

        last_tensor = inputs
        # if necessary, concatenate with zeros for FFT
        if input_shape[-1] == 1:
            last_tensor = ConcatenateZero(name='concat_zero')(last_tensor)

        # input -> kspace via FFT
        last_tensor = FFT(name='fft')(last_tensor)

        # input mask
        last_tensor_mask = inputs[1]

        # Under-sample and back to image space via IFFT
        last_tensor = UnderSample(name='under_sample_kspace')([last_tensor, last_tensor_mask])
        last_tensor = IFFT(name='under_sample_img')(last_tensor)

    else:  # Creates LOUPE
        assert model_type in ['v1', 'v2'], 'model_type should be unet, v1 or v2'

        # inputs
        inputs = Input(shape=input_shape, name='input_layer')

        last_tensor = inputs
        # if necessary, concatenate with zeros for FFT
        if input_shape[-1] == 1:
            last_tensor = ConcatenateZero(name='concat_zero')(last_tensor)

        # input -> kspace via FFT
        last_tensor = FFT(name='fft')(last_tensor)

        # build probability mask
        prob_mask_tensor = ProbMask(name='prob_mask', slope=pmask_slope, initializer=pmask_init)(last_tensor)

        if model_type == 'v2':
            assert sparsity is not None, 'for this model, need desired sparsity to be specified'
            prob_mask_tensor = RescaleProbMap(sparsity, name='prob_mask_scaled')(prob_mask_tensor)

        else:
            assert sparsity is None, 'for v1 model, cannot specify sparsity'

        # Realization of probability mask
        thresh_tensor = RandomMask(name='random_mask')(prob_mask_tensor)
        last_tensor_mask = ThresholdRandomMask(slope=sample_slope, name='sampled_mask')(
            [prob_mask_tensor, thresh_tensor])

        # Under-sample and back to image space via IFFT
        last_tensor = UnderSample(name='under_sample_kspace')([last_tensor, last_tensor_mask])
        last_tensor = IFFT(name='under_sample_img')(last_tensor)

    # hard-coded UNet
    recon_tensor = _unet_from_tensor(last_tensor, filt, kern, output_nb_feats=1, name='recon_unet')

    # complex absolute layer
    abs_tensor = ComplexAbs(name='complex_addition')(last_tensor)

    # recon output from model
    recon_tensor = Add(name='recon_final')([abs_tensor, recon_tensor])

    # segmentation output
    seg_tensor = _unet_from_tensor(recon_tensor, filt, kern, output_nb_feats=n_class, name='seg_unet')

    # prepare and output a model as necessary
    outputs = [recon_tensor, seg_tensor]
    if model_type == 'v1':
        outputs += [last_tensor_mask]

    return keras.models.Model(inputs, outputs)


def _unet_from_tensor(tensor, filt, kern, name=None, output_nb_feats=1):
    """
    UNet in nnUNet style
    """

    # start first convolution of UNet
    conv1 = Conv2D(filt, kern, padding='same')(tensor)
    conv1 = InstanceNormalization()(conv1)
    conv1 = LeakyReLU()(conv1)
    conv1 = Conv2D(filt, kern, padding='same')(conv1)
    conv1 = InstanceNormalization()(conv1)
    conv1 = LeakyReLU()(conv1)

    conv2 = Conv2D(filt * 2, kern, strides=2, padding='same')(conv1)
    conv2 = InstanceNormalization()(conv2)
    conv2 = LeakyReLU()(conv2)
    conv2 = Conv2D(filt * 2, kern, padding='same')(conv2)
    conv2 = InstanceNormalization()(conv2)
    conv2 = LeakyReLU()(conv2)

    conv3 = Conv2D(filt * 4, kern, strides=2, padding='same')(conv2)
    conv3 = InstanceNormalization()(conv3)
    conv3 = LeakyReLU()(conv3)
    conv3 = Conv2D(filt * 4, kern, padding='same')(conv3)
    conv3 = InstanceNormalization()(conv3)
    conv3 = LeakyReLU()(conv3)

    conv4 = Conv2D(filt * 8, kern, strides=2, padding='same',)(conv3)
    conv4 = InstanceNormalization()(conv4)
    conv4 = LeakyReLU()(conv4)
    conv4 = Conv2D(filt * 8, kern, padding='same')(conv4)
    conv4 = InstanceNormalization()(conv4)
    conv4 = LeakyReLU()(conv4)

    conv5 = Conv2D(filt * 16, kern, strides=2, padding='same')(conv4)
    conv5 = InstanceNormalization()(conv5)
    conv5 = LeakyReLU()(conv5)
    conv5 = Conv2D(filt * 16, kern, padding='same')(conv5)
    conv5 = InstanceNormalization()(conv5)
    conv5 = LeakyReLU()(conv5)

    sub1 = Conv2DTranspose(filt * 8, kernel_size=2, strides=2)(conv5)
    concat1 = Concatenate(axis=-1)([conv4, sub1])

    conv6 = Conv2D(filt * 8, kern, padding='same')(concat1)
    conv6 = InstanceNormalization()(conv6)
    conv6 = LeakyReLU()(conv6)
    conv6 = Conv2D(filt * 8, kern, padding='same')(conv6)
    conv6 = InstanceNormalization()(conv6)
    conv6 = LeakyReLU()(conv6)

    sub2 = Conv2DTranspose(filt * 4, kernel_size=2, strides=2)(conv6)
    concat2 = Concatenate(axis=-1)([conv3, sub2])

    conv7 = Conv2D(filt * 4, kern, padding='same')(concat2)
    conv7 = InstanceNormalization()(conv7)
    conv7 = LeakyReLU()(conv7)
    conv7 = Conv2D(filt * 4, kern, padding='same')(conv7)
    conv7 = InstanceNormalization()(conv7)
    conv7 = LeakyReLU()(conv7)

    sub3 = Conv2DTranspose(filt * 2, kernel_size=2, strides=2)(conv7)
    concat3 = Concatenate(axis=-1)([conv2, sub3])

    conv8 = Conv2D(filt * 2, kern, padding='same')(concat3)
    conv8 = InstanceNormalization()(conv8)
    conv8 = LeakyReLU()(conv8)
    conv8 = Conv2D(filt * 2, kern, padding='same')(conv8)
    conv8 = InstanceNormalization()(conv8)
    conv8 = LeakyReLU()(conv8)

    sub4 = Conv2DTranspose(filt, kernel_size=2, strides=2)(conv8)
    concat4 = Concatenate(axis=-1)([conv1, sub4])

    conv9 = Conv2D(filt, kern, padding='same')(concat4)
    conv9 = InstanceNormalization()(conv9)
    conv9 = LeakyReLU()(conv9)
    conv9 = Conv2D(filt, kern, padding='same')(conv9)
    conv9 = InstanceNormalization()(conv9)
    conv9 = LeakyReLU()(conv9)
    conv9 = Conv2D(output_nb_feats, 1, padding='same', name=name)(conv9)

    return conv9
