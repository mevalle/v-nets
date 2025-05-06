# Importing the copy module for creating shallow or deep copies of objects
import copy

# Importing the math module for mathematical functions
import math
import numpy as np
import tensorflow as tf

# Importing activation functions from Keras
from keras.src import activations  
# Importing constraints for layer weights
from keras.src import constraints  
# Importing initializers for layer weights
from keras.src import initializers  
# Importing operations module for various tensor operations
from keras.src import ops  
# Importing regularizers for applying penalties on layer weights
from keras.src import regularizers  
# Importing function to standardize data format
from keras.src.backend import standardize_data_format  
# Importing InputSpec for defining input specifications for layers
from keras.src.layers.input_spec import InputSpec  
# Importing base Layer class for creating custom layers
from keras.src.layers.layer import Layer  
# Importing function to compute output shape of convolutional layers
from keras.src.ops.operation_utils import compute_conv_output_shape  
# Importing functions to standardize padding arguments
from keras.src.utils.argument_validation import standardize_padding  
# Importing function to standardize tuple arguments
from keras.src.utils.argument_validation import standardize_tuple 


# Vector-valued Basic Layers:

class V_Dense(Layer):
    def __init__(self, 
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 name = "V-Dense",
                 algebra = np.stack([np.array([[1.0,0.0],[0.0,-1.0]]),np.array([[0.0,1.0],[1.0,0.0]])],axis=-1), # Complex Numbers
                ):
        super(V_Dense, self).__init__()
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.name = name
        self.algebra = np.float32(algebra)
        self.algdim = algebra.shape[2]

    def build(self, input_shape):
        assert input_shape[-1] % self.algdim == 0
        input_dim = input_shape[-1] // self.algdim

        # Real-part of the weights
        self.W = self.add_weight(
            shape=(input_dim, self.units,self.algdim),
            initializer="glorot_normal",
            trainable=True,
        )

        if self.use_bias:
            # Bias complex-valued
            self.b = self.add_weight(
                shape=(self.algdim * self.units,),
                initializer="zeros",
                trainable=True,
            )

    def call(self, inputs):
        W = tf.reduce_sum(
            [tf.experimental.numpy.kron(self.algebra[i,:,:],self.W[:,:,i]) for i in range(self.algdim)]
            ,axis=0)

        outputs = tf.matmul(inputs, W)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.b)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

class V_Conv2D(Layer):
    """Vector-valued 2D convolution layer.

    This layer creates a vector-valued convolution kernel that is convolved with the layer
    input over a 2D spatial (or temporal) dimension (height and width) to
    produce a tensor of outputs. 
    The feature channels are supposed to be grouped as follows
    [real part of channel 1,..., real part of channel C, imag1 part of channel 1,...., imag1 part of channel C,....] 
    If `use_bias` is True, a bias vector is created and added to the outputs. 
    Finally, if `activation` is not `None`, it is applied to the outputs as well.

    Args:
        filters: int, the dimension of the output space (the number of filters
            in the convolution).
        kernel_size: int or tuple/list of 2 integer, specifying the size of the
            convolution window.
        strides: int or tuple/list of 2 integer, specifying the stride length
            of the convolution. `strides > 1` is incompatible with
            `dilation_rate > 1`.
        padding: string, either `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input. When `padding="same"` and
            `strides=1`, the output has the same size as the input.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape
            `(batch_size, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        dilation_rate: int or tuple/list of 2 integers, specifying the dilation
            rate to use for dilated convolution.
        groups: A positive int specifying the number of groups in which the
            input is split along the channel axis. Each group is convolved
            separately with `filters // groups` filters. The output is the
            concatenation of all the `groups` results along the channel axis.
            Input channels and `filters` must both be divisible by `groups`.
        activation: Activation function. If `None`, no activation is applied.
        use_bias: bool, if `True`, bias will be added to the output.
        kernel_initializer: Initializer for the convolution kernel. If `None`,
            the default initializer (`"glorot_uniform"`) will be used.
        bias_initializer: Initializer for the bias vector. If `None`, the
            default initializer (`"zeros"`) will be used.
        kernel_regularizer: Optional regularizer for the convolution kernel.
        bias_regularizer: Optional regularizer for the bias vector.
        activity_regularizer: Optional regularizer function for the output.
        kernel_constraint: Optional projection function to be applied to the
            kernel after being updated by an `Optimizer` (e.g. used to implement
            norm constraints or value constraints for layer weights). The
            function must take as input the unprojected variable and must return
            the projected variable (which must have the same shape). Constraints
            are not safe to use when doing asynchronous distributed training.
        bias_constraint: Optional projection function to be applied to the
            bias after being updated by an `Optimizer`.
        algebra: The algebra (or hypercomplex algebra). Default - complex numbers;
        

    Input shape: Currently only accepts 4D tensors with shape: 
        `(batch_size, height, width, (algebra dimension)*channels)`
    The feature channels are supposed to be grouped. For example, 
    for a complex-valued image with C feature channels, the input must be structured sequentially 
    first with the real part of the C channels followed by the imaginary part of the same C channels.
    
    Output shape: A 4D tensor with shape: 
        `(batch_size, new_height, new_width, (algebra dimension)*filters)`
    Like the input, the feature channels are grouped. For example, 
    for a complex-valued image with C feature channels, the input must be structured sequentially 
    first with the real part of the C channels followed by the imaginary part of the same C channels.

    Returns:
        A 4D tensor representing `activation(conv2d(inputs, kernel) + bias)`.

    Raises:
        ValueError: when both `strides > 1` and `dilation_rate > 1`.

    Example:

    >>> x = np.random.rand(4, 10, 10, 128)
    >>> y = keras.layers.Conv2D(32, 3, algebra=get_algebra("ComplexNumbers"), activation='relu')(x)
    >>> print(y.shape)
    (4, 8, 8, 64)
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1),
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        lora_rank=None,
        rank = 2,
        algebra = np.stack([np.array([[1.0,0.0],[0.0,-1.0]]),np.array([[0.0,1.0],[1.0,0.0]])],axis=-1), # Complex Numbers
        **kwargs,
    ):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.rank = rank
        self.filters = filters
        self.groups = groups
        self.kernel_size = standardize_tuple(kernel_size, rank, "kernel_size")
        self.strides = standardize_tuple(strides, rank, "strides")
        self.dilation_rate = standardize_tuple(
            dilation_rate, rank, "dilation_rate"
        )
        self.padding = standardize_padding(padding, allow_causal=rank == 1)
        self.data_format = standardize_data_format(data_format)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.lora_rank = lora_rank
        self.lora_enabled = False
        self.input_spec = InputSpec(min_ndim=self.rank + 2)
        self.data_format = self.data_format
        self.algebra = algebra
        self.algdim = algebra.shape[2]

        if self.filters is not None and self.filters <= 0:
            raise ValueError(
                "Invalid value for argument `filters`. Expected a strictly "
                f"positive value. Received filters={self.filters}."
            )

        if self.groups <= 0:
            raise ValueError(
                "The number of groups must be a positive integer. "
                f"Received: groups={self.groups}."
            )

        if self.filters is not None and self.filters % self.groups != 0:
            raise ValueError(
                "The number of filters must be evenly divisible by the "
                f"number of groups. Received: groups={self.groups}, "
                f"filters={self.filters}."
            )

        if not all(self.kernel_size):
            raise ValueError(
                "The argument `kernel_size` cannot contain 0. Received "
                f"kernel_size={self.kernel_size}."
            )

        if not all(self.strides):
            raise ValueError(
                "The argument `strides` cannot contains 0. Received "
                f"strides={self.strides}"
            )

        if max(self.strides) > 1 and max(self.dilation_rate) > 1:
            raise ValueError(
                "`strides > 1` not supported in conjunction with "
                f"`dilation_rate > 1`. Received: strides={self.strides} and "
                f"dilation_rate={self.dilation_rate}"
            )

    def build(self, input_shape):
        if self.data_format == "channels_last":
            channel_axis = -1
            input_channel = input_shape[-1]
        else:
            raise ValueError(
                "The current version only accepts channels_last!"
            )
            channel_axis = 1
            input_channel = input_shape[1]
        self.input_spec = InputSpec(
            min_ndim=self.rank + 2, axes={channel_axis: input_channel}
        )
        if input_channel % (self.groups*self.algdim) != 0:
            raise ValueError(
                "The number of input channels must be evenly divisible by "
                f"the number of groups times the algebra dimension. Received {self.groups*self.algdim}, but the "
                f"input has {input_channel} channels (full input shape is "
                f"{input_shape})."
            )
        kernel_shape = self.kernel_size + (
            input_channel // (self.groups*self.algdim),
            self.filters, self.algdim,
        )

        # compute_output_shape contains some validation logic for the input
        # shape, and make sure the output shape has all positive dimensions.
        self.compute_output_shape(input_shape)

        self._kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters*self.algdim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.bias = None
        self.built = True
        if self.lora_rank:
            self.enable_lora(self.lora_rank)

    @property
    def kernel(self):
        if not self.built:
            raise AttributeError(
                "You must build the layer before accessing `kernel`."
            )
        if self.lora_enabled:
            return self._kernel + ops.matmul(
                self.lora_kernel_a, self.lora_kernel_b
            )
        return self._kernel

    def convolution_op(self, inputs, kernel):
        return ops.conv(
            inputs,
            kernel,
            strides=list(self.strides),
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format,
        )

    def call(self, inputs):
        W = sum([tf.experimental.numpy.kron(self.algebra[k,:,:].reshape(1,1,self.algdim,self.algdim),self.kernel[:,:,:,:,k]) for k in range(self.algdim)])
        # print("W = ",W)
        # print("algebra = ",self.algebra)
        outputs = self.convolution_op(
            inputs,
            kernel = tf.cast(W,dtype=self.dtype),
        )
        if self.use_bias:
            if self.data_format == "channels_last":
                bias_shape = (1,) * (self.rank + 1) + (self.filters*self.algdim,)
            else:
                bias_shape = (1, self.filters*self.algdim) + (1,) * self.rank
            bias = ops.reshape(self.bias, bias_shape)
            outputs += bias

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return compute_conv_output_shape(
            input_shape,
            self.algdim*self.filters,
            self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

    def enable_lora(
        self, rank, a_initializer="he_uniform", b_initializer="zeros"
    ):
        if self.kernel_constraint:
            raise ValueError(
                "Lora is incompatible with kernel constraints. "
                "In order to enable lora on this layer, remove the "
                "`kernel_constraint` argument."
            )
        if not self.built:
            raise ValueError(
                "Cannot enable lora on a layer that isn't yet built."
            )
        if self.lora_enabled:
            raise ValueError(
                "lora is already enabled. "
                "This can only be done once per layer."
            )
        self._tracker.unlock()
        self.lora_kernel_a = self.add_weight(
            name="lora_kernel_a",
            shape=self._kernel.shape[:-1] + (rank,),
            initializer=initializers.get(a_initializer),
            regularizer=self.kernel_regularizer,
        )
        self.lora_kernel_b = self.add_weight(
            name="lora_kernel_b",
            shape=(rank, self.filters),
            initializer=initializers.get(b_initializer),
            regularizer=self.kernel_regularizer,
        )
        self._kernel.trainable = False
        self._tracker.lock()
        self.lora_enabled = True
        self.lora_rank = rank

    def save_own_variables(self, store):
        # Do nothing if the layer isn't yet built
        if not self.built:
            return
        target_variables = [self.kernel]
        if self.use_bias:
            target_variables.append(self.bias)
        for i, variable in enumerate(target_variables):
            store[str(i)] = variable

    def load_own_variables(self, store):
        if not self.lora_enabled:
            self._check_load_own_variables(store)
        # Do nothing if the layer isn't yet built
        if not self.built:
            return
        target_variables = [self._kernel]
        if self.use_bias:
            target_variables.append(self.bias)
        for i, variable in enumerate(target_variables):
            variable.assign(store[str(i)])
        if self.lora_enabled:
            self.lora_kernel_a.assign(ops.zeros(self.lora_kernel_a.shape))
            self.lora_kernel_b.assign(ops.zeros(self.lora_kernel_b.shape))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "data_format": self.data_format,
                "dilation_rate": self.dilation_rate,
                "groups": self.groups,
                "activation": activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": regularizers.serialize(
                    self.bias_regularizer
                ),
                "activity_regularizer": regularizers.serialize(
                    self.activity_regularizer
                ),
                "kernel_constraint": constraints.serialize(
                    self.kernel_constraint
                ),
                "bias_constraint": constraints.serialize(self.bias_constraint),
            }
        )
        if self.lora_rank:
            config["lora_rank"] = self.lora_rank
        return config

    def _check_load_own_variables(self, store):
        all_vars = self._trainable_variables + self._non_trainable_variables
        if len(store.keys()) != len(all_vars):
            if len(all_vars) == 0 and not self.built:
                raise ValueError(
                    f"Layer '{self.name}' was never built "
                    "and thus it doesn't have any variables. "
                    f"However the weights file lists {len(store.keys())} "
                    "variables for this layer.\n"
                    "In most cases, this error indicates that either:\n\n"
                    "1. The layer is owned by a parent layer that "
                    "implements a `build()` method, but calling the "
                    "parent's `build()` method did NOT create the state of "
                    f"the child layer '{self.name}'. A `build()` method "
                    "must create ALL state for the layer, including "
                    "the state of any children layers.\n\n"
                    "2. You need to implement "
                    "the `def build_from_config(self, config)` method "
                    f"on layer '{self.name}', to specify how to rebuild "
                    "it during loading. "
                    "In this case, you might also want to implement the "
                    "method that generates the build config at saving time, "
                    "`def get_build_config(self)`. "
                    "The method `build_from_config()` is meant "
                    "to create the state "
                    "of the layer (i.e. its variables) upon deserialization.",
                )
            raise ValueError(
                f"Layer '{self.name}' expected {len(all_vars)} variables, "
                "but received "
                f"{len(store.keys())} variables during loading. "
                f"Expected: {[v.name for v in all_vars]}"
            )
            
            
class V_DepthwiseConv2D(Layer):
    """2D depthwise convolution layer.

    Depthwise convolution is a type of convolution in which each input vector-valued channel
    is convolved with a different vector-valued kernel (called a depthwise kernel). You can
    understand depthwise convolution as the first step in a depthwise separable convolution.

    It is implemented via the following steps:

    - Split the input into individual vector-valued channels. 
    - Convolve each channel with an individual depthwise vector-valued kernel with
      `depth_multiplier` output channels.
    - Concatenate the convolved outputs along the channels axis.

    Unlike a regular 2D convolution, depthwise convolution does not mix
    information across different input channels.
    
    *** The feature channels are supposed to be grouped as follows
        [real part of channel 1,..., real part of channel C, 
            imag1 part of channel 1,...., imag1 part of channel C,....]   ****

    The `depth_multiplier` argument determines how many filters are applied to
    one input channel. As such, it controls the amount of output channels that
    are generated per input channel in the depthwise step.

    Args:
        kernel_size: int or tuple/list of 2 integer, specifying the size of the
            depthwise convolution window.
        strides: int or tuple/list of 2 integer, specifying the stride length
            of the depthwise convolution. `strides > 1` is incompatible with
            `dilation_rate > 1`.
        padding: string, either `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input. When `padding="same"` and
            `strides=1`, the output has the same size as the input.
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel. The total number of depthwise convolution
            output channels will be equal to `input_channel * depth_multiplier`.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file
            at `~/.keras/keras.json`.
            If you never set it, then it will be `"channels_last"`.
        dilation_rate: int or tuple/list of 2 integers, specifying the dilation
            rate to use for dilated convolution.
        activation: Activation function. If `None`, no activation is applied.
        use_bias: bool, if `True`, bias will be added to the output.
        depthwise_initializer: Initializer for the convolution kernel.
            If `None`, the default initializer (`"glorot_uniform"`)
            will be used.
        bias_initializer: Initializer for the bias vector. If `None`, the
            default initializer (`"zeros"`) will be used.
        depthwise_regularizer: Optional regularizer for the convolution kernel.
        bias_regularizer: Optional regularizer for the bias vector.
        activity_regularizer: Optional regularizer function for the output.
        depthwise_constraint: Optional projection function to be applied to the
            kernel after being updated by an `Optimizer` (e.g. used to implement
            norm constraints or value constraints for layer weights). The
            function must take as input the unprojected variable and must return
            the projected variable (which must have the same shape). Constraints
            are not safe to use when doing asynchronous distributed training.
        bias_constraint: Optional projection function to be applied to the
            bias after being updated by an `Optimizer`.
        algebra: The underlying algebra. Defult - complex numbers.

    Input shape:

    - If `data_format="channels_last"`:
        A 4D tensor with shape: `(batch_size, height, width, channels)`
    - If `data_format="channels_first"`:
        A 4D tensor with shape: `(batch_size, channels, height, width)`

    Output shape:

    - If `data_format="channels_last"`:
        A 4D tensor with shape:
        `(batch_size, new_height, new_width, channels * depth_multiplier)`
    - If `data_format="channels_first"`:
        A 4D tensor with shape:
        `(batch_size, channels * depth_multiplier, new_height, new_width)`

    Returns:
        A 4D tensor representing
        `activation(depthwise_conv2d(inputs, kernel) + bias)`.

    Raises:
        ValueError: when both `strides > 1` and `dilation_rate > 1`.

    Example:

    >>> x = np.random.rand(4, 10, 10, 12)
    >>> y = keras.layers.DepthwiseConv2D(3, 3, activation='relu')(x)
    >>> print(y.shape)
    (4, 8, 8, 36)
    """

    def __init__(
        self,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        depth_multiplier=1,
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        depthwise_initializer="glorot_uniform",
        bias_initializer="zeros",
        depthwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        bias_constraint=None,
        rank=2,
        name = None,
        trainable=True,
        algebra = np.stack([np.array([[1.0,0.0],[0.0,-1.0]]),np.array([[0.0,1.0],[1.0,0.0]])],axis=-1), # Complex Numbers
        **kwargs,
    ):
        super().__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs,
        )
        self.rank = rank
        self.depth_multiplier = depth_multiplier
        self.kernel_size = standardize_tuple(kernel_size, rank, "kernel_size")
        self.strides = standardize_tuple(strides, rank, "strides")
        self.dilation_rate = standardize_tuple(
            dilation_rate, rank, "dilation_rate"
        )
        self.padding = standardize_padding(padding)
        self.data_format = standardize_data_format(data_format)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=self.rank + 2)
        self.data_format = self.data_format
        self.algebra = algebra
        self.algdim = algebra.shape[2]

        if self.depth_multiplier is not None and self.depth_multiplier <= 0:
            raise ValueError(
                "Invalid value for argument `depth_multiplier`. Expected a "
                "strictly positive value. Received "
                f"depth_multiplier={self.depth_multiplier}."
            )

        if not all(self.kernel_size):
            raise ValueError(
                "The argument `kernel_size` cannot contain 0. Received "
                f"kernel_size={self.kernel_size}."
            )

        if not all(self.strides):
            raise ValueError(
                "The argument `strides` cannot contains 0. Received "
                f"strides={self.strides}"
            )

        if max(self.strides) > 1 and max(self.dilation_rate) > 1:
            raise ValueError(
                "`strides > 1` not supported in conjunction with "
                f"`dilation_rate > 1`. Received: strides={self.strides} and "
                f"dilation_rate={self.dilation_rate}"
            )

    def build(self, input_shape):
        if self.data_format == "channels_last":
            channel_axis = -1
            input_channel = input_shape[-1]
        else:
            raise ValueError(
                "The current version only accepts channels_last!"
            )
            channel_axis = 1
            input_channel = input_shape[1]
        self.input_spec = InputSpec(
            min_ndim=self.rank + 2, axes={channel_axis: input_channel}
        )
        depthwise_shape = self.kernel_size + (
            input_channel//self.algdim,
            self.depth_multiplier,
            self.algdim
        )
        self.kernel = self.add_weight(
            name="kernel",
            shape=depthwise_shape,
            initializer=self.depthwise_initializer,
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint,
            trainable=True,
            dtype=self.dtype,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.depth_multiplier * input_channel * self.algdim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.bias = None
        self.built = True

    def _get_input_channel(self, input_shape):
        if self.data_format == "channels_last":
            input_channel = input_shape[-1]
        else:
            input_channel = input_shape[1]
        return input_channel

    def call(self, inputs):
        input_channel = self._get_input_channel(inputs.shape)
        
        W = sum([tf.experimental.numpy.kron(self.algebra[k,:,:].reshape(1,1,self.algdim,self.algdim),self.kernel[:,:,:,:,k]) for k in range(self.algdim)])
        out_expanded = ops.depthwise_conv(
            inputs,
            kernel = tf.cast(W,dtype=self.dtype),
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format,
        )
        outputs = sum([out_expanded[:,:,:,input_channel*i:input_channel*(i+1)] for i in range(self.algdim)])
        
        # Convert output from interleaved format [real_1, imag_1, real_2, imag_2, ..., real_N, imag_N] 
        # to separate real and imaginary parts format [real_1, ..., real_N, imag_1, ..., imag_N]
        # Create a list of indices for the last dimension of the outputs tensor
        ind = [i for i in range(outputs.shape[-1])]
        # Rearrange the indices to separate real and imaginary parts based
        ind = np.hstack([ind[i::self.algdim] for i in range(self.algdim)])
        # Gather the outputs tensor using the rearranged indices along the last dimension
        outputs = tf.gather(outputs, indices=ind, axis=-1)

        if self.use_bias:
            if self.data_format == "channels_last":
                bias_shape = (1,) * (self.rank + 1) + (
                    self.depth_multiplier * input_channel,
                )
            else:
                bias_shape = (1, self.depth_multiplier * input_channel) + (
                    1,
                ) * self.rank
            bias = ops.reshape(self.bias, bias_shape)
            outputs += bias

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_channel = self._get_input_channel(input_shape)
        return compute_conv_output_shape(
            input_shape,
            self.depth_multiplier * input_channel,
            self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depth_multiplier": self.depth_multiplier,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "data_format": self.data_format,
                "dilation_rate": self.dilation_rate,
                "activation": activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "depthwise_initializer": initializers.serialize(
                    self.depthwise_initializer
                ),
                "bias_initializer": initializers.serialize(
                    self.bias_initializer
                ),
                "depthwise_regularizer": regularizers.serialize(
                    self.depthwise_regularizer
                ),
                "bias_regularizer": regularizers.serialize(
                    self.bias_regularizer
                ),
                "activity_regularizer": regularizers.serialize(
                    self.activity_regularizer
                ),
                "depthwise_constraint": constraints.serialize(
                    self.depthwise_constraint
                ),
                "bias_constraint": constraints.serialize(self.bias_constraint),
            }
        )
        return config

# Some useful functions for manipulating the load and printing algebras;

def get_algebra(name="ComplexNumbers"):
    # Put the name in lowecase and without spaces:
    name = name.lower().replace(" ","")
    if name == "complexnumbers" or name == "complexnumber":
        return np.stack([np.array([[1.0,0.0],[0.0,-1.0]]),np.array([[0.0,1.0],[1.0,0.0]])],axis=-1) # Complex Numbers
    if name == "hyperbolicnumbers" or name == "hyperbolicnumber":
        return np.stack([np.array([[1.0,0.0],[0.0,1.0]]),np.array([[0.0,1.0],[1.0,0.0]])],axis=-1) # Complex Numbers
    if name == "dualnumbers" or name == "dualnumber":
        return np.stack([np.array([[1.0,0.0],[0.0,0.0]]),np.array([[0.0,1.0],[1.0,0.0]])],axis=-1) # Dual Numbers
    if name == "quaternions" or name == "quaternion":
        return 1.0*np.stack([np.array([[1,0,0,0],
                                      [0,-1,0,0],
                                      [0,0,-1,0],
                                      [0,0,0,-1]]),
                            np.array([[0,1,0,0],
                                      [1,0,0,0],
                                      [0,0,0,1],
                                      [0,0,-1,0]]),
                           np.array([[0,0,1,0],
                                     [0,0,0,-1],
                                     [1,0,0,0],
                                     [0,1,0,0]]),
                           np.array([[0,0,0,1],
                                     [0,0,1,0],
                                     [0,-1,0,0],
                                     [1,0,0,0]])],axis=-1)
    if name == "hyperbolicquaternions" or name == "hyperbolicquaternion":
        return 1.0*np.stack([np.array([[1,0,0,0],
                                      [0,1,0,0],
                                      [0,0,1,0],
                                      [0,0,0,1]]),
                            np.array([[0,1,0,0],
                                      [1,0,0,0],
                                      [0,0,0,1],
                                      [0,0,-1,0]]),
                           np.array([[0,0,1,0],
                                     [0,0,0,-1],
                                     [1,0,0,0],
                                     [0,1,0,0]]),
                           np.array([[0,0,0,1],
                                     [0,0,1,0],
                                     [0,-1,0,0],
                                     [1,0,0,0]])],axis=-1)
    if name == "dualcomplexnumbers" or name == "dualcomplexnumber":
        return 1.0*np.stack([np.array([[1,0,0,0],
                              [0,-1,0,0],
                              [0,0,0,0],
                              [0,0,0,0]]),
                            np.array([[0,1,0,0],
                                      [1,0,0,0],
                                      [0,0,0,0],
                                      [0,0,0,0]]),
                           np.array([[0,0,1,0],
                                     [0,0,0,-1],
                                     [1,0,0,0],
                                     [0,-1,0,0]]),
                           np.array([[0,0,0,1],
                                     [0,0,1,0],
                                     [0,1,0,0],
                                     [1,0,0,0]])],axis=-1)
                                     
def print_algebra(algebra):
    for k in range(algebra.shape[-1]):
        print("Bilinear form of the %d-th component." % k)
        print(algebra[:,:,k])


