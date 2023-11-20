"""
Using Multi-Headed Attention to improve resilience to device heterogeneity

Consider changing name to MASHER. 

Sub-project refers to the implementation of AdLoc

Notes to contributors:
- Keep all dependencies within sub-project
- Usable functions should be importable from outside sub-project
- Enable high flexibility when possible from outside sub-project

!pip install tensorflow_addons
"""

# TODO:
# Function(s) to build model as per paper
# Ability to provide custom model structure as input
# Main model should be interchangeable
# Model baseline model description should be provided from outside of this class

import tensorflow as tf


class MaskedGaussianNoise(tf.keras.layers.GaussianNoise):
    """Apply Gausisan Noise to values that are non-zerp"""

    def __init__(self, stddev, mask_val=0.0, seed=None, **kwargs):
        super(MaskedGaussianNoise, self).__init__(stddev, seed=seed, **kwargs)

        self.mask_val = tf.constant(mask_val)

    def call(self, inputs, training=False):

        # get noised output
        noised = super().call(inputs, training=training)

        # get indices where mask applies
        # do not change values where mask is given
        return tf.where(
            inputs == self.mask_val,  # where True, take x; else take y
            x=inputs,
            y=noised,
        )


class MaskedRandomContrast(tf.keras.layers.RandomContrast):
    def __init__(self, factor, is_img=True, mask_val=0.0, seed=None, **kwargs):
        super(MaskedRandomContrast, self).__init__(factor, seed=seed, **kwargs)

        self.mask_val = tf.constant(mask_val)
        self.is_img = is_img

    def call(self, inputs, training=False):

        # get noised output
        if self.is_img:
            noised = super().call(inputs, training=training)
        else:
            # get base shape
            base_shape = tf.shape(inputs)
            
            # convert input to image format
            # we do not change the original image var,
            # as it needs to be in its original shape to apply mask later
            input_img = tf.reshape(inputs, (base_shape[0], 1, base_shape[1], 1))
            
            # call super() function and reshape back to base shape
            noised = super().call(input_img, training=training)
            noised = tf.reshape(noised, base_shape)
            
        # get indices where mask applies
        # do not change values where mask is given
        return tf.where(
            inputs == self.mask_val,  # where True, take x; else take y
            x=inputs,
            y=noised,
        )


class MaskedRandomBrightness(tf.keras.layers.Layer):
    def __init__(self, max_delta, is_img=True, mask_val=0.0, seed=None, **kwargs):
        super(MaskedRandomBrightness, self).__init__(**kwargs)

        self.mask_val = tf.constant(mask_val)
        self.is_img = is_img
        self.max_delta = max_delta

    def call(self, inputs):

        # get noised output
        if self.is_img:
            noised = tf.image.random_brightness(inputs, self.max_delta)
        else:
            # get base shape
            base_shape = tf.shape(inputs)
            
            # convert input to image format
            # we do not change the original image var,
            # as it needs to be in its original shape to apply mask later
            input_img = tf.reshape(inputs, (base_shape[0], 1, base_shape[1], 1))
            
            # call super() function and reshape back to base shape
            noised = tf.image.random_brightness(input_img, self.max_delta)
            noised = tf.reshape(noised, base_shape)
            
        # get indices where mask applies
        # do not change values where mask is given
        return tf.where(
            inputs == self.mask_val,  # where True, take x; else take y
            x=inputs,
            y=noised,
        )


class MaskedDropout(tf.keras.layers.Dropout):
    def __init__(self, factor, mask_val=0.0, **kwargs):
        super(MaskedDropout, self).__init__(factor, **kwargs)

        self.mask_val = tf.constant(mask_val)

    def call(self, inputs, training=False):

        # get noised output
        droped = super().call(inputs, training=training)

        # get indices where mask applies
        # do not change values where mask is given
        return tf.where(
            inputs == self.mask_val,  # where True, take x; else take y
            x=inputs,
            y=droped,
        )



