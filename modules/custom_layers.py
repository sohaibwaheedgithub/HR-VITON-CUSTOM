import sys
import tensorflow as tf 
import tensorflow_addons as tfa


# Custom Layer for Warping feature map using flow map
class Warping_Layer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainable = False
       
    def call(self, image, flow):
        outputs = tfa.image.dense_image_warp(image=image, flow=flow)
        return outputs
    
    

    
class SpatiallyAdaptiveNormalization(tf.keras.layers.Layer):
    def __init__(self, filters, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.beta_conv = tf.keras.layers.Conv2D(self.filters, 3, padding='same')
        self.gamma_conv = tf.keras.layers.Conv2D(self.filters, 3, padding='same')
        self.shared_conv = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')
        self.BatchNormalization = tf.keras.layers.BatchNormalization(axis=[0, 1, 2], center=False, scale=False)
        
    def call(self, input_tensor, seg_map):
        standardized = self.BatchNormalization(input_tensor)
        shared_gamma_beta = self.shared_conv(seg_map)
        gamma = self.gamma_conv(shared_gamma_beta)
        beta = self.beta_conv(shared_gamma_beta)
        return gamma * standardized + beta
    
    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config, 
            'filters': self.filters
            }
        
        
class NormTanhActivation(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def call(self, inputs):
        return (inputs + 1) / 2
        
    
    
    
    
if __name__ == '__main__':
    pass
    
    
    