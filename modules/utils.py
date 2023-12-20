import sys
import math
import tensorflow as tf
from functools import partial
from constants import hparams
import tensorflow_addons as tfa
from custom_losses import LSGANLoss
from custom_layers import Warping_Layer, SpatiallyAdaptiveNormalization


def scaling_layer(input: tf.Tensor, filters: int, scaling_type=None, upscale=(2, 2)) -> tf.Tensor:
    if scaling_type == 'down':
        scaled_input = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=2,
            padding='same',
            use_bias=False
        )(input)
    elif scaling_type == 'up':
        scaled_input = tf.keras.layers.UpSampling2D(size=upscale, interpolation='bilinear')(input)
        scaled_input = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=1,
            strides=1
        )(scaled_input)
    else:
        scaled_input = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=1,
            strides=1
        )(input)
    
    return scaled_input




def ResBlock(input: tf.Tensor, n_filters: int, scaling_type=None) -> tf.Tensor:
    '''
    This function is to create residual block for condition generator. Number of 
    channels of input passed to it must be equal to the number of filters passed
    '''
    # scaling layer
    scaled_input = scaling_layer(input, scaling_type=scaling_type, filters=n_filters)
    
    # first conv layer of res unit along with batch normalization and activation
    layer1 = tf.keras.layers.Conv2D(
        filters=n_filters,
        kernel_size=3,
        strides=1,
        padding='same',
        use_bias=False
    )(scaled_input)
    bn_layer1 = tf.keras.layers.BatchNormalization()(layer1)
    act_layer1 = tf.keras.layers.Activation('relu')(bn_layer1)
    
    # second conv layer of res unit along with batch normalization
    layer2 = tf.keras.layers.Conv2D(
        filters=n_filters,
        kernel_size=3,
        strides=1,
        padding='same',
        use_bias=False
    )(act_layer1)
    bn_layer2 = tf.keras.layers.BatchNormalization()(layer2)
            
    # Skip connection along with final activation
    skip_connection = tf.keras.layers.Add()([bn_layer2, scaled_input])
    final_layer = tf.keras.layers.Activation('relu')(skip_connection)
    
    return final_layer



def Feature_Fusion_Block(
    seg_decode_filters: int,
    cloth_resblock: tf.Tensor = None,
    cloth_last_rep: tf.Tensor = None,  # rep for representation
    flow_map: tf.Tensor = None, 
    seg_feature: tf.Tensor = None, 
    seg_resblock: tf.Tensor = None
    ) -> None:
    '''Abbreviated variable names in operations block are short for full varaible names
       abbreviations in a variable name are in order from first(calculated) to last'''
    
    '''======== Initial placeholders definitions ========'''
    cloth_resblock_conv = tf.keras.layers.Conv2D(
        filters=384,
        kernel_size=1,
        strides=1,
        padding='same'
    )(cloth_resblock)
    scaled_cloth_last_rep = scaling_layer(cloth_last_rep, filters=cloth_last_rep.shape[-1], scaling_type='up')
    scaled_flow_map = scaling_layer(flow_map, filters=flow_map.shape[-1], scaling_type='up')
    
    '''======== Main Operations ========'''
    # Bitwise addition between 'scaled_cloth_last_rep' and 'cloth_resblock_conv'
    added_crc_sclr = tf.keras.layers.Add()([cloth_resblock_conv, scaled_cloth_last_rep])
    # Warping added_crc_sclr using 2d Deformation Field
    warped_sfm_oclr = Warping_Layer()(added_crc_sclr, scaled_flow_map)  
    # Concatenating 'seg_feature', 'seg_resblock' and 'warped_sfm_oclr
    concat_sf_sr_wsc = tf.keras.layers.Concatenate()([seg_feature, seg_resblock, warped_sfm_oclr])

    seg_feature_conv = tf.keras.layers.Conv2D(
        filters=384,
        kernel_size=3,
        strides=1,
        padding='same'
    )(seg_feature)
    
    concat_wsc_sfc = tf.keras.layers.Concatenate()([warped_sfm_oclr, seg_feature_conv])
    
    concat_wsc_sfc_conv = tf.keras.layers.Conv2D(
        filters=2,
        kernel_size=3,
        strides=1,
        padding='same'
    )(concat_wsc_sfc)
    
    output_flow_map = tf.keras.layers.Add()([scaled_flow_map, concat_wsc_sfc_conv])
    output_seg_feature = ResBlock(concat_sf_sr_wsc, n_filters=seg_decode_filters, scaling_type='up')
    
    return output_flow_map, output_seg_feature, added_crc_sclr




def ImageGeneratorEncoder():
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=[1024, 768, 9], batch_size=hparams['ig_batch_size']),
        tf.keras.layers.Conv2D(64, 3, 2, padding='same', use_bias=False),
        tfa.layers.InstanceNormalization(),
        tf.keras.layers.Activation(tf.nn.leaky_relu),
        tf.keras.layers.Conv2D(128, 3, 2, padding='same', use_bias=False),
        tfa.layers.InstanceNormalization(),
        tf.keras.layers.Activation(tf.nn.leaky_relu),
        tf.keras.layers.Conv2D(256, 3, 2, padding='same', use_bias=False),
        tfa.layers.InstanceNormalization(),
        tf.keras.layers.Activation(tf.nn.leaky_relu),
        tf.keras.layers.Conv2D(512, 3, 2, padding='same', use_bias=False),
        tfa.layers.InstanceNormalization(),
        tf.keras.layers.Activation(tf.nn.leaky_relu),
        tf.keras.layers.Conv2D(1024, 3, 2, padding='same', use_bias=False),
        tfa.layers.InstanceNormalization(),
        tf.keras.layers.Activation(tf.nn.leaky_relu),
        tf.keras.layers.Conv2D(1024, 3, 2, padding='same', use_bias=False),
        tfa.layers.InstanceNormalization(),
        tf.keras.layers.Activation(tf.nn.leaky_relu),
        tf.keras.layers.Conv2D(1024, 3, 2, padding='same', use_bias=False),
        tfa.layers.InstanceNormalization(),
        tf.keras.layers.Activation(tf.nn.leaky_relu)
    ])




def Custom_SPADE_ResBlock(input: tf.Tensor, seg_mask: tf.Tensor, n_filters: int, scaling_type='up') -> tf.Tensor:
    # scaling layer
    scaled_input = scaling_layer(input, scaling_type=scaling_type, filters=n_filters)
    scaled_seg_mask = tf.keras.layers.experimental.preprocessing.Resizing(scaled_input.shape[1], scaled_input.shape[2])(seg_mask)
    # first conv layer of res unit along with spatially adpative normalization and activation
    layer1 = tf.keras.layers.Conv2D(
        filters=n_filters,
        kernel_size=3,
        strides=1,
        padding='same',
        use_bias=False
    )(scaled_input)
    spade_layer1 = SpatiallyAdaptiveNormalization(filters=n_filters)(layer1, scaled_seg_mask)
    act_layer1 = tf.keras.layers.Activation('relu')(spade_layer1)
    
    # second conv layer of res unit along with batch normalization
    layer2 = tf.keras.layers.Conv2D(
        filters=n_filters,
        kernel_size=3,
        strides=1,
        padding='same',
        use_bias=False
    )(act_layer1)
    spade_layer2 = SpatiallyAdaptiveNormalization(filters=n_filters)(layer2, scaled_seg_mask)
            
    # Skip connection along with final activation
    spade_scaled_input = SpatiallyAdaptiveNormalization(filters=n_filters)(scaled_input, scaled_seg_mask)
    skip_connection = tf.keras.layers.Add()([spade_layer2, spade_scaled_input])
    final_layer = tf.keras.layers.Activation('relu')(skip_connection)
    
    return final_layer




class CG_PostProcessing():    # CG for Condition Generator
    def __init__(self):
        self.mr_condition_mask = tf.Variable(tf.zeros(shape=(hparams['cg_batch_size'], 256, 192, hparams['image_parse_classes']), dtype=tf.bool))   # mr for misalignment removal
        self.mr_condition_mask[:, :, :, 2].assign(tf.broadcast_to(tf.constant(True), shape=[hparams['cg_batch_size'], 256, 192]))
        self.cloth_mask_zeros = tf.zeros(shape=(hparams['cg_batch_size'], 256, 192, 1))
        self.cloth_ones = tf.ones(shape=(hparams['cg_batch_size'], 256, 192, 3)) #* 255 #When pixel range is 0-255
        
            
    def misalignment_remover(self, final_seg_feature, warped_cloth_mask):
        temp_tensor = final_seg_feature * warped_cloth_mask
        segmentation_mask_logits = tf.where(self.mr_condition_mask, temp_tensor, final_seg_feature)
        segmentation_mask = tf.keras.activations.softmax(segmentation_mask_logits)
        return segmentation_mask
    
    
    def occlusion_handler(self, segmentation_mask_scores, warped_cloth_mask, warped_cloth):
        segmentation_mask = tf.argmax(segmentation_mask_scores, axis=-1)[..., tf.newaxis]   
        condition_mask = tf.logical_or(
            tf.logical_or(
                tf.equal(segmentation_mask, 1),
                tf.equal(segmentation_mask, 6)
            ),
            tf.equal(segmentation_mask, 7)
        )
        pred_cloth_mask = tf.where(condition_mask, self.cloth_mask_zeros, warped_cloth_mask)
        pred_cloth = tf.where(condition_mask, self.cloth_ones, warped_cloth)
        
        return pred_cloth_mask, pred_cloth
    
    
    def cloth_warper(self, image, flow_map):
        return tfa.image.dense_image_warp(image, flow_map)
    
    
    def discriminator_rejection_sampling(self, disc_output):
        return disc_output / (hparams['DRS_normalization_constant'] * (1 - disc_output))
        
  
    
    

class MultiScale_Discriminator():
    def __init__(self, learning_rate, loss, spectral_norm=False):
        self.partial_conv = partial(tf.keras.layers.Conv2D, kernel_size=4, strides=2, padding='same', use_bias=False)
        self.loss = loss
        self.spectral_norm = spectral_norm
        self.learning_rate = learning_rate
            
        
    def custom_conv_layer(self, filters):
        output = self.partial_conv(filters=filters)
        if self.spectral_norm:
            output = tfa.layers.SpectralNormalization(output)
        return output
        
    
    def build_compiled_discriminator(self, input_shape, final_kernel_size):
        
        discriminator = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            self.custom_conv_layer(filters=64),
            tf.keras.layers.Activation(tf.nn.leaky_relu, name='output_1'),
            self.custom_conv_layer(filters=128),
            tfa.layers.InstanceNormalization(),
            tf.keras.layers.Activation(tf.nn.leaky_relu, name='output_2'),
            tf.keras.layers.Dropout(0.95),
            self.custom_conv_layer(filters=256),
            tfa.layers.InstanceNormalization(),
            tf.keras.layers.Activation(tf.nn.leaky_relu, name='output_3'),
            tf.keras.layers.Dropout(0.98),
            self.custom_conv_layer(filters=512),
            tfa.layers.InstanceNormalization(),
            tf.keras.layers.Activation(tf.nn.leaky_relu, name='output_4'),
            tf.keras.layers.Dropout(0.98),
            tf.keras.layers.Conv2D(1, final_kernel_size, padding='valid', activation='sigmoid'),
            tf.keras.layers.Flatten(name='final_output')
        ])
        
        if self.spectral_norm:
            discriminator = tf.keras.Model(
                inputs=[discriminator.input], 
                outputs={
                    'output_1': discriminator.layers[1].output,
                    'output_2': discriminator.layers[4].output,
                    'output_3': discriminator.layers[8].output,
                    'output_4': discriminator.layers[12].output,
                    'final_output': discriminator.layers[15].output,
                }
            )
         
        discriminator.compile(
            loss=self.loss,
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.5, beta_2=0.999)
        )
        
        return discriminator

    
    def preprocess_input(self, training_phase: str, fake_images, disc_no, real_images=None):
        # For training on both real and fake images
        width_divisor = 3
        height_divisor = 4
        if training_phase == 'discriminator_training':
            if disc_no == 2:
                width_divisor = 2
                height_divisor = 2
                
            patched_real_images = tf.reshape(
                tf.image.extract_patches(
                    images = real_images,
                    sizes=[1, int(real_images.shape[1] // height_divisor), int(real_images.shape[2] // width_divisor), 1],
                    strides=[1, int(real_images.shape[1] // height_divisor), int(real_images.shape[2] // width_divisor), 1],
                    rates=[1, 1, 1, 1],
                    padding='VALID'
                ),
                shape=[-1, int(real_images.shape[1] // height_divisor), int(real_images.shape[2] // width_divisor), real_images.shape[-1]]
            )
            patched_fake_images = tf.reshape(
                tf.image.extract_patches(
                    images = fake_images,
                    sizes=[1, int(fake_images.shape[1] // height_divisor), int(fake_images.shape[2] // width_divisor), 1],
                    strides=[1, int(fake_images.shape[1] // height_divisor), int(fake_images.shape[2] // width_divisor), 1],
                    rates=[1, 1, 1, 1],
                    padding='VALID'
                ),
                shape=[-1, int(fake_images.shape[1] // height_divisor), int(fake_images.shape[2] // width_divisor), fake_images.shape[-1]]
            )
            return patched_real_images, patched_fake_images
        # For training only on fake images
        else:
            if disc_no == 2:
                width_divisor = 2
                height_divisor = 2
            patched_fake_images = tf.reshape(
                tf.image.extract_patches(
                    images = fake_images,
                    sizes=[1, int(fake_images.shape[1] // height_divisor), int(fake_images.shape[2] // width_divisor), 1],
                    strides=[1, int(fake_images.shape[1] // height_divisor), int(fake_images.shape[2] // width_divisor), 1],
                    rates=[1, 1, 1, 1],
                    padding='VALID'
                ),
                shape=[-1, int(fake_images.shape[1] // height_divisor), int(fake_images.shape[2] // width_divisor), fake_images.shape[-1]]
            )       
            
            return patched_fake_images
        
        
    '''def preprocess_input(self, training_phase: str, fake_seg_mask, disc_no, real_seg_mask=None):
        # For training on both real and fake images
        if training_phase == 'discriminator_training':
            if disc_no == 2:
                real_seg_mask = tf.image.resize(real_seg_mask, size=[128, 96])
                fake_seg_mask = tf.image.resize(fake_seg_mask, size=[128, 96])
            patched_real_seg_mask = tf.reshape(
                tf.image.extract_patches(
                    images = real_seg_mask,
                    sizes=[1, int(real_seg_mask.shape[2] // 3), int(math.ceil(real_seg_mask.shape[1] * 0.27)), 1],
                    strides=[1, int(real_seg_mask.shape[2] // 3), int(math.ceil(real_seg_mask.shape[1] * 0.27)), 1],
                    rates=[1, 1, 1, 1],
                    padding='VALID'
                ),
                shape=[-1, int(real_seg_mask.shape[2] // 3), int(math.ceil(real_seg_mask.shape[1] * 0.27)), hparams['image_parse_classes']]
            )
            patched_fake_seg_mask = tf.reshape(
                tf.image.extract_patches(
                    images = fake_seg_mask,
                    sizes=[1, int(fake_seg_mask.shape[2] // 3), int(math.ceil(fake_seg_mask.shape[1] * 0.27)), 1],
                    strides=[1, int(fake_seg_mask.shape[2] // 3), int(math.ceil(fake_seg_mask.shape[1] * 0.27)), 1],
                    rates=[1, 1, 1, 1],
                    padding='VALID'
                ),
                shape=[-1, int(fake_seg_mask.shape[2] // 3), int(math.ceil(fake_seg_mask.shape[1] * 0.27)), hparams['image_parse_classes']]
            )
            return patched_real_seg_mask, patched_fake_seg_mask
        # For training only on fake images
        else:
            if disc_no == 2:
                fake_seg_mask = tf.image.resize(fake_seg_mask, size=[128, 96])
            patched_fake_seg_mask = tf.reshape(
                tf.image.extract_patches(
                    images = fake_seg_mask,
                    sizes=[1, int(fake_seg_mask.shape[2] // 3), int(math.ceil(fake_seg_mask.shape[1] * 0.27)), 1],
                    strides=[1, int(fake_seg_mask.shape[2] // 3), int(math.ceil(fake_seg_mask.shape[1] * 0.27)), 1],
                    rates=[1, 1, 1, 1],
                    padding='VALID'
                ),
                shape=[-1, int(fake_seg_mask.shape[2] // 3), int(math.ceil(fake_seg_mask.shape[1] * 0.27)), hparams['image_parse_classes']]
            )       
            
            return patched_fake_seg_mask'''

            
    
    
if __name__ == '__main__':
    md = MultiScale_Discriminator(spectral_norm=True)
    d = md.build_compiled_discriminator([256, 256, 3])
    
    
    pr, pf = md.preprocess_input('discriminator_training', tf.random.uniform((1, 256, 192, 17)), 1, tf.random.uniform((1, 256, 192, 17)))
    

