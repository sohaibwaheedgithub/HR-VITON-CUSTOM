import sys
import tensorflow as tf
from constants import hparams
from custom_layers import NormTanhActivation
from utils import ImageGeneratorEncoder, Custom_SPADE_ResBlock



class Image_Generator():
    def __init__(self):
        self.model = None
    
    
    def build_model(self):
        densepose = tf.keras.layers.Input(shape=[1024, 768, 3], batch_size=hparams['ig_batch_size'])
        warped_cloth = tf.keras.layers.Input(shape=[1024, 768, 3], batch_size=hparams['ig_batch_size'])
        cloth_agnostic = tf.keras.layers.Input(shape=[1024, 768, 3], batch_size=hparams['ig_batch_size'])
        seg_map = tf.keras.layers.Input(shape=[1024, 768, hparams['image_parse_classes']], batch_size=hparams['ig_batch_size'])
        
        fused_input = tf.keras.layers.Concatenate(axis=-1)([densepose, warped_cloth, cloth_agnostic])  
        
        # image encoder
        fused_input_encoder = ImageGeneratorEncoder()(fused_input)
    
        pre_conv = tf.keras.layers.Conv2D(
            filters=1024,
            kernel_size=3,
            padding='same',
            activation='relu'    # If using SPADE ResBlock, set use_bias=False 
        )(fused_input_encoder)
        # Custom SPADE ResBlock 1
        custom_spade_resblock = Custom_SPADE_ResBlock(pre_conv, seg_map, 1024)
        resized_fused_input = tf.keras.layers.experimental.preprocessing.Resizing(custom_spade_resblock.shape[1], custom_spade_resblock.shape[2])(fused_input)
        concat_csr = tf.keras.layers.Concatenate(axis=-1)([custom_spade_resblock, resized_fused_input])
        # Custom SPADE ResBlock 2
        custom_spade_resblock = Custom_SPADE_ResBlock(concat_csr, seg_map, 1024)
        resized_fused_input = tf.keras.layers.experimental.preprocessing.Resizing(custom_spade_resblock.shape[1], custom_spade_resblock.shape[2])(fused_input)
        concat_csr = tf.keras.layers.Concatenate(axis=-1)([custom_spade_resblock, resized_fused_input])
        # Custom SPADE ResBlock 3
        custom_spade_resblock = Custom_SPADE_ResBlock(concat_csr, seg_map, 1024)
        resized_fused_input = tf.keras.layers.experimental.preprocessing.Resizing(custom_spade_resblock.shape[1], custom_spade_resblock.shape[2])(fused_input)
        concat_csr = tf.keras.layers.Concatenate(axis=-1)([custom_spade_resblock, resized_fused_input])
        # Custom SPADE ResBlock 4
        custom_spade_resblock = Custom_SPADE_ResBlock(concat_csr, seg_map, 512)
        resized_fused_input = tf.keras.layers.experimental.preprocessing.Resizing(custom_spade_resblock.shape[1], custom_spade_resblock.shape[2])(fused_input)
        concat_csr = tf.keras.layers.Concatenate(axis=-1)([custom_spade_resblock, resized_fused_input])
        # Custom SPADE ResBlock 5
        custom_spade_resblock = Custom_SPADE_ResBlock(concat_csr, seg_map, 256)
        resized_fused_input = tf.keras.layers.experimental.preprocessing.Resizing(custom_spade_resblock.shape[1], custom_spade_resblock.shape[2])(fused_input)
        concat_csr = tf.keras.layers.Concatenate(axis=-1)([custom_spade_resblock, resized_fused_input])
        # Custom SPADE ResBlock 6
        custom_spade_resblock = Custom_SPADE_ResBlock(concat_csr, seg_map, 128)
        resized_fused_input = tf.keras.layers.experimental.preprocessing.Resizing(custom_spade_resblock.shape[1], custom_spade_resblock.shape[2])(fused_input)
        concat_csr = tf.keras.layers.Concatenate(axis=-1)([custom_spade_resblock, resized_fused_input])
        # Custom SPADE ResBlock 7
        custom_spade_resblock = Custom_SPADE_ResBlock(concat_csr, seg_map, 64)
        resized_fused_input = tf.keras.layers.experimental.preprocessing.Resizing(custom_spade_resblock.shape[1], custom_spade_resblock.shape[2])(fused_input)
        concat_csr = tf.keras.layers.Concatenate(axis=-1)([custom_spade_resblock, resized_fused_input])
        # Final Custom SPADE ResBlock 
        custom_spade_resblock = Custom_SPADE_ResBlock(concat_csr, seg_map, 32, scaling_type=None)
        
        # BottleNeck Convolutional Layer
        final_conv = tf.keras.layers.Conv2D(3, 3, padding='same', activation='sigmoid')(custom_spade_resblock)
        
        self.model = tf.keras.Model(
            inputs={
                'seg_mask': seg_map,
                'cloth_agnostic': cloth_agnostic,
                'warped_cloth': warped_cloth,
                'image_densepose': densepose
            },
            outputs=[final_conv]
        )
    
    
if __name__ == '__main__':
    from data_preparation import IG_Data_Preparation
    
    data_preparation = IG_Data_Preparation(cg_checkpoint='models\cg_models\cg_generator_epoch_80.h5')
    train_dataset = data_preparation.prepare_dataset()
    
    ig = Image_Generator()
    ig.build_model()
    
    iterator = train_dataset.__iter__()
    batch = iterator.__next__()
    batch.pop('filtered_image')
    
    print(ig.model(batch))
    