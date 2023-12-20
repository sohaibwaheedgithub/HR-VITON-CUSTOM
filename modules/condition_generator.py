import sys
import utils
import custom_layers
import custom_losses
from PIL import Image
import tensorflow as tf
from constants import hparams


# Function to create both encoders
    

class Condition_Generator():
    def __init__(self):
        self.model = None
        
        
    def build_model(self, input_shape):
        '''============================ Clothing Encoder ============================'''
        # inputs for clothing encoder
        cloth = tf.keras.layers.Input(shape=input_shape+[3], batch_size=hparams['cg_batch_size'])
        cloth_mask = tf.keras.layers.Input(shape=input_shape+[1], batch_size=hparams['cg_batch_size'])
        
        # concatenating inputs
        clothing_encoder_input = tf.keras.layers.Concatenate(axis=-1)([cloth, cloth_mask])
        
        # clothing encoder architecture
        CE_res_block_1 = utils.ResBlock(clothing_encoder_input, n_filters=96, scaling_type='down')
        CE_res_block_2 = utils.ResBlock(CE_res_block_1, n_filters=192, scaling_type='down')
        CE_res_block_3 = utils.ResBlock(CE_res_block_2, n_filters=384, scaling_type='down')
        CE_res_block_4 = utils.ResBlock(CE_res_block_3, n_filters=384, scaling_type='down')
        CE_res_block_5 = utils.ResBlock(CE_res_block_4, n_filters=384, scaling_type='down')
            
        '''=========================== Segmentation Encoder ==========================='''
        # inputs for segmentation encoder
        parse_agnostic = tf.keras.layers.Input(shape=input_shape+[hparams['parse_agnostic_classes']], batch_size=hparams['cg_batch_size'])
        densepose = tf.keras.layers.Input(shape=input_shape+[3], batch_size=hparams['cg_batch_size'])
        # concatenating inputs
        segmentation_encoder_input = tf.keras.layers.Concatenate(axis=-1)([parse_agnostic, densepose])
        # segmentation encoder architecture
        SE_res_block_1 = utils.ResBlock(segmentation_encoder_input, n_filters=96, scaling_type='down')
        SE_res_block_2 = utils.ResBlock(SE_res_block_1, n_filters=192, scaling_type='down')
        SE_res_block_3 = utils.ResBlock(SE_res_block_2, n_filters=384, scaling_type='down')
        SE_res_block_4 = utils.ResBlock(SE_res_block_3, n_filters=384, scaling_type='down')
        SE_res_block_5 = utils.ResBlock(SE_res_block_4, n_filters=384, scaling_type='down')
    
         
        # concatenating clothing and segmentation encoders final outputs
        concatenated_CE_SE = tf.keras.layers.Concatenate(axis=-1)([CE_res_block_5, SE_res_block_5])
        # first flow map generation of flow pathway
        flow_map_0 = tf.keras.layers.Conv2D(
            filters=2,
            kernel_size=3,
            strides=1,
            padding='same'
        )(concatenated_CE_SE)
        
        # generating first segmentation feature of segmentation pathway
        seg_feature_0 = utils.ResBlock(utils.ResBlock(SE_res_block_5, n_filters=768), n_filters=384, scaling_type='up')
        
        '''=========================== Feature Fusion Blocks ==========================='''
        
        # Block 1
        flow_map_1, seg_feature_1, cloth_last_rep = utils.Feature_Fusion_Block(
            seg_decode_filters=384,
            cloth_resblock=CE_res_block_4,
            cloth_last_rep=CE_res_block_5,
            flow_map=flow_map_0,
            seg_feature=seg_feature_0,
            seg_resblock=SE_res_block_4
        )
        # Block 2
        flow_map_2, seg_feature_2, cloth_last_rep = utils.Feature_Fusion_Block(
            seg_decode_filters=384,
            cloth_resblock=CE_res_block_3,
            cloth_last_rep=cloth_last_rep,
            flow_map=flow_map_1,
            seg_feature=seg_feature_1,
            seg_resblock=SE_res_block_3
        )
        # Block 3
        flow_map_3, seg_feature_3, cloth_last_rep = utils.Feature_Fusion_Block(
            seg_decode_filters=192,
            cloth_resblock=CE_res_block_2,
            cloth_last_rep=cloth_last_rep,
            flow_map=flow_map_2,
            seg_feature=seg_feature_2,
            seg_resblock=SE_res_block_2
        )
        # Block 4
        flow_map_4, seg_feature_4, cloth_last_rep = utils.Feature_Fusion_Block(
            seg_decode_filters=96,
            cloth_resblock=CE_res_block_1,
            cloth_last_rep=cloth_last_rep,
            flow_map=flow_map_3,
            seg_feature=seg_feature_3,
            seg_resblock=SE_res_block_1
        )
        
        # Bottleneck Layer for converting final "seg_feature" into Raw Segmentation Mask
        seg_bottleneck = tf.keras.layers.Conv2D(
            filters=hparams['image_parse_classes'],
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu'      
        )(seg_feature_4)
        
        
        '''=========================== Cloth And Cloth Mask Warpings ==========================='''
        
        # Upsampling Flow Maps
        flow_map_1_up = utils.scaling_layer(flow_map_1, filters=flow_map_1.shape[-1], scaling_type='up', upscale=(16, 16))
        flow_map_2_up = utils.scaling_layer(flow_map_2, filters=flow_map_2.shape[-1], scaling_type='up', upscale=(8, 8))
        flow_map_3_up = utils.scaling_layer(flow_map_3, filters=flow_map_3.shape[-1], scaling_type='up', upscale=(4, 4))
        flow_map_4_up = utils.scaling_layer(flow_map_4, filters=flow_map_4.shape[-1], scaling_type='up', upscale=(2, 2))
        
        
        self.model = tf.keras.Model(
            inputs={
                'cloth': cloth, 
                'cloth_mask': cloth_mask,
                'image_densepose': densepose,
                'image_parse_agnostic': parse_agnostic
            }, 
            outputs= {
                "flow_map_1_up": flow_map_1_up,
                "flow_map_2_up": flow_map_2_up,
                "flow_map_3_up": flow_map_3_up,
                "flow_map_4_up": flow_map_4_up,
                "seg_bottleneck": seg_bottleneck,
            }
        )
        
    
        
        

        
        
if __name__ == '__main__':
    condition_generator = Condition_Generator()
    condition_generator.build_model(input_shape=[256, 192])
    cg_post_processor = utils.CG_PostProcessing()
    
    
    