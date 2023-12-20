import sys
from tqdm import tqdm
import tensorflow as tf
from custom_losses import LSGANLoss
from custom_layers import Warping_Layer
from data_preparation import Data_Preparation
from utils import CG_PostProcessing, MultiScale_Discriminator



condition_generator = tf.keras.models.load_model(
    filepath='models/cg_models/cg_generator_epoch_500.h5',
    custom_objects={'Warping_Layer': Warping_Layer}
)
cg_postprocessor = CG_PostProcessing()

# Just for pre-processing discriminators' inputs
multiscale_discriminator = MultiScale_Discriminator(learning_rate=0.0002, loss=LSGANLoss())

discriminator_1 = tf.keras.models.load_model(
    'models/cg_models/cg_disc_1_epoch_500.h5', 
    custom_objects={
        'leaky_relu': tf.nn.leaky_relu, 
        'LSGANLoss': LSGANLoss
    }
)

discriminator_2 = tf.keras.models.load_model(
    'models/cg_models/cg_disc_2_epoch_500.h5', 
    custom_objects={
        'leaky_relu': tf.nn.leaky_relu, 
        'LSGANLoss': LSGANLoss
    }
)


data_preparation = Data_Preparation()
train_dataset = data_preparation.prepare_cg_datasets()

max_ratio = 0
for batch in tqdm(train_dataset):        
    real_seg_mask = batch.pop('image_parse')
    # Generating fake segmentation masks
    generator_output = condition_generator(batch)
    
    warped_cloth_mask_4 = cg_postprocessor.cloth_warper(
        image=batch['cloth_mask'],
        flow_map=generator_output['flow_map_4_up']
    )      
    
    fake_seg_mask = cg_postprocessor.misalignment_remover(
        generator_output['seg_bottleneck'],
        warped_cloth_mask_4,
    )   
    
    # Preprocessing and training discriminator 1
    patched_fake_seg_mask = multiscale_discriminator.preprocess_input('generator_training', fake_seg_mask, 1)
    patched_fake_seg_mask = tf.random.uniform((12, 64, 70, 12))
    
    D_x = tf.reduce_mean(discriminator_1(patched_fake_seg_mask))
    
    ratio = D_x / (1 - D_x)
    
    if ratio > max_ratio:
        max_ratio = ratio
        
    
    patched_fake_seg_mask = multiscale_discriminator.preprocess_input('generator_training', fake_seg_mask, 2)
    patched_fake_seg_mask = tf.random.uniform((5, 32, 35, 12))
    
    D_x = tf.reduce_mean(discriminator_2(patched_fake_seg_mask))
    
    ratio = D_x / (1 - D_x)
    
    if ratio > max_ratio:
        max_ratio = ratio