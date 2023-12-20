# Importing Libraries
import os
import sys
import numpy as np
import custom_losses
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from constants import hparams, lip_label_colours
from data_preparation import Data_Preparation
from condition_generator import Condition_Generator
from utils import MultiScale_Discriminator, CG_PostProcessing


# Defining Dataset
data_preparation = Data_Preparation()
train_dataset = data_preparation.prepare_cg_datasets()


# Defining Generator
condition_generator = Condition_Generator()
condition_generator.build_model(input_shape=[256, 192])

# Defining Generator Post Processor
cg_postprocessor = CG_PostProcessing()

# Defining And Compiling Discriminators 
multiscale_discriminator = MultiScale_Discriminator(learning_rate=0.0002, loss=custom_losses.LSGANLoss())
discriminator_1 = multiscale_discriminator.build_compiled_discriminator(
    input_shape=[64, 64, hparams['image_parse_classes']],
    final_kernel_size=int(64/16)
)
discriminator_2 = multiscale_discriminator.build_compiled_discriminator(
    input_shape=[128, 96, hparams['image_parse_classes']],
    final_kernel_size=(int(128/16), int(96/16))
)

# Defining Generator Optimizer
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)


# Defining Loss Functions
L1_Loss = custom_losses.L1Loss()
VGG_Loss = custom_losses.VGGLoss()
LSGAN_Loss = custom_losses.LSGANLoss()
Total_Variation_Loss = custom_losses.TotalVariationLoss()
Crossentropy_Loss = tf.keras.losses.SparseCategoricalCrossentropy()


# Logs directory creation for tensorboard visiualization
current_time = datetime.now().strftime('%d-%m-%Y--%H-%M-%S')
train_log_dir = os.path.join('logs/gradient_tape', current_time, 'train')
valid_log_dir = os.path.join('logs/gradient_tape', current_time, 'valid')
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)


# Defining Generator Training Step Function

generator_fake_ids = tf.ones((8 * hparams['cg_batch_size'],))

@tf.function
def train_step(X_batch, real_seg_mask):
    with tf.GradientTape() as tape:
        outputs = condition_generator.model(X_batch)
        
        warped_cloth_mask_1 = cg_postprocessor.cloth_warper(image=X_batch['cloth_mask'], flow_map=outputs['flow_map_1_up'])
        warped_cloth_mask_2 = cg_postprocessor.cloth_warper(image=X_batch['cloth_mask'], flow_map=outputs['flow_map_2_up'])
        warped_cloth_mask_3 = cg_postprocessor.cloth_warper(image=X_batch['cloth_mask'], flow_map=outputs['flow_map_3_up'])
        warped_cloth_mask_4 = cg_postprocessor.cloth_warper(image=X_batch['cloth_mask'], flow_map=outputs['flow_map_4_up'])
        
        warped_cloth_1 = cg_postprocessor.cloth_warper(image=X_batch['cloth'], flow_map=outputs['flow_map_1_up'])
        warped_cloth_2 = cg_postprocessor.cloth_warper(image=X_batch['cloth'], flow_map=outputs['flow_map_2_up'])
        warped_cloth_3 = cg_postprocessor.cloth_warper(image=X_batch['cloth'], flow_map=outputs['flow_map_3_up'])
        warped_cloth_4 = cg_postprocessor.cloth_warper(image=X_batch['cloth'], flow_map=outputs['flow_map_4_up'])
        
        fake_seg_mask = cg_postprocessor.misalignment_remover(
            outputs['seg_bottleneck'],
            warped_cloth_mask_4
        )    
        
        pred_cloth_mask, pred_cloth = cg_postprocessor.occlusion_handler(
            segmentation_mask_scores=fake_seg_mask,
            warped_cloth_mask=warped_cloth_mask_4,
            warped_cloth=warped_cloth_4
        )
        
        
        # Calculating L1 loss
        l1_loss = L1_Loss(
            y_true = tf.repeat(X_batch['cloth_mask'][:, tf.newaxis, ...], repeats=5, axis=1),
            y_pred = tf.stack([
                warped_cloth_mask_1,
                warped_cloth_mask_2,
                warped_cloth_mask_3,
                warped_cloth_mask_4,
                pred_cloth_mask
                ],
                axis=1
            )
        )
        # Calculating VGG loss
        vgg_loss = VGG_Loss(
            y_true = tf.cast(X_batch['cloth'] * 255, tf.uint8),
            y_pred = tf.cast(tf.stack([
                warped_cloth_1,
                warped_cloth_2,
                warped_cloth_3,
                warped_cloth_4,
                pred_cloth
                ], 
                axis=1
            ) * 255, tf.uint8)  
        )
        # Calculating Total-Variation loss
        tv_loss = Total_Variation_Loss(None, y_pred=outputs['flow_map_4_up'])
        # Calculating Pixel-Wise Crossentropy loss
        
        ce_loss = Crossentropy_Loss(
            y_true=tf.argmax(real_seg_mask, axis=-1),
            y_pred=fake_seg_mask
        )
        # Calculating LSGAN loss
        # Preprocessing fake_seg_mask, giving into discriminators as inputs and calculating the lsgan losses of both discriminators
        patched_fake_seg_mask = multiscale_discriminator.preprocess_input('generator_training', fake_seg_mask, 1)
        d1_y_pred = discriminator_1(patched_fake_seg_mask)
        lsgan_loss_1 = LSGAN_Loss(generator_fake_ids, d1_y_pred)
        
        patched_fake_seg_mask = multiscale_discriminator.preprocess_input('generator_training', fake_seg_mask, 2)
        d2_y_pred = discriminator_2(patched_fake_seg_mask)
        lsgan_loss_2 = LSGAN_Loss(generator_fake_ids, d2_y_pred)
        # Adding lsgan losses to get the final loss
        total_lsgan_loss = lsgan_loss_1 + lsgan_loss_2
        # Calculating Total Try On Condition Generator Loss
        tocg_loss = hparams['lambda_ce']*ce_loss + hparams['lambda_lsgan']*total_lsgan_loss + hparams['lambda_l1']*l1_loss + vgg_loss + hparams['lambda_tv']*tv_loss
    gradients = tape.gradient(tocg_loss, condition_generator.model.trainable_weights)
    generator_optimizer.apply_gradients(zip(gradients, condition_generator.model.trainable_weights))
    
    del tape
    
    return {
        'l1_loss': l1_loss,
        'vgg_loss': vgg_loss,
        'tv_loss': tv_loss,
        'ce_loss': ce_loss,
        'lsgan_loss_1': lsgan_loss_1,
        'lsgan_loss_2': lsgan_loss_2,
        'tocg_loss': tocg_loss,
        'pred_cloth_mask': pred_cloth_mask,
        'pred_cloth': pred_cloth,
        'real_seg_mask': real_seg_mask,
        'fake_seg_mask': fake_seg_mask
    }
    
    
def print_status_bar(d1_loss, d2_loss, losses):
    print('D1 Loss: {:.02f} || D2 Loss: {:.02f} || L1 Loss: {:.02f} || VGG Loss: {:.02f} || TV Loss: {:.02f} || CE Loss: {:.02f} || G1 Loss: {:.02f} || G2 Loss: {:.02f}'.format(
        d1_loss,
        d2_loss,
        losses["l1_loss"], 
        losses["vgg_loss"],
        losses["tv_loss"],
        losses["ce_loss"],
        losses["lsgan_loss_1"],
        losses["lsgan_loss_2"]
        )
    )    
 


print('============ Training Condition Generator ============')
  
# Defining Training Parameters
n = 5
cloth_fig, cloth_ax = plt.subplots(n, n, figsize=(10, 12))
cloth_mask_fig, cloth_mask_ax = plt.subplots(n, n, figsize=(10, 12))
seg_mask_fig, seg_mask_ax = plt.subplots(n, n, figsize=(10, 12))
progress_fig, progress_ax = plt.subplots(n, 3, figsize=(10, 15))
row = 0
fig_no = 1
checkpoint = 1

plot_index = 1
replay_buffer_1 = None
replay_buffer_2 = None

for epoch in range(1, hparams['cg_epochs']+1):
    print('Training')
    
    for batch in tqdm(train_dataset, desc=f'Epoch: {epoch}/{hparams["cg_epochs"]}'):        
        '''========= PHASE 1: Mutiscale Discriminator Training ========='''
        real_seg_mask = batch.pop('image_parse')
        # Generating fake segmentation masks
        generator_output = condition_generator.model(batch)
        
        warped_cloth_mask_4 = cg_postprocessor.cloth_warper(
            image=batch['cloth_mask'],
            flow_map=generator_output['flow_map_4_up']
        )      
        
        fake_seg_mask = cg_postprocessor.misalignment_remover(
            generator_output['seg_bottleneck'],
            warped_cloth_mask_4,
        )   
        
        # Preprocessing and training discriminator 1
        patched_real_seg_mask, patched_fake_seg_mask = multiscale_discriminator.preprocess_input('discriminator_training', fake_seg_mask, 1, real_seg_mask)
        
        # For Implementing "Experience Replay" to avoid "Mode Collapse" Problem
        if not replay_buffer_1 == None:
            replay_buffer_1 = tf.concat([replay_buffer_1[3:], patched_fake_seg_mask[:3]], axis=0)
        else:
            replay_buffer_1 = patched_fake_seg_mask
            
        real_fake_seg_mask = tf.concat([patched_real_seg_mask, replay_buffer_1], axis=0).numpy()
        real_ids = tf.ones(shape=(patched_real_seg_mask.shape[0],))
        fake_ids = tf.zeros(shape=(replay_buffer_1.shape[0],))
        real_fake_ids = tf.concat([real_ids, fake_ids], axis=0).numpy()
        # Shuffling discriminator input
        random_indices = np.random.permutation(real_fake_seg_mask.shape[0])
        real_fake_seg_mask = real_fake_seg_mask[random_indices]
        real_fake_ids = real_fake_ids[random_indices]
        
        d1_loss = discriminator_1.train_on_batch(x=real_fake_seg_mask, y=real_fake_ids, return_dict=True)['loss']

        # Preprocessing and training discriminator 2
        patched_real_seg_mask, patched_fake_seg_mask = multiscale_discriminator.preprocess_input('discriminator_training', fake_seg_mask, 2, real_seg_mask)

        if not replay_buffer_2 == None:
            replay_buffer_2 = tf.concat([replay_buffer_2[1:], patched_fake_seg_mask[:1]], axis=0)
        else:
            replay_buffer_2 = patched_fake_seg_mask
            
        real_fake_seg_mask = tf.concat([patched_real_seg_mask, replay_buffer_2], axis=0).numpy()
        real_ids = tf.ones(shape=(patched_real_seg_mask.shape[0],))
        fake_ids = tf.zeros(shape=(replay_buffer_2.shape[0],))
        real_fake_ids = tf.concat([real_ids, fake_ids], axis=0).numpy()
        # Suffling discriminator input
        random_indices = np.random.permutation(real_fake_seg_mask.shape[0])
        real_fake_seg_mask = real_fake_seg_mask[random_indices]
        real_fake_ids = real_fake_ids[random_indices]
        
        d2_loss =discriminator_2.train_on_batch(x=real_fake_seg_mask, y=real_fake_ids, return_dict=True)['loss']
        
        '''========= PHASE 2: Condition Generator Training ========='''
        
        losses = train_step(batch, real_seg_mask)
    
    
    if epoch % (checkpoint * n) == 0:
        progress_fig, progress_ax = plt.subplots(n, 3, figsize=(10, 15))
        row = 0
        fig_no += 1
        
    
    if epoch % checkpoint == 0:    
        # Saving discriminator 1
        tf.keras.models.save_model(discriminator_1, f'models/cg_disc_1_epoch_{epoch}.h5')
        # Saving discriminator 2
        tf.keras.models.save_model(discriminator_2, f'models/cg_disc_2_epoch_{epoch}.h5')
        # Saving discriminator 1
        tf.keras.models.save_model(condition_generator.model, f'models/cg_generator_epoch_{epoch}.h5')
        
        # Saving Cloth Masks Plots
        progress_ax[row, 0].imshow(tf.image.resize(losses['pred_cloth_mask'][0], size=[1024, 768]), cmap='gray')
        progress_ax[row, 0].axis('off')
        
        # Saving Cloth Plots
        progress_ax[row, 1].imshow(tf.cast(tf.image.resize(losses['pred_cloth'][0] * 255, size=[1024, 768]), dtype=tf.uint8))
        progress_ax[row, 1].axis('off')
        progress_ax[row, 1].set_title(f"Epoch: {epoch}")
        
        # Saving Segmentation Masks Plots
        fake_seg_mask = tf.argmax(fake_seg_mask[0], axis=-1).numpy().astype(np.uint8) 
        fake_seg_image = np.array(lip_label_colours)[fake_seg_mask]
        progress_ax[row, 2].imshow(tf.cast(tf.image.resize(fake_seg_image, size=[1024, 768]), tf.uint8))
        progress_ax[row, 2].axis('off')
        
        plt.tight_layout()
        progress_fig.savefig(f'progress_figure_{fig_no}.png')
        row += 1  
       
    
    # Displaying Training Progress Status Bar
    print_status_bar(d1_loss, d2_loss, losses)
        
    # Writing summary for train step
    with train_summary_writer.as_default():
        tf.summary.scalar('Train D1 Loss', d1_loss, step=epoch)
        tf.summary.scalar('Train D2 Loss', d2_loss, step=epoch)
        tf.summary.scalar('Train L1 Loss', losses['l1_loss'], step=epoch)
        tf.summary.scalar('Train VGG Loss', losses['vgg_loss'], step=epoch)
        tf.summary.scalar('Train TV_Loss', losses['tv_loss'], step=epoch)
        tf.summary.scalar('Train CE Loss', losses['ce_loss'], step=epoch)
        tf.summary.scalar('Train G1 Loss', losses['lsgan_loss_1'], step=epoch)
        tf.summary.scalar('Train G2 Loss', losses['lsgan_loss_2'], step=epoch)
        tf.summary.scalar('Train TOCG Loss', losses['tocg_loss'], step=epoch)
    
    # reseting the dataset iterator
    data_preparation.im_names_iterator = data_preparation.im_names.__iter__()