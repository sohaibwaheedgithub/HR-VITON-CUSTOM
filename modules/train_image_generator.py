import sys
import numpy as np
import custom_losses
from tqdm import tqdm
import tensorflow as tf
from constants import hparams
import matplotlib.pyplot as plt
from utils import MultiScale_Discriminator
from image_generator import Image_Generator
from data_preparation import IG_Data_Preparation


# Defining Dataset
data_preparation = IG_Data_Preparation()
train_dataset = data_preparation.prepare_dataset()


# Defining Image Generator
image_generator = Image_Generator()
image_generator.build_model()

# Defining And Compiling Discriminators
multiscale_discriminator = MultiScale_Discriminator(learning_rate=0.0004, loss={'final_output': custom_losses.HingeGANLoss(disc_loss=True)}, spectral_norm=True)
discriminator_1 = multiscale_discriminator.build_compiled_discriminator(
    input_shape=[256, 256, 3],
    final_kernel_size=int(256/16)
)
discriminator_2 = multiscale_discriminator.build_compiled_discriminator(
    input_shape=[512, 384, 3],
    final_kernel_size=(int(512/16), int(384/16))
)


# Defining Generator Optimizer
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.999)


# Defining Loss Functions
FMLoss = custom_losses.FeatureMatchingLoss()
IG_VGGLoss = custom_losses.IG_VGGLOSS()
GenHingGANLoss = custom_losses.HingeGANLoss(disc_loss=False)



@tf.function
def train_step(X_batch, real_images):
    with tf.GradientTape() as tape:
        fake_images = image_generator.model(X_batch)
        
        # Feature Matching Loss and Hinge GAN Loss for discriminator 1
        patched_real_imgs, patched_fake_imgs = multiscale_discriminator.preprocess_input('discriminator_training', fake_images, 1, real_images)
        disc_real_outputs = discriminator_1(patched_real_imgs)
        disc_fake_outputs = discriminator_1(patched_fake_imgs)
        
        disc1_fm_loss = FMLoss(disc_real_outputs, disc_fake_outputs)
        disc1_hinge_gan_loss = GenHingGANLoss(None, disc_fake_outputs['final_output'])
        
        # Feature Matching Loss and Hinge GAN Loss for discriminator 2
        patched_real_imgs, patched_fake_imgs = multiscale_discriminator.preprocess_input('discriminator_training', fake_images, 2, real_images)
        disc_real_outputs = discriminator_2(patched_real_imgs)
        disc_fake_outputs = discriminator_2(patched_fake_imgs)
        
        disc2_fm_loss = FMLoss(disc_real_outputs, disc_fake_outputs)
        disc2_hinge_gan_loss = GenHingGANLoss(None, disc_fake_outputs['final_output'])
        
        # Merging Feature Matching Losses and Hinge GAN Losses of both discriminators
        total_fm_loss = disc1_fm_loss + disc2_fm_loss
        total_hinge_gan_loss = disc1_hinge_gan_loss + disc2_hinge_gan_loss
        
        # VGG Loss
        ig_vgg_loss = IG_VGGLoss(real_images, fake_images)
        
        total_loss = total_hinge_gan_loss + hparams['ig_lambda_vgg_loss']*ig_vgg_loss + hparams['ig_lambda_fm_loss']*total_fm_loss
        
    gradients = tape.gradient(total_loss, image_generator.model.trainable_weights)
    generator_optimizer.apply_gradients(zip(gradients, image_generator.model.trainable_weights))
    
    del tape
    
    return {
        'disc1_fm_loss': disc1_fm_loss,
        'disc2_fm_loss': disc2_fm_loss,
        'disc1_hinge_gan_loss': disc1_hinge_gan_loss,
        'disc2_hinge_gan_loss': disc2_hinge_gan_loss,
        'vgg_loss': ig_vgg_loss  
    }
        

def print_status_bar(d1_loss, d2_loss, losses):
    print('D1 Loss: {:.02f} || D1 FM Loss: {:.02f} || G1 Loss: {:.02f} |||| D2 Loss: {:.02f} || D2 FM Loss: {:.02f} || G2 Loss: {:.02f} || VGG Loss: {:.02f}'.format(
        d1_loss,
        losses["disc1_fm_loss"], 
        losses["disc1_hinge_gan_loss"],
        d2_loss,
        losses["disc2_fm_loss"],
        losses["disc2_hinge_gan_loss"],
        losses["vgg_loss"],
        )
    )
    
        
        
print('============ Training Condition Generator ============')       
    
row = 0
col = 0
checkpoint = 1
replay_buffer_1 = None
replay_buffer_2 = None
figure, axes = plt.subplots(3, 3, figsize=(10, 15))
for epoch in range(1, hparams['ig_epochs']+1):
    print('Training')
    
    for batch in tqdm(train_dataset, desc=f'Epoch: {epoch}/{hparams["ig_epochs"]}'):        
        '''========= PHASE 1: Mutiscale Discriminator Training =========''' 
        real_images = batch.pop('filtered_image')
        
        fake_images = image_generator.model(batch)
        
        patched_real_images, patched_fake_images = multiscale_discriminator.preprocess_input('discriminator_training', fake_images, 1, real_images)
        
        # For Implementing "Experience Replay" to avoid "Mode Collapse" Problem
        if not replay_buffer_1 == None:
            replay_buffer_1 = tf.concat([replay_buffer_1[3:], patched_fake_images[:3]], axis=0)
        else:
            replay_buffer_1 = patched_fake_images
            
        real_fake_images = tf.concat([patched_real_images, replay_buffer_1], axis=0).numpy()
        real_ids = tf.ones(shape=(patched_real_images.shape[0],))
        fake_ids = tf.zeros(shape=(replay_buffer_1.shape[0],))
        real_fake_ids = tf.concat([real_ids, fake_ids], axis=0).numpy()
        # Shuffling discriminator input
        random_indices = np.random.permutation(real_fake_images.shape[0])
        real_fake_images = real_fake_images[random_indices]
        real_fake_ids = real_fake_ids[random_indices]
        
        d1_loss = discriminator_1.train_on_batch(x=real_fake_images, y=real_fake_ids, return_dict=True)['loss']

        # Preprocessing and training discriminator 2
        patched_real_images, patched_fake_images = multiscale_discriminator.preprocess_input('discriminator_training', fake_images, 2, real_images)
    
        if not replay_buffer_2 == None:
            replay_buffer_2 = tf.concat([replay_buffer_2[1:], patched_fake_images[:1]], axis=0)
        else:
            replay_buffer_2 = patched_fake_images
            
        real_fake_images = tf.concat([patched_real_images, replay_buffer_2], axis=0).numpy()
        real_ids = tf.ones(shape=(patched_real_images.shape[0],))
        fake_ids = tf.zeros(shape=(replay_buffer_2.shape[0],))
        real_fake_ids = tf.concat([real_ids, fake_ids], axis=0).numpy()
        # Suffling discriminator input
        random_indices = np.random.permutation(real_fake_images.shape[0])
        real_fake_images = real_fake_images[random_indices]
        real_fake_ids = real_fake_ids[random_indices]
        
        d2_loss =discriminator_2.train_on_batch(x=real_fake_images, y=real_fake_ids, return_dict=True)['loss']
        
        '''========= PHASE 2: Image Generator Training ========='''
        
        losses = train_step(batch, real_images)
    
    
    if epoch % checkpoint == 0:    
        # Saving discriminator 1
        tf.keras.models.save_model(discriminator_1, f'models/ig_disc_1_epoch_{epoch}.h5')
        # Saving discriminator 2
        tf.keras.models.save_model(discriminator_2, f'models/ig_disc_2_epoch_{epoch}.h5')
        # Saving image generator
        tf.keras.models.save_model(image_generator.model, f'models/ig_generator_epoch_{epoch}.h5')
    
        axes[row, col].imshow(fake_images[0])
        axes[row, col].axis('off')
        
        figure.savefig('ig_progress.png')
        
        col += 1
        if col == 3:
            row += 1
            if row == 3:
                sys.exit(0)
            col = 0
            
    print_status_bar(d1_loss, d2_loss, losses)

    # reseting the dataset iterator
    data_preparation.im_names_iterator = data_preparation.im_names.__iter__()
        
        

