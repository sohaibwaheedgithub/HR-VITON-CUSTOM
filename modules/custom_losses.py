import sys
import tensorflow as tf
from constants import hparams


class L1Loss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
        self.weights_vector = tf.constant([1., 2., 3., 4., 1.])[:, tf.newaxis]
    
    def call(self, y_true, y_pred):
        loss_matrix = tf.keras.losses.mae(y_true, y_pred)
        loss_matrix = tf.reduce_mean(loss_matrix, axis=[2, 3])
        loss = loss_matrix @ self.weights_vector
        return loss
    


    
class VGGLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
        self.L1_loss = L1Loss()
        self.vgg19 = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
        
    def call(self, y_true, y_pred):
        # preprocess ground truths
        y_true_preprocessed = tf.repeat(
            self.vgg19(
                tf.keras.applications.vgg19.preprocess_input(y_true)
            )[:, tf.newaxis, ...], 
            repeats=5, 
            axis=1
        )
        # preprocess predictions
        y_pred_preprocessed = tf.reshape(
            self.vgg19(
                tf.keras.applications.vgg19.preprocess_input(
                    tf.reshape(
                        y_pred, 
                        shape=(hparams['cg_batch_size'] * 5, 256, 192, 3)
                    )
                )
            ), 
            shape=(hparams['cg_batch_size'], 5, 8, 6, 512)
        )
        loss = self.L1_loss(y_true_preprocessed, y_pred_preprocessed)
        return loss
    
    
    
    
class LSGANLoss(tf.keras.losses.Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def call(self, y_true, y_pred):
        true_fn = lambda x: tf.square(x - 1)
        false_fn = lambda x: tf.square(x)
        y_true = tf.cast(y_true, tf.bool)
        loss = tf.where(y_true, true_fn(y_pred), false_fn(y_pred))
        loss /= 2
        return loss
    
    
    
class TotalVariationLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
        
    def call(self, y_true, y_pred):
        loss = tf.image.total_variation(y_pred)
        return loss
            
         
         
class FeatureMatchingLoss(tf.keras.losses.Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
        
    
    def call(self, real_outputs, fake_outputs):
        '''index = tf.constant(0)
        fm_loss = tf.constant(0.)
        condition = lambda index, _: tf.less(index, tf.constant(5))
        loop_body = lambda index, fm_loss: (
            index + 1,
            fm_loss + tf.reduce_mean(tf.keras.losses.mae(real_outputs[index], fake_outputs[index]))
        )   
        fm_loss = tf.while_loop(
            cond=condition,
            body=loop_body,
            loop_vars=[index, fm_loss]
        )[1]'''
        
        fm_loss = 0
        fm_loss += tf.reduce_mean(tf.keras.losses.mae(real_outputs['output_1'], fake_outputs['output_1']))
        fm_loss += tf.reduce_mean(tf.keras.losses.mae(real_outputs['output_2'], fake_outputs['output_2']))
        fm_loss += tf.reduce_mean(tf.keras.losses.mae(real_outputs['output_3'], fake_outputs['output_3']))
        fm_loss += tf.reduce_mean(tf.keras.losses.mae(real_outputs['output_4'], fake_outputs['output_4']))
        fm_loss += tf.reduce_mean(tf.keras.losses.mae(real_outputs['final_output'], fake_outputs['final_output']))
        fm_loss /= 5
        
        return fm_loss
    
    
class IG_VGGLOSS(tf.keras.losses.Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vgg19 = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
        
    def call(self, real_imgs, fake_imgs):
        processed_real_imgs = tf.keras.applications.vgg19.preprocess_input(real_imgs)
        processed_fake_imgs = tf.keras.applications.vgg19.preprocess_input(fake_imgs)
        loss = tf.reduce_mean(tf.keras.losses.mae(processed_real_imgs, processed_fake_imgs))
        return loss
        
        
        
        
class HingeGANLoss(tf.keras.losses.Loss):
    def __init__(self, disc_loss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.disc_loss = disc_loss
        
    def call(self, labels, disc_output):
        if self.disc_loss:
            true_tensor = tf.math.maximum(0., 1-disc_output)
            false_tensor = tf.math.maximum(0., 1+disc_output)
            labels = tf.cast(labels, tf.bool)
            loss = tf.where(labels, true_tensor, false_tensor)
            return loss
        else:
            loss = -1 * disc_output
            return loss
        

            
            

        
        
        
            
            
            
if __name__ == '__main__':
    from utils import MultiScale_Discriminator
    from image_generator import Image_Generator
    
    real_images = tf.random.uniform((hparams['ig_batch_size'], 1024, 768, 3))
    #fake_images = generator.model(inputs)    
    fake_images = tf.random.uniform((hparams['ig_batch_size'], 1024, 768, 3))
    
    
    pil_image = tf.keras.preprocessing.image.load_img(r'temp_train\image-parse-v3\a_man1.png')
    real_images = tf.keras.preprocessing.image.img_to_array(pil_image)[tf.newaxis, ...]
    
    multiscale_discriminator = MultiScale_Discriminator(learning_rate=0.0004, loss={'final_output': HingeGANLoss(disc_loss=True)}, spectral_norm=True)
    discriminator = multiscale_discriminator.build_compiled_discriminator(
        input_shape=[256, 256, 3],
        final_kernel_size=int(256/16)
    ) 
    
    patched_real_images, patched_fake_images = multiscale_discriminator.preprocess_input('discriminator_training', fake_images, 1, real_images)
    
    real_outputs = discriminator(patched_real_images)
    fake_outputs = discriminator(patched_fake_images)
    
    
    real_outputs = {
        'output_1': tf.constant([2., 0.]), 
        'output_2': tf.constant([2., 0.]), 
        'output_3': tf.constant([2., 0.]),
        'output_4': tf.constant([2., 0.]), 
        'final_output': tf.constant([2., 0.])
    }
    fake_outputs = {
        'output_1': tf.constant([1., 1.]), 
        'output_2': tf.constant([1., 1.]), 
        'output_3': tf.constant([1., 1.]),
        'output_4': tf.constant([1., 1.]), 
        'final_output': tf.constant([1., 1.])
    }
    
    loss_fn = FeatureMatchingLoss()
    fm_loss = loss_fn(real_outputs, fake_outputs)
    
    print(fm_loss)
    '''import matplotlib.pyplot as plt
    
    figure, axes = plt.subplots(1, 4, figsize=(10, 3))
    for col, patch in enumerate(patched_real_images):
        axes[col].imshow(tf.cast(patch, tf.uint8))
        axes[col].axis('off')
    
    figure.show()
    plt.show()'''

    
    '''loss = HingeGANLoss(disc_loss=False)
    y_true = tf.constant([[0], [0], [1]])
    y_pred = tf.constant([[0.9], [0.3], [0.9]], tf.float32)
    
    print(loss(y_true, y_pred))'''
    
    
    