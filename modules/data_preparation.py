import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import CG_PostProcessing
from custom_layers import Warping_Layer
from constants import hparams, lip_label_colours



class Data_Preparation():
    def __init__(self):
        self.im_names = os.listdir(os.path.join(hparams['data_dir'], 'image'))
        self.im_names_iterator = self.im_names.__iter__()
    
    
    def cg_generator(self):
        for _ in range(self.im_names.__len__()):
            image_name = self.im_names_iterator.__next__()
            cloth = tf.io.read_file(os.path.join(hparams['data_dir'], 'cloth', image_name))
            cloth_mask = tf.io.read_file(os.path.join(hparams['data_dir'], 'cloth-mask', image_name))
            image_densepose = tf.io.read_file(os.path.join(hparams['data_dir'], 'image-densepose', image_name))
            image_parse_agnostic = tf.constant(os.path.join(hparams['data_dir'], 'image-parse-agnostic-v3.2', image_name.replace('jpg', 'png')))
            image_parse = tf.constant(os.path.join(hparams['data_dir'], 'image-parse-v3', image_name.replace('jpg', 'png')))

            yield {
                'cloth': cloth,
                'cloth_mask': cloth_mask,
                'image_densepose': image_densepose,
                'image_parse_agnostic': image_parse_agnostic,
                'image_parse': image_parse
            }
          
            
    def generate_cg_dataset(self):
        cg_dataset = tf.data.Dataset.from_generator(
            generator=self.cg_generator,
            output_signature={
                'cloth': tf.TensorSpec(shape=(), dtype=tf.string),
                'cloth_mask': tf.TensorSpec(shape=(), dtype=tf.string),
                'image_densepose': tf.TensorSpec(shape=(), dtype=tf.string),
                'image_parse_agnostic': tf.TensorSpec(shape=(), dtype=tf.string),
                'image_parse': tf.TensorSpec(shape=(), dtype=tf.string)
            }
        )
        return cg_dataset
        
    
    def cg_decode_resize(self, encoded_image, shape):
        decoded_image = tf.io.decode_image(encoded_image)
        decoded_image.set_shape(shape)
        processed_image = tf.image.resize(decoded_image, size=[256, 192])
        return processed_image
    
    
    def decode_parse_img(self, image_path, depth):
        image = Image.open(image_path.numpy().decode('utf-8'))
        image = np.array(image)[..., np.newaxis]
        image = tf.image.resize(image, size=[256, 192])
        image = tf.cast(image, dtype=tf.uint8)
        image = tf.one_hot(image, depth=depth, on_value=1., off_value=0.)
        image = tf.squeeze(image, axis=-2)
        return image  
        
    
    
    def cg_preprocess(self, items_dict):
        cloth = self.cg_decode_resize(items_dict['cloth'], [1024, 768, 3]) / 255.
        cloth_mask = self.cg_decode_resize(items_dict['cloth_mask'], [1024, 768, 1]) / 255.
        image_densepose = self.cg_decode_resize(items_dict['image_densepose'], [1024, 768, 3]) / 255.
        image_parse_agnostic = tf.py_function(self.decode_parse_img, [items_dict['image_parse_agnostic'], hparams['parse_agnostic_classes']], [tf.float32])
        image_parse = tf.py_function(self.decode_parse_img, [items_dict['image_parse'], hparams['image_parse_classes']], [tf.float32])
        
        return {
            'cloth': cloth,
            'cloth_mask': cloth_mask,
            'image_densepose': image_densepose,
            'image_parse_agnostic': image_parse_agnostic[0],
            'image_parse': image_parse[0]
        }

    
    def preprocess_dataset(self, dataset: tf.data.Dataset):
        dataset = dataset.shuffle(32)
        dataset = dataset.batch(batch_size=hparams['cg_batch_size'], drop_remainder=True)
        return dataset.prefetch(tf.data.AUTOTUNE)
        
        
    def prepare_cg_datasets(self):
        dataset = self.generate_cg_dataset()
        dataset = dataset.map(self.cg_preprocess, num_parallel_calls=5)
        train_size = int(hparams['cg_dataset_size'] * 1.0)
        train_dataset = dataset.take(train_size)
        #valid_dataset = dataset.skip(train_size)    
        train_dataset = self.preprocess_dataset(train_dataset)
        #valid_dataset = self.preprocess_dataset(valid_dataset)
        return train_dataset
    
    
    def visualize_dataset(self):
        dataset = self.prepare_cg_datasets()
        figure = plt.figure(figsize=(15, 10))
        idx = 1
        for batch in dataset.take(5):
            figure.add_subplot(5, 5, idx)
            image_parse = np.array(lip_label_colours, np.uint8)[tf.argmax(batch['image_parse'][0], axis=-1)]
            plt.imshow(image_parse)
            plt.axis('off')
            plt.title('Image Parse')
            idx += 1
            
            figure.add_subplot(5, 5, idx)
            parse_agnostic = np.array(lip_label_colours, np.uint8)[tf.argmax(batch['image_parse_agnostic'][0], axis=-1)]
            plt.imshow(parse_agnostic)
            plt.axis('off')
            plt.title('Parse Agnostic')
            idx += 1
            
            figure.add_subplot(5, 5, idx)
            image_densepose = batch['image_densepose'][0]
            plt.imshow(image_densepose)
            plt.axis('off')
            plt.title('Image Densepose')
            idx += 1
            
            figure.add_subplot(5, 5, idx)
            cloth = batch['cloth'][0]
            plt.imshow(cloth)
            plt.axis('off')
            plt.title('Cloth')
            idx += 1
            
            figure.add_subplot(5, 5, idx)
            cloth_mask = batch['cloth_mask'][0]
            plt.imshow(cloth_mask, cmap='gray')
            plt.axis('off')
            plt.title('Cloth Mask')
            idx += 1
            
                
        plt.tight_layout()
        plt.show()
    
            
            
    
    
    
class IG_Data_Preparation():
    def __init__(self, cg_checkpoint='models/cg_models/cg_generator_epoch_500.h5'):
        self.im_names = os.listdir(os.path.join(hparams['data_dir'], 'image'))
        self.im_names_iterator = self.im_names.__iter__()
        self.condition_generator = tf.keras.models.load_model(
            filepath=cg_checkpoint,
            custom_objects={'Warping_Layer': Warping_Layer}
        )
        self.cg_postprocessor = CG_PostProcessing()
        
        
    def load_preprocess_img(self, image_path, grayscale=False, target_size=[256, 192]):
        image = tf.keras.preprocessing.image.load_img(image_path, grayscale=grayscale, target_size=target_size, interpolation='bilinear')
        image = tf.keras.preprocessing.image.img_to_array(image)[tf.newaxis, ...] / 255.
        return image
    
    
    def load_parse_img(self, image_path, depth, size=[256, 192]):
        image = Image.open(image_path.numpy().decode('utf-8'))
        image = np.array(image)[..., np.newaxis]
        image = tf.image.resize(image, size=size)
        image = tf.cast(image, dtype=tf.uint8)
        image = tf.one_hot(image, depth=depth, on_value=1., off_value=0.)
        image = tf.squeeze(image, axis=-2)[tf.newaxis, ...]
        return image
    
    
    def ig_generator(self):
        for _ in range(self.im_names.__len__()):
            image_name = self.im_names_iterator.__next__()
            cloth = self.load_preprocess_img(os.path.join(hparams['data_dir'], 'cloth', image_name))
            image = self.load_preprocess_img(os.path.join(hparams['data_dir'], 'image', image_name), target_size=[1024, 768])
            cloth_mask = self.load_preprocess_img(os.path.join(hparams['data_dir'], 'cloth-mask', image_name), grayscale=True)
            image_densepose = self.load_preprocess_img(os.path.join(hparams['data_dir'], 'image-densepose', image_name))
            image_parse_agnostic = tf.py_function(
                self.load_parse_img, 
                [os.path.join(hparams['data_dir'], 'image-parse-agnostic-v3.2', image_name.replace('jpg', 'png')), hparams['parse_agnostic_classes']],
                [tf.float32]
            )[0]
            face_parse = tf.py_function(
                self.load_parse_img, 
                [os.path.join(hparams['data_dir'], 'face-parse', image_name.replace('jpg', 'png')), 2, [1024, 768]],
                [tf.float32]
            )[0]
            
            cg_inputs = {
                'cloth': cloth,
                'cloth_mask': cloth_mask,
                'image_densepose': image_densepose,
                'image_parse_agnostic': image_parse_agnostic
            }
            
            cg_outputs = self.condition_generator(cg_inputs)
            
            
            warped_cloth_mask_4 = self.cg_postprocessor.cloth_warper(
                image=cg_inputs['cloth_mask'],
                flow_map=cg_outputs['flow_map_4_up']
            )
            
            warped_cloth_4 = self.cg_postprocessor.cloth_warper(
                image=cg_inputs['cloth'],
                flow_map=cg_outputs['flow_map_4_up']
            )
            
            seg_mask = self.cg_postprocessor.misalignment_remover(
                cg_outputs['seg_bottleneck'],
                warped_cloth_mask_4,
            )
            
            _, warped_cloth = self.cg_postprocessor.occlusion_handler(
                seg_mask,
                warped_cloth_mask_4,
                warped_cloth_4
            )
            
            # To generate Cloth Agnostic Person Image
            res_seg_mask = tf.image.resize(seg_mask, size=[1024, 768], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            res_seg_mask_ids = tf.broadcast_to(tf.argmax(res_seg_mask, axis=-1)[..., tf.newaxis], shape=[1, 1024, 768, 3])
            # To remove face and neck from image
            face_mask_ids = tf.broadcast_to(tf.argmax(face_parse, axis=-1)[..., tf.newaxis], shape=[1, 1024, 768, 3])
            white_background = tf.ones_like(image)
            filtered_image = tf.where(face_mask_ids == 1, white_background, image)
            # To put gray mask on arms and upper clothes
            gray_background = tf.ones_like(image) * 0.5
            cloth_agnostic = tf.where(
                tf.logical_or(
                    tf.logical_or(res_seg_mask_ids==2, res_seg_mask_ids==6),
                    tf.equal(res_seg_mask_ids, 7) 
                ),
                gray_background, 
                filtered_image
            )
            
            yield {
                'seg_mask': res_seg_mask[0],
                'cloth_agnostic': cloth_agnostic[0],
                'warped_cloth': tf.image.resize(warped_cloth, size=[1024, 768], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[0],
                'image_densepose': tf.image.resize(image_densepose, size=[1024, 768], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[0],
                'filtered_image': tf.where(face_mask_ids == 1, white_background, image)[0]
            }
            
                  
    def generate_dataset(self):
        dataset = tf.data.Dataset.from_generator(
            self.ig_generator,
            output_signature={
                'seg_mask': tf.TensorSpec(shape=[1024, 768, hparams['image_parse_classes']], dtype=tf.float32),
                'cloth_agnostic': tf.TensorSpec(shape=[1024, 768, 3], dtype=tf.float32),
                'warped_cloth': tf.TensorSpec(shape=[1024, 768, 3], dtype=tf.float32),
                'image_densepose': tf.TensorSpec(shape=[1024, 768, 3], dtype=tf.float32),
                'filtered_image': tf.TensorSpec(shape=[1024, 768, 3], dtype=tf.float32)
            }
        )
        return dataset
    
    
    def preprocess_dataset(self, dataset: tf.data.Dataset):
        dataset = dataset.shuffle(32)
        dataset = dataset.batch(batch_size=hparams['ig_batch_size'], drop_remainder=True)
        return dataset.prefetch(tf.data.AUTOTUNE)
    
    
    def prepare_dataset(self):
        dataset = self.generate_dataset()
        train_size = int(hparams['ig_dataset_size'] * 1.0)
        train_dataset = dataset.take(train_size)
        #valid_dataset = dataset.skip(train_size)
        train_dataset = self.preprocess_dataset(train_dataset)
        #valid_dataset = self.preprocess_dataset(valid_dataset)
        return train_dataset
    
    
    def visualize_dataset(self):
        dataset = self.prepare_dataset()
        figure = plt.figure(figsize=(15, 10))
        idx = 1
        for batch in dataset.take(5):
            figure.add_subplot(5, 5, idx)
            seg_image = np.array(lip_label_colours, np.uint8)[tf.argmax(batch['seg_mask'][0], axis=-1)]
            plt.imshow(seg_image)
            plt.axis('off')
            plt.title('Seg Mask')
            idx += 1
            
            figure.add_subplot(5, 5, idx)
            cloth_agnostic = batch['cloth_agnostic'][0]
            plt.imshow(cloth_agnostic)
            plt.axis('off')
            plt.title('Cloth Agnostic')
            idx += 1
            
            figure.add_subplot(5, 5, idx)
            warped_cloth = batch['warped_cloth'][0]
            plt.imshow(warped_cloth)
            plt.axis('off')
            plt.title('Warped Cloth')
            idx += 1
            
            figure.add_subplot(5, 5, idx)
            image_densepose = batch['image_densepose'][0]
            plt.imshow(image_densepose)
            plt.axis('off')
            plt.title('Image DensePose')
            idx += 1
            
            figure.add_subplot(5, 5, idx)
            filtered_image = batch['filtered_image'][0]
            plt.imshow(filtered_image)
            plt.axis('off')
            plt.title('Filtered Image')
            idx += 1
            
                
        plt.tight_layout()
        plt.show()
        
        
        
        
            
            
            

        
     

    
if __name__ == '__main__':
    a = IG_Data_Preparation()
    a.visualize_dataset()
    
    
    
    
    '''pred_seg_mask = a.ig_generator()[0, ..., 0]
    pred_seg_mask = tf.cast(pred_seg_mask, tf.uint8)
    pred_seg_palette = np.array(lip_label_colours, dtype=np.uint8)
    pred_seg_image = pred_seg_palette[pred_seg_mask]
    pred_seg_image = Image.fromarray(pred_seg_image)
    pred_seg_image.show()'''
    

    
        
    
    
    
        
        
    