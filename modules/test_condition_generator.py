import sys
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import CG_PostProcessing
from constants import lip_label_colours
from custom_layers import Warping_Layer
from data_preparation import Data_Preparation



data_preparation = Data_Preparation()
train_dataset = data_preparation.prepare_cg_datasets()

cg_postprocessor = CG_PostProcessing()


model = tf.keras.models.load_model(
    filepath='models/cg_models/cg_generator_epoch_80.h5',
    custom_objects={'Warping_Layer': Warping_Layer}
)

for i in range(5):
    
    figure, axes = plt.subplots(3, 3, figsize=(10, 8))
    for row, batch in train_dataset.shuffle(32).take(3).enumerate():
        _ = batch.pop('image_parse')

        model_output = model(batch)
        
        warped_cloth_mask_4 = cg_postprocessor.cloth_warper(
            image=batch['cloth_mask'],
            flow_map=model_output['flow_map_4_up']
        ) 
        
        warped_cloth_4 = cg_postprocessor.cloth_warper(
            image=batch['cloth'],
            flow_map=model_output['flow_map_4_up']
        )      
        
        fake_seg_mask = cg_postprocessor.misalignment_remover(
            model_output['seg_bottleneck'],
            warped_cloth_mask_4,
        )
        
        
        pred_cloth_mask, pred_cloth = cg_postprocessor.occlusion_handler(
                segmentation_mask_scores=fake_seg_mask,
                warped_cloth_mask=warped_cloth_mask_4,
                warped_cloth=warped_cloth_4
        )
        

        axes[row, 0].imshow(pred_cloth_mask[0], cmap='gray')
        axes[row, 0].axis('off')
        axes[row, 0].set_title('Cloth Mask')
        
        axes[row, 1].imshow(pred_cloth[0])
        axes[row, 1].axis('off')
        axes[row, 1].set_title('Cloth')
        
        fake_seg_img = np.array(lip_label_colours, dtype=np.uint8)
        fake_seg_img = fake_seg_img[tf.argmax(fake_seg_mask[0], axis=-1)]
        axes[row, 2].imshow(fake_seg_img)
        axes[row, 2].axis('off')
        axes[row, 2].set_title('Seg Mask')
        
        plt.tight_layout()
        figure.show()
        
    data_preparation.im_names_iterator = data_preparation.im_names.__iter__()

    plt.show()
