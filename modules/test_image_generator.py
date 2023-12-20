from PIL import Image
import tensorflow as tf
from data_preparation import IG_Data_Preparation
from custom_layers import SpatiallyAdaptiveNormalization, NormTanhActivation


data_preparation = IG_Data_Preparation(cg_checkpoint='models/cg_models/cg_generator_epoch_80.h5')
train_dataset = data_preparation.prepare_dataset()


model = tf.keras.models.load_model(
    filepath='models/ig_models/ig_generator_epoch_60.h5', 
    custom_objects={
        'leaky_relu': tf.nn.leaky_relu,
        'SpatiallyAdaptiveNormalization': SpatiallyAdaptiveNormalization,
        'NormTanhActivation': NormTanhActivation
    }
)



for batch in train_dataset:
    _ = batch.pop('filtered_image')

    model_output = model(batch)

    image = Image.fromarray((model_output[0].numpy() * 255).astype('uint8'))
    image.show()