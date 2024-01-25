import os
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import gradio as gr

from utils.prediction import get_sr_image
from utils.config import config

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Add, Lambda
from tensorflow.python.keras.layers import PReLU

from utils.normalization import normalize_01, denormalize_m11

upsamples_per_scale = {
    2: 1,
    4: 2,
    8: 3
}


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


def upsample(x_in, num_filters):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = Lambda(pixel_shuffle(scale=2))(x)
    return PReLU(shared_axes=[1, 2])(x)


def residual_block(block_input, num_filters, momentum=0.8):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(block_input)
    x = BatchNormalization(momentum=momentum)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = Add()([block_input, x])
    return x


def build_srresnet(scale=4, num_filters=64, num_res_blocks=16):
    if scale not in upsamples_per_scale:
        raise ValueError(f"available scales are: {upsamples_per_scale.keys()}")

    num_upsamples = upsamples_per_scale[scale]

    lr = Input(shape=(None, None, 3))
    x = Lambda(normalize_01)(lr)

    x = Conv2D(num_filters, kernel_size=9, padding='same')(x)
    x = x_1 = PReLU(shared_axes=[1, 2])(x)

    for _ in range(num_res_blocks):
        x = residual_block(x, num_filters)

    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x_1, x])

    for _ in range(num_upsamples):
        x = upsample(x, num_filters * 4)

    x = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(x)
    sr = Lambda(denormalize_m11)(x)

    return Model(lr, sr)



def use_srresnet(image):
    
    lr = image
    
    model = build_srresnet(scale=4)
    model_key = f"srresnet"
    weights_directory = os.path.abspath(f"weights/{model_key}")
    os.makedirs(weights_directory, exist_ok=True)
    weights_file = f'{weights_directory}/generator.h5'
    model.load_weights(weights_file)

    sr = get_sr_image(model, lr)
    
    return sr

def use_srgan(image):
        
        lr = image
        
        model = build_srresnet(scale=4)
        model_key = f"srgan"
        weights_directory = os.path.abspath(f"weights/{model_key}")
        os.makedirs(weights_directory, exist_ok=True)
        weights_file = f'{weights_directory}/generator.h5'
        model.load_weights(weights_file)
    
        sr = get_sr_image(model, lr)
        
        return sr


with gr.Blocks() as demo:

    image = gr.Image(label="Input low resolution image")
    with gr.Row():
        
        button_use_srresnet = gr.Button("Use SRResNet")
        button_use_srgan = gr.Button("Use SRGAN")
    
    with gr.Row():
        output_srresnet = gr.Image(label="Output SRResNet")
        output_srgan = gr.Image(label="Output SRGAN")

    button_use_srresnet.click(fn=use_srresnet, inputs=[image], outputs=[output_srresnet],api_name="SRResNet")
    button_use_srgan.click(fn=use_srgan, inputs=[image], outputs=[output_srgan],api_name="SRGAN")

demo.launch()