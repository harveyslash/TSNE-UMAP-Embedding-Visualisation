from PIL import Image
import glob
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import click
import json


def images_to_sprite(data):
    """
    Creates the sprite image along with any necessary padding
    Source : https://github.com/tensorflow/tensorflow/issues/6322
    Args:
      data: NxHxW[x3] tensor containing the images.
    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) - min).transpose(3, 0, 1, 2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) / max).transpose(3, 0, 1, 2)

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
               (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
                  constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
                                                           + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data


def populate_img_arr(images_paths, size=(100, 100), should_preprocess=False):
    """
    Get an array of images for a list of image paths
    Args:
        size: the size of image , in pixels
        should_preprocess: if the images should be processed (according to InceptionV3 requirements)
    Returns:
        arr: An array of the loaded images
    """
    arr = []
    for i, img_path in enumerate(images_paths):
        img = image.load_img(img_path, target_size=size)
        x = image.img_to_array(img)
        arr.append(x)
    arr = np.array(arr)
    if should_preprocess:
        arr = preprocess_input(arr)
    return arr


@click.command()
@click.option('--data', help='Data folder,has to end with /')
@click.option('--name', default="Visualisation", help='Name of visualisation')
@click.option('--sprite_size', default=100, help='Size of sprite')
@click.option('--tensor_name', default="tensor.bytes", help='Name of Tensor file')
@click.option('--sprite_name', default="sprites.png", help='Name of sprites file')
@click.option('--model_input_size', default=299, help='Size of inputs to model')
def main(data, name, sprite_size, tensor_name, sprite_name, model_input_size):
    
    if not data.endswith('/'):
        raise ValueError('Makesure --name ends with a "/"')
    
    images_paths = glob.glob(data + "*.jpg")
    images_paths.extend(glob.glob(data + "*.JPG"))
    images_paths.extend(glob.glob(data + "*.png"))

    model = InceptionV3(include_top=False, pooling='avg')

    img_arr = populate_img_arr(images_paths, size=(model_input_size, model_input_size), should_preprocess=True)
    preds = model.predict(img_arr, batch_size=64)
    preds.tofile("./oss_data/" + tensor_name)

    raw_imgs = populate_img_arr(images_paths, size=(sprite_size, sprite_size), should_preprocess=False)
    sprite = Image.fromarray(images_to_sprite(raw_imgs).astype(np.uint8))
    sprite.save('./oss_data/' + sprite_name)

    oss_json = json.load(open('./oss_data/oss_demo_projector_config.json'))
    tensor_shape = [raw_imgs.shape[0], model.output_shape[1]]
    single_image_dim = [raw_imgs.shape[1], raw_imgs.shape[2]]

    json_to_append = {"tensorName": name,
                      "tensorShape": tensor_shape,
                      "tensorPath": "./oss_data/" + tensor_name,
                      "sprite": {"imagePath": "./oss_data/" + sprite_name,
                                 "singleImageDim": single_image_dim}}
    oss_json['embeddings'].append(json_to_append)
    with open('oss_data/oss_demo_projector_config.json', 'w+') as f:
        json.dump(oss_json, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
