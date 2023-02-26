from typing import Union
import PIL
from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True

def process_image(
    image_path:str, 
    rescale_size:Union[list, tuple],
    upper_bound:int = 10,
    debug:bool = False
) -> Union[np.array, tuple]:
    r"""
    scale the image resolution to predetermined resolution and return
    it as numpy
    
    args:
        image_path (:obj:`str`):
            path to file
        rescale_size (:obj:`list` or `tuple`):
            width and height target
        upper_bound (:obj:`int`, *optional*, defaults to 10):
            major axis obund (not important, just set it as high as possible)
        debug (:obj:`bool`, *optional*, defaults to `False`):
            will return tuple (np.array, PIL.Image)

    return: pd.DataFrame or (np.array, PIL.Image)
    """
    image = Image.open(image_path)

    # get smallest and largest res from image
    minor_axis_value = min(image.size)
    minor_axis = image.size.index(minor_axis_value)
    major_axis_value = max(image.size)
    major_axis = image.size.index(major_axis_value)
    major_axis_rescale = rescale_size[minor_axis]/image.size[minor_axis] * image.size[major_axis]

    # scale image
    # TODO: instead using thumbnail use rescale with integer scale
    sampling_algo = PIL.Image.LANCZOS
    if major_axis_rescale >= rescale_size[major_axis] and minor_axis == 0:
        # ("x is smaller than y and y is larger than y scaled")
        image.thumbnail((rescale_size[minor_axis], rescale_size[major_axis] * upper_bound), sampling_algo)
    elif major_axis_rescale < rescale_size[major_axis] and minor_axis == 0:
        # ("x is smaller than y but y is smaller than y scaled")
        image.thumbnail((rescale_size[minor_axis] * upper_bound, rescale_size[major_axis]), sampling_algo)
    elif major_axis_rescale >= rescale_size[major_axis] and minor_axis == 1:
        # ("y is smaller than x and x is larger than x scaled")
        image.thumbnail((rescale_size[major_axis] * upper_bound, rescale_size[minor_axis]), sampling_algo)
    else:
        # ("y is smaller than x but x is smaller than x scaled")
        image.thumbnail((rescale_size[major_axis], rescale_size[minor_axis] * upper_bound), sampling_algo)

    # warning
    if max(image.size) < max(rescale_size):
        print(f"[WARN] image {image_path} is smaller than designated batch, zero pad will be added")
    
    if minor_axis == 0 or  major_axis_rescale < rescale_size[major_axis]:
        # left and right same crop top and bottom
        top = (image.size[1] - rescale_size[1])//2
        bottom = (image.size[1] + rescale_size[1])//2

        # remainder add
        bottom_remainder = (top  + bottom)
        # left, top, right, bottom
        image = image.crop((0, top, image.size[0], bottom))
    else:
        # top and bottom same crop the left and right
        left = (image.size[0] - rescale_size[0])//2
        right = (image.size[0] + rescale_size[0])//2
        # left, top, right, bottom
        image = image.crop((left, 0, right, image.size[1]))

    # cheeky resize to catch missmatch 
    image = image.resize(rescale_size, resample=sampling_algo)
    # for some reason np flip width and height
    np_image = np.array(image)
    # normalize
    np_image = np_image/127.5 - 1
    # height width channel to channel height weight
    np_image = np.transpose(np_image, (2,0,1))
    # add batch axis
    # np_image = np.expand_dims(np_image, axis=0)

    if debug:
        return (np_image, image)
    else:
        return (np_image)

