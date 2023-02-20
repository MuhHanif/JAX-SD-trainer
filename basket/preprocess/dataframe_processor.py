import pandas as pd
import numpy as np
import random

def discrete_scale_to_equal_area(
    dataframe:pd.DataFrame,
    image_height_col_name:str,
    image_width_col_name:str,
    new_image_height_col_name:str,
    new_image_width_col_name:str,
    maximum_area:int = 512 ** 2,
    nearest_multiple:int = 64,
    extreme_aspect_ratio_clip:float = 4.0,
    aspect_ratio_clamping:float = 2.0,
    return_with_helper_columns:bool = False
) -> pd.DataFrame:
    r"""
    scale the image resolution to nearest multiple value 
    with less or equal to the maximum area constraint

    note:
      this code assumes that the image is larger than maximum area
      if the image is smaller than maximum area it will get scaled up
    
    args:
      dataframe (:obj:`pd.DataFrame`):
        input dataframe
      image_height_col_name (:obj:`str`):
        target column height
      image_width_col_name (:obj:`str`):
        target column width
      new_image_height_col_name (:obj:`str`):
        column name for new height value
      new_image_width_col_name (:obj:`str`):
        column name for new width value
      maximum_area (:obj:`int`, *optional*, defaults to 512 ** 2):
        maximum pixel area to be compared with
      nearest_multiple (:obj:`int`, *optional*, defaults to 64):
        rounding value
      extreme_aspect_ratio_clip (:obj:`float`, *optional*, defaults to 4.0):
        drop images that have width/height or height/width 
        beyond threshold value
      aspect_ratio_clamping (:obj:`float`, *optional*, defaults to 2.0):
        crop images that have width/height or height/width 
        beyond threshold value 
      return_with_helper_columns (:obj:`bool`, *optional*, defaults to `False`):
        return pd.DataFramw with helper columns (for debugging purposes)

    return: pd.DataFrame
    """
    clamped_height = "_clamped_height"
    clamped_width = "_clamped_width"

    error_message = f"extreme_aspect_ratio_clip ({extreme_aspect_ratio_clip}) is less than aspect_ratio_clamping ({aspect_ratio_clamping})"
    assert extreme_aspect_ratio_clip > aspect_ratio_clamping , error_message

    # drop ridiculous aspect ratio
    dataframe = dataframe[dataframe[image_height_col_name] / dataframe[image_width_col_name] <=extreme_aspect_ratio_clip]
    dataframe = dataframe[dataframe[image_width_col_name] / dataframe[image_height_col_name] <=extreme_aspect_ratio_clip]

    # clamp aspect ratio
    dataframe[clamped_height] = dataframe[image_height_col_name]
    dataframe[clamped_width] = dataframe[image_width_col_name]
    loc_boolean_map = dataframe[clamped_height] / dataframe[clamped_width] >= aspect_ratio_clamping
    dataframe.loc[loc_boolean_map, clamped_height] = dataframe.loc[loc_boolean_map, clamped_width] * aspect_ratio_clamping
    loc_boolean_map = dataframe[clamped_width] / dataframe[clamped_height] >= aspect_ratio_clamping
    dataframe.loc[loc_boolean_map, clamped_width] = dataframe.loc[loc_boolean_map, clamped_height] * aspect_ratio_clamping

    #create square area scaling
    image_area = dataframe[clamped_height] * dataframe[clamped_width]
    image_area = (maximum_area / image_area) ** (1/2)

    # rescaling width and height
    new_height = (dataframe[clamped_height] * image_area) // nearest_multiple * nearest_multiple
    new_width = (dataframe[clamped_width] * image_area) // nearest_multiple * nearest_multiple

    # insert column to the dataframe
    dataframe[new_image_height_col_name] = new_height
    dataframe[new_image_width_col_name] = new_width

    # square special case
    loc_boolean_map = dataframe[clamped_height] == dataframe[clamped_width]
    dataframe.loc[loc_boolean_map, [new_image_width_col_name, new_image_height_col_name]] = maximum_area ** (1/2)
    
    # remove helper columns
    if not return_with_helper_columns:
      dataframe = dataframe.drop(columns=[clamped_height,clamped_width])

    return dataframe

def resolution_bucketing_batch(
    dataframe:pd.DataFrame,
    image_height_col_name:str,
    image_width_col_name:str,
    seed:int = 0,
    bucket_batch_size:int = 8,
    bucket_group_col_name = "bucket_group"
) -> pd.DataFrame:
    r"""
    create aspect ratio bucket and batch it

    note:
        non full batch will get dropped

    args:
        dataframe (:obj:`pd.DataFrame`):
            input dataframe
        image_height_col_name (:obj:`str`):
            target column height
        image_width_col_name (:obj:`str`):
            target column width
        seed (:obj:`int`, *optional*, defaults to 0):
            rng seed for reproductibility
        bucket_batch_size (:obj: `int`, *optional*, default to 8):
            size of the bucket batch, non full batch will get dropped
        bucket_group_col_name (:obj:`str`):
            bucket column name to store randomized order

    return: pd.DataFrame
    """
    dataframe = dataframe.groupby([image_height_col_name, image_width_col_name])

    # store first batch for JAX to compile 
    first_batch = pd.DataFrame()
    remainder_batch = pd.DataFrame()

    # helper counter
    group_count = 0

    # increment bucket grouping order for the next group
    group_max_index = 0

    for group, data in dataframe:

        # helper counter
        group_count = group_count + 1

        # shuffle rows within group
        data = data.sample(frac=1, random_state=seed)
        
        # create ordered index for generating bucket batch
        data = data.reset_index()
        data[bucket_group_col_name] = data.index // bucket_batch_size + group_max_index 

        # strip tail end bucket because it's not full bucket
        tail_end_length = len(data.loc[data[bucket_group_col_name] == data[bucket_group_col_name].max()])
        if tail_end_length < bucket_batch_size:
            data = data.iloc[:-tail_end_length,:]

        # build first batch for JAX to compile 
        first_batch = pd.concat([first_batch, data.iloc[-bucket_batch_size:,:]])

        # remainder batch
        data = data.iloc[:-bucket_batch_size,:]
        remainder_batch = pd.concat([remainder_batch, data])

        # increment bucket grouping order for the next group
        group_max_index = data[bucket_group_col_name].max() + 1
        
    # shuffling bucket
    bucket_order = remainder_batch[bucket_group_col_name]
    np.random.seed(seed)
    bucket_order_array = bucket_order.unique()
    np.random.shuffle(bucket_order_array)

    # replacing order of the bucket
    replace_dict = dict(zip(bucket_order.unique(), bucket_order_array)) 
    bucket_order = bucket_order.map(replace_dict)

    # shifting bucket index to make room for the first batch
    remainder_batch[bucket_group_col_name] = bucket_order + group_count

    # shuffling first batch bucket
    first_batch_order = first_batch[bucket_group_col_name]
    np.random.seed(seed)
    first_batch_order_array = first_batch_order.unique()
    np.random.shuffle(first_batch_order_array)

    # replacing order of the first batch bucket
    replace_dict = dict(zip(first_batch_order_array, list(range(len(first_batch_order_array))))) 
    first_batch[bucket_group_col_name] = first_batch_order.map(replace_dict)

    #combine both batch 
    dataframe = pd.concat([first_batch, remainder_batch])

    # restore original index back
    dataframe = dataframe.set_index(dataframe["index"], drop=True)

    # create multi level index for bucket s it can be accessed with loc
    dataframe = dataframe.set_index(dataframe[bucket_group_col_name], append=True)
    dataframe = dataframe.swaplevel(0,1)

    return dataframe

def tag_suffler_to_comma_separated(tags:str, seed:int) -> str:
    r"""
    suffle and reformat tag from `this_is a_tag to_suffle`
    to `to suffle, a tag, this is`

    args:
        tags (:obj:`str`):
            tag string
        seed (:obj:`int`, *optional*, defaults to 0):
            rng seed for reproductibility

    return: str
    """

    tags = tags.replace(" ",",").replace("_"," ").split(",")
    random.Random(len(tags)+seed).shuffle(tags)
    tags = ", ".join(tags)
    return(tags)
  
def tag_suffler_to_space_separated(tags:str, seed:int) -> str:
    r"""
    suffle and reformat tag from `this_is a_tag to_suffle`
    to `to_suffle a_tag this_is`

    args:
        tags (:obj:`str`):
            tag string
        seed (:obj:`int`, *optional*, defaults to 0):
            rng seed for reproductibility

    return: str
    """

    tags = tags.split(" ")
    random.Random(len(tags)+seed).shuffle(tags)
    tags = " ".join(tags)
    return(tags)
