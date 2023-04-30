import pandas as pd
import numpy as np
import random
import jax
import os

from basket.preprocess.dataframe_processor import (
    discrete_scale_to_equal_area,
    resolution_bucketing_batch_with_chunking,
    tag_suffler_to_comma_separated,
    scale_by_minimum_axis,
)

epoch = 0

for epoch in range(10):
    seed = 42 + epoch

    # pandas bucketing
    csv_file = f"/root/project/JAX-SD-trainer/basket/laion_aesthetics_1024_33M_{epoch+1}.parquet"
    image_dir = f"/root/project/dataset/dataset_{epoch+1}"
    batch_num = 2
    batch_size = jax.device_count() * batch_num
    maximum_resolution_area = [576**2, 704**2, 832**2, 960**2, 1088**2]
    bucket_lower_bound_resolution = [384, 512, 576, 704, 832]
    maximum_axis = 1024
    minimum_axis = 512
    # if true maximum_resolution_area and bucket_lower_bound_resolution not used
    # else maximum_axis and minimum_axis is not used
    use_ragged_batching = False
    repeat_batch = 10

    # batch generator (dataloader)
    image_folder = f"/root/project/dataset/dataset_{epoch+1}"
    image_name_col = "file"
    orig_width_height = ["WIDTH", "HEIGHT"]
    width_height = ["new_image_width", "new_image_height"]
    caption_col = "TEXT"
    hash_col = "hash"
    extension = ".webp"
    token_concatenate_count = 3
    token_length = 75 * token_concatenate_count + 2

    # ensure image exist
    data = pd.read_parquet(csv_file)  # .sample(200000, random_state=1)
    # create file name column so this script can retrieve those files
    data[image_name_col] = data[hash_col].astype("str") + extension
    image_list = os.listdir(image_dir)
    data = data.loc[data[image_name_col].isin(image_list)]

    image_properties = zip(maximum_resolution_area, bucket_lower_bound_resolution)
    store_multiple_aspect_ratio = []

    for aspect_ratio in image_properties:
        data_processed = discrete_scale_to_equal_area(
            dataframe=data,
            image_width_col_name=orig_width_height[0],
            image_height_col_name=orig_width_height[1],
            new_image_width_col_name=width_height[0],
            new_image_height_col_name=width_height[1],
            max_res_area=aspect_ratio[0],
            bucket_lower_bound_res=aspect_ratio[1],
            extreme_aspect_ratio_clip=2.0,
            aspect_ratio_clamping=2.0,
            return_with_helper_columns=True,
        )
        store_multiple_aspect_ratio.append(data_processed)

    data_processed = pd.concat(store_multiple_aspect_ratio)

    # this function should provide me with evenly cutted dataframe
    # but somehow it's not working with parquet file, idk why!
    data_processed = resolution_bucketing_batch_with_chunking(
        dataframe=data_processed,
        image_width_col_name=width_height[0],
        image_height_col_name=width_height[1],
        seed=seed,
        bucket_batch_size=batch_size,
        repeat_batch=repeat_batch,
        bucket_group_col_name="bucket_group",
    )

    assert (
        len(data_processed) % batch_size == 0
    ), f"DATA IS NOT CLEANLY DIVISIBLE BY {batch_size} {len(data_processed)%batch_size}"
