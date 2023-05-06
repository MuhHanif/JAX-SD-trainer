import random
import time
import logging
import sys
from queue import Queue as thread_queue
from multiprocessing import Process, Queue, Value, Lock
import os
import pandas as pd
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm
import numpy as np

from transformers import T5Tokenizer

from basket.preprocess.batch_processor import (
    generate_batch,
    process_image,
    tokenize_text,
    worker_batching,
)
from basket.preprocess.dataframe_processor import (
    discrete_scale_to_equal_area,
    resolution_bucketing_batch_with_chunking,
    tag_suffler_to_comma_separated,
    scale_by_minimum_axis,
)

# master seed
epoch = 0
seed = 42 + epoch
steps_offset = 0
# pandas bucketing
csv_file = f"/home/user/laion_aesthetics_1024_33M_1_c{epoch+1}.parquet"
image_dir = f"/home/user/data_dump/laion"
batch_num = 4
batch_size = jax.device_count() * batch_num
maximum_resolution_area = [512**2]  # [576**2, 704**2, 832**2, 960**2, 1088**2]
bucket_lower_bound_resolution = [256]  # [384, 512, 576, 704, 832]
maximum_axis = 1024
minimum_axis = 512
# if true maximum_resolution_area and bucket_lower_bound_resolution not used
# else maximum_axis and minimum_axis is not used
use_ragged_batching = False
repeat_batch = 10
shuffle_tags = False

# batch generator (dataloader)
worker_count = 5
image_folder = f"/home/user/data_dump/laion"
image_name_col = "file"
orig_width_height = ["WIDTH", "HEIGHT"]
width_height = ["new_image_width", "new_image_height"]
caption_col = "TEXT"
hash_col = "hash"
extension = ".webp"
token_concatenate_count = 1
token_length = 512  # 75 * token_concatenate_count + 2
debug = True
base_model_name = "sd1.5-t5-e"
model_dir = f"/home/user/data_dump/{base_model_name}{epoch}"  # continue from last model

# ===============[pandas batching & bucketing]=============== #
# ensure image exist
data = pd.read_parquet(csv_file)
# create file name column so this script can retrieve those files
data[image_name_col] = data[hash_col].astype("str") + extension
image_list = os.listdir(image_dir)
data = data.loc[data[image_name_col].isin(image_list)]

# create bucket resolution
if use_ragged_batching:
    data_processed = scale_by_minimum_axis(
        dataframe=data,
        image_width_col=orig_width_height[0],
        image_height_col=orig_width_height[1],
        new_image_width_col=width_height[0],
        new_image_height_col=width_height[1],
        target_minimum_scale=minimum_axis,
        target_maximum_scale=maximum_axis,
    )

else:
    # check guard
    assert len(maximum_resolution_area) == len(
        bucket_lower_bound_resolution
    ), "list count not match!"
    # multiple aspect ratio training!
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
            return_with_helper_columns=False,
        )
        store_multiple_aspect_ratio.append(data_processed)

    data_processed = pd.concat(store_multiple_aspect_ratio)

# generate bucket batch and provide starting batch
# with all possible image resolution to make sure jax compile everything in one go
data_processed = resolution_bucketing_batch_with_chunking(
    dataframe=data_processed,
    image_width_col_name=width_height[0],
    image_height_col_name=width_height[1],
    seed=seed,
    bucket_batch_size=batch_size,
    repeat_batch=repeat_batch,
    bucket_group_col_name="bucket_group",
)


# NOTE: SHUFFLE TAGS MUST NOT BE PERFORMED FOR LAION DATASET!
# shuffle tags
def shuffle(tags, seed):
    tags = tags.split(",")
    random.Random(len(tags) * seed).shuffle(tags)
    tags = ",".join(tags)
    return tags


if shuffle_tags:
    logging.info("shuffling captions")
    data_processed[caption_col] = data_processed[caption_col].apply(
        lambda x: shuffle(x, seed)
    )

logging.info("creating bucket and dataloader sequence")
# ===============[load model to CPU]=============== #

tokenizer = T5Tokenizer.from_pretrained(model_dir, subfolder="tokenizer")
# ===============[simple dataloader]=============== #


# spawn dataloader in another core
def generate_batch_wrapper(
    lock: Lock,
    count: Value,
    worker_order: int,
    numb_of_worker: int,
    list_of_batch: list,
    queue: Queue,
    print_debug: bool = False,
):
    internal_queue = thread_queue(numb_of_worker)
    # loop until queue is full
    # list of batch is the entiere batch assigned to this worker
    print(list_of_batch)
    for batch in list_of_batch:
        tic = time.time()
        current_batch = generate_batch(
            process_image_fn=process_image,
            tokenize_text_fn=tokenize_text,
            tokenizer=tokenizer,
            dataframe=data_processed.iloc[
                batch * batch_size : batch * batch_size + batch_size
            ],
            folder_path=image_folder,
            image_name_col=image_name_col,
            caption_col=caption_col,
            caption_token_length=token_length,
            width_col=width_height[0],
            height_col=width_height[1],
            tokenizer_path=model_dir,
            batch_slice=token_concatenate_count,
        )
        toc = time.time()
        if print_debug and queue.full():
            print("queue is full!")
            print(round(toc - tic, 2))
        # put task in queue

        if not internal_queue.full():
            internal_queue.put([[current_batch, batch]])
            if print_debug:
                print(round(toc - tic, 2))
                print(f"putting task {batch} into internal queue")

        if internal_queue.full():
            # blocking while loop until counter is incremented from another worker
            if count.value % numb_of_worker != worker_order:
                while count.value % numb_of_worker != worker_order:
                    time.sleep(1)
                    print(f"{worker_order} modulo is not 0, sleeping")
            else:
                print(
                    f"===========================[{worker_order}]==========================="
                )
                with lock:
                    print(
                        "internal_queue is full, pushing everything to the main queue"
                    )

                    while internal_queue.empty() == False:
                        data = internal_queue.get()

                        queue.put(data)
                        print(f"dumping {data[0][1]} into main queue{queue.qsize()}")

                    count.value = count.value + 1
                    print(count.value, batch)


# ===============[training loop]=============== #

logging.info("start training")

# get group index as batch order
assert (
    len(data_processed) % batch_size == 0
), f"DATA IS NOT CLEANLY DIVISIBLE BY {batch_size} {len(data_processed)%batch_size}"
batch_order = list(range(0, len(data_processed) // batch_size))

# this just an ordered list now XD [0,1,2, ..., n]
batch_order = batch_order[steps_offset:]

# perfom short training run for debugging purposes
if debug:
    batch_order = batch_order[: worker_count * 10]
    save_step = 100
    average_loss_step_count = 20

training_step = 0

train_step_progress_bar = tqdm(
    total=len(batch_order), desc="Training...", position=1, leave=False
)

# loop counter
train_metric = None
sum_train_metric = 0
global_step = 0
checkpoint_counter = 0

# store training array here
batch_queue = Queue(maxsize=100)

sharded_batch_order = worker_batching(
    batch_order=batch_order, internal_queue_length=worker_count
)

count = Value("i", 0)
lock = Lock()

# spawn another process for processing images
batch_workers = []
for order, batch_shard in enumerate(sharded_batch_order):
    batch_processor = Process(
        target=generate_batch_wrapper,
        args=[
            lock,
            count,
            order,
            len(sharded_batch_order),
            batch_shard,
            batch_queue,
            debug,
        ],
    )
    batch_workers.append(batch_processor)

for worker in batch_workers:
    worker.start()

# dummy storage
store_data = []
# black hole loop
for x in batch_order:
    # grab training array from queue
    start = time.time()
    current_batch = batch_queue.get()
    print(f"this main queue length is now {batch_queue.qsize()}")
    print(len(current_batch))
    # dummy storage
    store_data.append(current_batch)
    stop = time.time()
    print(f"{x} swallowed took {round(stop-start,2)}s")
    print([data[0][1] for data in store_data])
