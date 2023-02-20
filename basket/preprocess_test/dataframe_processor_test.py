import pandas as pd

"""a collection of test function"""

# test suite for resolution_bucketing_batch function
def test_ensure_single_res_each_batch(
    dataframe:pd.DataFrame,
    bucket_group_col_name = "bucket_group",
    image_height_col_name="new_image_height",
    image_width_col_name="new_image_width",
) -> None:

    error_flag = False
    
    min_value = dataframe[bucket_group_col_name].min()
    max_value = dataframe[bucket_group_col_name].max()
    
    for x in range(min_value, max_value):
        width_unique_count = len(dataframe.loc[x][image_width_col_name].unique())
        height_unique_count = len(dataframe.loc[x][image_height_col_name].unique())

        if width_unique_count > 1 or height_unique_count > 1:
            print(
                "FOUND BATCH WITH MULTIPLE RES AT", 
                x, 
                dataframe.loc[x][image_width_col_name].unique(), 
                dataframe.loc[x][image_height_col_name].unique())
            error_flag = True

    if not error_flag:
        print("PASS")
    else:
        print("FAILED")
    pass
