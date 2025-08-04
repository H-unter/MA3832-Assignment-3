
with open("token.txt", "r") as file:
    hugging_face_token = file.read().strip() # must have credentials to access the dataset

login(token=hugging_face_token)
raw_dataset = load_dataset('competitions/aiornot') # hugging face method to import the data

dataset = raw_dataset['train'] # remove the unused 'test' set with no labels

# split dataset into:
# - small_train (10% of original)
# - train (70% of original)
# - test (20% of original)

# First split: small_train (10%) and remainder (90%)
split_1 = dataset.train_test_split(train_size=0.1, seed=RANDOM_SEED)
small_train = split_1["train"]
remainder = split_1["test"]
split_2 = remainder.train_test_split(train_size=0.7 / 0.9, seed=RANDOM_SEED)
train = split_2["train"]
test = split_2["test"]

is_any_data_unused = not(small_train.num_rows + train.num_rows + test.num_rows == dataset.num_rows)
assert not is_any_data_unused, "Some data is unused in the splits."

def format_dataset(hugging_face_dataset, batch_size=32, shuffle=True):
    """convert a hugging face dataset to a TensorFlow dataset"""
    dataset = hugging_face_dataset.with_format(type='tf', columns=['image', 'label'], output_all_columns=True)
    dataset = dataset.to_tf_dataset(columns='image', label_cols='label', batch_size=batch_size, shuffle=shuffle)

    # normalse the image channels to be in the range [0, 1] rather than [0, 255]
    dataset = dataset.map(lambda image, label: (tf.cast(image, tf.float32) / 255.0, label), num_parallel_calls=tf.data.AUTOTUNE)
    # return prefetched dataset
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

small_train_tf = format_dataset(small_train, batch_size=32, shuffle=True)
train_tf = format_dataset(train, batch_size=32, shuffle=True)
test_tf = format_dataset(test, batch_size=32, shuffle=False)