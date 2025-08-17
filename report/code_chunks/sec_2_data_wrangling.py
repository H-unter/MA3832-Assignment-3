
with open("token.txt", "r") as file:
    hugging_face_token = file.read().strip() # must have credentials to access the dataset

raw_dataset = load_dataset('competitions/aiornot')
raw_dataset = raw_dataset['train'] # remove the unused 'test' set with no labels
dataset_length = len(raw_dataset)

# first holdout 500 images for final model evaluation
holdout_test = raw_dataset.select(range(500))
dataset = raw_dataset.select(range(500, len(raw_dataset)))

# split dataset into:
# - small_train (10% of original)
# - train (70% of original)
# - test (20% of original)

# first split: small_train (10%) and remainder (90%)
split_1 = dataset.train_test_split(train_size=0.1, seed=RANDOM_SEED)
small_train = split_1["train"]
remainder = split_1["test"]

# second split: train (70%) and test (20%)
split_2 = remainder.train_test_split(train_size=0.7 / 0.9, seed=RANDOM_SEED)
train = split_2["train"]
test = split_2["test"]

is_any_data_unused = not(holdout_test.num_rows + small_train.num_rows + train.num_rows + test.num_rows == dataset.num_rows)
assert not is_any_data_unused, "Some data is unused in the splits."

def to_numpy(dataset):
    images = np.stack([np.asarray(img) for img in dataset["image"]])
    labels = np.array(dataset["label"])
    return images, labels

x_small, y_small = to_numpy(small_train)
x_train, y_train = to_numpy(train)
x_test, y_test = to_numpy(test)
x_holdout, y_holdout = to_numpy(holdout_test)

sampler = RandomUnderSampler(random_state=RANDOM_SEED)

def resample_images_and_labels(images, labels, sampler):
    flat_images = images.reshape((images.shape[0], -1))
    resampled_flat, resampled_labels = sampler.fit_resample(flat_images, labels)
    resampled_images = resampled_flat.reshape((-1,) + images.shape[1:])
    return resampled_images, resampled_labels

x_small_resampled, y_small_resampled = resample_images_and_labels(x_small, y_small, sampler)
x_train_resampled, y_train_resampled = resample_images_and_labels(x_train, y_train, sampler)
x_test_resampled, y_test_resampled = resample_images_and_labels(x_test, y_test, sampler)

# save to npz file and upload to s3 bucket