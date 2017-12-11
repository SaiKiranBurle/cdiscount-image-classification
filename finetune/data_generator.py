import os, io
import io
import os
import struct
import threading
from collections import defaultdict

import bson
import numpy as np
import pandas as pd
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import load_img, img_to_array
from tqdm import *

lock = threading.Lock()


DATA_DIR = '/Users/sai/dev/datasets/cdiscount-kaggle/'

# TRAIN_PATH = os.path.join(DATA_DIR, 'train.bson')
# NUM_TRAIN_SAMPLES = 7069896

TRAIN_PATH = os.path.join(DATA_DIR, 'train_example.bson')
NUM_TRAIN_SAMPLES = 82

NUM_CLASSES = 5270

TEST_PATH = os.path.join(DATA_DIR, 'test.bson')
NUM_TEST_SAMPLES = 1768182

CATEGORIES_PATH = os.path.join(DATA_DIR, "category_names.csv")

BATCH_SIZE = 128


class BSONIterator(Iterator):
    def __init__(self, bson_file, images_df, offsets_df, num_class,
                 image_data_generator, lock, target_size=(180, 180),
                 with_labels=True, batch_size=32, shuffle=False, seed=None):

        self.file = bson_file
        self.images_df = images_df
        self.offsets_df = offsets_df
        self.with_labels = with_labels
        self.samples = len(images_df)
        self.num_class = num_class
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.image_shape = self.target_size + (3,)

        print("Found %d images belonging to %d classes." % (self.samples, self.num_class))

        super(BSONIterator, self).__init__(self.samples, batch_size, shuffle, seed)
        self.lock = lock

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        if self.with_labels:
            batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())

        for i, j in enumerate(index_array):
            # Protect file and dataframe access with a lock.
            with self.lock:
                image_row = self.images_df.iloc[j]
                product_id = image_row["product_id"]
                offset_row = self.offsets_df.loc[product_id]

                # Read this product's data from the BSON file.
                self.file.seek(offset_row["offset"])
                item_data = self.file.read(offset_row["length"])

            # Grab the image from the product.
            item = bson.BSON(item_data).decode()
            img_idx = image_row["img_idx"]
            bson_img = item["imgs"][img_idx]["picture"]

            # Load the image.
            img = load_img(io.BytesIO(bson_img), target_size=self.target_size)

            # Preprocess the image.
            x = img_to_array(img)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)

            # Add the image and the label to the batch (one-hot encoded).
            batch_x[i] = x
            if self.with_labels:
                batch_y[i, image_row["category_idx"]] = 1

        if self.with_labels:
            return batch_x, batch_y
        else:
            return batch_x

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)


def make_category_tables():

    categories_df = pd.read_csv(CATEGORIES_PATH, index_col="category_id")

    # Maps the category_id to an integer index. This is what we'll use to
    # one-hot encode the labels.
    categories_df["category_idx"] = pd.Series(range(len(categories_df)), index=categories_df.index)
    categories_df.to_csv("categories_idx.csv")

    cat2idx = {}
    idx2cat = {}
    for ir in categories_df.itertuples():
        category_id = ir[0]
        category_idx = ir[4]
        cat2idx[category_id] = category_idx
        idx2cat[category_idx] = category_id
    return cat2idx, idx2cat


def gen_offsets(bson_path, num_records, with_categories):
    rows = {}
    with open(bson_path, "rb") as f, tqdm(total=num_records) as pbar:
        offset = 0
        while True:
            item_length_bytes = f.read(4)
            if len(item_length_bytes) == 0:
                break

            length = struct.unpack("<i", item_length_bytes)[0]

            f.seek(offset)
            item_data = f.read(length)
            assert len(item_data) == length

            item = bson.BSON(item_data).decode()
            product_id = item["_id"]
            num_imgs = len(item["imgs"])

            row = [num_imgs, offset, length]
            if with_categories:
                row += [item["category_id"]]
            rows[product_id] = row

            offset += length
            f.seek(offset)
            pbar.update()

    columns = ["num_imgs", "offset", "length"]
    if with_categories:
        columns += ["category_id"]

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "product_id"
    df.columns = columns
    df.sort_index(inplace=True)
    return df


def make_val_set(df, cat2idx, split_percentage, drop_percentage):
    # Find the product_ids for each category.
    category_dict = defaultdict(list)
    for ir in tqdm(df.itertuples()):
        category_dict[ir[4]].append(ir[0])

    train_list = []
    val_list = []
    with tqdm(total=len(df)) as pbar:
        for category_id, product_ids in category_dict.items():
            category_idx = cat2idx[category_id]

            # Randomly remove products to make the dataset smaller.
            keep_size = int(len(product_ids) * (1. - drop_percentage))
            if keep_size < len(product_ids):
                product_ids = np.random.choice(product_ids, keep_size, replace=False)

            # Randomly choose the products that become part of the validation set.
            val_size = int(len(product_ids) * split_percentage)
            if val_size > 0:
                val_ids = np.random.choice(product_ids, val_size, replace=False)
            else:
                val_ids = []

            # Create a new row for each image.
            for product_id in product_ids:
                row = [product_id, category_idx]
                for img_idx in range(df.loc[product_id, "num_imgs"]):
                    if product_id in val_ids:
                        val_list.append(row + [img_idx])
                    else:
                        train_list.append(row + [img_idx])
                pbar.update()

    columns = ["product_id", "category_idx", "img_idx"]
    train_df = pd.DataFrame(train_list, columns=columns)
    val_df = pd.DataFrame(val_list, columns=columns)
    return train_df, val_df


def run_once():
    cat2idx, idx2cat = make_category_tables()
    train_offsets_df = gen_offsets(TRAIN_PATH, NUM_TRAIN_SAMPLES, True)
    train_offsets_df.to_csv("train_offsets.csv")
    train_images_df, val_images_df = make_val_set(train_offsets_df, cat2idx, split_percentage=0.2,
                                                  drop_percentage=0.)
    print("Number of training images:", len(train_images_df))
    print("Number of validation images:", len(val_images_df))
    print("Total images:", len(train_images_df) + len(val_images_df))
    train_images_df.to_csv("train_images.csv")
    val_images_df.to_csv("val_images.csv")


def get_generator():
    # categories_df = pd.read_csv("categories_idx.csv", index_col=0)

    train_offsets_df = pd.read_csv("train_offsets.csv", index_col=0)
    train_images_df = pd.read_csv("train_images.csv", index_col=0)
    val_images_df = pd.read_csv("val_images.csv", index_col=0)

    train_bson_file = open(TRAIN_PATH, "rb")
    # num_train_images = len(train_images_df)
    # num_val_images = len(val_images_df)

    # Tip: use ImageDataGenerator for data augmentation and preprocessing.
    train_datagen = ImageDataGenerator()
    train_gen = BSONIterator(train_bson_file, train_images_df, train_offsets_df,
                             NUM_CLASSES, train_datagen, lock,
                             batch_size=BATCH_SIZE, shuffle=True)

    val_datagen = ImageDataGenerator()
    val_gen = BSONIterator(train_bson_file, val_images_df, train_offsets_df,
                           NUM_CLASSES, val_datagen, lock,
                           batch_size=BATCH_SIZE, shuffle=True)
    return train_gen, val_gen


if __name__ == "__main__":
    # run_once()
    train_gen, val_gen = get_generator()
    from IPython import embed
    embed()
