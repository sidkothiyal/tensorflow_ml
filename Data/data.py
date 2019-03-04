import os
import numpy as np
from random import  shuffle
import tensorflow as tf
from Data.image import Image


class Data:
    def __init__(self, data_location=None, data_type="img", classification_type="folder", debug=False):
        self.debug = debug
        if data_location is None:
            print("Must specify location to pick data from")
            return

        self.location = data_location
        self.classification_type = classification_type
        self.data_type = data_type
        self.labels = []

    def to_int64_tf_feature(self, val):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[val]))

    def to_int64list_tf_feature(self,val):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=val))

    def to_byte_tf_feature(self, val):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[val]))

    def to_float_tf_feature(self, val):
        return tf.train.Feature(float_list=tf.train.FloatList(value=val))

    def store_tf_record(self, output_file, dataset, feature_name_list):
        file_writer = tf.python_io.TFRecordWriter(output_file)

        if self.data_type == "img":
            img_locs, labels = zip(*dataset)
            for i, img_loc in enumerate(img_locs):

                img = self.to_float_tf_feature(self.get_image_data(img_loc).flatten())
                label = labels[i]

                if type(label) == int:
                    label = self.to_int64_tf_feature(label)
                else:
                    label = self.to_byte_tf_feature(label)

                feature = {
                    'img': img,
                    'label': label
                }

                tf_feature = tf.train.Example(features=tf.train.Features(feature=feature))

                file_writer.write(tf_feature.SerializeToString())

        file_writer.close()

    def get_data(self):
        pass

    def write_data(self, train=0., val=0., test=0., shuffle_class_data=True, label_type="name", feature_list=None):
        train_dataset = []
        test_dataset = []
        validation_dataset = []

        if train == 0. and test == 0.:
            print("Both training and testing data cannot be empty")
            exit()

        if (train + test) > 1.:
            print("Sum of training and test data cannot be more than 1")
            exit()

        if val == 0:
            test = 1. - train

        if (test + train) < 1. :
            val = 1 - (test + train)

        if self.classification_type == "folder":
            for data in self.classified_by_folder(self.location, label_type=label_type):
                if self.debug:
                    print(data)

                if shuffle_class_data:
                    shuffle(data)
                train_dataset.extend(data[: int(train * len(data))])
                if val != 0:
                    validation_dataset.extend(data[int(train * len(data)): int((train + val) * len(data))])
                test_dataset.extend(data[int((train + val) * len(data)):])

                if self.debug:
                    print(test_dataset)

        if self.data_type == "img":
            if feature_list is None:
                feature_list = ["img", "label"]
            self.store_tf_record("train.tfrecords", train_dataset, feature_name_list=feature_list)
            if val != 0:
                self.store_tf_record("validation.tfrecords", validation_dataset, feature_name_list=feature_list)
            self.store_tf_record("test.tfrecords", test_dataset, feature_name_list=feature_list)

    def classified_by_folder(self, folder, label_type="number"):
        for class_label in os.listdir(folder):
            data = []
            label = []
            if class_label not in self.labels:
                self.labels.append(class_label)

            class_data_folder_absolute_path = folder + "/" + class_label
            for data_file in os.listdir(class_data_folder_absolute_path):
                file_loc = class_data_folder_absolute_path + "/" + data_file
                data.append(file_loc)
                if label_type == "number":
                    label.append(len(self.labels) - 1)
                else:
                    try:
                        label.append(int(class_label))
                    except:
                        label.append(class_label)

            yield list(zip(data, label))

    def classified_with_csv(self, filename_first=True):
        pass

    def classified_with_name(self):
        pass

    def divide_dataset(self, train_percent=80):
        pass

    # For pickle
    def get_pickle_data(self):
        pass

    # For images
    def get_image_data(self, path):
        image = Image(path)
        return image.get_image()

    # For speech
    def get_speech_data(self):
        pass


if __name__ == "__main__":
    mnist = Data(data_location="/home/siddharth/Desktop/datasets/mnist/trainingSet", classification_type="folder")
    mnist.write_data(train=0.8)
