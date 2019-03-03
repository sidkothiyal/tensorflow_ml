import os
import numpy as np
from Data.image import Image


class Data:
    def __init__(self, data_location=None, train=None, test=None, data_type="img", classification_type="folder", debug=False):
        self.debug = debug
        if data_location is None:
            print("Must specify location to pick data from")
            return

        self.location = data_location
        self.classification_type = classification_type
        self.data_type = data_type
        self.labels = []

        if train is not None and test is not None:
            self.train_folder = train
            self.test_folder = test
            self.divided = True

    def get_data(self, label_type="one_hot"):
        if self.classification_type == "folder":
            if self.divided:
                train_folder_absolute_path = self.location + "/" + self.train_folder
                train_data, train_label_data = self.classified_by_folder(train_folder_absolute_path, label_type=label_type)
                if label_type == "one_hot":
                    one_hot_labels = np.zeros((len(train_data), len(self.labels)))
                    for i, label_loc in enumerate(train_label_data):
                        one_hot_labels[i, label_loc] = 1
                        if self.debug:
                            if i < 5:
                                print(one_hot_labels[i])
                    train_label_data = one_hot_labels

                if self.debug:
                    print(train_data.shape, train_label_data.shape)

                test_folder_absolute_path = self.location + "/" + self.test_folder
                test_data, _ = self.classified_by_folder(test_folder_absolute_path, use_foldername_as_class_label=False)
                if self.debug:
                    print(test_data.shape)

                return self.labels, train_data, train_label_data, test_data

    def classified_by_folder(self, folder, use_foldername_as_class_label=True, label_type="one_hot"):
        data = []
        label_data = []

        if use_foldername_as_class_label:
            for class_label in os.listdir(folder):
                if class_label not in self.labels:
                    self.labels.append(class_label)

                class_data_folder_absolute_path = folder + "/" + class_label
                for data_file in os.listdir(class_data_folder_absolute_path):
                    if self.data_type == "img":
                        data.append(self.get_image_data(class_data_folder_absolute_path + "/" + data_file))

                    if label_type == "one_hot" or label_type == "position":
                        label_data.append(len(class_label) - 1)
                    else:
                        label_data.append(class_label)

        else:
            for data_file in os.listdir(folder):
                if self.data_type == "img":
                    data.append(self.get_image_data(folder + "/" + data_file))

        data = np.array(data)
        if len(label_data) != 0:
            label_data = np.array(label_data)

        return data, label_data

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
    mnist = Data(data_location="/home/siddharth/Desktop/datasets/mnist/", train="trainingSet", test="testSet", classification_type="folder")
    mnist.get_data()
