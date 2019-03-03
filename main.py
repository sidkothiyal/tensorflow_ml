from Data.data import Data


if __name__ == "__main__":
    mnist = Data(data_location="/home/siddharth/Desktop/datasets/mnist/", train="trainingSet", test="testSet", classification_type="folder", debug=True)
    labels, train_data, train_label_data, test_data = mnist.get_data()
