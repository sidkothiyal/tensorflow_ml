from PIL import Image as pil
import numpy as np

class Image:
    def __init__(self, location, label=None):
        self.label = label
        self.location = location

    def get_image(self, return_type="np"):
        if return_type == "np":
            img = pil.open(self.location)
            pix = np.array(img)
            return pix

if __name__ == "__main__":
    image = Image(location="/home/siddharth/Desktop/datasets/mnist/trainingSet/0/img_1.jpg")
    print(image.get_image().shape)
