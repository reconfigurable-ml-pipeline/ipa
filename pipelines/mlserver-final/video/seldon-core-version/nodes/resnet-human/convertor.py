import numpy as np
import os
from PIL import Image
import pathlib

path = pathlib.Path(__file__).parent.resolve()
iamge_name = "input-sample.npy"
file_path = os.path.join(path, iamge_name)

array = np.load(file_path)
im = Image.fromarray(array)
im.save("input-sample.JPEG")
