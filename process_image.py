from src.FHT import ImageProcessor, NearestNeighborInterpolation, BilinearInterpolation
import sys
import os
from PIL import Image

if sys.argv[1] == "all":
    for name in os.listdir("./images"):
        print(name)
        image_name = "./images/" + name
        im_proc = ImageProcessor(image_name)
        result = im_proc.RotateCurreтеImage(NearestNeighborInterpolation)
        Image.fromarray(result).save("./results/" + name.split('.')[0] + "_nearest_neighbour.jpg")
        result = im_proc.RotateCurreтеImage(BilinearInterpolation)
        Image.fromarray(result).save("./results/" + name.split('.')[0] + "_bilinear.jpg")
else:
    image_name = sys.argv[1].split("/")[-1]
    im_proc = ImageProcessor(sys.argv[1])
    result = im_proc.RotateCurreтеImage(NearestNeighborInterpolation)
    Image.fromarray(result).save("./results/" + image_name.split('.')[0] + "_nearest_neighbour.jpg")
    result = im_proc.RotateCurreтеImage(BilinearInterpolation)
    Image.fromarray(result).save("./results/" + image_name.split('.')[0] + "_bilinear.jpg")
