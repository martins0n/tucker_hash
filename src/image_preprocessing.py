import imageio
import cv2
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import data, img_as_float, color


def preprocessing(image, size=(256,256)):
    demo_im = resize(image, size, anti_aliasing=True)
    L = color.rgb2lab(demo_im)[:,:,0]
    blur = cv2.GaussianBlur(L,(5,5),0)
    return blur
