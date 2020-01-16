import pytest
from PIL import Image
from thash.tucker_hash import tucker_hash
from itertools import combinations


def test_image_house():
    assert str(tucker_hash(Image.open('./data/Pasadena-Houses/calhouse_0081.jpg'))) == '4f9f00007fff03fff000' 


def test_initial_state_ind():
    img = Image.open('./data/Pasadena-Houses/calhouse_0081.jpg')
    arr_img = (tucker_hash(img) for _ in range(200))
    arr_img = map(lambda x: abs(x[0] - x[1]), combinations(arr_img, 2))
    assert sum(arr_img) == 0
