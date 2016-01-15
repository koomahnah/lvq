import Image
import math

def load_image(name):
    im = Image.open(name)
    return list(im.getdata())

def print_image(image, size):
    for i in range(0,size):
        print image[i],
        if i == 0:
            continue
        elif (i % math.sqrt(size)) == (math.sqrt(size) - 1):
            print

def letter_from_32x8(vec):
    bit_array = [0] * 256
    for i in range(0,32):
        for j in range(0,8):
            bit_array[i*8+j] = (vec[i] >> (7 - j)) & 1
    return bit_array
