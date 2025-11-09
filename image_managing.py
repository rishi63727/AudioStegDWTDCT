from PIL import Image
import numpy as np
import math
from utils import imodule, coprime
import sys

#Return numpy array from a Image file
def loadImage(path=""):
    if path == "":
        sys.exit("LOAD IMAGE: Path must not be None!")
    img = Image.open(path)
    return img

#Show image from array or path
def showImage(img):
    if type(img) is not str:
        img.show()
    else:
        if img == "":
            sys.exit("SHOW IMAGE: Path must not be None!")
        Image.open(img).show()

#Save image from array or Image file
def saveImage(img, path):
    if path == "":
        sys.exit("SAVE IMAGE: Path must not be None!")
    img.save(path)

def grayscale(img):
    return img.convert(mode="L")

#Return binarized image
def binarization(img):
    return img.convert(mode="1", dither=0)

#Return image size
def imgSize(img):
    if type(img) is np.ndarray:
        width, height = (img.shape[1], img.shape[0])
    else:
        width, height = img.size
    return width, height

#Arnold transform
def arnoldTransform(img, iteration):
    width, height = imgSize(img)
    if width != height:
        # Make it square by cropping to smaller dimension
        min_size = min(width, height)
        img = img.crop((0, 0, min_size, min_size))
        width = height = min_size
        print(f"  Arnold: Cropped to square {width}x{height}")
    
    side = width
    toTransform = img.copy()
    transformed = img.copy()
    
    for iter in range(iteration):
        for i in range(side):
            for j in range(side):
                newX = (i + j) % side
                newY = (i + 2*j) % side
                value = toTransform.getpixel(xy=(i, j))
                transformed.putpixel(xy=(newX, newY), value=value)
        toTransform = transformed.copy()

    return transformed

#Inverse Arnold transform
def iarnoldTransform(img, iteration):
    width, height = imgSize(img)
    if width != height:
        min_size = min(width, height)
        img = img.crop((0, 0, min_size, min_size))
        width = height = min_size
    
    side = width
    transformed = img.copy()
    toTransform = img.copy()
    
    for iter in range(iteration):
        for i in range(side):
            for j in range(side):
                newX = (2*i - j) % side
                newY = (-i + j) % side
                value = toTransform.getpixel(xy=(i, j))
                transformed.putpixel(xy=(newX, newY), value=value)
        toTransform = transformed.copy()
    
    return transformed

#2D lower triangular mapping - OPTIMIZED VERSION
def lowerTriangularMappingTransform(img, iteration, c, a=-1, d=-1):
    width, height = imgSize(img)
    coprime_mode = "first"
    
    # Get coprime values
    if a == -1 and d == -1:
        a = coprime(width, coprime_mode)
        d = coprime(height, coprime_mode)
    
    print(f"  Lower triangular: width={width}, height={height}, a={a}, c={c}, d={d}, iterations={iteration}")
    
    # Convert to numpy array for faster processing
    img_array = np.array(img)
    
    for iter in range(iteration):
        new_array = np.zeros_like(img_array)
        
        for i in range(width):
            for j in range(height):
                newX = (a * i) % width
                newY = (c * i + d * j) % height
                new_array[newY, newX] = img_array[j, i]
        
        img_array = new_array.copy()
    
    # Convert back to PIL Image
    return Image.fromarray(img_array)

#2D inverse lower triangular mapping - OPTIMIZED VERSION
def ilowerTriangularMappingTransform(img, iteration, c, a=-1, d=-1):
    width, height = imgSize(img)
    coprime_mode = "first"
    
    if a == -1 and d == -1:
        a = coprime(width, coprime_mode)
        d = coprime(height, coprime_mode)
    
    print(f"  Inverse lower triangular: width={width}, height={height}")
    
    # Convert to numpy array
    img_array = np.array(img)
    
    ia = imodule(a, width)
    id = imodule(d, height)
    
    for iter in range(iteration):
        new_array = np.zeros_like(img_array)
        
        for i in range(width):
            for j in range(height):
                newX = (ia * i) % width
                newY = (id * (j + (math.ceil(c * width / height) * height) - (c * newX))) % height
                new_array[newY, newX] = img_array[j, i]
        
        img_array = new_array.copy()
    
    return Image.fromarray(img_array)

#2D upper triangular mapping - OPTIMIZED VERSION
def upperTriangularMappingTransform(img, iteration, c, a=-1, d=-1):
    width, height = imgSize(img)
    coprime_mode = "first"
    
    if a == -1 and d == -1:
        a = coprime(width, coprime_mode)
        d = coprime(height, coprime_mode)
    
    print(f"  Upper triangular: width={width}, height={height}, a={a}, c={c}, d={d}")
    
    # Convert to numpy array
    img_array = np.array(img)
    
    for iter in range(iteration):
        new_array = np.zeros_like(img_array)
        
        for i in range(width):
            for j in range(height):
                newX = (a * i + c * j) % width
                newY = (d * j) % height
                new_array[newY, newX] = img_array[j, i]
        
        img_array = new_array.copy()
    
    return Image.fromarray(img_array)

#2D inverse upper triangular mapping - OPTIMIZED VERSION
def iupperTriangularMappingTransform(img, iteration, c, a=-1, d=-1):
    width, height = imgSize(img)
    coprime_mode = "first"
    
    if a == -1 and d == -1:
        a = coprime(width, coprime_mode)
        d = coprime(height, coprime_mode)
    
    print(f"  Inverse upper triangular: width={width}, height={height}")
    
    # Convert to numpy array
    img_array = np.array(img)
    
    ia = imodule(a, width)
    id = imodule(d, height)
    
    for iter in range(iteration):
        new_array = np.zeros_like(img_array)
        
        for i in range(width):
            for j in range(height):
                newY = (id * j) % height
                newX = (ia * (i + (math.ceil(c * height / width) * width) - (c * newY))) % width
                new_array[newY, newX] = img_array[j, i]
        
        img_array = new_array.copy()
    
    return Image.fromarray(img_array)

def mappingTransform(mode, img, iteration, c, a=-1, d=-1):
    if mode == "lower":
        return lowerTriangularMappingTransform(img, iteration, c, a, d)
    elif mode == "upper":
        return upperTriangularMappingTransform(img, iteration, c, a, d)
    else:
        sys.exit("MAPPING TRANSFORM: Mode must be lower or upper!")

def imappingTransform(mode, img, iteration, c, a=-1, d=-1):
    if mode == "lower":
        return ilowerTriangularMappingTransform(img, iteration, c, a, d)
    elif mode == "upper":
        return iupperTriangularMappingTransform(img, iteration, c, a, d)
    else:
        sys.exit("MAPPING TRANSFORM: Mode must be lower or upper!")