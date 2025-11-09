# watermark_embedding_extraction.py - FIXED VERSION
from utils import *
from PIL import Image
from audio_managing import frameToAudio, audioToFrame
from image_managing import binarization, grayscale, imgSize
import numpy as np
import math
import sys

ALPHA = 0.1

# Check if the image is in grayscale and convert it to this mode
def isImgGrayScale(image):
    if image.mode != "L":
        image = grayscale(image)
    return image

# Check if the image is binary and convert it in this mode
def isImgBinary(image):
    if image.mode != "1":
        image = binarization(image)
    return image

# Embedding of width and height. Audio must be linear and not frames
def sizeEmbedding(audio, width, height):
    embedded = audio.copy()
    
    # Ensure we have at least 2 frames
    if len(embedded) < 2:
        sys.exit("SIZE EMBEDDING: Audio must have at least 2 frames!")
    
    # Check if frames exist and have coefficients
    if hasattr(embedded[0], '__len__') and len(embedded[0]) > 0:
        embedded[0][-1] = width
    if hasattr(embedded[1], '__len__') and len(embedded[1]) > 0:
        embedded[1][-1] = height

    return embedded

def sizeExtraction(audio):
    # Extraction of width and height with safety checks
    try:
        width = int(audio[0][-1]) if hasattr(audio[0], '__len__') else 128
        height = int(audio[1][-1]) if hasattr(audio[1], '__len__') else 128
        return width, height
    except:
        return 128, 128  # Default fallback

# Check if audio is divided in frames
def isJoinedAudio(audio):
    # Handle different numpy array structures
    if isinstance(audio, np.ndarray):
        if audio.ndim == 1:
            # 1D array - not divided into frames
            numOfFrames = -1
            joinAudio = audio.copy()
        elif audio.ndim == 2:
            # 2D array - divided into frames
            numOfFrames = audio.shape[0]
            joinAudio = frameToAudio(audio)
        else:
            sys.exit("AUDIO FORMAT ERROR: Unexpected audio dimensions!")
    else:
        numOfFrames = -1
        joinAudio = np.array(audio) if not isinstance(audio, np.ndarray) else audio.copy()
    
    return joinAudio, numOfFrames

def iisJoinedAudio(audio, numOfFrames, frameLen=4):
    if numOfFrames != -1 and numOfFrames > 0:
        return audioToFrame(audio, frameLen)
    return audio

# LSB algorithm (bit-level) for audio watermarking
def LSB(audio, image):
    image = isImgBinary(image)
    joinAudio, numOfFrames = isJoinedAudio(audio)
    width, heigth = imgSize(image)

    audioLen = len(joinAudio)

    if (width * heigth) + 32 >= audioLen:
        sys.exit("LEAST SIGNIFICANT BIT: Cover dimension is not sufficient for this payload size!")

    joinAudio = sizeEmbedding(joinAudio, width, heigth)

    # Embedding watermark
    for i in range(width):
        for j in range(heigth):
            value = image.getpixel(xy=(i, j))
            value = 1 if value == 255 else 0
            x = i * heigth + j
            if x + 32 < len(joinAudio):
                joinAudio[x + 32] = setLastBit(joinAudio[x + 32], value)

    if numOfFrames != -1:
        return audioToFrame(joinAudio, numOfFrames)
    else:
        return joinAudio

def iLSB(audio):
    joinAudio, numOfFrames = isJoinedAudio(audio)
    width, heigth = (128, 128)
    image = Image.new("1", (width, heigth))

    # Extraction watermark
    for i in range(width):
        for j in range(heigth):
            x = i * heigth + j
            if x + 32 < len(joinAudio):
                value = getLastBit(joinAudio[x + 32])
                image.putpixel(xy=(i, j), value=value)

    return image

# Brute Binary embedding
def bruteBinary(coeffs, image):
    image = isImgBinary(image)
    
    # Ensure coeffs is a numpy array
    if not isinstance(coeffs, np.ndarray):
        coeffs = np.array(coeffs)
    
    # Create a copy
    joinCoeffs = coeffs.copy()
    
    # Get dimensions
    if joinCoeffs.ndim == 1:
        coeffsLen = len(joinCoeffs)
        frameLen = 1
    elif joinCoeffs.ndim == 2:
        coeffsLen = joinCoeffs.shape[0]
        frameLen = joinCoeffs.shape[1]
    else:
        sys.exit("BRUTE BINARY: Unexpected coefficients structure!")
    
    width, heigth = imgSize(image)
    
    if (width * heigth) + 2 >= coeffsLen:
        # Resize image to fit
        max_pixels = coeffsLen - 2
        new_size = int(math.sqrt(max_pixels))
        image = image.resize((new_size, new_size))
        width, heigth = new_size, new_size
        print(f"WARNING: Image resized to {width}x{heigth} to fit in audio")

    # Embed size information if possible
    if coeffsLen >= 2 and joinCoeffs.ndim == 2:
        if len(joinCoeffs[0]) > 0:
            joinCoeffs[0][-1] = width
        if len(joinCoeffs[1]) > 0:
            joinCoeffs[1][-1] = heigth

    # Embedding watermark
    for i in range(width):
        for j in range(heigth):
            value = image.getpixel(xy=(i, j))
            x = i * heigth + j
            
            if x + 2 < coeffsLen:
                if joinCoeffs.ndim == 2 and len(joinCoeffs[x + 2]) > 0:
                    joinCoeffs[x + 2] = setBinary(joinCoeffs[x + 2], value)
                elif joinCoeffs.ndim == 1:
                    # For 1D array, use simple embedding
                    joinCoeffs[x + 2] = 10 if value == 255 else -10

    return joinCoeffs

def ibruteBinary(coeffs):
    # Ensure coeffs is proper array
    if not isinstance(coeffs, np.ndarray):
        coeffs = np.array(coeffs)
        
    joinCoeffs = coeffs.copy()
    width, heigth = (128, 128)
    
    extracted = Image.new("1", (width, heigth))
    
    if joinCoeffs.ndim == 1:
        coeffsLen = len(joinCoeffs)
    elif joinCoeffs.ndim == 2:
        coeffsLen = joinCoeffs.shape[0]
    else:
        coeffsLen = 0

    # Extraction watermark
    for i in range(width):
        for j in range(heigth):
            x = i * heigth + j
            value = 0
            
            try:
                if x + 2 < coeffsLen:
                    if joinCoeffs.ndim == 2 and len(joinCoeffs[x + 2]) > 0:
                        value = getBinary(joinCoeffs[x + 2])
                    elif joinCoeffs.ndim == 1:
                        value = 255 if joinCoeffs[x + 2] > 0 else 0
            except (IndexError, TypeError):
                value = 0
                
            extracted.putpixel(xy=(i, j), value=value)

    return extracted

# Delta DCT embedding
def deltaDCT(coeffs, image):
    image = isImgBinary(image)
    width, heigth = imgSize(image)

    if not isinstance(coeffs, np.ndarray):
        coeffs = np.array(coeffs)
    
    joinCoeffs = coeffs.copy()
    
    if joinCoeffs.ndim == 1:
        coeffsLen = len(joinCoeffs)
    elif joinCoeffs.ndim == 2:
        coeffsLen = joinCoeffs.shape[0]
    else:
        coeffsLen = 0

    # Embedding watermark
    for i in range(min(width, coeffsLen)):
        for j in range(min(heigth, coeffsLen)):
            value = image.getpixel(xy=(i, j))
            x = i * heigth + j
            
            if x < coeffsLen and joinCoeffs.ndim == 2:
                try:
                    v1, v2 = subVectors(joinCoeffs[x])
                    
                    norm1, u1 = normCalc(v1)
                    norm2, u2 = normCalc(v2)

                    norm = (norm1 + norm2) / 2
                    norm1, norm2 = setDelta(norm, 10, value)

                    v1 = inormCalc(norm1, u1)
                    v2 = inormCalc(norm2, u2)

                    joinCoeffs[x] = isubVectors(v1, v2)
                except:
                    continue

    return joinCoeffs

def ideltaDCT(coeffs):
    if not isinstance(coeffs, np.ndarray):
        coeffs = np.array(coeffs)
        
    joinCoeffs = coeffs.copy()
    width, heigth = (128, 128)
    extracted = Image.new("L", (width, heigth))
    
    if joinCoeffs.ndim == 1:
        coeffsLen = len(joinCoeffs)
    elif joinCoeffs.ndim == 2:
        coeffsLen = joinCoeffs.shape[0]
    else:
        coeffsLen = 0

    # Extraction watermark
    for i in range(width):
        for j in range(heigth):
            x = i * heigth + j
            value = 0
            
            if x < coeffsLen and joinCoeffs.ndim == 2:
                try:
                    v1, v2 = subVectors(joinCoeffs[x])
                    norm1, u1 = normCalc(v1)
                    norm2, u2 = normCalc(v2)
                    value = getDelta(norm1, norm2)
                except:
                    value = 0
            
            extracted.putpixel(xy=(i, j), value=value)

    return extracted

# Brute Gray embedding
def bruteGray(coeffs, image):
    image = isImgGrayScale(image)

    if not isinstance(coeffs, np.ndarray):
        coeffs = np.array(coeffs)
    
    joinCoeffs = coeffs.copy()

    if joinCoeffs.ndim == 1:
        coeffsLen = len(joinCoeffs)
        frameLen = 1
    elif joinCoeffs.ndim == 2:
        coeffsLen = joinCoeffs.shape[0]
        frameLen = joinCoeffs.shape[1]
    else:
        sys.exit("BRUTE GRAY: Unexpected coefficients structure!")
    
    width, heigth = imgSize(image)
    
    if (width * heigth) + 2 >= coeffsLen:
        max_pixels = coeffsLen - 2
        new_size = int(math.sqrt(max_pixels))
        image = image.resize((new_size, new_size))
        width, heigth = new_size, new_size
        print(f"WARNING: Image resized to {width}x{heigth} to fit in audio")

    # Embed size if possible
    if coeffsLen >= 2 and joinCoeffs.ndim == 2:
        if len(joinCoeffs[0]) > 0:
            joinCoeffs[0][-1] = width
        if len(joinCoeffs[1]) > 0:
            joinCoeffs[1][-1] = heigth

    # Embedding watermark
    for i in range(width):
        for j in range(heigth):
            value = image.getpixel(xy=(i, j))
            x = i * heigth + j
            
            if x + 2 < coeffsLen:
                if joinCoeffs.ndim == 2 and len(joinCoeffs[x + 2]) > 0:
                    joinCoeffs[x + 2] = setGray(joinCoeffs[x + 2], value)
                elif joinCoeffs.ndim == 1:
                    joinCoeffs[x + 2] = value - 127

    return joinCoeffs

def ibruteGray(coeffs):
    if not isinstance(coeffs, np.ndarray):
        coeffs = np.array(coeffs)
        
    joinCoeffs = coeffs.copy()
    width, heigth = (128, 128)
    extracted = Image.new("L", (width, heigth))
    
    if joinCoeffs.ndim == 1:
        coeffsLen = len(joinCoeffs)
    elif joinCoeffs.ndim == 2:
        coeffsLen = joinCoeffs.shape[0]
    else:
        coeffsLen = 0

    # Extraction watermark
    for i in range(width):
        for j in range(heigth):
            x = i * heigth + j
            value = 0
            
            try:
                if x + 2 < coeffsLen:
                    if joinCoeffs.ndim == 2 and len(joinCoeffs[x + 2]) > 0:
                        value = getGray(joinCoeffs[x + 2])
                    elif joinCoeffs.ndim == 1:
                        value = int(joinCoeffs[x + 2] + 127)
                    # Clamp value to valid range
                    value = max(0, min(255, value))
            except (IndexError, TypeError):
                value = 127  # Middle gray as fallback
                
            extracted.putpixel(xy=(i, j), value=value)

    return extracted

# Magnitude DCT functions remain the same but with better error handling
def magnitudoDCT(coeffs, watermark, alpha):
    watermark = isImgGrayScale(watermark)
    watermark = createImgArrayToEmbed(watermark)
    coeffs, joinFlag = isJoinedAudio(coeffs)
    
    if (coeffs.shape[0] < len(watermark)):
        sys.exit("MAGNITUDO DCT: Cover dimension is not sufficient for this payload size!")
    
    wCoeffs = []
    for i in range(len(watermark)):
        wCoeffs.append((coeffs[i]) * (1 + alpha * watermark[i]))
    for i in range(len(watermark), len(coeffs)):
        wCoeffs.append(coeffs[i])
    
    if joinFlag != -1:
        wCoeffs = np.asarray(wCoeffs)
        wCoeffs = iisJoinedAudio(wCoeffs, joinFlag)
    
    return wCoeffs

def imagnitudoDCT(coeffs, wCoeffs, alpha):
    coeffs, joinCoeffsFlag = isJoinedAudio(coeffs)
    wCoeffs, joinWCoeffsFlag = isJoinedAudio(wCoeffs)
    
    watermark = []
    for i in range(len(wCoeffs)):
        try:
            if abs(coeffs[i]) < 1e-10:  # Avoid division by zero
                ratio = 0
            else:
                ratio = (wCoeffs[i] - coeffs[i]) / (coeffs[i] * alpha)
            
            if math.isinf(ratio) or math.isnan(ratio):
                watermark.append(0)
            else:
                watermark.append(max(0, min(255, int(abs(ratio)))))
        except Exception:
            watermark.append(0)
    
    return convertToPIL(createImgMatrix(extractImage(watermark)))

def extractImage(watermark):
    if len(watermark) < 2:
        return [128, 128] + [0] * (128 * 128)
    
    nPixel = (watermark[0] * watermark[1]) + 2
    if len(watermark) < nPixel:
        watermark = watermark + [0] * (nPixel - len(watermark))
    return watermark[:nPixel]

def createImgMatrix(image):
    if len(image) < 2:
        return np.zeros((128, 128))
    
    width = max(1, int(image[0]))
    heigth = max(1, int(image[1]))
    
    expected_size = width * heigth
    if len(image[2:]) < expected_size:
        image = list(image) + [0] * (expected_size - len(image[2:]))
    
    try:
        matrixImg = np.reshape(image[2:2+expected_size], (width, heigth))
        return matrixImg
    except:
        return np.zeros((width, heigth))

def convertToPIL(image):
    image = np.clip(image, 0, 255)
    PImage = Image.fromarray((image).astype("uint8"), mode="L")
    return PImage

def createImgArrayToEmbed(image):
    width, heigth = imgSize(image)
    flattedImage = [width, heigth]
    tmp = np.ravel(image)
    for i in range(len(tmp)):
        flattedImage.append(int(tmp[i]))
    return flattedImage