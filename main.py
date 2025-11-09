import numpy as np
import argparse
import os
import sys
import traceback
from scipy.io import wavfile
import image_managing as im
import audio_managing as am
import watermark_embedding_extraction as watermark
from utils import makeFileName, ImageToFlattedArray, fixSizeImg
import metrics as m
import attacks as a

#audio
T_AUDIO_PATH = 0
T_SAMPLERATE = 1
LEN_FRAMES = 4

#DWT
WAVELETS_LEVEL = 1
WAVELET_TYPE = "db1"
WAVELET_MODE = "symmetric"

#scrambling
SCRAMBLING_TECHNIQUES = ["arnold", "lower", "upper"]
BINARY = 0
GRAYSCALE = 1
NO_ITERATIONS = 1
TRIANGULAR_PARAMETERS = [5, 3, 1] #c,a,d

#embedding
ALPHA = 0.1

#attack
CUTOFF_FREQUENCY = 22050

def getAudio(path):
    print(f"  Reading audio from: {path}")
    tupleAudio = am.readWavFile(path)
    audioData = am.audioData(tupleAudio)
    print(f"  Audio shape: {audioData.shape}, Is mono: {am.isMono(audioData)}")
    
    if am.isMono(audioData) == False:
        print("  Converting to mono...")
        tupleAudio = am.joinAudioChannels(path)
        audioData = am.audioData(tupleAudio)
        print(f"  Converted to mono, shape: {audioData.shape}")
    
    return audioData, tupleAudio

def getDWT(audioData, type, mode):
    print(f"  Applying DWT (type: {type}, mode: {mode})...")
    waveletsFamilies = am.getWaveletsFamilies()
    DWTFamilies = am.filterWaveletsFamilies(waveletsFamilies)
    waveletsModes = am.getWaveletsModes()
    coeffs = am.DWT(audioData, DWTFamilies[DWTFamilies.index(type)], waveletsModes[waveletsModes.index(mode)], WAVELETS_LEVEL)
    print(f"  DWT completed, coeffs length: {len(coeffs)}")
    return coeffs

def getScrambling(path, type, mode = BINARY):
    print(f"  Loading and scrambling image: {path}")
    print(f"  Scrambling type: {type}, mode: {mode}")
    
    image = im.loadImage(path)
    print(f"  Original image size: {im.imgSize(image)}, mode: {image.mode}")
    
    if mode == BINARY:
        image = im.binarization(image)
        print(f"  Converted to binary")
    else:
        image = im.grayscale(image)
        print(f"  Converted to grayscale")
    
    # Resize image to a reasonable size for embedding
    width, height = im.imgSize(image)
    max_size = 128  # Limit watermark size
    if width > max_size or height > max_size:
        # Make it square and resize
        new_size = min(max_size, min(width, height))
        image = image.resize((new_size, new_size))
        print(f"  Resized image to: {im.imgSize(image)}")
    
    if type == "arnold":
        print("  Applying Arnold transform...")
        image = im.arnoldTransform(image, NO_ITERATIONS)
    elif type == "lower" or type == "upper":
        print(f"  Applying {type} triangular mapping...")
        image = im.mappingTransform(type, image, NO_ITERATIONS, TRIANGULAR_PARAMETERS[0], TRIANGULAR_PARAMETERS[1], TRIANGULAR_PARAMETERS[2])
    
    print(f"  Final scrambled image size: {im.imgSize(image)}")
    return image

def getiScrambling(payload, type):
    print(f"  Inverse scrambling (type: {type})...")
    if type == "arnold":
        image = im.iarnoldTransform(payload, NO_ITERATIONS)
    elif type == "lower" or type == "upper":
        image = im.imappingTransform(type, payload, NO_ITERATIONS, TRIANGULAR_PARAMETERS[0], TRIANGULAR_PARAMETERS[1], TRIANGULAR_PARAMETERS[2])
    return image

def getStego(data, tupleAudio, outputAudioPath):
    print(f"  Saving stego audio to: {outputAudioPath}")
    nData = am.normalizeForWav(data)
    
    # Check if outputAudioPath is a full path or needs prefix handling
    if os.path.isabs(outputAudioPath) or os.path.dirname(outputAudioPath):
        # It's a full path or relative path with directory
        # Just save directly without using makeFileName
        directory = os.path.dirname(outputAudioPath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        wavfile.write(outputAudioPath, tupleAudio[T_SAMPLERATE], nData)
        print(f"  Stego audio saved successfully to: {outputAudioPath}")
    else:
        # It's just a prefix, use the old method
        am.saveWavFile(tupleAudio[T_AUDIO_PATH], tupleAudio[T_SAMPLERATE], nData, outputAudioPath)
        print(f"  Stego audio saved successfully")

def getPayload(image, outputImagePath):
    print(f"  Saving payload image to: {outputImagePath}")
    im.saveImage(image, outputImagePath)
    print(f"  Payload image saved successfully")
    
def embedding(audioPath, imagePath, outputAudioPath, scramblingMode, imageMode, embeddingMode, frames = 1):
    try:
        print("\n--- EMBEDDING PROCESS ---")
        
        #1 load audio file
        print("Step 1: Loading audio...")
        audioData, tupleAudio = getAudio(audioPath)
        print(f"Audio data shape: {audioData.shape}, dtype: {audioData.dtype}")

        #2 run DWT on audio file
        print("\nStep 2: Applying DWT...")
        DWTCoeffs = getDWT(audioData, WAVELET_TYPE, WAVELET_MODE)
        cA, cD1 = DWTCoeffs #level 1
        print(f"DWT coefficients - cA shape: {cA.shape}, cD1 shape: {cD1.shape}")

        #3 divide by frame & #4 run DCT on DWT coeffs
        print("\nStep 3-4: Framing and DCT...")
        if frames != 0:
            print(f"  Dividing into frames of length {LEN_FRAMES}")
            cA = am.audioToFrame(cA, LEN_FRAMES)
            print(f"  Number of frames: {cA.shape[0]}")
            DCTCoeffs = np.copy(cA)

            for i in range(cA.shape[0]):
                DCTCoeffs[i] = am.DCT(cA[i])
            print(f"  DCT applied to all frames")
        else:
            print("  Applying DCT without framing")
            DCTCoeffs = am.DCT(cA)
        
        print(f"DCT coefficients shape: {DCTCoeffs.shape}")

        #5 scrambling image watermark
        print("\nStep 5: Scrambling watermark image...")
        payload = getScrambling(imagePath, scramblingMode, imageMode)
        payload_width, payload_height = im.imgSize(payload)
        print(f"Payload size: {payload_width}x{payload_height} = {payload_width * payload_height} pixels")
        print(f"Available DCT frames: {DCTCoeffs.shape[0]}")
        
        # Check if we have enough space
        required_frames = (payload_width * payload_height) + 2
        if required_frames > DCTCoeffs.shape[0]:
            print(f"WARNING: Not enough frames! Required: {required_frames}, Available: {DCTCoeffs.shape[0]}")
            # Resize payload to fit
            max_pixels = DCTCoeffs.shape[0] - 2
            new_size = int(np.sqrt(max_pixels))
            payload = payload.resize((new_size, new_size))
            print(f"Resized payload to: {new_size}x{new_size}")
    
        #6 embed watermark image
        print(f"\nStep 6: Embedding watermark (mode: {embeddingMode})...")
        if embeddingMode == "magnitudo":
            wCoeffs = watermark.magnitudoDCT(DCTCoeffs, payload, ALPHA)
        elif embeddingMode == "lsb":
            wCoeffs = watermark.LSB(DCTCoeffs, payload)
        elif embeddingMode == "delta":
            wCoeffs = watermark.deltaDCT(DCTCoeffs, payload)
        elif embeddingMode == "bruteBinary":
            wCoeffs = watermark.bruteBinary(DCTCoeffs, payload)
        elif embeddingMode == "bruteGray":
            wCoeffs = watermark.bruteGray(DCTCoeffs, payload)
        else:
            raise ValueError(f"Unknown embedding mode: {embeddingMode}")
        
        print(f"Watermark embedded, wCoeffs shape: {wCoeffs.shape}")

        #7 run iDCT and #8 join audio frames
        print("\nStep 7-8: Inverse DCT and joining frames...")
        if frames != 0:
            iWCoeffs = np.copy(wCoeffs)
            for i in range(wCoeffs.shape[0]):
                iWCoeffs[i] = am.iDCT(wCoeffs[i])
            print(f"  iDCT applied to all frames")
            
            iWCoeffs = am.frameToAudio(iWCoeffs)
            print(f"  Frames joined, shape: {iWCoeffs.shape}")
        else:
            iWCoeffs = am.iDCT(wCoeffs)
            print(f"  iDCT applied, shape: {iWCoeffs.shape}")

        #9 run iDWT
        print("\nStep 9: Inverse DWT...")
        DWTCoeffs = iWCoeffs, cD1 #level 1
        iWCoeffs = am.iDWT(DWTCoeffs, WAVELET_TYPE, WAVELET_MODE)
        print(f"iDWT completed, shape: {iWCoeffs.shape}")

        #10 save new audio file
        print("\nStep 10: Saving stego audio...")
        getStego(iWCoeffs, tupleAudio, outputAudioPath)
        
        print("--- EMBEDDING COMPLETED SUCCESSFULLY ---\n")
        return wCoeffs
        
    except Exception as e:
        print(f"\n!!! ERROR IN EMBEDDING !!!")
        print(f"Error: {str(e)}")
        print(f"Traceback:\n{traceback.format_exc()}")
        raise
    
def extraction(stegoAudio, audio, outputImagePath, scramblingMode, embeddingMode, frames = 1):
    try:
        print("\n--- EXTRACTION PROCESS ---")
        
        #1 load audio file
        print("Step 1: Loading audio files...")
        audioData, tupleAudio = getAudio(audio)
        stegoAudioData, stegoTupleAudio = getAudio(stegoAudio)

        #2 run DWT on audio file
        print("\nStep 2: Applying DWT to both audio files...")
        DWTCoeffs = getDWT(audioData, WAVELET_TYPE, WAVELET_MODE)
        cA, cD1 = DWTCoeffs #level 1

        stegoDWTCoeffs = getDWT(stegoAudioData, WAVELET_TYPE, WAVELET_MODE)
        stegocA, stegocD1 = stegoDWTCoeffs #level 1
        
        #3 divide by frame & #4 run DCT on DWT coeffs
        print("\nStep 3-4: Framing and DCT...")
        if frames != 0:
            cA = am.audioToFrame(cA, LEN_FRAMES)
            DCTCoeffs = np.copy(cA)
            for i in range(cA.shape[0]):
                DCTCoeffs[i] = am.DCT(cA[i])
            
            stegocA = am.audioToFrame(stegocA, LEN_FRAMES)
            stegoDCTCoeffs = np.copy(stegocA)
            for i in range(stegocA.shape[0]):
                stegoDCTCoeffs[i] = am.DCT(stegocA[i])
            
            print(f"DCT coefficients shape: {DCTCoeffs.shape}")
            print(f"Stego DCT coefficients shape: {stegoDCTCoeffs.shape}")
        else:
            DCTCoeffs = am.DCT(cA)
            stegoDCTCoeffs = am.DCT(stegocA)

        #5 extract image watermark
        print(f"\nStep 5: Extracting watermark (mode: {embeddingMode})...")
        if embeddingMode == "magnitudo":
            payload = watermark.imagnitudoDCT(DCTCoeffs, stegoDCTCoeffs, ALPHA)
        elif embeddingMode == "lsb":
            payload = watermark.iLSB(stegoDCTCoeffs)
        elif embeddingMode == "delta":
            payload = watermark.ideltaDCT(stegoDCTCoeffs)
        elif embeddingMode == "bruteBinary":
            payload = watermark.ibruteBinary(stegoDCTCoeffs)
        elif embeddingMode == "bruteGray":
            payload = watermark.ibruteGray(stegoDCTCoeffs)
        else:
            raise ValueError(f"Unknown embedding mode: {embeddingMode}")
        
        print(f"Payload extracted, size: {im.imgSize(payload)}")
        
        #6 inverse scrambling of payload
        print("\nStep 6: Inverse scrambling...")
        payload = getiScrambling(payload, scramblingMode)
        
        #7 save image
        print("\nStep 7: Saving extracted image...")
        getPayload(payload, outputImagePath)
        
        print("--- EXTRACTION COMPLETED SUCCESSFULLY ---\n")
        
    except Exception as e:
        print(f"\n!!! ERROR IN EXTRACTION !!!")
        print(f"Error: {str(e)}")
        print(f"Traceback:\n{traceback.format_exc()}")
        raise
    
def compareWatermark(wOriginal, wExtracted, imgMode):
    try:
        print("\n--- COMPARING WATERMARKS ---")
        wOriginal = im.loadImage(wOriginal)
        if imgMode == "GRAYSCALE":
            wOriginal = im.grayscale(wOriginal)
        else:
            wOriginal = im.binarization(wOriginal)
        wExtracted = im.loadImage(wExtracted)
        
        if(im.imgSize(wOriginal) != im.imgSize(wExtracted)):
            print(f"Size mismatch: Original {im.imgSize(wOriginal)}, Extracted {im.imgSize(wExtracted)}")
            wExtracted = fixSizeImg(wOriginal, wExtracted, imgMode)
            
        wOriginal = ImageToFlattedArray(wOriginal)
        wExtracted = ImageToFlattedArray(wExtracted)
        
        p = m.correlationIndex(wOriginal, wExtracted)
        psnr = m.PSNR(wOriginal, wExtracted)
        
        result = m.binaryDetection(p, 0.7)
        print(f"Correlation: {p[0]:.4f}, Detection: {result}, PSNR: {psnr:.2f}")
        
        return result, psnr
    except Exception as e:
        print(f"Error comparing watermarks: {str(e)}")
        return False, 0.0

def compareAudio(audio, stegoAudio):
    try:
        print("\n--- COMPARING AUDIO ---")
        audio = am.audioData(am.readWavFile(audio))
        stegoAudio = am.audioData(am.readWavFile(stegoAudio))
        snr = m.SNR(audio)
        snrStego = m.SNR(stegoAudio)
        print(f"SNR Original: {snr:.2f}, SNR Stego: {snrStego:.2f}")
        return snr, snrStego
    except Exception as e:
        print(f"Error comparing audio: {str(e)}")
        return 0.0, 0.0

def attackStego(stegoAudio):
    stegoAudio = am.readWavFile(stegoAudio)
    tAmplitude = [0.5, 2]
    for i in range(len(tAmplitude)):
        getStego(a.amplitudeScaling(stegoAudio[2], tAmplitude[i]), stegoAudio, "amplitude{}".format(tAmplitude[i]))
    sampleRates = [int(stegoAudio[T_SAMPLERATE]*0.75), int(stegoAudio[T_SAMPLERATE]*0.5), int(stegoAudio[T_SAMPLERATE]*0.25), int(stegoAudio[T_SAMPLERATE])+1]
    for i in range(len(sampleRates)):
        a.resampling(stegoAudio[T_AUDIO_PATH], sampleRates[i])
    nLPFilter = [2, 4, 6]
    tupleFFT = am.FFT(stegoAudio)
    indexCutoff = am.indexFrequency(tupleFFT[1], stegoAudio[T_SAMPLERATE], CUTOFF_FREQUENCY)
    for i in range(len(nLPFilter)):
        getStego(am.iFFT(a.butterLPFilter(tupleFFT[0], indexCutoff, nLPFilter[i])), stegoAudio, "butter{}".format(nLPFilter[i]))
    sigmaGauss = [0.00005, 0.0001, 0.00015, 0.0002]
    for i in range(len(sigmaGauss)):
        getStego(a.gaussianNoise(am.audioData(stegoAudio), sigmaGauss[i]), stegoAudio, "gauss{}".format(sigmaGauss[i]))

def main():
    outputDir = opt.output + "/"
    stegoImage = outputDir + opt.embedding_mode + "-" + opt.watermark
    stegoAudio = outputDir + "stego-" + opt.embedding_mode + "-" + opt.source
    wCoeffs = embedding(opt.source, opt.watermark, outputDir + "stego-" + opt.embedding_mode, opt.scrambling_mode, opt.type_watermark, opt.embedding_mode, 0)
    extraction(stegoAudio, opt.source, stegoImage, opt.scrambling_mode, opt.embedding_mode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='', help='audio input')
    parser.add_argument('--watermark', type=str, default='', help='watermark to embed')
    parser.add_argument('--type-watermark', type=str, default='BINARY', choices=['BINARY','GRAYSCALE'], help='Type of watermark')
    parser.add_argument('--embedding-mode', type=str, default='bruteBinary', choices=['delta','bruteBinary',"bruteGray"], help='Embedding mode')
    parser.add_argument('--scrambling-mode', type=str, default='lower', choices=['arnold','lower',"upper"], help='Scrambling mode')    
    parser.add_argument('--output', type=str, default='Output', help='output folder')  

    opt = parser.parse_args()
    
    if os.path.isdir(opt.output) == False:
        os.mkdir(opt.output)
    if os.path.isdir(opt.source):
        sys.exit("Source must not be a dir!")
    if opt.source == '' or opt.watermark == '':
        sys.exit("Input must not be empty!")
    else:
        print(opt)
        main()