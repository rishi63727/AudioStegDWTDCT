from flask import Flask, render_template, request, send_file, jsonify, flash, redirect, url_for
import os
import sys
import shutil
import traceback
from werkzeug.utils import secure_filename
import main as watermark_main
from utils import ImageToFlattedArray, fixSizeImg
import image_managing as im
import metrics as m

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Allowed extensions
ALLOWED_AUDIO_EXTENSIONS = {'wav'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/embed', methods=['POST'])
def embed_watermark():
    try:
        print("\n=== STARTING EMBED PROCESS ===")
        
        # Check if files are present
        if 'audio_file' not in request.files or 'watermark_file' not in request.files:
            flash('Please upload both audio and watermark files')
            return redirect(url_for('index'))
        
        audio_file = request.files['audio_file']
        watermark_file = request.files['watermark_file']
        
        print(f"Audio file: {audio_file.filename}")
        print(f"Watermark file: {watermark_file.filename}")
        
        if audio_file.filename == '' or watermark_file.filename == '':
            flash('Please select files to upload')
            return redirect(url_for('index'))
        
        # Validate file types
        if not allowed_file(audio_file.filename, ALLOWED_AUDIO_EXTENSIONS):
            flash('Audio file must be in WAV format')
            return redirect(url_for('index'))
        
        if not allowed_file(watermark_file.filename, ALLOWED_IMAGE_EXTENSIONS):
            flash('Watermark file must be an image (PNG, JPG, JPEG)')
            return redirect(url_for('index'))
        
        # Get form parameters
        watermark_type = request.form.get('watermark_type', 'BINARY')
        embedding_mode = request.form.get('embedding_mode', 'bruteBinary')
        scrambling_mode = request.form.get('scrambling_mode', 'lower')
        
        print(f"Watermark type: {watermark_type}")
        print(f"Embedding mode: {embedding_mode}")
        print(f"Scrambling mode: {scrambling_mode}")
        
        # Save uploaded files
        audio_filename = secure_filename(audio_file.filename)
        watermark_filename = secure_filename(watermark_file.filename)
        
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
        watermark_path = os.path.join(app.config['UPLOAD_FOLDER'], watermark_filename)
        
        print(f"Saving audio to: {audio_path}")
        print(f"Saving watermark to: {watermark_path}")
        
        audio_file.save(audio_path)
        watermark_file.save(watermark_path)
        
        # Prepare output paths
        output_dir = app.config['OUTPUT_FOLDER']
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert watermark type to integer
        image_mode = watermark_main.BINARY if watermark_type == 'BINARY' else watermark_main.GRAYSCALE
        
        print(f"Image mode: {image_mode}")
        
        # Create output filenames
        stego_audio_filename = f"stego-{embedding_mode}-{audio_filename}"
        actual_stego_path = os.path.join(output_dir, stego_audio_filename)
        
        print(f"Output stego path: {actual_stego_path}")
        
        print("Starting embedding process...")
        
        # Embed watermark - pass the full output path without prefix
        try:
            watermark_main.embedding(
                audio_path,
                watermark_path,
                actual_stego_path,  # Pass full path instead of just prefix
                scrambling_mode,
                image_mode,
                embedding_mode,
                frames=0
            )
            print("Embedding completed successfully")
        except Exception as embed_error:
            print(f"Error during embedding: {str(embed_error)}")
            print(traceback.format_exc())
            raise
        
        # Check if stego file was created
        if not os.path.exists(actual_stego_path):
            print(f"ERROR: Stego file not found at {actual_stego_path}")
            flash('Error: Watermarked audio file was not created')
            return redirect(url_for('index'))
        
        print(f"Stego file created: {actual_stego_path}")
        
        # Create extracted image filename
        extracted_image_filename = f"{embedding_mode}-{watermark_filename}"
        extracted_image_path = os.path.join(output_dir, extracted_image_filename)
        
        print(f"Extracting watermark to: {extracted_image_path}")
        
        # Extract watermark for verification
        try:
            watermark_main.extraction(
                actual_stego_path,
                audio_path,
                extracted_image_path,
                scrambling_mode,
                embedding_mode,
                frames=0
            )
            print("Extraction completed successfully")
        except Exception as extract_error:
            print(f"Error during extraction: {str(extract_error)}")
            print(traceback.format_exc())
            raise
        
        # Check if extracted image was created
        if not os.path.exists(extracted_image_path):
            print(f"WARNING: Extracted image not found at {extracted_image_path}")
            # Create a placeholder image
            from PIL import Image
            placeholder = Image.new('L', (128, 128), color=128)
            placeholder.save(extracted_image_path)
            print("Created placeholder extracted image")
        
        print("Calculating metrics...")
        
        # Calculate metrics with error handling
        try:
            correlation_result, psnr = watermark_main.compareWatermark(
                watermark_path,
                extracted_image_path,
                watermark_type
            )
            print(f"Correlation: {correlation_result}, PSNR: {psnr}")
        except Exception as metric_error:
            print(f"Error calculating watermark metrics: {str(metric_error)}")
            correlation_result = False
            psnr = 0.0
        
        try:
            snr_original, snr_stego = watermark_main.compareAudio(
                audio_path,
                actual_stego_path
            )
            print(f"SNR Original: {snr_original}, SNR Stego: {snr_stego}")
        except Exception as snr_error:
            print(f"Error calculating SNR: {str(snr_error)}")
            snr_original = 0.0
            snr_stego = 0.0
        
        print("=== EMBED PROCESS COMPLETED ===\n")
        
        return render_template('result.html',
                             stego_audio=stego_audio_filename,
                             extracted_image=extracted_image_filename,
                             correlation=correlation_result,
                             psnr=round(psnr, 2),
                             snr_original=round(snr_original, 2),
                             snr_stego=round(snr_stego, 2),
                             watermark_type=watermark_type,
                             embedding_mode=embedding_mode,
                             scrambling_mode=scrambling_mode)
    
    except Exception as e:
        print(f"\n!!! CRITICAL ERROR !!!")
        print(f"Error: {str(e)}")
        print(f"Traceback:\n{traceback.format_exc()}")
        print("!!! END ERROR !!!\n")
        flash(f'Error processing files: {str(e)}')
        return redirect(url_for('index'))

@app.route('/extract', methods=['POST'])
def extract_watermark():
    try:
        print("\n=== STARTING EXTRACT PROCESS ===")
        
        if 'stego_audio' not in request.files or 'original_audio' not in request.files:
            flash('Please upload both stego and original audio files')
            return redirect(url_for('index'))
        
        stego_audio_file = request.files['stego_audio']
        original_audio_file = request.files['original_audio']
        
        print(f"Stego audio: {stego_audio_file.filename}")
        print(f"Original audio: {original_audio_file.filename}")
        
        if stego_audio_file.filename == '' or original_audio_file.filename == '':
            flash('Please select files to upload')
            return redirect(url_for('index'))
        
        # Validate file types
        if not allowed_file(stego_audio_file.filename, ALLOWED_AUDIO_EXTENSIONS):
            flash('Audio files must be in WAV format')
            return redirect(url_for('index'))
        
        if not allowed_file(original_audio_file.filename, ALLOWED_AUDIO_EXTENSIONS):
            flash('Audio files must be in WAV format')
            return redirect(url_for('index'))
        
        # Get form parameters
        embedding_mode = request.form.get('extract_embedding_mode', 'bruteBinary')
        scrambling_mode = request.form.get('extract_scrambling_mode', 'lower')
        
        print(f"Embedding mode: {embedding_mode}")
        print(f"Scrambling mode: {scrambling_mode}")
        
        # Save uploaded files
        stego_filename = secure_filename(stego_audio_file.filename)
        original_filename = secure_filename(original_audio_file.filename)
        
        stego_path = os.path.join(app.config['UPLOAD_FOLDER'], stego_filename)
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        
        print(f"Saving stego to: {stego_path}")
        print(f"Saving original to: {original_path}")
        
        stego_audio_file.save(stego_path)
        original_audio_file.save(original_path)
        
        # Prepare output path
        output_dir = app.config['OUTPUT_FOLDER']
        os.makedirs(output_dir, exist_ok=True)
        
        extracted_image_filename = f"extracted-{embedding_mode}-{stego_filename.rsplit('.', 1)[0]}.png"
        extracted_image_path = os.path.join(output_dir, extracted_image_filename)
        
        print(f"Extracting to: {extracted_image_path}")
        
        # Extract watermark
        try:
            watermark_main.extraction(
                stego_path,
                original_path,
                extracted_image_path,
                scrambling_mode,
                embedding_mode,
                frames=0
            )
            print("Extraction completed successfully")
        except Exception as extract_error:
            print(f"Error during extraction: {str(extract_error)}")
            print(traceback.format_exc())
            raise
        
        # Check if extracted image was created
        if not os.path.exists(extracted_image_path):
            print(f"WARNING: Extracted image not found")
            from PIL import Image
            placeholder = Image.new('L', (128, 128), color=128)
            placeholder.save(extracted_image_path)
            print("Created placeholder image")
        
        print("=== EXTRACT PROCESS COMPLETED ===\n")
        
        return render_template('extract_result.html',
                             extracted_image=extracted_image_filename,
                             embedding_mode=embedding_mode,
                             scrambling_mode=scrambling_mode)
    
    except Exception as e:
        print(f"\n!!! CRITICAL ERROR !!!")
        print(f"Error: {str(e)}")
        print(f"Traceback:\n{traceback.format_exc()}")
        print("!!! END ERROR !!!\n")
        flash(f'Error extracting watermark: {str(e)}')
        return redirect(url_for('index'))

@app.route('/download/<folder>/<filename>')
def download_file(folder, filename):
    try:
        if folder not in ['uploads', 'outputs']:
            flash('Invalid folder')
            return redirect(url_for('index'))
        
        file_path = os.path.join(folder, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            flash('File not found')
            return redirect(url_for('index'))
    except Exception as e:
        print(f"Error downloading file: {str(e)}")
        flash('Error downloading file')
        return redirect(url_for('index'))

if __name__ == '__main__':
    print("=" * 50)
    print("Audio Watermarking System Starting...")
    print("=" * 50)
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Output folder: {app.config['OUTPUT_FOLDER']}")
    print("Server starting on http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)