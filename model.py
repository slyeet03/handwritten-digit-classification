import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob

def load_model(model_path):
    """Load the saved model"""
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image_path):
    """Preprocess a single image for prediction"""
    try:
        # Load image
        img = Image.open(image_path)
        
        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')
        
        # Resize to 28x28
        img = img.resize((28, 28))
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize pixel values to [0, 1]
        img_array = img_array / 255.0
        
        # Reshape to (1, 28, 28, 1) for model input
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array, img
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None, None

def get_all_images(folder_path):
    """Get all image files from the folder"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff', '*.webp']
    image_files = []
    
    for extension in image_extensions:
        pattern = os.path.join(folder_path, extension)
        image_files.extend(glob.glob(pattern))
        # Also check for uppercase extensions
        pattern = os.path.join(folder_path, extension.upper())
        image_files.extend(glob.glob(pattern))
    
    return sorted(image_files)

def predict_all_images(model, folder_path):
    """Predict all images in the folder and return results"""
    image_files = get_all_images(folder_path)
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return []
    
    print(f"Found {len(image_files)} images in {folder_path}")
    
    results = []
    
    for i, image_path in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        # Preprocess image
        processed_img, original_img = preprocess_image(image_path)
        
        if processed_img is not None:
            # Make prediction
            prediction = model.predict(processed_img, verbose=0)
            predicted_digit = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            results.append({
                'path': image_path,
                'filename': os.path.basename(image_path),
                'predicted_digit': predicted_digit,
                'confidence': confidence,
                'original_image': original_img,
                'probabilities': prediction[0]
            })
            
            print(f"  -> Predicted: {predicted_digit}, Confidence: {confidence:.3f}")
        else:
            print(f"  -> Failed to process image")
    
    return results

def display_results(results):
    """Display prediction results in a grid"""
    if not results:
        print("No results to display")
        return
    
    n_images = len(results)
    
    # Calculate grid size
    cols = 4
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    
    # Handle single row case
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif n_images == 1:
        axes = np.array([[axes]])
    
    for i, result in enumerate(results):
        row = i // cols
        col = i % cols
        
        ax = axes[row, col]
        
        # Display image
        ax.imshow(result['original_image'], cmap='gray')
        ax.set_title(f"File: {result['filename'][:15]}...\nPredicted: {result['predicted_digit']}\nConfidence: {result['confidence']:.3f}")
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(n_images, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

def display_detailed_results(results):
    """Display detailed results with probability distributions"""
    if not results:
        return
    
    print("\n" + "="*60)
    print("DETAILED PREDICTION RESULTS")
    print("="*60)
    
    for i, result in enumerate(results, 1):
        print(f"\nImage {i}: {result['filename']}")
        print(f"Predicted Digit: {result['predicted_digit']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("Probability distribution:")
        
        for digit in range(10):
            prob = result['probabilities'][digit]
            bar = "█" * int(prob * 20)  # Visual bar
            print(f"  Digit {digit}: {prob:.4f} {bar}")

def save_results_to_file(results, output_file="prediction_results.txt"):
    """Save results to a text file"""
    if not results:
        return
    
    with open(output_file, 'w') as f:
        f.write("MNIST Digit Prediction Results\n")
        f.write("=" * 40 + "\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"Image {i}: {result['filename']}\n")
            f.write(f"Predicted Digit: {result['predicted_digit']}\n")
            f.write(f"Confidence: {result['confidence']:.4f}\n")
            f.write("Probabilities: " + ", ".join([f"{i}:{p:.3f}" for i, p in enumerate(result['probabilities'])]) + "\n")
            f.write("-" * 40 + "\n")
    
    print(f"Results saved to {output_file}")

def main():
    """Main function to run predictions"""
    # Configuration
    MODEL_PATH = "model.keras"  # Path to your saved model
    TEST_FOLDER = "test"        # Folder containing test images
    
    print("MNIST Digit Prediction Script")
    print("=" * 40)
    
    # Load model
    model = load_model(MODEL_PATH)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Check if test folder exists
    if not os.path.exists(TEST_FOLDER):
        print(f"Test folder '{TEST_FOLDER}' not found. Exiting.")
        return
    
    # Predict all images
    results = predict_all_images(model, TEST_FOLDER)
    
    if results:
        print(f"\nSuccessfully processed {len(results)} images")
        
        # Display results
        display_results(results)
        
        # Display detailed results
        display_detailed_results(results)
        
        # Save results to file
        save_results_to_file(results)
        
        # Summary statistics
        avg_confidence = np.mean([r['confidence'] for r in results])
        print(f"\nSummary:")
        print(f"Total images processed: {len(results)}")
        print(f"Average confidence: {avg_confidence:.4f}")
        
        # Count predictions by digit
        digit_counts = {}
        for result in results:
            digit = result['predicted_digit']
            digit_counts[digit] = digit_counts.get(digit, 0) + 1
        
        print("Predicted digits distribution:")
        for digit in range(10):
            count = digit_counts.get(digit, 0)
            print(f"  Digit {digit}: {count} images")
    
    else:
        print("No images were successfully processed.")

if __name__ == "__main__":
    main()