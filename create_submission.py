import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import math

# --- Parameters ---
MODEL_PATH = 'deepfake_detector_finetuned.h5'
TEST_DIR = 'test_images/'
IMAGE_SIZE = (224, 224)

# --- Load Model ---
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# --- Get Class Labels ---
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory('dataset/train', target_size=IMAGE_SIZE, class_mode='binary')
class_indices = train_generator.class_indices
labels_map = {v: k for k, v in class_indices.items()}
print(f"Label mapping used: {labels_map}")

# --- Predict on Test Images and Collect Results ---
print("Starting predictions on test images...")
predictions_data = []
test_image_files = [f for f in os.listdir(TEST_DIR) if f.endswith(('png', 'jpg', 'jpeg'))]

for filename in sorted(test_image_files):
    try:
        img_path = os.path.join(TEST_DIR, filename)
        img = image.load_img(img_path, target_size=IMAGE_SIZE)
        img_array = image.img_to_array(img)
        img_array /= 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        pred_prob = model.predict(img_batch, verbose=0)[0][0]
        pred_label_index = 1 if pred_prob > 0.5 else 0
        pred_label_name = labels_map[pred_label_index]

        predictions_data.append({
            'filename': filename, 'label': pred_label_name,
            'confidence': pred_prob, 'image_obj': img
        })
        print(f"  - Predicted {filename} as: {pred_label_name} ({pred_prob:.4f})")

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        predictions_data.append({'filename': filename, 'label': 'error', 'confidence': 0, 'image_obj': None})

# --- Plot all images in a grid AFTER the loop is done ---
print("\nDisplaying all predictions in a grid...")
num_images = len(predictions_data)
if num_images > 0:
    cols = 4
    rows = math.ceil(num_images / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten()

    for i, data in enumerate(predictions_data):
        ax = axes[i]
        if data['image_obj']:
            ax.imshow(data['image_obj'])
            title_color = 'red' if data['label'] == 'fake' else 'green'
            ax.set_title(f"Pred: {data['label'].upper()}\nConf: {data['confidence']:.2f}", color=title_color)
        else:
            ax.text(0.5, 0.5, 'Error', ha='center', va='center')
        ax.axis('off')

    for j in range(num_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# --- Create Submission File ---
print("\nCreating submission file...")
submission_list = [{'image': d['filename'], 'label': d['label']} for d in predictions_data]
submission_df = pd.DataFrame(submission_list)
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully!")