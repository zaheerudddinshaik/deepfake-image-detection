import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- Parameters ---
MODEL_PATH = 'deepfake_detector_best.h5'
VAL_DIR = 'dataset/validation'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# --- Load Model and Data ---
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Use a non-augmented data generator for evaluation
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False  # Important: Do not shuffle for evaluation
)

# --- Make Predictions ---
print("Making predictions on validation data...")
# Get the true labels
y_true = validation_generator.classes

# Get the predicted probabilities
y_pred_probs = model.predict(validation_generator)

best_accuracy = 0
best_threshold = 0.5

for threshold in np.arange(0.1, 0.9, 0.01):
    y_pred_temp = (y_pred_probs > threshold).astype(int)
    current_accuracy = np.mean(y_pred_temp == y_true.reshape(-1, 1))
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        best_threshold = threshold

print(f"Best Threshold found: {best_threshold:.2f}")
print(f"Best Accuracy on Validation Set: {best_accuracy*100:.2f}%")

# Convert probabilities to class labels (0 or 1)
y_pred = (y_pred_probs > 0.5).astype(int).reshape(-1)

# --- Generate Reports ---
print("\n--- Classification Report ---")
# Get class names from the generator
class_names = list(validation_generator.class_indices.keys())
print(classification_report(y_true, y_pred, target_names=class_names))


print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_true, y_pred)
print(cm)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
