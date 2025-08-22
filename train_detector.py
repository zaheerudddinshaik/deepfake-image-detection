# train_detector.py

# --- Part 1: Imports and Setup ---
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import albumentations as A
import cv2

# Define constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
TRAIN_DIR = 'dataset/train'
VAL_DIR = 'dataset/validation'
EPOCHS = 20
# --- Part 2: Data Preparation ---
print("Setting up data generators...")

# Training data generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

transform = A.Compose([
    A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]), # Ensure image is correct size first
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(p=0.2),
    # This line is corrected with the new parameter names
    A.CoarseDropout(num_holes=8, max_h_size=25, max_w_size=25, p=0.3),
    # This line is corrected with the new function name
    A.ImageCompression(quality_lower=85, quality_upper=100, p=0.5),
    # Normalization should be the last step for augmentation
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


# Validation data generator (only normalization)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Flow data from directories
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

print("Data generators created.")
# --- Part 3: Model Building ---
print("Building the model...")

# Load the base model (MobileNetV2) without the top layer
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMAGE_SIZE, 3)
)

# Freeze the base model layers
base_model.trainable = False

# Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x) # Sigmoid for binary output

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

print("Model built successfully.")
model.summary()
# --- Part 4: Compile and Train ---
print("Compiling the model...")

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
# Saves the best model found during training
checkpoint = ModelCheckpoint(
    "deepfake_detector_best.h5",
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)
# Stops training early if there is no improvement
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', # The metric to watch
    factor=0.2,         # Reduce learning rate by a factor of 5 (1 * 0.2)
    patience=3,         # Reduce if there's no improvement for 3 epochs
    min_lr=1e-6         # The minimum learning rate
)
print("Starting training...")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

print("Training finished.")
# --- Part 5: Save and Evaluate ---
print("Saving final model and plotting history...")

# Save the final trained model
model.save("deepfake_detector_final.h5")

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.savefig('training_history.png')
plt.show()

print("All steps completed. Your best model is saved as 'deepfake_detector_best.h5'")
