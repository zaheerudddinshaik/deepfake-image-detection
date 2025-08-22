import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# --- Parameters ---
MODEL_PATH = 'deepfake_detector_best.h5'
TRAIN_DIR = 'dataset/train'
VAL_DIR = 'dataset/validation'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# --- Load the Data Generators (same as before) ---
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, rotation_range=20, width_shift_range=0.2,
    height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode='nearest'
)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    VAL_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
)

# --- Load your already trained model ---
print("Loading the base trained model...")
model = tf.keras.models.load_model(MODEL_PATH)

# --- THE FINAL, CORRECT WAY TO PREPARE FOR FINE-TUNING ---
# We work directly on the flat list of layers in the model.
print("Configuring model layers for fine-tuning...")

# First, we freeze all layers in the model.
for layer in model.layers:
    layer.trainable = False

# Then, we unfreeze the last 40 layers. This allows the model to adapt
# its more complex feature detectors and the classifier head.
for layer in model.layers[-80:]:
    layer.trainable = True

# --- Re-compile the model with a VERY LOW learning rate ---
print("Re-compiling model for fine-tuning...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # 10x smaller than before
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- Continue Training (Fine-Tuning) ---
fine_tune_checkpoint = ModelCheckpoint(
    "deepfake_detector_finetuned.h5",
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

print("Starting fine-tuning...")
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[fine_tune_checkpoint]
)

print("\nFine-tuning complete! Your new, improved model is saved as 'deepfake_detector_finetuned.h5'")
