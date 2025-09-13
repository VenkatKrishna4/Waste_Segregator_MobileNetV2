# train.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type: ignore
from tensorflow.keras.applications import MobileNetV2 #type: ignore
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout  #type: ignore
from tensorflow.keras.models import Model   #type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau #type: ignore
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import json
import os

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
DATA_DIR = "waste_raw/waste_dataset"

train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "val")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(IMG_SIZE,IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True
)
val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(IMG_SIZE,IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

num_classes = train_generator.num_classes

label_map = {v:k for k,v in train_generator.class_indices.items()}
with open("labels.json","w") as f:
    json.dump(label_map, f)

y = train_generator.classes
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = {i: w for i,w in enumerate(class_weights)}

# model
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
out = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base.input, outputs=out)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# callbacks
callbacks = [
    ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

steps_per_epoch = max(1, train_generator.samples // BATCH_SIZE)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    class_weight=class_weights
)

base.trainable = True

fine_tune_at = len(base.layers) - 50
for layer in base.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

fine_epochs = 10
history_fine = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=fine_epochs,
    validation_data=val_generator,
    callbacks=callbacks,
    class_weight=class_weights
)

# final model
model.save("waste_model.h5")
print("Saved waste_model.h5 and labels.json")