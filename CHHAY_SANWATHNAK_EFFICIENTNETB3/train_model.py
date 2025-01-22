import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import tensorflow.keras.backend as K
from tensorflow.keras.utils import register_keras_serializable

# Define paths to your dataset
train_dir = 'push-to-github/datasets/'
validation_dir = 'push-to-github/datasets/'

# Enhanced data augmentation with more diverse transformations
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest',
    validation_split=0.2,
    # Add noise augmentation
    preprocessing_function=lambda x: x + np.random.normal(0, 0.05, x.shape)
)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Create generators with class balancing
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(260, 260),
    batch_size=32,
    class_mode='categorical',
    shuffle=True,
    subset='training'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(260, 260),
    batch_size=32,
    class_mode='categorical',
    shuffle=True,
    subset='validation'
)

# Custom temperature scaling layer for confidence calibration
@register_keras_serializable()
class TemperatureScaling(tf.keras.layers.Layer):
    def __init__(self, temperature=1.0, **kwargs):
        super(TemperatureScaling, self).__init__(**kwargs)
        self.temperature = self.add_weight(
            name='temperature',
            shape=(),  # scalar
            initializer=tf.keras.initializers.Constant(temperature),
            trainable=True
        )
    
    def call(self, inputs):
        return inputs / self.temperature
    
    def get_config(self):
        config = super(TemperatureScaling, self).get_config()
        config.update({'temperature': float(self.temperature.numpy())})
        return config

# Custom loss function with confidence penalty
@register_keras_serializable()
class ConfidencePenaltyLoss(tf.keras.losses.Loss):
    def __init__(self, penalty_weight=0.1, **kwargs):
        super().__init__(**kwargs)
        self.penalty_weight = penalty_weight
    
    def call(self, y_true, y_pred):
        # Standard categorical crossentropy
        cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        # Confidence penalty (entropy term)
        entropy = -tf.reduce_sum(y_pred * tf.math.log(y_pred + K.epsilon()), axis=-1)
        return cce - self.penalty_weight * entropy
    
    def get_config(self):
        config = super().get_config()
        config.update({'penalty_weight': self.penalty_weight})
        return config

# Enhanced model architecture with attention and regularization
def create_model(num_classes, input_shape=(260, 260, 3)):
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Add attention mechanism
    attention = tf.keras.layers.MultiHeadAttention(
        num_heads=4, key_dim=64
    )(base_model.output, base_model.output)
    
    x = GlobalAveragePooling2D()(attention)
    x = BatchNormalization()(x)
    
    # Deeper architecture with skip connections
    x1 = Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x1 = Dropout(0.5)(x1)
    x1 = BatchNormalization()(x1)
    
    x2 = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x1)
    x2 = Dropout(0.4)(x2)
    x2 = BatchNormalization()(x2)
    
    # Skip connection
    x = tf.keras.layers.Concatenate()([x1, x2])
    
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.3)(x)
    
    # Pre-softmax logits
    logits = Dense(num_classes)(x)
    
    # Temperature scaling for better calibration
    scaled_logits = TemperatureScaling()(logits)
    
    # Softmax activation
    predictions = tf.keras.layers.Activation('softmax')(scaled_logits)
    
    return Model(inputs=base_model.input, outputs=predictions)

# Create model
num_classes = len(train_generator.class_indices)
model = create_model(num_classes)

# Custom callback for monitoring prediction confidence
class ConfidenceMonitor(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        predictions = self.model.predict(validation_generator)
        max_confidences = np.max(predictions, axis=1)
        avg_confidence = np.mean(max_confidences)
        print(f"\nEpoch {epoch + 1} - Average confidence: {avg_confidence:.4f}")

# Enhanced training setup
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    ConfidenceMonitor()
]

# Custom confidence threshold for prediction
def predict_with_confidence(model, image, confidence_threshold=0.85):
    prediction = model.predict(image)
    max_confidence = np.max(prediction)
    predicted_class = np.argmax(prediction)
    
    if max_confidence < confidence_threshold:
        return "Unknown", max_confidence
    else:
        class_names = list(train_generator.class_indices.keys())
        return class_names[predicted_class], max_confidence

# Two-phase training with custom loss
# Phase 1: Train top layers
print("Phase 1: Training top layers...")
for layer in model.layers[:-8]:  # Freeze most layers initially
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=ConfidencePenaltyLoss(penalty_weight=0.1),
    metrics=['accuracy']
)

history_1 = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=15,
    callbacks=callbacks
)

# Phase 2: Fine-tuning
print("Phase 2: Fine-tuning the model...")
for layer in model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=ConfidencePenaltyLoss(penalty_weight=0.1),
    metrics=['accuracy']
)

history_2 = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10,
    callbacks=callbacks
)

# Example usage for prediction
def process_and_predict(model, image_path, confidence_threshold=0.85):
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(260, 260)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)
    img_array /= 255.0
    
    predicted_class, confidence = predict_with_confidence(
        model, img_array, confidence_threshold
    )
    
    return predicted_class, confidence

# After training is complete, no need for additional save since ModelCheckpoint will handle it