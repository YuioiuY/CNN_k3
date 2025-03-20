import os
import requests
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 10
EPOCHS = 30  
FINE_TUNE_EPOCHS = 15
PATH = r"\dogs" # <- возможно потребуется полный путь

def load_stanford_dogs_data(data_dir=PATH):
    images_dir = data_dir
    url = "https://storage.yandexcloud.net/academy.ai/stanford_dogs.zip"
    zip_path = os.path.join(data_dir, "stanford_dogs.zip")

    os.makedirs(data_dir, exist_ok=True)
    
    if not os.path.exists(images_dir) or not os.listdir(images_dir):
        print(f"Датасет не найден в {images_dir}. Начинаем загрузку...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Загрузка завершена!")
            print("Распаковка stanford_dogs.zip...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print("Распаковка завершена!")
            os.remove(zip_path)
    
    images_dir = os.path.join(data_dir, "Images")
    print(f"Проверяемый путь: {images_dir}")
    
    if not os.path.exists(images_dir):
        raise ValueError(f"Директория {images_dir} не найдена.")
    
    all_breeds = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
    if not all_breeds:
        raise ValueError(f"В директории {images_dir} нет папок с породами.")
    
    print(f"Найдено {len(all_breeds)} пород")
    selected_breeds = np.random.choice(all_breeds, NUM_CLASSES, replace=False)
    print(f"Выбранные породы: {list(selected_breeds)}")
    
    image_paths = []
    labels = []
    
    for i, breed in enumerate(selected_breeds):
        breed_path = os.path.join(images_dir, breed)
        breed_images = [os.path.join(breed_path, img) 
                       for img in os.listdir(breed_path) 
                       if img.endswith(('.jpg', '.jpeg', '.png'))]
        image_paths.extend(breed_images)
        labels.extend([i] * len(breed_images))
    
    print(f"Загружено {len(image_paths)} изображений для {len(set(labels))} пород")
    return image_paths, labels


data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.3),
    layers.RandomContrast(0.3),
    layers.RandomBrightness(0.2),
    layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
])

def process_image(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = preprocess_input(img)  
    return img, label

def create_dataset(image_paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def build_model():
    base_model = EfficientNetV2S(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)  
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model, base_model

def main():
    image_paths, labels = load_stanford_dogs_data()
    
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=0.2, random_state=42
    )
    
    train_ds = create_dataset(train_paths, train_labels)
    val_ds = create_dataset(val_paths, val_labels)
    test_ds = create_dataset(test_paths, test_labels)
    
    model, base_model = build_model()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ]
    )
    
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) // 2  
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    fine_tune_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ]
    )
    
    train_loss, train_accuracy = model.evaluate(train_ds)
    val_loss, val_accuracy = model.evaluate(val_ds)
    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    model.save('stanford_dogs_model.keras')
    
    plt.plot(history.history['accuracy'] + fine_tune_history.history['accuracy'])
    plt.plot(history.history['val_accuracy'] + fine_tune_history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    main()