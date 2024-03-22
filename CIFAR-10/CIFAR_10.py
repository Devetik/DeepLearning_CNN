import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import cifar10
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
EPOCH = 15
PATIENCE = 2
ACTIVATION = "relu"

"""Extraction des données de CIFAR_10"""
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#permet d'afficher un des échantillons
# single_image = x_train[0]
# plt.imshow(single_image)
# plt.show()

"""normalisation des valeurs des images"""
x_train = x_train / 255
x_test = x_test / 255

"""conversion des labels (y) en "one hot encoding"""
from keras.utils import to_categorical
y_cat_train = to_categorical(y_train, num_classes=10)
y_cat_test = to_categorical(y_test, num_classes=10)


# #batch_size, width, height, color_channel (1 for black/white and 3 for colors)
# x_train = x_train.reshape(50000,32,32,3)
# x_test = x_test.reshape(10000,32,32,3)

"""Création du model"""
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(4,4), padding="same", input_shape=(32,32,3), activation=ACTIVATION))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=32, kernel_size=(4,4), padding="same", input_shape=(32,32,3), activation=ACTIVATION))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(256, activation=ACTIVATION))
model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

"""Entrainement des données"""
if not os.path.exists(os.path.join(CUR_DIR, f"myModel_{ACTIVATION}_{EPOCH}_{PATIENCE}.h5")):
    from keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(monitor="val_loss", patience=PATIENCE)
    model.fit(x_train, y_cat_train, epochs=EPOCH, validation_data=(x_test, y_cat_test), callbacks=[early_stop])
    model.save(os.path.join(CUR_DIR, f"myModel_{ACTIVATION}_{EPOCH}_{PATIENCE}.h5"))
    metrics = pd.DataFrame(model.history.history)

    # Sauvegardez le DataFrame dans un fichier CSV
    metrics_file_path = os.path.join(CUR_DIR, f"metrics_{ACTIVATION}_{EPOCH}_{PATIENCE}.csv")
    metrics.to_csv(metrics_file_path, index=False)
else:
    from keras.models import load_model
    model = load_model(os.path.join(CUR_DIR, f"myModel_{ACTIVATION}_{EPOCH}_{PATIENCE}.h5"))
    # Chargement de l'historique d'entraînement
    metrics = pd.read_csv(os.path.join(CUR_DIR, f"metrics_{ACTIVATION}_{EPOCH}_{PATIENCE}.csv"))


"""affichage des metrics du model"""
metrics[["loss", "val_loss"]].plot()
metrics[["accuracy", "val_accuracy"]].plot()
print(model.evaluate(x_test, y_cat_test, verbose=0))
plt.show()

from sklearn.metrics import classification_report, confusion_matrix
predictions = np.argmax(model.predict(x_test), axis=1)
print(classification_report(y_test, predictions))


