import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

"""
    # Téléchargement et extraction du fichier zip
    import requests, zipfile, io
    zip_file_url = 'https://moncoachdata.com/wp-content/uploads/cell_images.zip'
    r = requests.get(zip_file_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()
"""

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(CUR_DIR, "cell_images")
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

EPOCH = 15
PATIENCE = 2
ACTIVATION = "relu"
BATCH_SIZE = 16

para_cell = train_dir+'\\parasitized\\C59P20thinF_IMG_20150803_113809_cell_41.png'
para_img = imread(para_cell)
plt.imshow(para_img)
plt.show()


dim1 = []
dim2 = []

for image_filename in os.listdir(test_dir + "/uninfected"):
    img = imread(test_dir+"/uninfected" + "/" + image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)

"""
Randomisation des images afin d'augmenter notre dataset
Note: -les images doivent être groupé dans des dossiers individuels (un dossier par classe)
"""    
from keras.preprocessing.image import ImageDataGenerator
image_gen = ImageDataGenerator(rotation_range=20, 
                               height_shift_range=0.1,
                               width_shift_range=0.1,
                               shear_range=0.1,
                               zoom_range=0.1,
                               horizontal_flip=True,
                               fill_mode="nearest")

image_gen.flow_from_directory(train_dir)
image_gen.flow_from_directory(test_dir)
image_shape = (130,130,3)

"""Création du model"""
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=image_shape, activation=ACTIVATION))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=image_shape, activation=ACTIVATION))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=image_shape, activation=ACTIVATION))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation=ACTIVATION))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])



train_image_gen = image_gen.flow_from_directory(train_dir, 
                                                target_size=image_shape[:2],
                                                color_mode="rgb",
                                                batch_size=BATCH_SIZE,
                                                class_mode="binary")

test_image_gen = image_gen.flow_from_directory(train_dir, 
                                                target_size=image_shape[:2],
                                                color_mode="rgb",
                                                batch_size=BATCH_SIZE,
                                                class_mode="binary",
                                                shuffle=False)

print(train_image_gen.class_indices) #parasitized:0 uninfected:1

"""Entrainement des données"""
if not os.path.exists(os.path.join(CUR_DIR, f"myModel_{ACTIVATION}_{EPOCH}_{PATIENCE}.h5")):
    from keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(monitor="val_loss", patience=PATIENCE)
    model.fit(train_image_gen, epochs=EPOCH, validation_data=test_image_gen, callbacks=[early_stop])
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
#print(model.evaluate(train_image_gen, test_image_gen, verbose=0))
plt.show()

pred = model.predict(test_image_gen)

from sklearn.metrics import classification_report, confusion_matrix
predictions = pred > 0.5 #permet de modifier la sensibilité des résultats
print(classification_report(test_image_gen.classes, predictions))

"""Analyse d'une autre image"""
from keras.preprocessing import image
my_image = image.load_img(para_cell, target_size=image_shape) #conversion en (130,130,3)
my_img_array = image.img_to_array(my_image)
my_img_array = np.expand_dims(my_image, axis=0) #conversion en (1,130,130,3) pour avoir un seul batch d'image
print(model.predict(my_img_array)) # [[0.02744671]] donc cellule infecté