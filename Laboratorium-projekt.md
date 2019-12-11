# Przykład projektu w Python

## Stworzenie szkieletu repozytorium


1. Tworzymy katalog i inicjujemy repozytorium:

```
mkdir mnist-demo
cd mnist-demo
git init
```

2. Tworzymy strukturę plików:

```
mkdir data   # katalog z danymi
mkdir mnist  # nazwa biblioteki
touch setup.py  # to będzie plik instalacyjny biblioteki
```

Zawartość pliku `setup.py` (przykładowa):

```python
from setuptools import setup, find_packages

# biblioteki które zostaną autmatycznie zainstalowane
requirements = """
tensorflow
"""

setup(name='mnist',   # nazwa projektu
      install_requires=requirements,
      packages=find_packages())
```

3. Tworzymy `.gitignore`

```
echo "data/" >> .gitignore
```

4. Plik README.md

```
touch README.md
nano README.md
```

to będzie podstawowa dokumentacja projektu. Warto tam wpisać:
- skąd wziąć dane
- co należy zainstalować (poza zależnościami pythonowymi)
- jak uruchomić skrypty (zreprodukować wyniki)
- itp.

4. Git commit

```
git add mnist README.md setup.py .gitignore
git commit -m "initial commit"
```

## Dane

1. Pobieramy dane:

```
cd data
curl -L https://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz -o mnist_png.tar.gz
```

2. Rozpakowujemy:

```
tar -zxf mnist_png.tar.gz
```

Warto zapoznać się ze strukturą plików, czy wszystko rozumiemy, otworzyć kilka przykładowych,
np.:

```
xdg-open mnist_png/training/5/16553.png  # linux
open mnist_png/training/5/16553.png  # macos
```

## Wirtualne środowisko w python

Poza katalogiem repozytorium (w katalogu domowym, albo katalog wyżej), tworzymy
python virtual env:

```
cd ..
python3 -m venv mnist-venv
source mnist-venv/bin/activate
```

## Wczytywanie danych w pythonie

1. Tworzymy przykładową bilbiotekę

```python
# mnist/load.py
import imageio
import os
from glob import glob


def load_images(path):
    """ Loads all png images from the given path """
    return [imageio.imread(f)
            for f in glob(os.path.join(path, "*.png"))]


def load_data(path):
    """ Loads all images and labels under given path.

    assumes dir structure:

    0/images.png
    1/images.png
    2/images.png
    etc.
    """
    labels = []
    images = []
    for cls in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
        imgs = load_images(os.path.join(path, cls))
        images.extend(imgs)
        labels.extend([int(cls)] * len(imgs))
    return images, labels
```

2. Do wymaganych zależności w `setup.py` trzeba dodać `imageio`.

3. Instalujemy paczkę za pomocą pythonowego managera pakietów:

```
pip install -U -e .
```

(różnice między `pip install -U .` a `pip install -U -e .`).

4. Git commit

```
git add setup.py mnist/load.py
git commit -m "data loader"
```

5. (Opcjonalnie) uaktualnienie listy gitignore:

```
echo "__pycache__/" >> .gitignore
echo "*.egg-info/*" >> .gitignore
git add .gitignore
git commit -m "update .gitignore"
```

## Tworzymy model

1. Dodajemy model przerabiany na Lab 5:


```python
# mnist/model.py
from tensorflow import keras


def model1():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
```

2. Dodajemy nowe zależności do `setup.py`
```
requirements = """
tensorflow
imageio
pyyaml  # new (for model.save)
h5py    # new (for model.save)
click   # new (for cli)
"""
```

3. Dodajemy skrypt uruchamiający uczenie się:

```python
#!/usr/bin/env python

import click
import tensorflow as tf

import mnist.models
import mnist.load


@click.command()
def train():
    print("Loading data...")
    imgs, labels = mnist.load.load_data("../data/mnist_png/training")
    
    # Dane muszą być w formacie tf.Tensor
    X = tf.convert_to_tensor(imgs)
    y = tf.convert_to_tensor(labels)

    model = mnist.models.model1()
    print("Training data...")
    model.fit(X, y, epochs=10)

    model.save("model.h5")
    print("Model saved to file model.h5")


if __name__ == '__main__':
    train()
```

4. Skrypt liczący dokładność na zbiorze testowym:

5. Uruchomienie `train.py` i `test.py`

6. Git add & commit

## Usprawnianie

1. Dodać konfigurowalną nazwę modelu w `train.py` i `test.py`

2. Dodać tasowanie zbioru treningowego (i etykiet!)

3. Konfigurowanie niektórych hiperparametrów z linii poleceń

4. Więcej niż jeden model (CNN z lab 5.)

```
model_cnn = keras.Sequential([
    keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
    keras.layers.Conv2D(64, 3, activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(10, activation='softmax')
])
model_cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```
