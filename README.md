import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline

from google.colab import files
uploaded = files.upload()

df = pd.read_csv(list(uploaded.keys())[0])

df.head()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(df[['age','affordibility']],df.bought_insurance,train_size=0.8)

x_train

x_train_scaled = x_train.copy()
x_train_scaled['age'] = x_train_scaled['age']/100
x_test_scaled =x_test.copy()
x_test_scaled['age']= x_test_scaled['age']/100
model = keras.Sequential([
    keras.layers.Dense(1, input_shape=(2,), activation='sigmoid', kernel_initializer='ones', bias_initializer='zero')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train_scaled,y_train,epochs=500)

model.evaluate(x_test_scaled,y_test)

model.predict(x_test_scaled)

coef, intercept = model.get_weights()

coef, intercept
