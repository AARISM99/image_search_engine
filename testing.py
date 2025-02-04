from skimage import io
import time
import cv2
import joblib
from matplotlib import pyplot as plt
import numpy as np
from keras.models import load_model
from keras import backend as K
from keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from keras.applications.vgg16 import preprocess_input
from sklearn.neighbors import NearestNeighbors as knn


optimizer = Adam(learning_rate=0.001) 

model = load_model("content/encoder_model.h5")
model.compile(optimizer=optimizer, loss='mse') 



