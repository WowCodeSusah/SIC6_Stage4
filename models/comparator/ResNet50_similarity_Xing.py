# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 12:46:51 2020
@author: x.liang@greenwich.ac.uk
Image Similarity using ResNet50
"""
import os
import numpy as np
from models.comparator.resnet50 import ResNet50
import keras

#from keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
#from scipy.spatial import distance

'''
def get_feature_vector(img):
 img1 = cv2.resize(img, (224, 224))
 feature_vector = feature_model.predict(img1.reshape(1, 224, 224, 3))
 return feature_vector
'''
# avg_pool (AveragePooling2D) output shape: (None, 1, 1, 2048)
# Latest Keras version causing no 'flatten_1' issue; output shape:(None,2048) 
def get_feature_vector_fromPIL(img):
 feature_vector = feature_model.predict(img)
 a, b, c, n = feature_vector.shape
 feature_vector= feature_vector.reshape(b,n)
 return feature_vector

def calculate_similarity_cosine(vector1, vector2):
 #return 1 - distance.cosine(vector1, vector2)
 return cosine_similarity(vector1, vector2)

# This distance can be in range of [0,∞]. And this distance is converted to a [0,1]
def calculate_similarity_euclidean(vector1, vector2):
 return 1/(1 + np.linalg.norm(vector1- vector2))  

image_input = keras.layers.Input(shape=(224, 224, 3))
feature_model = ResNet50(input_tensor=image_input, include_top=False,weights='imagenet')

def compare(original, comparison):
	data = [original, comparison]
	img_data_list=[]
	for dataset in data:
			img_path = dataset
			img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
			x = keras.preprocessing.image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			x = keras.applications.imagenet_utils.preprocess_input(x)
			img_data_list.append(x)

	image_similarity_euclidean = calculate_similarity_euclidean(get_feature_vector_fromPIL(img_data_list[0]), get_feature_vector_fromPIL(img_data_list[1]))
	image_similarity_cosine = calculate_similarity_cosine(get_feature_vector_fromPIL(img_data_list[0]), get_feature_vector_fromPIL(img_data_list[1]))

	print('ResNet50 image similarity_euclidean: ',image_similarity_euclidean)
	print('ResNet50 image similarity_cosine: {:.2f}%'.format(image_similarity_cosine[0][0]*100))
     
	return image_similarity_cosine[0][0]
