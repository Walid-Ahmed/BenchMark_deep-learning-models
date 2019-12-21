import numpy as np

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

from keras.preprocessing import image

from keras.models import load_model
import cv2
import os



def loadBenchMarksModel(networkName):
	if (networkName=="InceptionV3"):
		filepathModel=os.path.join("models","InceptionV3_ImageNet_299px.h5")
		model=load_model(filepathModel)
		print('InceptionV3 loaded')
		return model

	if (networkName=="VGG16"):
		filepathModel=os.path.join("models","VGG16_ImageNet_224px.h5")
		model=load_model(filepathModel)
		print('VGG16 model loaded')
		return model

	if (networkName=="VGG19"):
		filepathModel=os.path.join("models","VGG19_ImageNet_224px.h5")
		model=load_model(filepathModel)
		print('VGG19 model loaded')
		return model

	if (networkName=="ResNet50"):
		filepathModel=os.path.join("models","ResNet50_ImageNet_224px.h5")
		model=load_model(filepathModel)
		print('ResNet50 model loaded')
		return model

	if (networkName=="InceptionResNetV2"):
		filepathModel=os.path.join("models","InceptionResNetV2_ImageNet_299px.h5")
		model=load_model(filepathModel)
		print('InceptionResNetV2 model loaded')
		return model

	if (networkName=="InceptionResNetV2"):
		filepathModel=os.path.join("models","InceptionResNetV2_ImageNet_299px.h5")
		model=load_model(filepathModel)
		print('InceptionResNetV2 model loaded')
		return model

	if (networkName=="MobileNet"):
		filepathModel=os.path.join("models","MobileNet_ImageNet_299px.h5")
		model=load_model(filepathModel)
		print('[INFO] MobileNet model loaded from file {}'.format(filepathModel) )
		return model

	if (networkName=="NASNetLarge"):
		filepathModel=os.path.join("models","NASNetLarge_ImageNet_224px.h5")
		model=load_model(filepathModel)
		print('NASNetLarge model loaded')
		return model


	if (networkName=="NASNetMobile"):
		filepathModel=os.path.join("models","NASNetMobile_ImageNet_224px.h5")
		model=load_model(filepathModel)
		print('NASNetMobile model loaded')
		return model


	if (networkName=="DenseNet121"):
		filepathModel=os.path.join("models","DenseNet121_ImageNet_224px.h5")
		model=load_model(filepathModel)
		print('DenseNet121 model loaded')
		return model

	if (networkName=="DenseNet169"):
		filepathModel=os.path.join("models","DenseNet169_ImageNet_224px.h5")
		model=load_model(filepathModel)
		print('DenseNet169 model loaded')
		return model

	if (networkName=="DenseNet201"):
		filepathModel=os.path.join("models","DenseNet201_ImageNet_224px.h5")
		model=load_model(filepathModel)
		print('DenseNet201 model loaded')
		return model




