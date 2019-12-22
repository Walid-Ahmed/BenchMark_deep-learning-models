

#python deepModelsSaver.py --networkName InceptionV3    
#python deepModelsSaver.py --networkName all

import numpy as np

from keras.preprocessing import image

#from imagenet_utils import preprocess_input, decode_predictions
from keras.applications.inception_v3 import InceptionV3

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
#from imagenet_utils import preprocess_input, decode_predictions
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16


from keras.applications.nasnet import NASNetLarge
from keras.applications.nasnet import NASNetMobile
from keras.applications.inception_resnet_v2  import InceptionResNetV2
from keras.applications.mobilenet import MobileNet


from keras.applications.densenet import DenseNet121
from keras.applications.densenet import DenseNet169
from keras.applications.densenet import DenseNet201

import argparse
import os
import deepModelsSaver

from keras.models import load_model



def getModel(networkName):

	if (networkName=="InceptionV3"):

		filepathModel=os.path.join("models","InceptionV3_ImageNet_299px.h5")

		if(os.path.exists(filepathModel)):
			print("[INFO] Model {} already exists".format(networkName))
			model=load_model(filepathModel)

		else:	
			model = InceptionV3(weights='imagenet')
			model.save(filepathModel)
			model.summary()
			print("[INFO] Model {} saved  to file {} ".format(networkName,filepathModel))
		return model



	if (networkName=="ResNet50"):

		filepathModel=os.path.join("models","ResNet50_ImageNet_224px.h5")
		if(os.path.exists(filepathModel)):
			print("[INFO] Model {} already exists".format(networkName))
			model=load_model(filepathModel)


		else:	
			model= ResNet50(weights='imagenet')
			model.save(filepathModel)
			model.summary()
			print("[INFO] Model {} saved  to file {} ".format(networkName,filepathModel))
		return model




	if (networkName=="VGG16"):

		model= VGG16(weights='imagenet')
		filepathModel=os.path.join("models","VGG16_ImageNet_224px.h5")
		if(os.path.exists(filepathModel)):
			print("[INFO] Model {} already exists".format(networkName))
			model=load_model(filepathModel)

		else:	
			model= VGG16(weights='imagenet')
			model.save(filepathModel)
			model.summary()
			print("[INFO] Model {} saved  to file {} ".format(networkName,filepathModel))
		return model




	if (networkName=="VGG19"):
		filepathModel=os.path.join("models","VGG19_ImageNet_224px.h5")
		if(os.path.exists(filepathModel)):
			print("[INFO] Model {} already exists".format(networkName))
			model=load_model(filepathModel)

		else:	
			model= VGG19(weights='imagenet')

			model.save(filepathModel)
			model.summary()
			print("[INFO] Model {} saved  to file {} ".format(networkName,filepathModel))
		return model



	if (networkName=="NASNetMobile"):

		filepathModel=os.path.join("models","NASNetMobile_ImageNet_224px.h5")
		if(os.path.exists(filepathModel)):
			print("[INFO] Model {} already exists".format(networkName))
			model=load_model(filepathModel)

		else:	
			model =NASNetMobile(input_shape=(224, 224, 3), include_top=True, weights='imagenet')

			model.save(filepathModel)
			model.summary()
			print("[INFO] Model {} saved ".format(networkName))
		return model

	if (networkName=="NASNetLarge"):

		filepathModel=os.path.join("models","NASNetLarge_ImageNet_224px.h5")
		if(os.path.exists(filepathModel)):
			print("[INFO] Model {} already exists".format(networkName))
			model=load_model(filepathModel)

		else:	
			model =NASNetLarge(input_shape=(331, 331, 3), include_top=True, weights='imagenet')

			model.save(filepathModel)
			model.summary()
			print("[INFO] Model {} saved  to file {} ".format(networkName,filepathModel))
		return model

		


	if (networkName=="InceptionResNetV2"):
		filepathModel=os.path.join("models","InceptionResNetV2_ImageNet_299px.h5")
		if(os.path.exists(filepathModel)):
			print("[INFO] Model {} already exists".format(networkName))
			model=load_model(filepathModel)

		else:	
			model =InceptionResNetV2(include_top=True, weights='imagenet')

			model.save(filepathModel)
			model.summary()
			print("[INFO] Model {} saved  to file {} ".format(networkName,filepathModel))
		return model



	if (networkName=="MobileNet"):
		filepathModel=os.path.join("models","MobileNet_ImageNet_299px.h5")
		if(os.path.exists(filepathModel)):
			print("[INFO] Model {} already exists".format(networkName))
			model=load_model(filepathModel)

		else:	
			model =InceptionResNetV2(include_top=True, weights='imagenet')

			model.save(filepathModel)
			model.summary()
			print("[INFO] Model {} saved  to file {} ".format(networkName,filepathModel))
		return model

		




	if (networkName=="DenseNet121"):
		filepathModel=os.path.join("models","DenseNet121_ImageNet_224px.h5")
		if(os.path.exists(filepathModel)):
			print("[INFO] Model {} already exists".format(networkName))
			model=load_model(filepathModel)

		else:	
			model =DenseNet121(include_top=True, weights='imagenet')

			model.save(filepathModel)
			model.summary()
			print("[INFO] Model {} saved ".format(networkName))
		return model

		

	if (networkName=="DenseNet169"):
		filepathModel=os.path.join("models","DenseNet169_ImageNet_224px.h5")
		if(os.path.exists(filepathModel)):
			print("[INFO] Model {} already exists".format(networkName))
			model=load_model(filepathModel)

		else:	
			model =DenseNet169(include_top=True, weights='imagenet')

			model.save(filepathModel)
			model.summary()
			print("[INFO] Model {} saved  to file {} ".format(networkName,filepathModel))
		return model


	if (networkName=="DenseNet201"):
		filepathModel=os.path.join("models","DenseNet201_ImageNet_224px.h5")
		if(os.path.exists(filepathModel)):
			print("[INFO] Model {} already exists".format(networkName))
			model=load_model(filepathModel)

		else:	
			model =DenseNet201(include_top=True, weights='imagenet')

			model.save(filepathModel)
			model.summary()
			print("[INFO] Model {} saved  to file {} ".format(networkName,filepathModel))
		return model

	

if __name__ == "__main__":



    if not os.path.exists('models'):
	    os.makedirs('models')


	# construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--networkName", required=True, help="name of network")



    #read the arguments
    args = vars(ap.parse_args())
    networkName=args["networkName"]

    if (networkName=="all"):
    	nets=["InceptionV3","ResNet50","VGG16","VGG19","NASNetMobile","NASNetLarge","InceptionResNetV2","MobileNet","DenseNet121","DenseNet169","DenseNet201"]
    	for networkName in nets:
    		getModel(networkName)


    else:	
	    getModel(networkName)




