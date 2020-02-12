#This script is used to predict  classes of  images in folder using a deep benchmark model


#python benchMarkModelPredictorFolder.py --networkName NASNetLarge --folder_path imagenet_images






import numpy as np

from keras.preprocessing import image
#from keras.applications.mobilenet import preprocess_input, decode_predictions
from keras.applications import mobilenet

from keras.applications import resnet50 
from keras.applications import inception_resnet_v2
from keras.applications import inception_v3 
from keras.applications import  nasnet
from keras.applications import vgg19 
from keras.applications import densenet

#from deepModelsLoader import loadBenchMarksModel

import argparse

import cv2

import os
import deepModelsSaver
from util import paths

import random

fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
width,height=int(960/2),int(540/2)

k=1


def predict(networkName,model,img_path,gt):




	if (networkName=="MobileNet"):
		img = image.load_img(img_path, target_size=(299, 299))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = mobilenet.preprocess_input(x)
		preds = model.predict(x)
		# decode the results into a list of tuples (class, description, probability)
		# (one such list for each sample in the batch)
		print('Predicted:', mobilenet.decode_predictions(preds, top=1)[0])
		k= (mobilenet.decode_predictions(preds))[0]
		inID, label, probability=k[0]

	elif(networkName=="ResNet50"):	
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = resnet50.preprocess_input(x)
		preds = model.predict(x)
		# decode the results into a list of tuples (class, description, probability)
		# (one such list for each sample in the batch)
		print('Predicted:', resnet50.decode_predictions(preds, top=1)[0])
		k= (resnet50.decode_predictions(preds))[0]
		inID, label, probability=k[0]

	elif(networkName=="InceptionResNetV2"):
		img = image.load_img(img_path, target_size=(299, 299))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = inception_resnet_v2.preprocess_input(x)
		preds = model.predict(x)
		# decode the results into a list of tuples (class, description, probability)
		# (one such list for each sample in the batch)
		print('Predicted:', inception_resnet_v2.decode_predictions(preds, top=1)[0])
		k= (inception_resnet_v2.decode_predictions(preds))[0]
		inID, label, probability=k[0]

	elif(networkName=="InceptionV3"):
		img = image.load_img(img_path, target_size=(299, 299))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = inception_v3.preprocess_input(x)
		preds = model.predict(x)
		# decode the results into a list of tuples (class, description, probability)
		# (one such list for each sample in the batch)
		print('Predicted:', inception_v3.decode_predictions(preds, top=1)[0])
		k= (inception_v3.decode_predictions(preds))[0]
		inID, label, probability=k[0]		

	elif(networkName=="NASNetLarge"):
		img = image.load_img(img_path, target_size=(331, 331))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = nasnet.preprocess_input(x)
		preds = model.predict(x)
		# decode the results into a list of tuples (class, description, probability)
		# (one such list for each sample in the batch)
		print('Predicted:', nasnet.decode_predictions(preds, top=1)[0])
		k= (nasnet.decode_predictions(preds))[0]
		inID, label, probability=k[0]	

	elif(networkName=="NASNetMobile"):
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = nasnet.preprocess_input(x)
		preds = model.predict(x)
		# decode the results into a list of tuples (class, description, probability)
		# (one such list for each sample in the batch)
		print('Predicted:', nasnet.decode_predictions(preds, top=1)[0])
		k= (nasnet.decode_predictions(preds))[0]
		inID, label, probability=k[0]		
	
	elif(networkName=="VGG19"):
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = vgg19.preprocess_input(x)
		preds = model.predict(x)
		# decode the results into a list of tuples (class, description, probability)
		# (one such list for each sample in the batch)
		print('Predicted:', vgg19.decode_predictions(preds, top=1)[0])
		k= (vgg19.decode_predictions(preds))[0]
		inID, label, probability=k[0]		
	
	elif(networkName=="VGG16"):
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = vgg19.preprocess_input(x)
		preds = model.predict(x)
		# decode the results into a list of tuples (class, description, probability)
		# (one such list for each sample in the batch)
		print('Predicted:', vgg19.decode_predictions(preds, top=1)[0])
		k= (vgg19.decode_predictions(preds))[0]
		inID, label, probability=k[0]		

	elif(networkName=="DenseNet121"):
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = densenet.preprocess_input(x)
		preds = model.predict(x)
		# decode the results into a list of tuples (class, description, probability)
		# (one such list for each sample in the batch)
		print('Predicted:', densenet.decode_predictions(preds, top=1)[0])
		k= (densenet.decode_predictions(preds))[0]
		inID, label, probability=k[0]	


	elif(networkName=="DenseNet169"):
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = densenet.preprocess_input(x)
		preds = model.predict(x)
		# decode the results into a list of tuples (class, description, probability)
		# (one such list for each sample in the batch)
		print('Predicted:', densenet.decode_predictions(preds, top=1)[0])
		k= (densenet.decode_predictions(preds))[0]
		inID, label, probability=k[0]	

	elif(networkName=="DenseNet201"):
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = densenet.preprocess_input(x)
		preds = model.predict(x)
		# decode the results into a list of tuples (class, description, probability)
		# (one such list for each sample in the batch)
		print('Predicted:', densenet.decode_predictions(preds, top=1)[0])
		k= (densenet.decode_predictions(preds))[0]
		inID, label, probability=k[0]	




	# load the original image via OpenCV so we can draw on it and display
	# it to our screen later

	orig = cv2.imread(img_path)





	# display the predictions to our screen
	print("ImageNet ID: {}, Label: {}".format(inID, label))
	cv2.putText(orig, "Predicted Label: {} ".format(label), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	#cv2.putText(orig, "Actual Label: {} ".format(gt), (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	cv2.putText(orig, "Network: {}".format(networkName), (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	fileName=os.path.basename(img_path) 
	cv2.imshow("Classification Results", orig)
	#cv2.imwrite(os.path.join("results",fileName),orig)
	img=cv2.resize(orig,(width, height))
	video_creator.write(img)
	cv2.waitKey(10)




if __name__ == "__main__":



    if not os.path.exists('models'):
	    os.makedirs('models')


	# construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--networkName", required=True, help="name of network")
    ap.add_argument("--folder_path", required=True, help="folder_path")




    #read the arguments
    args = vars(ap.parse_args())
    networkName=args["networkName"]
    folder_path=args["folder_path"]
    fileName="demo_"+networkName+".mp4"
    video_creator = cv2.VideoWriter(fileName,fourcc, 1, (width,height))


    #model= loadBenchMarksModel(networkName)
    model=deepModelsSaver.getModel(networkName)
    img_paths=list(paths.list_images(folder_path))
    random.shuffle(img_paths) 


    for img_path in  img_paths:
    	groundTruth=(os.path.dirname(img_path))
    	gt=(os.path.basename(groundTruth))
    	predict(networkName,model,img_path,gt)
    	k=k+1
    	print(k)
    	if(k==100):
    		exit()



    video_creator.release()





