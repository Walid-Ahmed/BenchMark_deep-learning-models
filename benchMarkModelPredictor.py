

#python benchMarkModelPredictor.py --networkName MobileNet --img_path images/pers3.jpg
#python benchMarkModelPredictor.py --networkName ResNet50 --img_path images/pers3.jpg
#python benchMarkModelPredictor.py --networkName InceptionResNetV2 --img_path images/pers3.jpg
#python benchMarkModelPredictor.py --networkName InceptionV3  --img_path images/pers3.jpg
#python benchMarkModelPredictor.py --networkName  NASNetLarge --img_path images/dog_beagle.png  
#python benchMarkModelPredictor.py --networkName  NASNetMobile --img_path images/elph.jpeg
#python benchMarkModelPredictor.py --networkName  VGG19 --img_path images/pers3.jpg 
#python benchMarkModelPredictor.py --networkName  VGG16 --img_path images/pers3.jpg 
#python benchMarkModelPredictor.py --networkName  DenseNet121 --img_path images/pers3.jpg 
#python benchMarkModelPredictor.py --networkName  DenseNet169 --img_path images/pers3.jpg 
#python benchMarkModelPredictor.py --networkName  DenseNet201 --img_path images/pers3.jpg 






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



def predict(networkName,model,img_path):




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
	cv2.putText(orig, "Label: {} ".format(label), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	cv2.putText(orig, "Network: {}".format(networkName), (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	fileName=os.path.basename(img_path) 
	cv2.imshow("Classification Results", orig)
	cv2.imwrite(os.path.join("results",fileName),orig)
	cv2.waitKey(0)




if __name__ == "__main__":



    if not os.path.exists('models'):
	    os.makedirs('models')


	# construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--networkName", required=True, help="name of network")
    ap.add_argument("--img_path", required=True, help="img_path")




    #read the arguments
    args = vars(ap.parse_args())
    networkName=args["networkName"]
    img_path=args["img_path"]

    #model= loadBenchMarksModel(networkName)
    model=deepModelsSaver.getModel(networkName)
    img_paths=paths.list_images(imagenet_images)
    print(img_paths)

    exit()

    predict(networkName,model,img_path)







