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



'''

img_path = 'images/beer.png'
img_path = 'images/brown_bear.png'
img_path = 'images/dog_beagle.png'
img_path = 'images/keyboard.png'
img_path = 'images/ped.jpeg'
img_path = 'images/person_011.bmp'
img_path = 'images/space_shuttle.png'
img_path = 'images/vehicle.jpg'
img_path = 'images/vehicle2.jpg'
img_path = 'images/pers.jpg'
img_path = 'images/pers3.jpg'
img_path = 'images/monitor.png'
img_path = 'images/1475867400089.jpg'










img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


# load the original image via OpenCV so we can draw on it and display
# it to our screen later
orig = cv2.imread(img_path)


preds = model1.predict(x)
print('Predicted by InceptionV3:', decode_predictions(preds))


img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


preds = model2.predict(x)
print('Predicted  by VGG16:', decode_predictions(preds))
preds = model3.predict(x)
print('Predicted BY VGG19:', decode_predictions(preds))



preds = model4.predict(x)
print('Predicted by  ResNet50:', decode_predictions(preds))


preds = model4.predict(x)
(inID, label) = decode_predictions(preds)[0]





# display the predictions to our screen
print("ImageNet ID: {}, Label: {}".format(inID, label))
cv2.putText(orig, "Label: {}".format(label), (10, 30),
	cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv2.imshow("Classification", orig)
cv2.waitKey(0)

'''
