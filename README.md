# standard_deep-learning-models

This code in this repo enables you to download the benchmark deep learningKeras  models as files saved to your system. All these benchmark are trained on the imagenet dataset. </br>
Each available  network is identified by a unique I.D.  , the type  of network can be infereded from the i.d.


These are the avilabile I.D.s
["InceptionV3","ResNet50","VGG16","VGG19","NASNetMobile","NASNetLarge","InceptionResNetV2","MobileNet","DenseNet121","DenseNet169","DenseNet201"]

## Download benchmark model
#python deepModelsSaver.py --networkName InceptionV3    
#python deepModelsSaver.py --networkName all

## image classification using  a benchmark model 
#python benchMarkModelPredictor.py --networkName  NASNetLarge --img_path images/dog_beagle.png  


 ![Sample classiffication](https://github.com/Walid-Ahmed/standard_deep-learning-models/blob/master/results/elph.jpeg)
