# Benchmark_deep-learning-models

The code in this repo enables you to download the benchmark deep learning Keras  models as files saved to your system. All these benchmark models are pretrained on the imagenet dataset. </br>
Each available  network is identified by a unique I.D.  , the type  of network can be infereded from the i.d.

Thanks to Adrian Rosebrock for his inspiring post https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/

These are the list of avilabile I.D.s:
["InceptionV3","ResNet50","VGG16","VGG19","NASNetMobile","NASNetLarge","InceptionResNetV2","MobileNet","DenseNet121","DenseNet169","DenseNet201"]

## Download benchmark model
You can choose a certain model to save to your computer  (for ex InceptionV3) </br>
python deepModelsSaver.py --networkName InceptionV3    

you can also download all the models to your computer as follows </br>
python deepModelsSaver.py --networkName all

## image classification using  a benchmark model 
you can choose to classify a certain image with an imagenet label with a certain model as follows:  </br>
python benchMarkModelPredictor.py --networkName  NASNetLarge --img_path images/dog_beagle.png  


 ![Sample classiffication](https://github.com/Walid-Ahmed/standard_deep-learning-models/blob/master/results/elph.jpeg)

## Classification of a folder of images  using  a benchmark model 

you can choose to classify a collection of images in a folder  with an imagenet label with a certain model as follows:  </br>

python benchMarkModelPredictorFolder.py --networkName NASNetLarge --folder_path imagenet_images

A video will be created showing results of classification.


## Credits

Imagenet images where downloaded using https://github.com/mf1024/ImageNet-Datasets-Downloader
