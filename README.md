# Ironhack project 8 Scenes Landscapes CNN image classification

![GitHub Logo](https://www.publicdomainpictures.net/pictures/220000/nahled/landscape-with-a-lake-1493481278Ed8.jpg)

### Project description

In this project the idea was to build a model able to classify images, and to be more precise, the images themselves and not the pixels, or specific features in the images. A dataset of scenes/landscapes images was choosed because of its possible link to geography, and provided by Intel on Kaggle. A deep learning model, CNN (Convolutional Neural Network) was choosed because they usually work well for image classification because of how they're built. Different numbers and organisations of layers were tried, and each time the accuracies (training and validation) were compared. Eventually the best model's results were displayed with a confusion matrix

Steps of the project :
- Importation of the images and labels separated ; from 2 folders we get test images/labels and train images/labels
- Train images/labels are then separated in train images/labels and validation images labels 
      => we end up with test, train and validation datasets
- Number of images per datasets and classes are then shown on an horizontal barplot
- Different models were tested, the first model was much faster than the other ones, but with only around 50 % of accuracy. The other tests were in the range 50% to 75% of accuracy
The fourth version of the model was the best, but very close to the third
      - The performance of the models are compared with the evolution of their accuracies at different epoch
- Transformed images are created to train the fourth model
- Fourth model is runed with the new images to see if there is an improvments in the results
- Confusion matrix is built for the fourth model (the most efficient), to see numerically the confusion between classes of images

Also, regularly the predictions of the models (probabilities of classes for images) were checked randomly


### Libraries
os
pandas
numpy
matplotlib.pyplot
seaborn
sns.set()
tensorflow
keras
load_img from keras.preprocessing.image
img_to_array from keras.preprocessing.image
train_test_split from sklearn.model_selection
ImageDataGenerator from keras.preprocessing.image
confusion_matrix from sklearn.metrics

### Code details

TBD

### Links

Source page : https://www.kaggle.com/puneet6060/intel-image-classification

Presentation : https://docs.google.com/presentation/d/1fudDVPBgNvsxvW_Dswt_IH-L0lTufoGXpRqsyF5vOWQ/edit?usp=sharing

