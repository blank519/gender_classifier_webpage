# Gender Classifier Webpage
A basic Django web application which allows users to upload an image with a person's face and determines that person's gender.

## Environment
python==3.8.10  
PyTorch==1.9.1  
Django==4.2.9  

## Details
The web application runs on [http://localhost:8000/](http://localhost:8000/).  
The model used to predict gender is based on Transfer Learning with ResNet-18, trained on 58,658 images.  
Dataset: [Kaggle Gender Classification Dataset](https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset)

## Next Steps
- Training on a more diverse set of images with regards to age, ethnicity, etc, in order to improve overall accuracy
- Deployment