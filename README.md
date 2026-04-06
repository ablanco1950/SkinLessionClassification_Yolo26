# SkinLessionClassification_Yolo26
Classification of skin lesions (among 7 classes) using the file https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T  and the Yolo26 model. Only images with a confidence level greater than 0.9 are considered, representing two-thirds of the total.

This indicates that the model has an accuracy greater than 0.9.

Images that do not have a confidence level exceeding 0.9, according to the model, are referenced in a file for manual inspection.

According to the specifications of the download file, the 7 types of injuries to be detected are:

akiec : Actinic keratoses and intraepithelial carcinoma / Bowen’s disease

bkl : benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses

bcc: basal cell carcinoma

df: dermatofibroma

mel: melanoma

nv: melanocytic nevi

vasc: vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage

INSTALLATION:

All packages, if any are missing, can be installed with a simple pip in case the programs indicate their absence in the environment.
If not yet installed, this packages are:

pip install numpy

pip install pandas

pip install keras

pip install tensorflow

pip install opencv-python

pip install ultralytics

pip install scikit-learn

Download all the files that accompany this project in a single folder.

By downloading the file from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T   in the directory where the project is located, a file called dataverse_files.zip is obtained, which once decompressed as dataverse_files contains, among others, the files HAM10000_images_part1.zip and HAM10000_images_part2.zip, which once unzipped must be unified into a single HAM10000_images folder (through a simple copy and paste) in the same dataverse_files directory

In that folder: dataverse_files, the file ISIC2018_Task3_Test_Images.zip must be descompressed, which produces two nested directories named each one ISIC2018_Task3_Test_Images with 1115 images to test

Next, the structure necessary for the operation of yolo26 is created, consisting of a folder Dir_SkinCancer_yolo26 from which a folder called train and another called valid hang, each with a subfolder for each of the 7 classes, by executing:

python Create_DirSkinCancer_Yolo26.py

This structure is then filled from the images contained in dataverse_files\HAM10000_images and following the order indicated in the file dataverse_files\HAM10000_metadata, by executing:

python Fill_DirSkinCancer_Yolo26.py

Next, the structure necessary for the operation of yolo26 is created with the specific test file, consisting of a folder Dir_Test_SkinCancer_Yolo26 from which a folder called test hangs with subfolders for each of the 7 classes, by executing:

python Create_Test_DirSkinCancer_Yolo26.py

This structure is then filled from the images contained in dataverse_files\ISIC2018_Task3_Test_Images\ISIC2018_Task3_Test_Images and following the order indicated in the file dataverse_files\ISIC2018_Task3_Test_GroundTruth.csv, by executing:

python Fill_Test_DirSkinCancer_Yolo26.py

To avoid yolo26 errors, when you find a valid folder in which one of its subfolders does not have images, unzip the attached valid.zip and copy the resulting valid folder  over the folder Dir_SkinCancer_Yolo26 overwriting the old one, this way you ensure that all valid subfolders have at least some image.

EVALUATION

python Evaluate_SkinLessions_Yolo26.py

The model displays a list of images whose classes were correctly predicted and those that were not, resulting in an accuracy of 0.9275220372184133 (hit predictions / (hit predictions + error predictions)).

It also produces the confusion matrix and the classification report. Images with a prediction confidence level below 0.9 are not considered and appear in the SkinImagesRejected.txt file.

              precision    recall  f1-score   support

       akiec       0.60      0.64      0.62        14
       
         bcc       0.94      0.81      0.87        42
         
         bkl       0.89      0.87      0.88       124
         
          df       0.88      0.92      0.90        24
          
         mel       0.78      0.72      0.75        87
         
          nv       0.96      0.98      0.97       708
          
        vasc       0.89      0.73      0.80        22
        

    accuracy                           0.93      1021
    
     macro avg       0.85      0.81      0.83      1021
   
weighted avg       0.93      0.93      0.93      1021
    
  

  

TRAINING

The best.pt model was obtained through training

python Train_SkinLessions_Yolo26.py.

The LOGTrainSkinLessionClassification_50epoch.docx file, containing the 50-epoch log of the training process, is also included.

REFERENCES:

https://dataverse.harvard.edu/dataset.xhtml?

https://github.com/ablanco1950/SkinLesionDetection_Resnet_Pytorch
