# -*- coding: utf-8 -*-
"""

 Alfonso Blanco García , apr 2026
"""
######################################################################
# https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
######################################################################

CONF =0.9

from ultralytics import YOLO
# Load YOLO26 classification model (pretrained on ImageNet-like dataset)
#model = YOLO("runs\\classify\\train3\\weights\\best.pt")  # Ensure you have this model file or let Ultralytics auto-download it
model = YOLO("best.pt")  # Ensure you have this model file or let Ultralytics auto-download it


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import  ConfusionMatrixDisplay


TabSkinCancerComplete =[
            'akiec : Actinic keratoses and intraepithelial carcinoma / Bowen s disease', 
            'bkl : benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses',
            'bcc : basal cell carcinoma',
            'df : dermatofibroma',
            'mel : melanoma', 
            'nv : melanocytic nevi',
            'vasc : vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage,'    
           
           ]
TabSkinCancer =[ 'akiec',
            'bcc'  ,
            'bkl',      
            'df',       
            'mel',      
            'nv',    
            'vasc']
import torch
from torch import nn
import os
import re

import cv2

import numpy as np
import keras
import functools  
import time
inicio=time.time()

from torchvision import datasets, transforms, models
import torchvision.models as models
from PIL import Image


def find_classes(dir):
    classes = os.listdir(dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

classes, c_to_idx = find_classes('Dir_Test_SkinCancer_Yolo26/test')


print(classes, c_to_idx) 


def loadimagesTest():
    imgpath = "Dir_Test_SkinCancer_Yolo26\\test\\"

    print("Reading imagenes from ",imgpath)

    TabDirName=[]
    for root, dirnames, filenames in os.walk(imgpath):
         for dirname in dirnames:  
             print(dirname)
             TabDirName.append(dirname)

    TotImages=0
   
    TotImagesValid=0
    TabImagePath = []
    NameImages=[]
    Y=[]
    
    for i in range(len(TabDirName)):
    
        imgpath1=imgpath+ str(TabDirName[i])+"\\"
        print(imgpath1)
        
        # https://stackoverflow.com/questions/62137343/how-to-get-full-path-with-os-walk-function-in-python
        for root, dirnames, filenames in os.walk(imgpath1):
            for filename in filenames:  
                #print(filename)
                #if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            
                filepath = os.path.join(root, filename)
                # https://stackoverflow.com/questions/51810407/convert-image-into-1d-array-in-python
            
                #image = cv2.imread(filepath)
                #cv2.imshow("image",image)
                #cv2.waitKey(0)
                
                TabImagePath.append(filepath)
                NameImages.append(filename)
                Y.append(TabDirName[i])       
                TotImages+=1
                
                
    print( " Total images to test "  + str(TotImages))     

    return TabImagePath, Y, NameImages

# asking IA Bing 
# python example object classification with yolo26 
def classify_image(image_path):
    """
    Classify an image using YOLO26 pretrained classification model.
    """
    # Validate file path
    if not os.path.isfile(image_path):
        print(f"Error: File '{image_path}' not found.")
        return

    try:
      
        # Run inference
        results = model(image_path, verbose=False)

        # Display results
        for result in results:
            # result.probs contains class probabilities
            top_class_index = int(result.probs.top1)
            top_class_name = result.names[top_class_index]
            confidence = float(result.probs.top1conf)

            #print(f"Predicted Class: {top_class_name} (Confidence: {confidence:.2%})")

    except Exception as e:
        print(f"Error during classification: {e}")
    return top_class_index, top_class_name,confidence


###########################################################
# MAIN
##########################################################



TabImagePath, Y_test_confidence, imageName_test=loadimagesTest()
#print(TabImagePath)
#print(Y_test)

TotalHits=0
TotalFailures=0
TotalRejected=0
with open( "SkinImagesRejected.txt" ,"w") as  w:
    
    
    TabPredictions=[]
    TabConfidence=[]
    Y_test=[]
    for i in range(len(TabImagePath)):
        
        top_class_index, NameSkinCancerPredicted,confidence =classify_image(TabImagePath[i])
        #print(top_class_index)


        if confidence < CONF:
            lineaw=[]
            lineaw.append(imageName_test[i]) 
            lineaw.append(Y_test_confidence[i])
            lineaw.append(str(confidence))
            #NameSkinCancerTrue=Y_test_confidence[i]
            #lineaw.append(NameSkinCancerTrue)
            #lineaw.append(NameSkinCancerPredicted)
            lineaWrite =','.join(lineaw)
            lineaWrite=lineaWrite + "\n"
            w.write(lineaWrite)
            TotalRejected=TotalRejected+1
            continue

        
        TabConfidence.append(confidence)
        TabPredictions.append(NameSkinCancerPredicted)
        Y_test.append(Y_test_confidence[i])
        #TabPredictions.append(top_class_index)
        #print(TabPredictions)
        
        TabConfidence.append(confidence)        
        NameSkinCancerTrue=Y_test_confidence[i]
        
             
        
        
        if NameSkinCancerPredicted!=NameSkinCancerTrue:
            TotalFailures=TotalFailures + 1
            print("ERROR " + imageName_test[i]+ " is assigned Model " +  NameSkinCancerPredicted
                  + "  True Model " +  NameSkinCancerTrue)
                  
        else:
            print("HIT " + imageName_test[i]+ " is assigned model " +   NameSkinCancerPredicted)
                
          
                 
          
            TotalHits=TotalHits+1
        #lineaw=[]
        #lineaw.append(imageName_test[i]) 
        #lineaw.append(Y_test_confidence[i])
        #lineaw.append(NameSkinCancerTrue)
        #lineaw.append(NameSkinCancerPredicted)
        #lineaWrite =','.join(lineaw)
        #lineaWrite=lineaWrite + "\n"
        #w.write(lineaWrite)

    
print("")
print("Total hits = " + str(TotalHits))  
print("Total failures = " + str(TotalFailures) )     
print("Accuracy = " + str(TotalHits*100/(TotalHits + TotalFailures)) + "%")

print("")
print("Total Rejected because low confidence  " + str(TotalRejected))
print("")
#print(Y_test[0])
#print(TabPredictions[0])

print("Accuracy with unseen data:", accuracy_score(Y_test, TabPredictions))
print(classification_report(Y_test, TabPredictions,target_names=classes))

# Compute confusion matrix
cm = confusion_matrix(Y_test, TabPredictions)
print("")
# Optional: print raw matrix
print("Confusion Matrix (raw values):\n", cm)

import matplotlib.pyplot as plt
# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Skin Lessions Classification")
plt.show()
print("")



