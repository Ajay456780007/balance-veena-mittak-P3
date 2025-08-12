import cv2
import os
import numpy as np
from Feature_Extraction.Deep_structural_pattern import Deep_Structural_Pattern
from Feature_Extraction.Deep_color_based_pattern import Deep_color_based_pattern
from Feature_Extraction.Resnet151 import Resnet151
from Feature_Extraction.Statistical_green import Statistical_Green
from Feature_Extraction.ROI import ROI_Extraction
from Models.Gan import HSAGAN

def Feature_Extraction(img):
    DCBP = Deep_color_based_pattern(img)
    DSP = Deep_Structural_Pattern(img)
    R151 = Resnet151(img)
    SGP = Statistical_Green(img)
    return DCBP, DSP, R151, SGP


def Preprocessing(DB,save=True):
    if DB == "DB1":
        print("Processing Dataset1........................")
        folder_path1 = "Dataset/DB1/archive/Bacterialblight"
        folder_path2 = "Dataset/DB1/archive/Blast"
        folder_path3 = "Dataset/DB1/archive/Brownspot"
        folder_path4 = "Dataset/DB1/archive/Tungro"

        image_names1 = [names for names in os.listdir(folder_path1)]
        image_names2 = [names for names in os.listdir(folder_path2)]
        image_names3 = [names for names in os.listdir(folder_path3)]
        image_names4 = [names for names in os.listdir(folder_path4)]

        data1, data2, data3, data4 = [], [], [], []
        label1, label2, label3, label4 = [], [], [], []

        for i in image_names1:
            img_path = os.path.join(folder_path1, i)
            img1 = cv2.imread(img_path)
            img1 = cv2.resize(img1, (224, 224))
            roi= ROI_Extraction(img1)
            DCBP, DSP, R151, SGP = Feature_Extraction(roi)
            Extracted_features = np.concatenate((DCBP, DSP, R151, SGP), axis=0)
            Extracted_features = np.array(Extracted_features)
            data1.append(Extracted_features)
            label1.append(0)

        for j in image_names2:
            img_path = os.path.join(folder_path2, j)
            img2 = cv2.imread(img_path)
            img2 = cv2.resize(img2, (224, 224))
            roi = ROI_Extraction(img2)
            DCBP, DSP, R151, SGP = Feature_Extraction(roi)
            Extracted_features = np.concatenate((DCBP, DSP, R151, SGP), axis=0)
            Extracted_features = np.array(Extracted_features)
            data2.append(Extracted_features)
            label2.append(1)

        for l in image_names3:
            img_path = os.path.join(folder_path3, l)
            img3 = cv2.imread(img_path)
            img3 = cv2.resize(img3, (224, 224))
            roi = ROI_Extraction(img3)
            DCBP, DSP, R151, SGP = Feature_Extraction(roi)
            Extracted_features = np.concatenate((DCBP, DSP, R151, SGP), axis=0)
            Extracted_features = np.array(Extracted_features)
            data3.append(Extracted_features)
            label3.append(2)

        for k in image_names4:
            img_path = os.path.join(folder_path4, k)
            img4 = cv2.imread(img_path)
            img4 = cv2.resize(img4, (224, 224))
            roi = ROI_Extraction(img4)
            DCBP, DSP, R151, SGP = Feature_Extraction(roi)
            Extracted_features = np.concatenate((DCBP, DSP, R151, SGP), axis=0)
            Extracted_features = np.array(Extracted_features)
            data4.append(Extracted_features)
            label4.append(3)

        data_1 = np.concatenate((data1, data2, data3, data4), axis=1)
        label_1 = np.concatenate((label1, label2, label3, label4), axis=1)


        if save:
            np.save(f"Data_loader/DB1/{DB}data.npy", data_1)
            np.save(f"Data_loader/DB1/{DB}label.npy", label_1)

        print("The data saved successfully for Dataset1 🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️........")

    if DB == "DB3":
        print("Processing Dataset3........................")
        folder_path1 = "Dataset/DB3/rice_leaf_diseases/Bacterial leaf blight/"
        folder_path2 = "Dataset/DB3/rice_leaf_diseases/Brown spot/"
        folder_path3 = "Dataset/DB3/rice_leaf_diseases/Leaf smut/"

        image_names1 = [names for names in os.listdir(folder_path1)]
        image_names2 = [names for names in os.listdir(folder_path2)]
        image_names3 = [names for names in os.listdir(folder_path3)]
        data1, data2, data3 = [], [], []
        label1, label2, label3 = [], [], []
        for i in image_names1:
            img_path = os.path.join(folder_path1, i)
            img1 = cv2.imread(img_path)
            img1 = cv2.resize(img1, (224, 224))
            # img1_array = np.array(img1, dtype="float")
            # img1_array = img1_array / 255.0
            roi = ROI_Extraction(img1)
            DCBP, DSP, R151, SGP = Feature_Extraction(roi)
            Extracted_features = np.concatenate((DCBP, DSP, R151, SGP), axis=0)
            Extracted_features = np.array(Extracted_features)
            data1.append(Extracted_features)
            label1.append(0)

        for j in image_names2:
            img_path = os.path.join(folder_path2, j)
            img2 = cv2.imread(img_path)
            img2 = cv2.resize(img2, (224, 224))
            roi = ROI_Extraction(img2)
            DCBP, DSP, R151, SGP = Feature_Extraction(roi)
            Extracted_features = np.concatenate((DCBP, DSP, R151, SGP), axis=0)
            Extracted_features = np.array(Extracted_features)
            data2.append(Extracted_features)
            label2.append(1)

        for l in image_names3:
            img_path = os.path.join(folder_path3, l)
            img3 = cv2.imread(img_path)
            img3 = cv2.resize(img3, (224, 224))
            roi = ROI_Extraction(img3)
            DCBP, DSP, R151, SGP = Feature_Extraction(roi)
            Extracted_features = np.concatenate((DCBP, DSP, R151, SGP), axis=0)
            Extracted_features = np.array(Extracted_features)
            data3.append(Extracted_features)
            label3.append(2)

        data_2 = np.concatenate((data1, data2, data3), axis=1)
        label_2 = np.concatenate((label1, label2, label3), axis=1)

        if save:
            np.save(f"Data_loader/DB1/{DB}data.npy", data_2)
            np.save(f"Data_loader/DB1/{DB}label.npy", label_2)
        print("The data saved successfully for Dataset3 🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️........")

    if DB == "DB2":
        print("Processing Dataset2........................")
        folder_path1 = "Dataset/DB2/Rice Leaf Diseases Dataset/rice leaf diseases dataset/rice leaf diseases dataset/Bacterialblight"
        folder_path2 = "Dataset/DB2/Rice Leaf Diseases Dataset/rice leaf diseases dataset/rice leaf diseases dataset/Brownspot"
        folder_path3 = "Dataset/DB2/Rice Leaf Diseases Dataset/rice leaf diseases dataset/rice leaf diseases dataset/Leafsmut"

        img_names1 = [names for names in os.listdir(folder_path1)]
        img_names2 = [names for names in os.listdir(folder_path2)]
        img_names3 = [names for names in os.listdir(folder_path3)]

        data1, data2, data3 = [], [], []
        label1, label2, label3 = [], [], []
        for i in img_names1:
            img_path = os.path.join(folder_path1, i)
            img1 = cv2.imread(img_path)
            img1 = cv2.resize(img1, (224, 224))
            roi = ROI_Extraction(img1)
            DCBP, DSP, R151, SGP = Feature_Extraction(roi)
            # Extracted_features = np.concatenate((DCBP, DSP, R151, SGP), axis=0)
            # Extracted_features = np.array(Extracted_features)
            data1.append(np.array(roi))
            label1.append(0)

        for j in img_names2:
            img_path = os.path.join(folder_path2, j)
            img2 = cv2.imread(img_path)
            img2 = cv2.resize(img2, (897, 3081))
            roi = ROI_Extraction(img2)
            # DCBP, DSP, R151, SGP = Feature_Extraction(roi)
            # Extracted_features = np.concatenate((DCBP, DSP, R151, SGP), axis=0)
            # Extracted_features = np.array(Extracted_features)
            data2.append(np.array(roi))
            label2.append(1)

        for l in img_names3:
            img_path = os.path.join(folder_path3, l)
            img3 = cv2.imread(img_path)
            img3 = cv2.resize(img3, (897, 3081))
            roi = ROI_Extraction(img3)
            # DCBP, DSP, R151, SGP = Feature_Extraction(roi)
            # Extracted_features = np.concatenate((DCBP, DSP, R151, SGP), axis=0)
            # Extracted_features = np.array(Extracted_features)
            data3.append(np.array(roi))
            label3.append(2)

        data_3 = np.concatenate((data1, data2, data3), axis=1)
        label_3 = np.concatenate((label1, label2, label3), axis=1)
        print("The shape of the data is:",data_3.shape)
        print("The unique values in the labels is:",np.unique(label_3))

        HSAGAN(data_3,label2,DB)

        data_final3=[]
        label_final_3=[]
        for i in range(data_3.shape[0]):
            DCBP, DSP, R151, SGP = Feature_Extraction(data_3[i])
            Extracted_features = np.concatenate((DCBP, DSP, R151, SGP), axis=0)
            Extracted_features = np.array(Extracted_features)
            data_final3.append(Extracted_features)


        if save:
            np.save(f"Data_loader/DB1/{DB}data.npy", data_final3)
            np.save(f"Data_loader/DB1/{DB}label.npy", label_3)

        print("The data saved successfully for Dataset2 🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡")