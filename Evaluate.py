from sklearn.metrics import mean_squared_error, multilabel_confusion_matrix, accuracy_score, precision_score, \
    recall_score, f1_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import numpy as np

import math
import warnings

from tensorflow.python.ops.metrics_impl import recall

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def tt(value):
    power = math.ceil(math.log10(value) - 1)
    A1 = 100 ** (math.log10(value) - power)
    return A1


def main_est_parameters(y_true, pred):
    """
    :param y_true: true labels
    :param pred: predicted labels
    :return: performance metrics in list dtype
    """
    cm = multilabel_confusion_matrix(y_true, pred)
    cm = sum(cm)
    TP = cm[0, 0]  # True Positive
    FP = cm[0, 1]  # False Positive
    FN = cm[1, 0]  # False Negative
    TN = cm[1, 1]  # True Negative
    Acc = (TP + TN) / (TP + TN + FP + FN)
    Sen = TP / (TP + FN)
    Spe = TN / (TN + FP)
    Pre = TP / (TP + FP)
    Rec = TP / (TP + FN)
    F1score = 2 * (Pre * Rec) / (Pre + Rec)
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    return [Acc, Sen, Spe, F1score,Rec,Pre,TPR,FPR]


def main_est_parameters_mul(y_true, pred):
    acc = accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred, average='macro', zero_division=0)
    rec = recall_score(y_true, pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, pred, average='macro', zero_division=0)

    cm = confusion_matrix(y_true, pred)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    with np.errstate(divide='ignore', invalid='ignore'):
        specificity = TN / (TN + FP)
        fpr = FP / (FP + TN)
        tpr = TP / (TP + FN)

    spe = np.nanmean(specificity)
    fpr_macro = np.nanmean(fpr)
    tpr_macro = np.nanmean(tpr)

    return [acc, tpr_macro, spe, f1, rec, prec, tpr_macro, fpr_macro]



def Evaluation_Metrics1(y, y_pred):
    mse = tt(mean_squared_error(y, y_pred))
    rmse = np.sqrt(mse)
    mae = tt(mean_absolute_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    cor = np.sqrt(r2)
    return [mse, rmse, mae, r2, cor]


from sklearn.metrics import confusion_matrix

def compute_metrics(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds)
    num_classes = cm.shape[0]

    sensitivity_list = []
    specificity_list = []
    F1_score_list = []
    Pre_list = []
    Rec_list = []
    FPR_list = []

    for i in range(num_classes):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - (TP + FN + FP)

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        Pre = TP / (TP + FP) if (TP + FP) > 0 else 0
        Rec = TP / (TP + FN) if (TP + FN) > 0 else 0
        F1_score = 2 * (Pre * Rec) / (Pre + Rec) if (Pre + Rec) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

        F1_score_list.append(F1_score)
        Pre_list.append(Pre)
        Rec_list.append(Rec)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        FPR_list.append(FPR)

    accuracy = np.trace(cm) / np.sum(cm)
    avg_precision = np.mean(Pre_list)
    avg_recall = np.mean(Rec_list)
    avg_f1_score = np.mean(F1_score_list)
    avg_sensitivity = np.mean(sensitivity_list)
    avg_specificity = np.mean(specificity_list)
    TPR = avg_recall
    avg_FPR = np.mean(FPR_list)

    return [accuracy, avg_sensitivity, avg_specificity, avg_f1_score,
            avg_recall, avg_precision, TPR, avg_FPR]


def compute_metrics2(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds)

    num_classes = cm.shape[0]
    sensitivity_list = []
    specificity_list = []
    F1_score_list = []
    Pre_list = []
    Rec_list = []
    FPR_list=[]

    for i in range(num_classes):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - (TP + FN + FP)

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        Pre = TP / (TP + FP) if (TP + FP) > 0 else 0
        Rec = TP / (TP + FN) if (TP + FN) > 0 else 0
        F1_score = 2 * (Pre * Rec) / (Pre + Rec) if (Pre + Rec) > 0 else 0
        FPR=FP/(FP+TN) if (FP+TN)>0 else 0

        F1_score_list.append(F1_score)
        Pre_list.append(Pre)
        Rec_list.append(Rec)
        sensitivity_list.append(sensitivity)
        FPR_list.append(FPR)
        specificity_list.append(specificity)

    # Accuracy: total correct predictions / total predictions
    accuracy = np.trace(cm) / np.sum(cm)
    accuracy =accuracy

    avg_precision = np.mean(Pre_list)
    avg_precision = avg_precision
    avg_recall = np.mean(Rec_list)
    avg_recall = avg_recall
    avg_f1_score = np.mean(F1_score_list)
    avg_f1_score = avg_f1_score
    avg_sensitivity = np.mean(sensitivity_list)
    avg_sensitivity = avg_sensitivity
    avg_specificity = np.mean(specificity_list)
    avg_specificity = avg_specificity
    TPR = avg_recall
    avg_FPR= np.mean(FPR_list)


    return [accuracy, avg_sensitivity, avg_specificity, avg_f1_score, avg_recall, avg_precision,TPR,avg_FPR]


import numpy as np
from collections import Counter


def process_data(features, labels, max_samples=2000):
    # Initialize an empty list to store the processed features and labels
    processed_features = []
    processed_labels = []

    # Get the unique labels
    unique_labels = np.unique(labels)

    # Random seed to ensure reproducibility if needed
    np.random.seed(42)

    # Loop through each unique label in a complicated manner
    for label in unique_labels:
        # Get all indices where the label matches
        indices = np.where(labels == label)[0]
        num_samples = len(indices)

        # Select data based on the number of samples for the label
        # If the number of samples exceeds the threshold, sample it down
        if num_samples > max_samples:
            # Perform random sampling to limit the number of samples to max_samples
            sampled_indices = np.random.choice(indices, max_samples, replace=False)
        else:
            # Keep all the samples if below the threshold
            sampled_indices = indices

        # Fetch the corresponding feature data for the sampled indices
        sampled_features = features[sampled_indices]
        sampled_labels = labels[sampled_indices]

        # Append to the processed features and labels list
        processed_features.append(sampled_features)
        processed_labels.append(sampled_labels)

    # Convert processed features and labels lists into numpy arrays
    processed_features = np.vstack(processed_features)
    processed_labels = np.concatenate(processed_labels)

    # Shuffle the final dataset
    shuffle_indices = np.random.permutation(len(processed_labels))
    processed_features = processed_features[shuffle_indices]
    processed_labels = processed_labels[shuffle_indices]

    # Return the processed features and labels
    return processed_features, processed_labels



def balance_features(feat01, feat02, feat03, label01, max_samples_per_class=200):
    assert feat01.shape[0] == feat02.shape[0] == feat03.shape[0] == label01.shape[0], \
        "Mismatch in sample counts among features or labels."

    unique_classes = np.unique(label01)
    selected_indices = []

    for cls in unique_classes:
        cls_indices = np.where(label01 == cls)[0]
        if len(cls_indices) > max_samples_per_class:
            chosen = np.random.choice(cls_indices, max_samples_per_class, replace=False)
        else:
            chosen = cls_indices
        selected_indices.extend(chosen)

    selected_indices = np.array(selected_indices)
    np.random.shuffle(selected_indices)  # Optional: shuffle the final result

    # Apply the same indices to all features and labels
    feat01_balanced = feat01[selected_indices]
    feat02_balanced = feat02[selected_indices]
    feat03_balanced = feat03[selected_indices]
    label_balanced = label01[selected_indices]

    print(f"✅ Balanced shapes: feat01 {feat01_balanced.shape}, feat02 {feat02_balanced.shape}, "
          f"feat03 {feat03_balanced.shape}, labels {label_balanced.shape}")

    return feat01_balanced, feat02_balanced, feat03_balanced, label_balanced
