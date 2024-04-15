import shutil
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import os
from main_model import MainModel
from variant1 import Variant1
from variant2 import Variant2

def split_dataset(dataset_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    train_dir = '../Education-MindA.I.lytics/train'
    val_dir = '../Education-MindA.I.lytics/val'
    test_dir = '../Education-MindA.I.lytics/test'

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_files = []
    val_files = []
    test_files = []

    for cls in os.listdir(dataset_path):
        cls_files = [os.path.join(dataset_path, cls, file) for file in os.listdir(os.path.join(dataset_path, cls))]

        train_val_cls_files, test_cls_files = train_test_split(cls_files, test_size=test_ratio,
                                                               random_state=random_state)
        train_cls_files, val_cls_files = train_test_split(train_val_cls_files,
                                                          test_size=val_ratio / (train_ratio + val_ratio),
                                                          random_state=random_state)

        for file in train_cls_files:
            new_path = os.path.join(train_dir, cls, os.path.basename(file))
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            shutil.move(file, new_path)
            train_files.append(new_path)
        for file in val_cls_files:
            new_path = os.path.join(val_dir, cls, os.path.basename(file))
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            shutil.move(file, new_path)
            val_files.append(new_path)
        for file in test_cls_files:
            new_path = os.path.join(test_dir, cls, os.path.basename(file))
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            shutil.move(file, new_path)
            test_files.append(new_path)

    return train_files, val_files, test_files


def evaluate_model(model, data_loader, classes):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            predicted_classes = torch.argmax(outputs, dim=1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted_classes.numpy())

    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro')
    precision_micro = precision_score(y_true, y_pred, average='micro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')

    #plot_confusion_matrix(cm, classes, normalize=True)

    print("Confusion Matrix:")
    print(cm)
    print("Accuracy:", accuracy)
    print("Precision (Macro):", precision_macro)
    print("Precision (Micro):", precision_micro)
    print("Recall (Macro):", recall_macro)
    print("Recall (Micro):", recall_micro)
    print("F1-score (Macro):", f1_macro)
    print("F1-score (Micro):", f1_micro)

    return {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'precision_micro': precision_micro,
        'recall_macro': recall_macro,
        'recall_micro': recall_micro,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro
    }


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    #set normalize = True to create a normalized confusion matrix
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def k_fold(dataset_loader, num_epochs=10, k_folds=10, batch_size=32):

    kf = KFold(n_splits=k_folds, shuffle=True)

    total_accuracy = 0
    total_macro_precision = 0
    total_macro_recall = 0
    total_macro_f1_score = 0
    total_micro_precision = 0
    total_micro_recall = 0
    total_micro_f1_score = 0

    for fold, (train_indices, test_indices) in enumerate(kf.split(dataset_loader.dataset)):

        train_indices, val_indices = train_test_split(train_indices, test_size=0.15, random_state=42)
        train_dataset = torch.utils.data.Subset(dataset_loader.dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset_loader.dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset_loader.dataset, test_indices)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = Variant1()
        model.train_model(train_loader, val_loader, num_epochs=num_epochs, save_path_best=f'models/final_best_fold_{fold}', save_path_last=f'models/final_last_fold_{fold}')

        print(f'Fold {fold + 1}/{k_folds}:')
        best_model = Variant1()
        best_model.load_model(f'models/final_best_fold_{fold}')
        test_metrics = evaluate_model(best_model, test_loader, classes)

        total_accuracy += test_metrics["accuracy"]
        total_macro_precision += test_metrics["precision_macro"]
        total_macro_recall += test_metrics["recall_macro"]
        total_macro_f1_score += test_metrics["f1_macro"]
        total_micro_precision += test_metrics["precision_micro"]
        total_micro_recall += test_metrics["recall_micro"]
        total_micro_f1_score += test_metrics["f1_micro"]

    avg_accuracy = total_accuracy / k_folds
    avg_macro_precision = total_macro_precision / k_folds
    avg_macro_recall = total_macro_recall / k_folds
    avg_macro_f1_score = total_macro_f1_score / k_folds
    avg_micro_precision = total_micro_precision / k_folds
    avg_micro_recall = total_micro_recall / k_folds
    avg_micro_f1_score = total_micro_f1_score / k_folds

    print(f'Average across all folds:')
    print(f'  Accuracy: {avg_accuracy}')
    print(f'  Macro Precision: {avg_macro_precision}')
    print(f'  Macro Recall: {avg_macro_recall}')
    print(f'  Macro F1-score: {avg_macro_f1_score}')
    print(f'  Micro Precision: {avg_micro_precision}')
    print(f'  Micro Recall: {avg_micro_recall}')
    print(f'  Micro F1-score: {avg_micro_f1_score}')

if __name__ == "__main__":
    dataset_path = 'Raw_Dataset'
    classes = ['Angry', 'Focused', 'Neutral', 'Surprised']

    # create train, val, and test files
    # train_files, val_files, test_files = split_dataset(dataset_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    #create DataLoader objects for train, val, and test sets
    train_dataset = datasets.ImageFolder('train', transform=transform)
    val_dataset = datasets.ImageFolder('val', transform=transform)
    test_dataset = datasets.ImageFolder('test', transform=transform)
    dataset = datasets.ImageFolder('Dataset', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    dataset_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    #initialize  model
    model = Variant1()

    #train and save model
    # model.train_model(train_loader, val_loader, save_path_best='models/best_model_layers2_kernel7x7', save_path_last='models/last_model_layers2_kernel7x7')
    # model.save_model('model_layers2_kernel7x7_v2')

    k_fold(dataset_loader)
    #evaluate model using k-fold


    # #load the model for evaluation
    # loaded_model = Variant2()
    # loaded_model.load_model('models/best_model_layers2_kernel7x7')
    #
    # loaded_model.inference('test/Angry/images - 2020-11-06T001050.259_face.png', classes)

    #evaluate the loaded model on train, val, and test sets
    # train_metrics = evaluate_model(loaded_model, train_loader, classes)
    # val_metrics = evaluate_model(loaded_model, val_loader, classes)
    # test_metrics = evaluate_model(loaded_model, test_loader, classes)
