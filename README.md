
# Image Classification

This repository contains the implementation of a image classification model for 25 different images , specifically designed to handle a dataset with 25 classes. The model has been modified to accommodate input images of size 416x416.

## Table of Contents

1. [Model Architecture](#model-architecture)
2. [Dataset Preparation](#dataset-preparation)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Saving the Model](#saving-the-model)
6. [Plotting Metrics](#plotting-metrics)
7. [Usage](#usage)

## Model Architecture


- **Input:** Images of size 416x416.
- **Output:** 25 classes for classification.
- **Total Parameters:** 11,189,337.

### Key Components

- **Conv2d:** Initial convolution layer with 9,408 parameters.
- **BatchNorm2d:** Batch normalization applied after each convolution.
- **BasicBlock:** The core building block of the model, contains two convolutional layers followed by batch normalization and a residual connection.

## Model Architecture


    =================================================================
    Layer (type:depth-idx)                   Param #
    =================================================================
    ├─Conv2d: 1-1                            9,408
    ├─BatchNorm2d: 1-2                       128
    ├─Sequential: 1-3                        --
    |    └─BasicBlock: 2-1                   --
    |    |    └─Conv2d: 3-1                  36,864
    |    |    └─BatchNorm2d: 3-2             128
    |    |    └─Conv2d: 3-3                  36,864
    |    |    └─BatchNorm2d: 3-4             128
    |    |    └─Sequential: 3-5              --
    |    └─BasicBlock: 2-2                   --
    |    |    └─Conv2d: 3-6                  36,864
    |    |    └─BatchNorm2d: 3-7             128
    |    |    └─Conv2d: 3-8                  36,864
    |    |    └─BatchNorm2d: 3-9             128
    |    |    └─Sequential: 3-10             --
    ├─Sequential: 1-4                        --
    |    └─BasicBlock: 2-3                   --
    |    |    └─Conv2d: 3-11                 73,728
    |    |    └─BatchNorm2d: 3-12            256
    |    |    └─Conv2d: 3-13                 147,456
    |    |    └─BatchNorm2d: 3-14            256
    |    |    └─Sequential: 3-15             8,448
    |    └─BasicBlock: 2-4                   --
    |    |    └─Conv2d: 3-16                 147,456
    |    |    └─BatchNorm2d: 3-17            256
    |    |    └─Conv2d: 3-18                 147,456
    |    |    └─BatchNorm2d: 3-19            256
    |    |    └─Sequential: 3-20             --
    ├─Sequential: 1-5                        --
    |    └─BasicBlock: 2-5                   --
    |    |    └─Conv2d: 3-21                 294,912
    |    |    └─BatchNorm2d: 3-22            512
    |    |    └─Conv2d: 3-23                 589,824
    |    |    └─BatchNorm2d: 3-24            512
    |    |    └─Sequential: 3-25             33,280
    |    └─BasicBlock: 2-6                   --
    |    |    └─Conv2d: 3-26                 589,824
    |    |    └─BatchNorm2d: 3-27            512
    |    |    └─Conv2d: 3-28                 589,824
    |    |    └─BatchNorm2d: 3-29            512
    |    |    └─Sequential: 3-30             --
    ├─Sequential: 1-6                        --
    |    └─BasicBlock: 2-7                   --
    |    |    └─Conv2d: 3-31                 1,179,648
    |    |    └─BatchNorm2d: 3-32            1,024
    |    |    └─Conv2d: 3-33                 2,359,296
    |    |    └─BatchNorm2d: 3-34            1,024
    |    |    └─Sequential: 3-35             132,096
    |    └─BasicBlock: 2-8                   --
    |    |    └─Conv2d: 3-36                 2,359,296
    |    |    └─BatchNorm2d: 3-37            1,024
    |    |    └─Conv2d: 3-38                 2,359,296
    |    |    └─BatchNorm2d: 3-39            1,024
    |    |    └─Sequential: 3-40             --
    ├─Linear: 1-7                            12,825
    =================================================================
    Total params: 11,189,337
    Trainable params: 11,189,337
    Non-trainable params: 0
    =================================================================


## Dataset Preparation

### 1. Dataset Structure

Organize your dataset as follows:

    Dataset/
    ├── train/
    │   ├── class1/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   ├── class2/
    │   └── ...
    └── val/
        ├── class1/
        ├── class2/
        └── ...

### 2. Loading the Dataset

Use the following code to load and transform your dataset:


    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4731, 0.4819, 0.4018], std=[0.1925, 0.1915, 0.1963])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4706, 0.4802, 0.4020], std=[0.1907, 0.1898, 0.1950])
        ]),
    }
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4731, 0.4819, 0.4018], std=[0.1925, 0.1915, 0.1963])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4706, 0.4802, 0.4020], std=[0.1907, 0.1898, 0.1950])
        ]),
    }


## Training

Use the provided script to train the model:


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=initial_lr,steps_per_epoch=len(data_loaders['train']), epochs=num_epochs)

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    best_acc = 0.0
    early_stop_counter = 0
    best_model_wts = None
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(data_loaders[phase], desc=f"{phase} - Epoch {epoch+1}"):
                inputs = inputs.to(device, non_blocking=True) #Cuda or CPU
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()
                epoch_train_loss = running_loss / dataset_sizes['train']
                epoch_train_corrects = running_corrects.double() / dataset_sizes['train']
            else:
                epoch_val_loss = running_loss / dataset_sizes['val']
                epoch_val_corrects = running_corrects.double() / dataset_sizes['val']

        train_loss_history.append(epoch_train_loss)
        train_acc_history.append(epoch_train_corrects.item())
        val_loss_history.append(epoch_val_loss)
        val_acc_history.append(epoch_val_corrects.item())

        print(f'Train Loss: {epoch_train_loss:.4f} Accuracy: {epoch_train_corrects:.4f}')
        print(f'Validation Loss: {epoch_val_loss:.4f} Accuracy: {epoch_val_corrects:.4f}')

        if epoch_val_corrects > best_acc:
            best_acc = epoch_val_corrects
            best_model_wts = model.state_dict().copy()
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print("Early stopping done")
            model.load_state_dict(best_model_wts)
            break
## Evaluation

After training, evaluate the model's performance on the validation set:


    best_acc = 0.0
    early_stop_counter = 0
    best_model_wts = None
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')

        for phase in ['train', 'val']:

                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(data_loaders[phase], desc=f"{phase} - Epoch {epoch+1}"):
                inputs = inputs.to(device, non_blocking=True) #Cuda or CPU
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                epoch_val_loss = running_loss / dataset_sizes['val']
                epoch_val_corrects = running_corrects.double() / dataset_sizes['val']

        train_loss_history.append(epoch_train_loss)
        train_acc_history.append(epoch_train_corrects.item())
        val_loss_history.append(epoch_val_loss)
        val_acc_history.append(epoch_val_corrects.item())

        print(f'Train Loss: {epoch_train_loss:.4f} Accuracy: {epoch_train_corrects:.4f}')
        print(f'Validation Loss: {epoch_val_loss:.4f} Accuracy: {epoch_val_corrects:.4f}')
    print('Best Validation Accuracy: {:4f}'.format(best_acc))



## Saving the Model

### 1. Save the Model State

Save the trained model for later use:


    torch.save(model, 'image_classify.pth')


## Usage

To use the trained model for inference:

    model.load_state_dict(torch.load('image_classify.pth'))
    model.eval()

# Predict
    output = model(image)
    _, predicted_class = torch.max(output, 1)
    print(f'Predicted Class: {predicted_class.item()}')

## Plotting Metrics

### 1. Plot Accuracy

Visualize the training and validation accuracy:

    
    plt.plot(range(epochs), train_acc_history, label='Train Accuracy')
    plt.plot(range(epochs), val_acc_history, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Graph')
    plt.legend()

<img src="images\accuracyy.png" alt="Accuracy of model">

### 2. Plot Loss

Similarly, plot the training and validation loss:

    
    plt.plot(range(epochs), train_loss_history, label='Train Loss')
    plt.plot(range(epochs), val_loss_history, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.title('Training and Validation Loss Graph')
    plt.legend()

<img src="images\losss.png" alt="Loss of model">


### 3. Confusion Matrix, ROC and Classification Report

Similarly, plot the training and validation loss:

    predictions, test_labels, probabilities = prediction(test_loader, model, device)

    class_names = test_dataset.classes  
    report = classification_report(test_labels, predictions, target_names=class_names)
    print(report)

    cm_matrix = confusion_matrix(test_labels, predictions) 
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm_matrix, display_labels=class_names)

    n_classes = len(class_names)
    test_labels_one_hot = label_binarize(test_labels, classes=range(n_classes))

    roc_auc = roc_auc_score(test_labels_one_hot, probabilities, average='macro', multi_class='ovr')
    print(f"Macro-averaged ROC-AUC: {roc_auc:.2f}")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))  # Increase the figure size

    cm_display.plot(ax=ax1)
    ax1.set_title("Confusion Matrix")
    ax1.set_xticklabels(class_names, rotation=90)

    for i in range(n_classes):
        RocCurveDisplay.from_predictions(test_labels_one_hot[:, i], probabilities[:, i], ax=ax2, name=class_names[i])
    ax2.set_title("ROC Curves")

    ax3.axis('off')  
    classification_report_str = classification_report(test_labels, predictions, target_names=class_names)
    ax3.text(0.5, 0.5, classification_report_str, horizontalalignment='center', verticalalignment='center', fontsize=12, family='monospace')

    plt.tight_layout()
    plt.show()

    

<img src="images\ROC.png" alt="confusion_matrix_roc_and_classification_report">