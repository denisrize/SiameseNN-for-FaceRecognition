# SiameseNN-for-FaceRecognition: A Siamese neural network for one-shot facial recognition

## Authors

- [**Denis Rize**](https://github.com/denisrize)
- [**Adir Serruya**](https://github.com/Adirser)

## Table of Contents
- [Project Overview](#project-overview)
  - [Key Objectives](#key-objectives)
- [Data Analysis](#data-analysis)
- [Utils](#utils)
- [Loss Function – Contrastive Loss](#loss-function--contrastive-loss)
  - [How It Works](#how-it-works)
- [Data Preprocessing](#data-preprocessing)
  - [Transformations](#transformations)
  - [Dataset Instances](#dataset-instances)
  - [Data Loaders](#data-loaders)
- [Hyperparameters and Network Architecture](#hyperparameters-and-network-architecture)
  - [Original Architecture Insights](#original-architecture-insights)
  - [Chosen Hyperparameters](#chosen-hyperparameters)
  - [Regularization Methods](#regularization-methods)
- [Results](#results)
  - [Best Results – Train and Validation](#best-results--train-and-validation)
  - [ROC-AUC Curve](#roc-auc-curve)
  - [Best Run Accurate and Misclassified Examples](#best-run-accurate-and-misclassified-examples)
  - [Best Run with Linear Layers Replacement Result](#best-run-with-linear-layers-replacement-result)
- [Summary Table](#summary-table)
- [Conclusion](#conclusion)
- 
## Project Overview

The purpose of this assignment is to build a convolutional neural network (CNN) and apply it to a real-world problem—facial recognition—using one-shot learning. This assignment is based on the paper [**Siamese Neural Networks for One-shot Image Recognition**](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

### Key Objectives

1. **Implement a Convolutional Neural Network**:
   - Build a Siamese network architecture to perform one-shot learning for facial recognition.
   - Use the Labeled Faces in the Wild (LFW-a) dataset to train and test the model.

2. **Apply One-Shot Learning**:
   - The goal is to determine whether two facial images of previously unseen individuals represent the same person.
   - Follow the architecture and concepts outlined in the referenced scientific paper, while also exploring potential modifications.
   - 
![siamses](https://github.com/user-attachments/assets/eddcd7b1-3350-4445-bc9f-5b2905f5729d)


## Data Analysis
We are using the Labeled Faces in the Wild (LFW-a) dataset for this task which consists of 2200 training examples and 1000 test examples.
We divided our training into 2 sets – Training and Validation (80% - 20% division) so we are left with the following composition:

<img src="https://github.com/user-attachments/assets/4e187ced-7740-48ea-8f76-1a82304ae016" alt="sameperson" width="500"/>

In general, the dataset consists of 3095 unique individuals and 6400 face images and the distribution of the unique instances across the training-validation-test sets is as follows:

<img src="https://github.com/user-attachments/assets/6e9b3f57-38c5-47eb-856f-54223456103a" alt="unique" width="500"/>

## Utils
To enhance our model's robustness and performance, we incorporated several utility functions and classes:

- parse_pairs(file_name, data_dir): A function to load image pairs and their labels.
- class Cutout and class NoiseInjection: Data augmentation techniques to introduce regularization and improve the model’s robustness by mimicking real-world variations in training data, thereby preventing overfitting.
- imshow: For visual inspection of image pairs during development, helping in qualitative evaluation and debugging.
- plot_roc_curve: Employed to optimize the decision threshold on the validation set, maximizing accuracy by assessing true positive and false positive rates.

  ## Loss function – Contrastive Loss

  
For our Siamese network, we employ the **Contrastive Loss**, which is well-suited for learning fine-grained similarities and distinctions between pairs of inputs.

### How It Works

- **Euclidean Distance**: The loss function first calculates the Euclidean distance between the feature vectors `output1` and `output2`, which represent the deep features extracted from the two input images by the Siamese network.
  
- **Loss Calculation**:
  - **Similar Pairs (label = 0)**: The loss is proportional to the square of the distance between the outputs. This encourages the network to minimize the distance between feature vectors of similar pairs, making them more alike.
  - **Dissimilar Pairs (label = 1)**: The loss incorporates a margin, a hyperparameter that defines how far apart the feature vectors of dissimilar pairs should be. If the distance between the feature vectors is less than the margin, a penalty is applied proportional to the squared difference between the margin and the actual distance. This ensures that dissimilar pairs are separated by at least the margin in the feature space.
 
## Data Preprocessing

For the facial recognition task using our Siamese network, we employ a series of transformations to prepare the dataset for training and evaluation.

### Transformations

1. **Resize**: Each image is resized to 105x105 pixels to ensure uniformity in input dimensions across all images.
2. **Scaling**: Pixel values are scaled to be between 0 and 1.
3. **Random Horizontal Flip**: Images are randomly flipped horizontally to augment the data and introduce variability.
4. **ToTensor**: Converts the images to PyTorch tensors.
5. **Cutout**: Creates 3 holes in each image, each with a length of 5 pixels, to simulate occlusions.
6. **Noise Injection**: Adds random noise to the images with a mean of 0 and a standard deviation of 0.03.

### Dataset Instances

Using these transformations, we create instances of our custom `FacePairsDataset` class for the training, validation, and testing datasets. Each instance uses paired images (either similar or dissimilar) along with the above transformations to ensure consistency in data preprocessing across different stages of the model lifecycle.

### Data Loaders

- **Training DataLoader**: Batches of 64 images with shuffling enabled to provide a random order of data in each epoch, minimizing overfitting and improving model generalization.
- **Validation DataLoader**: Batches of 64 images with shuffling disabled to maintain consistency during validation.
- **Test DataLoader**: Batches of 64 images with shuffling disabled to maintain consistency during testing.

## Hyperparameters and Network Architecture

The hyperparameters for our Siamese Network were inspired by the architecture outlined in the original Siamese Network paper, which achieved effective results in face recognition tasks. Key considerations included the number of convolutional layers, initial number of filters, pooling types, activation functions, and the use of batch normalization and dropout.

![siameseney](https://github.com/user-attachments/assets/855b60c6-9a07-479a-bed4-e348446b004b)

### Original Architecture Insights

The original Siamese Network employed a progressively deeper and wider architecture:
- **Convolutional Layers**: Started with 64 channels, doubling at each subsequent layer (64 → 128 → 128 → 256).
- **Fully Connected Layers**: Followed the convolutional layers, leading to a high-dimensional representation (e.g., 4096).
- This approach captures increasingly complex features at various levels of abstraction.

### Chosen Hyperparameters

To align our network with the original architecture while exploring the efficiency of smaller configurations, we experimented with the following hyperparameters:

- **Number of Convolutional Layers**: [3, 4]
- **Initial Number of Filters**: [16, 64]
- **Pooling Type**: [Max pooling, Average pooling]
- **Activation Functions**: [ReLU, Leaky ReLU (default negative slope)]
- **Batch Normalization**: [True, False]
- **Learning Rate**: [0.005, 0.001]

These configurations allowed us to experiment with different architectural choices while maintaining a close resemblance to the effective design principles from the original paper.

### Regularization Methods

To enhance the robustness and generalization capability of our Siamese Network for face recognition, we employed several regularization techniques:

1. **Random Horizontal Flip**: Randomly flips images horizontally with a probability of 0.5 to simulate real-world facial orientation variations.
2. **Cutout**: Randomly masks out three square patches of 5 pixels each from the image, forcing the network to learn from the surrounding context.
3. **Noise Injection**: Adds Gaussian noise (mean = 0, std = 0.03) to images to simulate real-world variations and encourage the learning of robust features.
4. **Dropout**: Applies Dropout with a rate of 0.05 after each convolutional and fully connected layer to prevent overfitting by omitting random neurons during training.
5. **L2 Regularization**: Uses weight decay of 0.0001 in the optimizer to constrain the magnitude of the weights and prevent overfitting.

These regularization methods collectively contribute to a more robust and generalizable model by simulating various real-world conditions and preventing overfitting.

## Results

The hyperparameter tuning process involved testing various architecture configurations to identify the optimal design for our Siamese Network. We evaluated different combinations of the number of convolutional layers, the initial number of filters, pooling types, and activation functions. For each architecture configuration, we conducted multiple runs with different training configurations, including varying learning rates, epochs, and batch normalization settings. The table below presents the results of the best run for each architecture configuration, providing a concise overview of the top-performing setups.

![results](https://github.com/user-attachments/assets/3603f6ab-5a1f-40cf-acbe-7a4732c72265)

## Best Results – Train and Validation

**Model Configuration**: 4 Conv Layers, 16 Initial Filters, Average Pooling, LeakyReLU Activation

- **Learning Rate**: 0.005
- **Batch Norm**: True
- **Convergence Time**: 7.3 minutes
- **Early Stopping**: Epoch 73
- **Train ROC-AUC**: 0.76
- **Validation ROC-AUC**: 0.61

### Loss vs Accuracy:
(Val – purple, Train – green)
<img src="https://github.com/user-attachments/assets/e6cbbd68-a6b9-4b20-8141-10beac2de418" alt="best_loss" width="500"/>

The pattern in the graph indicates that the model is effectively learning from the training data, as evidenced by the decreasing training loss. However, the slight increase in validation loss suggests the onset of overfitting, where the model might begin to memorize the training data rather than generalize. Thanks to early stopping, we mitigated this risk, ensuring the model did not overfit and maintained strong performance, as reflected in the accuracy score.

### Accuracy vs Epoch
(Val – black, Train – orange)
<img src="https://github.com/user-attachments/assets/021dbca6-0e8f-4afc-889e-d1fe86acc4fd" alt="best_accuracy" width="500"/>

The accuracy graph displays an increasing trend for both the training and validation sets, indicating improvement in the model's performance over time. Specifically, the training accuracy reaches a score of 0.83, while the validation accuracy achieves a relatively high accuracy score of 0.68 (using not optimal threshold of 1.5). The relatively high validation accuracy suggests that the model maintains a good level of generalization, effectively balancing between learning from the training data and applying this knowledge to new unseen data.

### ROC-AUC curve:

<img src="https://github.com/user-attachments/assets/2462dd07-fc5c-45bc-8117-33b06af4ea6d" alt="best_roc_auc" width="400"/>

## Best run accurate and misclassified examples:
Using the optimal distance threshold of 1.33 from the validation ROC-AUC curve.

### True Positive ✅

<img src="https://github.com/user-attachments/assets/f1ce0c36-6216-4049-bfba-75348a30d305" alt="true_positive_example" width="500"/>

The images are indeed very similar, and the facial features are obvious, making it easy for the model to identify
the same features and determine that this is the same person.

### True Negative ✅

<img src="https://github.com/user-attachments/assets/6e499453-ff30-41d6-b4a6-24df2c23895b" alt="true_negative" width="500"/>

The colors and facial expressions vary significantly, and the facial features are distinctly clear, which makes it
easy for the model to determine that they are two different people.

### False Positive ❌

<img src="https://github.com/user-attachments/assets/95b46810-770d-453c-8ef8-e7e2a905f780" alt="False_positive" width="500"/>

It is apparent that these are not the same person, yet the network had high confidence that they were. This
may be because the facial features seem alike and the face angle, leading to a false positive case.

### False Negative ❌

<img src="https://github.com/user-attachments/assets/abc487cf-6819-4cd9-80a5-93ac94f17ee1" alt="same_person_misclassificatio" width="500"/>

Even though it is the same person, the network classified them as different. This may be because, in one of the
pictures, the individual is wearing a sunglasses, which makes it hard for the network to extract meaningful
features.

## Best run with linear layers replacement result:
As mentioned previously, we replaced our three smaller fully connected layers with a single large linear layer of 4096 neurons, similar to the architecture in the original paper.

<img src="https://github.com/user-attachments/assets/509dd6e9-2906-4acf-b9be-07a417d5e31a" alt="linear_val_roc" width="400"/>

## Summary Table

| Convergence Time (minutes) | Early Stopping | Train ROC-AUC | Validation ROC-AUC | Optimal Threshold | Test Accuracy with Optimal Threshold |
|----------------------------|----------------|---------------|--------------------|-------------------|--------------------------------------|
| 7.8                        | 72             | 0.62          | 0.64               | 0.89              | 0.66                                 |


## Conclusion

- **Model Comparison**:
  - Two models achieved the best test accuracy of 0.69. 
    - **Model 1** (highlighted in blue) uses a total of 128 filters, compared to **Model 2's** (highlighted in green) 512 filters. This saves 384 filters while maintaining the same test accuracy, reducing computational complexity and resource usage without compromising performance.
  - **ROC-AUC Scores**:
    - Model 1 had a lower train ROC-AUC (0.76) and validation ROC-AUC (0.61) compared to Model 2, which had a train ROC-AUC of 0.94 and a validation ROC-AUC of 0.71.

- **Comparison of Fully Connected Layers**:
  - We compared Model 1, which uses three smaller fully connected layers, with a modified version that uses a single large linear layer of 4096 neurons.
  - The model with three smaller linear layers achieved higher train ROC-AUC and test accuracy, while the model with a single large linear layer demonstrated better validation ROC-AUC, indicating a potential for better generalization.
  - Using three smaller fully connected layers instead of a single large linear layer with 4096 neurons reduces the number of parameters by 1,048,001. This reduction in parameters leads to lower computational complexity and resource usage while maintaining comparable performance.

The findings highlight the trade-offs between model complexity, computational efficiency, and generalization performance. When selecting a model, it is important to consider the specific requirements and constraints of the deployment environment. Opting for a less complex model can reduce resource usage without sacrificing accuracy, while careful consideration of layer architecture can further optimize the balance between performance and efficiency.



