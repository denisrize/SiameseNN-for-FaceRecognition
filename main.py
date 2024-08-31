import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, auc
from torchvision import datasets, transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
import os
import json

def parse_pairs(filename, data_dir):
    with open(filename, 'r') as f:
        lines = f.readlines()[1:]  # Skip the first line, which contains the number of pairs
    pairs = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            # Positive pair (same person)
            name, idx1, idx2 = parts
            img1_path = os.path.join(data_dir, f"{name}/{name}_{idx1.zfill(4)}.jpg")
            img2_path = os.path.join(data_dir, f"{name}/{name}_{idx2.zfill(4)}.jpg")
            pairs.append((img1_path, img2_path, 0))  # 0 indicates same person
        elif len(parts) == 4:
            # Negative pair (different people)
            name1, idx1, name2, idx2 = parts
            img1_path = os.path.join(data_dir, f"{name1}/{name1}_{idx1.zfill(4)}.jpg")
            img2_path = os.path.join(data_dir, f"{name2}/{name2}_{idx2.zfill(4)}.jpg")
            pairs.append((img1_path, img2_path, 1))  # 1 indicates different people
    return pairs

class NoiseInjection(object):
    def __init__(self, mean=0., std=1.):
        """
        Args:
            mean (float): Mean of the Gaussian noise.
            std (float): Standard deviation of the Gaussian noise.
        """
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with Gaussian noise added.
        """
        noise = torch.randn(img.size()) * self.std + self.mean
        img = img + noise
        return img
        
class Cutout(object):
    def __init__(self, n_holes, length):
        """
        Args:
            n_holes (int): Number of patches to cut out of each image.
            length (int): The length (in pixels) of each square patch.
        """
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class FacePairsDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        # Load images directly as grayscale
        img1 = Image.open(img1_path).convert('L')  # 'L' mode for grayscale
        img2 = Image.open(img2_path).convert('L')
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, label

# Define the model
class SiameseNetwork(nn.Module):
    def __init__(self, num_conv_layers=4, init_num_filters=64, pooling_type='max', dropout_rate=0.05, batch_norm=False, activation='relu', one_linear=False):
        super(SiameseNetwork, self).__init__()

        self.num_conv_layers = num_conv_layers
        self.init_num_filters = init_num_filters
        self.one_linear = one_linear
        # Define activation function
        if activation == 'relu':
            self.activation_fn = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu': 
            self.activation_fn = nn.LeakyReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        layers = []
        in_channels = 1
        num_filters = init_num_filters

        for i in range(num_conv_layers):
            layers.append(nn.Conv2d(in_channels, num_filters, kernel_size=4 if i == num_conv_layers-1 else 7))
            if batch_norm:
                layers.append(nn.BatchNorm2d(num_filters))
            layers.append(self.activation_fn)
            if pooling_type == 'max':
                layers.append(nn.MaxPool2d(2, stride=2))
            else:
                layers.append(nn.AvgPool2d(2, stride=2))
            layers.append(nn.Dropout2d(dropout_rate))
            in_channels = num_filters
            if i % 2 != 0:
                num_filters = num_filters * 2

        self.cnn1 = nn.Sequential(*layers)

        # Calculate the size of the input to the fully connected layer
        with torch.no_grad():
            self.feature_size = self._get_feature_size()

        # Fully connected layers
        if self.one_linear: # similar to the original paper
            self.fc1 = nn.Sequential(
            nn.Linear(self.feature_size, 4096),
            nn.BatchNorm1d(4096),
            self.activation_fn,
            nn.Linear(4096, 1)
        )
        else: # simpler model
            self.fc1 = nn.Linear(self.feature_size, 32)
            self.fc_dropout1 = nn.Dropout(dropout_rate)  # Adding dropout after the first fully connected layer
            self.fc2 = nn.Linear(32, 16)
            self.fc_dropout2 = nn.Dropout(dropout_rate)  # Adding dropout after the second fully connected layer
            self.fc3 = nn.Linear(16, 1)


    
    def _get_feature_size(self):
        x = torch.zeros(1, 1, 105, 105)
        x = self.cnn1(x)
        return x.view(1, -1).size(1)
    
    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.activation_fn(self.fc1(output))
        if self.one_linear:
            return output
        output = self.fc_dropout1(output)  
        output = self.activation_fn(self.fc2(output))
        output = self.fc_dropout2(output) 
        output = self.fc3(output)
        return output
    
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=3.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calculate the Euclidean distance and contrastive loss
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

def plot_roc_auc(all_labels_train, all_distances_train, all_labels_val, all_distances_val, log_dir, tag='Training/ROC_Curve'):
    writer = SummaryWriter(log_dir=log_dir)    
    # Compute ROC AUC curve and find the optimal threshold
    fpr_train, tpr_train, thresholds_train = roc_curve(all_labels_train, all_distances_train)
    roc_auc_train = roc_auc_score(all_labels_train, all_distances_train)
    fpr_val, tpr_val, thresholds_val = roc_curve(all_labels_val, all_distances_val)
    roc_auc_val = roc_auc_score(all_labels_val, all_distances_val)

    # Find the optimal threshold
    optimal_idx_train = np.argmax(tpr_train - fpr_train)
    optimal_threshold_train = thresholds_train[optimal_idx_train]
    optimal_idx_val = np.argmax(tpr_val - fpr_val)
    optimal_threshold__val = thresholds_val[optimal_idx_val]

    # Plot ROC curve
    roc_figure = plt.figure(figsize=(6, 6))
    plt.plot(fpr_train, tpr_train, color='blue', lw=2, label='Train ROC curve (area = %0.2f)' % roc_auc_train)
    plt.scatter(fpr_train[optimal_idx_train], tpr_train[optimal_idx_train], color='red', label=f'Train Optimal threshold {optimal_threshold_train:.2f}')

    plt.plot(fpr_val, tpr_val, color='blue', lw=2, label='Val ROC curve (area = %0.2f)' % roc_auc_val)
    plt.scatter(fpr_val[optimal_idx_val], tpr_val[optimal_idx_val], color='red', label=f'Val Optimal threshold {optimal_threshold__val:.2f}')

    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')

    # Convert matplotlib figure to a tensor image
    roc_figure.canvas.draw()
    roc_image = np.frombuffer(roc_figure.canvas.tostring_rgb(), dtype=np.uint8)
    roc_image = roc_image.reshape(roc_figure.canvas.get_width_height()[::-1] + (3,))
    roc_image = torch.tensor(roc_image).permute(2, 0, 1)

    writer.add_image(tag, roc_image)

    writer.close()

    return optimal_threshold_train, optimal_threshold__val

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, log_dir='./logs', collect_roc_data=True):
    writer = SummaryWriter(log_dir=log_dir)
    best_loss = float('inf')
    no_improvement_epochs = 0
    best_model_wts = None

    outputs_train, targets_train = [], []
    outputs_val, targets_val = [], []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device {device}")
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        total_loss, correct, total = 0, 0, 0

        for i, batch in enumerate(train_loader):
            img0, img1, labels = batch
            img0, img1, labels = img0.to(device), img1.to(device), labels.to(device)
            
            optimizer.zero_grad()
            output1, output2 = model(img0, img1)
            loss = criterion(output1, output2, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            dists = (output1 - output2).norm(p=2, dim=1)
            predictions = dists > 1.5
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # ROC data for last epoch
            if collect_roc_data and epoch == num_epochs - 1:
                outputs_train.extend(dists.detach().cpu().numpy())
                targets_train.extend(labels.detach().cpu().numpy())


        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        
        end_time = time.time()  # End timing the training phase
        epoch_time = end_time - start_time  # Calculate the duration for the training phase
        writer.add_scalar('Time/epoch', epoch_time, epoch)

        print(f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Training Acc: {train_acc:.4f}")
        
        model.eval()
        val_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                img0, img1, labels_val = batch
                img0, img1, labels_val = img0.to(device), img1.to(device), labels_val.to(device)
                
                output1, output2 = model(img0, img1)
                loss = criterion(output1, output2, labels_val)
                val_loss += loss.item()

                dists_val = (output1 - output2).norm(p=2, dim=1)
                predictions = dists_val > 1.5
                correct += (predictions == labels_val).sum().item()
                total += labels_val.size(0)

                if collect_roc_data and epoch == num_epochs - 1:
                    outputs_val.extend(dists_val.detach().cpu().numpy())
                    targets_val.extend(labels_val.detach().cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")

        writer.add_scalars('Loss', {'Train': train_loss, 'validation': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'Train': train_acc, 'validation': val_acc}, epoch)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = model.state_dict()
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
        
        if no_improvement_epochs >= 30:
            outputs_train.extend(dists.detach().cpu().numpy())
            targets_train.extend(labels.detach().cpu().numpy())
            outputs_val.extend(dists_val.detach().cpu().numpy())
            targets_val.extend(labels_val.detach().cpu().numpy())
            print("Early stopping due to no improvement in validation loss")
            break
        
    _, optimal_threshold__val = plot_roc_auc(targets_train, outputs_train, targets_val, outputs_val, log_dir)
            
    model.load_state_dict(best_model_wts)
    writer.close()
    return model, optimal_threshold__val

def test_model(model, test_loader, log_dir, threshold=1.5):
    writer = SummaryWriter(log_dir=log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []
    all_distances = []
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            img1, img2, labels, _, _ = data
            img1, img2 = img1.to(device), img2.to(device)
            output1, output2 = model(img1, img2)
            distances = F.pairwise_distance(output1, output2)
            predictions = (distances > threshold).float()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_distances.extend(distances.cpu().numpy())
    
    # Compute classification report
    report_str = classification_report(all_labels, all_predictions, target_names=['same', 'different'])

    # Log the classification report as text in TensorBoard
    writer.add_text('Test/Classification_Report', report_str)

    print("Test classification report: ", report_str)

    # Compute and log confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    cm_figure = plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['same', 'different'], yticklabels=['same', 'different'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Test Confusion Matrix')

    # Convert matplotlib figure to a tensor image
    cm_figure.canvas.draw()
    cm_image = np.frombuffer(cm_figure.canvas.tostring_rgb(), dtype=np.uint8)
    cm_image = cm_image.reshape(cm_figure.canvas.get_width_height()[::-1] + (3,))
    cm_image = torch.tensor(cm_image).permute(2, 0, 1)

    writer.add_image('Test/Confusion_Matrix', cm_image)

    # Compute ROC AUC curve and find the optimal threshold
    fpr, tpr, thresholds = roc_curve(all_labels, all_distances)
    roc_auc = roc_auc_score(all_labels, all_distances)

    # Find the optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Plot ROC curve
    roc_figure = plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='Test ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label='Optimal threshold {:.2f}'.format(optimal_threshold))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test ROC curve')
    plt.legend(loc='lower right')

    # Convert matplotlib figure to a tensor image
    roc_figure.canvas.draw()
    roc_image = np.frombuffer(cm_figure.canvas.tostring_rgb(), dtype=np.uint8)
    roc_image = roc_image.reshape(roc_figure.canvas.get_width_height()[::-1] + (3,))
    roc_image = torch.tensor(roc_image).permute(2, 0, 1)

    writer.add_image('Test/ROC_Curve', roc_image)

    writer.close()

def plot_dataset_distribution(train_pairs, val_pairs, test_pairs):
    # Class distribution
    # Create directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)

    # Create a df of train,val, and test with the number of samples for each class
    train_df = pd.DataFrame(train_pairs, columns=['img1', 'img2', 'label'])
    val_df = pd.DataFrame(val_pairs, columns=['img1', 'img2', 'label'])
    test_df = pd.DataFrame(test_pairs, columns=['img1', 'img2', 'label', 'person1', 'person2']).drop(columns=['person1', 'person2'])

    train_class_counts = train_df['label'].value_counts()
    val_class_counts = val_df['label'].value_counts()
    test_class_counts = test_df['label'].value_counts()

    class_counts = pd.DataFrame({ 'train': train_class_counts, 'val': val_class_counts, 'test': test_class_counts })
    class_counts = class_counts.fillna(0)
    class_counts = class_counts.astype(int)
    class_counts = class_counts.sort_index()

    # Create a bar plot, with a bar for each type of data (train, val, test) and each class
    class_counts.plot(kind='bar', figsize=(10, 6))
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    # Add the count at the top of each bar
    for i, (train, val, test) in enumerate(zip(class_counts['train'], class_counts['val'], class_counts['test'])):
        plt.text(i-0.2, train+8, str(train), color='black')
        plt.text(i-0.05, val+8, str(val), color='black')
        plt.text(i+0.15, test+8, str(test), color='black')
    plt.legend(title='Data Set')
    # Save the plot
    plt.savefig(os.path.join('plots', 'class_distribution.png'), bbox_inches='tight')
    plt.close()  

    # Unique instances ditribution
    train_unique = len(train_df['img1'].unique())
    val_unique = len(val_df['img1'].unique())
    test_unique = len(test_df['img1'].unique())

    # Create DataFrame
    unique_counts = pd.DataFrame({'train': [train_unique], 'val': [val_unique], 'test': [test_unique]}, index=['unique'])

    # Plot
    ax = unique_counts.plot(kind='bar', figsize=(10, 6))

    # Add bar values on top
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() + 0.05, p.get_height() + 0.1))

    # Titles and labels
    plt.title('Unique Instances')
    plt.xlabel('Set')
    plt.ylabel('Count')

    # Save the plot
    plt.savefig(os.path.join('plots', 'unique_instances.png'), bbox_inches='tight')
    plt.close()  

def imshow(img, text=None, is_positive=True, model_dir=None):
    npimg = img.cpu().numpy()
    plt.axis("off")
    
    if is_positive:
        border_color = 'green'
    else:
        border_color = 'red'
    
    fig, ax = plt.subplots()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    
    # Adding border around the image
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(4)
    
    # Save the plot
    plt.savefig(os.path.join(model_dir, f'{text}.png'), bbox_inches='tight')
    plt.close(fig)

def parse_pairs_test(filename, data_dir):
    with open(filename, 'r') as f:
        lines = f.readlines()[1:]  # Skip the first line, which contains the number of pairs
    pairs = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            # Positive pair (same person)
            name, idx1, idx2 = parts
            img1_path = os.path.join(data_dir, f"{name}/{name}_{idx1.zfill(4)}.jpg")
            img2_path = os.path.join(data_dir, f"{name}/{name}_{idx2.zfill(4)}.jpg")
            pairs.append((img1_path, img2_path, 0, name, name))  # 0 indicates same person, names are the same
        elif len(parts) == 4:
            # Negative pair (different people)
            name1, idx1, name2, idx2 = parts
            img1_path = os.path.join(data_dir, f"{name1}/{name1}_{idx1.zfill(4)}.jpg")
            img2_path = os.path.join(data_dir, f"{name2}/{name2}_{idx2.zfill(4)}.jpg")
            pairs.append((img1_path, img2_path, 1, name1, name2))  # 1 indicates different people, names are different
    return pairs

class FacePairsTestset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img1_path, img2_path, label, person1_name, person2_name = self.pairs[idx]
        img1 = Image.open(img1_path).convert('L')  # 'L' mode for grayscale
        img2 = Image.open(img2_path).convert('L')
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, label, person1_name, person2_name

def find_examples_for_person(model, test_dataset, optimal_threshold, model_dir):
        # Function to find accurate and misclassification examples for the same person
    test_dataloader = DataLoader(test_dataset, num_workers=6, batch_size=1, shuffle=False)
    dataiter = iter(test_dataloader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processed_persons = {}
    model.eval()
    
    with torch.no_grad():
        while True:
            try:
                img1, img2, label2, person1_name, person2_name = next(dataiter)
            except StopIteration:
                print("End of dataset reached. Exiting loop.")
                break
            
            for person_name in [person1_name, person2_name]:
                if person_name not in processed_persons:
                    processed_persons[person_name] = {'accurate': [], 'misclassified': []}
                
                concatenated = torch.cat((img1, img2), 0)
                img1, img2 = img1.to(device), img2.to(device)
                output1, output2 = model(img1, img2)
                euclidean_distance = F.pairwise_distance(output1, output2)
                
                if euclidean_distance < optimal_threshold and label2 == 0:
                    processed_persons[person_name]['accurate'].append((concatenated, 'Accurate, Distance: {:.2f}'.format(euclidean_distance.item()), True))
                elif euclidean_distance < optimal_threshold and label2 == 1:
                    processed_persons[person_name]['misclassified'].append((concatenated, 'Misclassification, Distance: {:.2f}'.format(euclidean_distance.item()), False))
                elif euclidean_distance > optimal_threshold and label2 == 0:
                    processed_persons[person_name]['misclassified'].append((concatenated, 'Misclassification, Distance: {:.2f}'.format(euclidean_distance.item()), False))
                elif euclidean_distance > optimal_threshold and label2 == 1:
                    processed_persons[person_name]['accurate'].append((concatenated, 'Accurate, Distance: {:.2f}'.format(euclidean_distance.item()), True))
                
                if processed_persons[person_name]['accurate'] and processed_persons[person_name]['misclassified']:
                    print(f"Found both accurate and misclassification examples for {person_name}")
                    for img, text, is_positive in processed_persons[person_name]['accurate'] + processed_persons[person_name]['misclassified']:
                        imshow(torchvision.utils.make_grid(img), text, is_positive=is_positive, model_dir=model_dir)
                    return
                
if __name__ == '__main__':

    # Load the data
    data_dir = r'./data_set/lfw2/lfw2'
    train_pairs = parse_pairs("pairsDevTrain.txt", data_dir)
    test_pairs = parse_pairs("pairsDevTest.txt", data_dir)

    train_pairs, val_pairs = train_test_split(train_pairs, test_size=0.2, random_state=42)

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((105, 105)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Cutout(n_holes=3, length=5),
        NoiseInjection(mean=0., std=0.03)
    ])

    transform_test = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor(),
    ])

    # Create datasets
    train_dataset = FacePairsDataset(train_pairs, transform=transform)
    val_dataset = FacePairsDataset(val_pairs, transform=transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Create test-set
    test_pairs = parse_pairs_test('pairsDevTest.txt', data_dir)
    test_dataset = FacePairsTestset(test_pairs, transform=transform_test)
    test_loader = DataLoader(test_dataset, num_workers=6, batch_size=64, shuffle=False)
    dataiter = iter(test_loader)

    # EDA
    plot_dataset_distribution(train_pairs, val_pairs, test_pairs)

    # Hyperparameters
    # learning_rates = [0.005, 0.001] 
    # num_conv_layers_list = [3, 4] 
    # init_num_filters_list = [16, 64] 
    # pooling_type = ['max', 'avg'] 
    # activations = ['relu', 'leaky_relu'] 
    # epochs = [200]
    # batch_norm = [False, True] 

    learning_rates = [0.007]
    num_conv_layers_list = [4] 
    init_num_filters_list = [16] 
    pooling_type = ['avg'] 
    activations = ['leaky_relu'] 
    epochs = [5]
    batch_norm = [True]

    # Create combinations of hyperparameters
    hyperparameter_combinations = list(itertools.product(
        learning_rates,
        num_conv_layers_list,
        init_num_filters_list,
        pooling_type,
        activations,
        epochs,
        batch_norm
    ))

    criterion = ContrastiveLoss()
    for combination in hyperparameter_combinations:
        lr, num_conv_layers, init_num_filters, pooling_type, activation, epochs, batch_norm = combination
        model_dir = f'lr_{lr}_num-conv_{num_conv_layers}_init_filter_{init_num_filters}_pooltype_{pooling_type}_act_{activation}_bnorm_{batch_norm}'
        
        log_dir = f'./logs/' + model_dir
        model_dir = 'best_models/' + model_dir

        # Save configuration to a JSON file
        config = {
            'learning_rate': lr,
            'num_conv_layers': num_conv_layers,
            'init_num_filters': init_num_filters,
            'pooling_type': pooling_type,
            'activation': activation,
            'epochs': epochs,
            'batch_norm': batch_norm
        }

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        with open(os.path.join(model_dir, 'config.json'), 'w') as config_file:
            json.dump(config, config_file, indent=4)

        model = SiameseNetwork(num_conv_layers=num_conv_layers,
                            init_num_filters=init_num_filters,
                            pooling_type=pooling_type,
                            activation=activation, batch_norm=batch_norm)
        
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

        print(f"Training with lr={lr}, num_conv_layers={num_conv_layers}, init_num_filters={init_num_filters} "
            f"pooling_type={pooling_type},"
            f"activation={activation}, batch_norm={batch_norm}")
        best_model, val_optimal_threshold = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=epochs, log_dir=log_dir, collect_roc_data=True)
        
        # Save best model
        torch.save(model.state_dict(), model_dir + '/best_siamese_model.pth')

        # Test the model with optimal validation threshold
        test_model(best_model, test_loader, log_dir, threshold=val_optimal_threshold)

        # Find examples of the model accurate and misclassification for the same person
        find_examples_for_person(model, test_dataset, val_optimal_threshold, model_dir)

        # Test the model with the original fully connected layer (1 layer with 4096 units)
        model_one_linear = SiameseNetwork(num_conv_layers=num_conv_layers,
                    init_num_filters=init_num_filters,
                    pooling_type=pooling_type,
                    activation=activation, batch_norm=batch_norm, one_linear=True)
        
        optimizer = optim.Adam(model_one_linear.parameters(), lr=lr, weight_decay=0.0001)

        print(f"Training with lr={lr}, num_conv_layers={num_conv_layers}, init_num_filters={init_num_filters} "
            f"pooling_type={pooling_type},"
            f"activation={activation}, batch_norm={batch_norm}, with one linear layer")
        
        best_model_one_linear, val_optimal_threshold_one_linear = train_model(model_one_linear, train_loader, val_loader, criterion, optimizer, num_epochs=epochs, log_dir=log_dir, collect_roc_data=True)
        
        # Save one linear layer model cnofig and results 
        model_dir_one_linear = model_dir + '_one_linear'
        log_dir_one_linear = log_dir + '_one_linear'

        if not os.path.exists(model_dir_one_linear):
            os.makedirs(model_dir_one_linear)

        with open(os.path.join(model_dir_one_linear, 'config.json'), 'w') as config_file:
            json.dump(config, config_file, indent=4)

        torch.save(model_one_linear.state_dict(), model_dir_one_linear + '/best_siamese_model.pth')

        # Test the model with optimal validation threshold
        test_model(best_model_one_linear, test_loader, log_dir_one_linear, threshold=val_optimal_threshold_one_linear)

        # Find examples of the model accurate and misclassification for the same person
        find_examples_for_person(model_one_linear, test_dataset, val_optimal_threshold_one_linear, model_dir_one_linear)
