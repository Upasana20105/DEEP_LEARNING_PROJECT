import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# Define transformations for the training and test sets
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # Normalize to [-1, 1]

# Define the Convolutional Neural Network (CNN)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 3 input channels (RGB), 6 output channels, 5x5 kernel
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 input channels, 16 output channels, 5x5 kernel
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 16 * 5 * 5 is the flattened size after conv and pool
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # 10 output classes (CIFAR-10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x))) # (batch_size, 6, 14, 14)
        x = self.pool(torch.relu(self.conv2(x))) # (batch_size, 16, 5, 5)
        x = torch.flatten(x, 1)                  # (batch_size, 16*5*5) = (batch_size, 400)
        x = torch.relu(self.fc1(x))              # (batch_size, 120)
        x = torch.relu(self.fc2(x))              # (batch_size, 84)  <--- THIS IS THE CORRECTED LINE
        x = self.fc3(x)                          # (batch_size, 10)
        return x

# --- Everything below this line that involves creating DataLoaders, Net instances,
# --- and running the training/evaluation should be inside the if __name__ == '__main__': block.

if __name__ == '__main__':
    # Load the training and test datasets
    # num_workers=0 is a quick fix if the problem persists, but the __name__ == '__main__' is the proper one.
    # For Windows, num_workers=0 is often recommended due to multiprocessing overhead.
    # If you have a powerful CPU and want to try multiprocessing, keep num_workers > 0.
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=0) # num_workers=2 is what causes the issue on Windows if not in main block

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=0) # Same for testloader

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print(f"Number of training samples: {len(trainset)}")
    print(f"Number of test samples: {len(testset)}")

    # Function to show an image (can stay outside, but moved for consistency)
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        # plt.show() # Removed plt.show() here as it blocks execution; it's called later

    # Get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Show images
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
    imshow(torchvision.utils.make_grid(images))
    plt.show() # Display the image after dataiter and imshow

    net = Net()

    # Check if GPU is available and move the model to GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    net.to(device)

    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 10
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    print("Starting training...")
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        correct_train = 0
        total_train = 0
        net.train() # Set model to training mode
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(trainloader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        net.eval() # Set model to evaluation mode
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        with torch.no_grad(): # Disable gradient calculation for validation
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss /= len(testloader)
        val_accuracy = 100 * correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch + 1}, '
              f'Train Loss: {train_loss:.3f}, Train Acc: {train_accuracy:.2f}%, '
              f'Validation Loss: {val_loss:.3f}, Validation Acc: {val_accuracy:.2f}%')

    print('Finished Training')

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # Evaluate per-class accuracy
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4): # batch size is 4
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print('\nAccuracy for each class:')
    for i in range(10):
        print(f'Accuracy of {classes[i]:5s} : {100 * class_correct[i] / class_total[i]:.2f} %')

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Function to visualize predictions
    def visualize_predictions(model, dataloader, classes, num_images=8):
        was_training = model.training
        model.eval() # Set model to evaluation mode
        images_so_far = 0
        fig = plt.figure(figsize=(15, 8))

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = fig.add_subplot(num_images // 4, 4, images_so_far, xticks=[], yticks=[])
                    # Unnormalize image for display
                    img = inputs.cpu().data[j] / 2 + 0.5
                    npimg = img.numpy()
                    ax.imshow(np.transpose(npimg, (1, 2, 0)))

                    ax.set_title(f'True: {classes[labels[j]]}\nPred: {classes[preds[j]]}',
                                 color='green' if preds[j] == labels[j] else 'red')

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)

    print("\nVisualizing sample predictions:")
    visualize_predictions(net, testloader, classes, num_images=8)
    plt.show() # Added plt.show() for the visualization of predictions