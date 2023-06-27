import os
import random
import torch
from torch import nn
import torchvision
from tqdm.notebook import tqdm

import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def walk_through_dir(dir_path) -> None:
    """
    Walks through dir_path returning its contents.
    Args:
    dir_path (str): target directory

    Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def say_hello() -> None:
    print("Hello World!")
    
def train_step(model: nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              device='cpu') -> tuple:
  """
  Performs one EPOCH of training on the provided data and labels.
  Args:
    model (torch.nn.Module): the neural network
    data_loader (torch.utils.data.DataLoader): the data loader
    loss_fn (torch.nn.Module): the loss function
    optimizer (torch.optim.Optimizer): the optimizer for the model
    device (str): where the model and data should be loaded (default: cpu)
  Returns:
    A tuple: The average loss and the accuracy values for this epoch.
  """

  # Set model to training mode  
  model.train()
  running_losses = 0.0
  running_accuracy = 0.0
  
  for batch, (images, labels) in enumerate(tqdm(data_loader), desc='Training'):
    # Load images and labels to device
    images = images.to(device)
    labels = labels.to(device)

    # Zero out the optimizer
    optimizer.zero_grad()

    # Perform forward pass
    outputs = model(images)

    # Compute loss
    loss = loss_fn(outputs, labels)
    running_losses += loss.item()
    
    # Perform backward pass
    loss.backward()

    # Perform optimization
    optimizer.step()

    # Compute accuracy
    running_accuracy += (outputs.argmax(dim=1) == labels).float().mean().item()
  
  # Return loss and accuracy values over 1 epoch
  return running_losses / len(data_loader), running_accuracy / len(data_loader)

def test_step(model: nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device='cpu') -> tuple:
  """
  Perfoms one EPOCH of testing on the provided data and labels.
  Args:
    model (torch.nn.Module): the neural network
    data_loader (torch.utils.data.DataLoader): the data loader
    loss_fn (torch.nn.Module): the loss function
    device (str): where the model and data should be loaded (default: cpu)
  Returns:
    A tuple: The average loss and the accuracy of this epoch.
  """
  # Set model to eval mode
  model.eval()
  running_loss = 0.0
  running_accuracy = 0.0
  
  with torch.inference_mode():
    for batch, (images, labels) in enumerate(tqdm(data_loader), desc='Testing'):
      images = images.to(device)
      labels = labels.to(device)

      outputs = model(images)
      loss = loss_fn(outputs, labels)
      running_loss += loss.item()
      
      accuracy = (outputs.argmax(dim=1) == labels).float().mean()
      running_accuracy += accuracy
    
  return running_loss / len(data_loader), running_accuracy / len(data_loader)

def plot_loss_curve(train_losses, test_losses, every_n=1) -> None:
  """
  Plots the loss curve for the model.
  
  Args:
    train_losses (list): list of training losses: shape (num_epochs, num_batchs)
    test_losses (list): list of testing losses: shape (num_epochs)
    every_n (int): plot every n-th training loss (default: 1)
    
  Returns:
    None
  """
  
  # Get number of epochs and batchs  
  num_epochs= len(train_losses)

  train_losses = np.array(train_losses)
  test_losses = np.array(test_losses)
  
  line_space = np.linspace(1, num_epochs, num_epochs)
  if every_n > 0 and len(test_losses) > 1:
    test_line_space = line_space[1::every_n]

  # Plot losses
  fig = plt.figure(figsize=(10, 6))
  plt.plot(line_space, train_losses, label='Training Loss')
  if every_n > 0 and len(test_losses) > 1:
    plt.plot(test_line_space, test_losses, label='Testing Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  fig.show()

def overfit_single_batch(model: nn.Module,
                         data_loader: torch.utils.data.DataLoader,
                         loss_fn: torch.nn.Module,
                         optimizer: torch.optim.Optimizer,
                         epochs: int = 1,
                         device='cpu') -> float:
  """
  Overfits the model on a single batch of data, returning the loss.
  Args:
    model (torch.nn.Module): the neural network
    data_loader (torch.utils.data.DataLoader): the data loader
    loss_fn (torch.nn.Module): the loss function
    optimizer (torch.optim.Optimizer): the optimizer for the model
    epochs (int): the number of epochs to train for (default: 1)
    device (str): where the model and data should be loaded (default: cpu)
  Returns:
    The loss value after training.
  """
  model.to(device)
  model.train()
  
  image, label = next(iter(data_loader))
  for i in tqdm(range(epochs)):
    preds = model(image.to(device))
    loss = loss_fn(preds, label.to(device))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print(loss.item())
  return loss.item()

def train_model(model: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                test_loader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                device='cpu',
                EPOCHS=1,
                test_every=1) -> tuple:
  """
  Trains the model for the specified number of epochs.
  Args:
    model (torch.nn.Module): the neural network
    train_loader (torch.utils.data.DataLoader): the data loader for training data 
    test_loader (torch.utils.data.DataLoader): the data loader for testing data
    loss_fn (torch.nn.Module): the loss function
    optimizer (torch.optim.Optimizer): the optimizer for the model
    device (str): where the model and data should be loaded (default: cpu)
    EPCHOS (int): number of epochs to train the model (default: 1)
    test_every (int): how frequently to test the model on the test set (default: 1)
  Returns:
    A tuple of lists:
      1. train_losses:shape(EPOCHS) The training loss over each epoch 
      2. test_losses:shape(EPOCHS) The testing loss over each epoch
  """
  
  train_losses = []
  train_accuracies = []
  
  test_losses = []
  test_accuracies = []
  
  model.to(device)
  
  for epoch in (range(EPOCHS)):
    print(f"---------------------EPOCH {epoch+1}---------------------")
    
    # Train Step
    print("Training...", end="")
    train_loss, train_accuracy = train_step(model, train_loader, loss_fn, optimizer, device)
    print(f"\tTrain Loss: {(train_loss):.4f}, Train Acc: {100*(train_accuracy):.4f}%")
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    # Test step
    if test_every!=0 and (epoch+1) % test_every == 0:
      print("Testing...", end="")
      test_loss, test_accuracy = test_step(model, test_loader, loss_fn, device)
      print(f"\tTest Loss: {(test_loss):.4f}, Test Acc: {100*(test_accuracy):.4f}%")
      test_losses.append(test_loss)
      test_accuracies.append(test_accuracy)
    
    print("\n")
    
  return train_losses, test_losses

def show_random_prediction(model: nn.Module,
                            dataset: torchvision.datasets) -> None:
  """
  Shows a random prediction from the dataset.
  Args:
    model (torch.nn.Module): the neural network
    dataset (torch.utils.data.Dataset): the dataset to predict
  Returns:
    None
  """
  CLASSES = dataset.classes
  image, label = random.choice(dataset)
  model.to('cpu').eval()
  with torch.inference_mode():
      pred = model(image.unsqueeze(0))
      pred = pred.argmax(dim=1).item()
      
      plt.imshow(image.permute(1, 2, 0), cmap='gray')
      plt.axis('off')
      plt.title(f"Predicted: {CLASSES[pred]}, Actual: {CLASSES[label]}", color="green" if pred == label else "red")
      
def predict(model: nn.Module,
            dataset: torch.utils.data.Dataset,
            device='cpu') -> list:
  """
  Predicts the labels for the provided dataset.
  Args:
    model (torch.nn.Module): the neural network
    dataset (torch.utils.data.Dataset): the dataset to predict
    device (str): where the model and data should be loaded (default: cpu)
  Returns:
    A list of the predicted labels.
  """
  model.to(device).eval()
  preds = []
  with torch.inference_mode():
    print("Predicting...")
    for image, label in tqdm(dataset):
        preds.append(model(image.unsqueeze(0).to(device)).argmax(dim=1).item())
  return preds

def plot_metrics(targets: list, preds: list, CLASSES: list) -> None:
  """
  Plots the confusion matrix and classification report.
  Args:
    targets (list): the true labels from the dataset
    preds (list): the predicted labels from the model
    CLASSES (list): the list of classes in dataset
  Returns:
    None
  """
  print(classification_report(y_true=targets, y_pred=preds, target_names=CLASSES, digits=4))
  cm = confusion_matrix(targets, preds)
  fig = px.imshow(cm, x=CLASSES, y=CLASSES, color_continuous_scale='mint', title='Confusion Matrix', labels=dict(x='Predicted', y='Actual', color='Count'),
                width=600, height=600)
  fig.show()