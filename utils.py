from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import os

class ICD9_Dataset(Dataset):
    def __init__(self, X, y):
       self.X = X
       self.y = y
       super().__init__()

    def __len__(self):
       return len(self.X)
    
    def __getitem__(self, index):
       return self.X[index], self.y[index]
    
def score_model(trues, outputs, threshold=0.2):
  with torch.no_grad():
    outputs = np.array(outputs.cpu()).ravel()
    trues = np.array(trues.cpu()).ravel()
    preds = np.array([outputs>=threshold], dtype=np.float32).ravel()

    true_positives = (
      trues[trues==1] == preds[trues==1]).sum()
    false_positives = (
      preds[preds==1] != trues[preds==1]).sum()
    false_negatives = (
      preds[preds==0] != trues[preds==0]).sum()
    true_negatives = (
      trues[trues==0] == preds[trues==0]).sum()


    precision = true_positives/(true_positives + false_positives+ 1e-10)
    recall = true_positives/(true_positives + false_negatives+ 1e-10)
    accuracy = (true_positives + true_negatives) / (true_positives + false_negatives + true_negatives + false_positives + 1e-10)
    f1 = 2*precision*recall/(precision+recall + 1e-10)

    return accuracy, precision, recall, f1


def evaluate(model, data_loader, criterion, device, threshold):

    model.eval()

    all_true_labels = []
    all_outputs = []

    losses = []
    metrics_ls = []

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating", ascii=True)
        for batch_idx, (X, y) in enumerate(progress_bar):
            y = y.to(device)
            out = model(X)

            all_outputs.extend(out.cpu().numpy())
            all_true_labels.extend(y.cpu().numpy())

            loss = criterion(out, y.float())
            losses.append(loss.item())

            metrics = score_model(y, out, threshold)
            metrics_ls.append(metrics)

    accuracy, prec, rec, f1 = zip(*metrics_ls)
    return np.mean(accuracy), np.mean(losses), np.mean(prec), np.mean(rec), np.mean(f1)

def train(model, data_loader, criterion, optimizer, device, threshold):

    model.train()
    losses = []
    metrics_ls = []
    progress_bar = tqdm(data_loader, desc="Training", ascii=True)

    for X, y in progress_bar:
        y = y.to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(X)

        loss = criterion(outputs, y.float())

        losses.append(loss.item())

        metrics = score_model(y, outputs, threshold=threshold)
        metrics_ls.append(metrics)

        loss.backward()
        optimizer.step()

    accuracy, prec, rec, f1 = zip(*metrics_ls)
    return np.mean(accuracy), np.mean(losses), np.mean(prec), np.mean(rec), np.mean(f1)

def plot_curves(train, valid, metric_name, filename, output_dir = './'):
    '''
    Plot learning curves with matplotlib. Training perplexity and validation perplexity are plot in the same figure
    :param train_perplexity_history: training perplexity history of epochs
    :param valid_perplexity_history: validation perplexity history of epochs
    :param filename: filename for saving the plot
    :return: None, save plot in the current directory
    '''
    epochs = range(len(train))
    plt.plot(epochs, train, label='train')
    plt.plot(epochs, valid, label='valid')

    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.title(metric_name + ' Curve - ' + filename)
    plt.savefig(os.path.join(output_dir, filename+ "_" + metric_name + '.png'))
    plt.show()

def train_and_plot(model, optimizer, scheduler, criterion, train_loader, valid_loader, device, filename='', epochs=10, threshold=0.2):
    train_accuracies = []
    valid_accuracies = []
    train_losses = []
    valid_losses = []

    for epoch_idx in range(epochs):
        print("-----------------------------------")
        print("Epoch %d" % (epoch_idx+1))
        print("-----------------------------------")

        train_accuracy, train_loss, train_precision, train_recall, train_f1 = train(model, train_loader, criterion, optimizer, device=device, threshold=threshold)
        scheduler.step(train_loss)

        valid_accuracy, valid_loss, valid_precision, valid_recall, valid_f1 = evaluate(model, valid_loader, criterion, device=device, threshold=threshold)

        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print("Training Loss: %.4f. Validation Loss: %.4f. " % (train_loss, valid_loss))
        print("Training Accuracy: %.4f. Validation Accuracy: %.4f. " % (train_accuracy, valid_accuracy))
        print("Training Precision: %.4f. Validation Precision: %.4f. " % (train_precision, valid_precision))
        print("Training Recall: %.4f. Validation Recall: %.4f. " % (train_recall, valid_recall))
        print("Training F1: %.4f. Validation F1: %.4f. " % (train_f1, valid_f1))

    plot_curves(train_accuracies, valid_accuracies, "Accuracy", filename, output_dir='outputs/')
    plot_curves(train_losses, valid_losses, "Loss", filename, output_dir='outputs/')

