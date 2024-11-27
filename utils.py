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
    outputs = np.array(outputs).ravel()
    trues = np.array(trues).ravel()
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
    f1 = 2*precision*recall/(precision+recall)

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

            loss = criterion(out.to(device), y.float())
            losses.append(loss.item())

            metrics = score_model(y.cpu(), out.cpu(), threshold)
            metrics_ls.append(metrics)

    accuracy, prec, rec, f1 = zip(*metrics_ls)
    return np.mean(accuracy), np.mean(losses), np.mean(prec), np.mean(rec), np.mean(f1)

def train(model, data_loader, criterion, optimizer, device, threshold):

    model.train()
    losses = []
    metrics_ls = []
    progress_bar = tqdm(data_loader, desc="Training", ascii=True)

    for X, y in progress_bar:
        y = y
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(X).to('cpu')

        loss = criterion(outputs, y.float())

        losses.append(loss.item())

        metrics = score_model(y.cpu(), outputs.cpu(), threshold=threshold)
        metrics_ls.append(metrics)

        loss.backward()
        optimizer.step()

    accuracy, prec, rec, f1 = zip(*metrics_ls)
    return np.mean(accuracy), np.mean(losses), np.mean(prec), np.mean(rec), np.mean(f1)

def plot_curves(train, valid, metric_name, filename, output_dir = './'):

    epochs = range(len(train))
    plt.plot(epochs, train, label='train')
    plt.plot(epochs, valid, label='valid')

    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.title(metric_name + ' Curve - ' + filename)
    plt.savefig(os.path.join(output_dir, filename+ "_" + metric_name + '.png'))
    plt.show()