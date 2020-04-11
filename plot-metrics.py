import json
import argparse
from pprint import pprint
import matplotlib.pyplot as plt

FILE = "metrics-logs.json"


def create(data):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    l1 = ax.plot(data['train_accuracies'], label='train_accuracies', c='steelblue')
    l2 = ax2.plot(data['train_losses'], label='train_losses', c='orange')
    l3 = ax.plot(data['val_accuracies'], label='val_accuracies', c='r')
    l4 = ax2.plot(data['val_losses'], label='val_losses', c='g')

    # set all labels to the same legend
    lns = l1+l2+l3+l4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax2.set_ylabel('Loss')

    plt.savefig('plot.png')

if __name__ == "__main__":
    with open(FILE, 'r') as f:
        data = json.loads(f.read())

    pprint(data)
    create(data)