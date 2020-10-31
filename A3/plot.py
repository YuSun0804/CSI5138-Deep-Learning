import os
from os import path
import numpy as np
import json
import matplotlib.pyplot as plt

def compute(models,states):
    loss = []
    accuracy = []
    val_loss = []
    val_accuracy = []
    files = []
    names = []
    for model in models:
        for state in states:    
            filename = "result/loss_"+model+"_"+str(state)+".out"
            files.append(filename)
            names.append(model+"_"+str(state))

    for filename in files:
        temp_loss = []
        temp_accuracy = []
        temp_val_loss = []
        temp_val_accuracy = []
        with open(filename) as f:
            for line in f:
                record = json.loads(line)
                temp_loss.append(record["loss"])
                temp_accuracy.append(record["accuracy"])
                temp_val_loss.append(record["val_loss"])
                temp_val_accuracy.append(record["val_accuracy"])
        loss.append(temp_loss)
        accuracy.append(temp_accuracy)
        val_loss.append(temp_val_loss)
        val_accuracy.append(temp_val_accuracy)
    return loss, accuracy, val_loss, val_accuracy ,names

states = [20 ,50, 100, 200, 500]
models = ["lstm"]
loss, accuracy, val_loss, val_accuracy, names = compute(models, states)
plt.figure()
for i in range(len(names)):
    name = names[i]
    plt.plot(val_accuracy[i], label=name+"_accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title("LSTM Test Accuracy")

plt.figure()
for i in range(len(names)):
    name = names[i]
    plt.plot(loss[i], label=name+"_loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')
plt.title("LSTM Training Loss")

states = [20 ,50, 100, 200, 500]
models = ["lstm_mean"]
loss, accuracy, val_loss, val_accuracy, names = compute(models, states)
plt.figure()
for i in range(len(names)):
    name = names[i]
    plt.plot(val_accuracy[i], label=name+"_accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title("LSTM With Mean Pooling Test Accuracy")

plt.figure()
for i in range(len(names)):
    name = names[i]
    plt.plot(loss[i], label=name+"_loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')
plt.title("LSTM With Mean Pooling Training Loss")

states = [20 ,50, 100, 200, 500]
models = ["rnn"]
loss, accuracy, val_loss, val_accuracy, names = compute(models, states)
plt.figure()
for i in range(len(names)):
    name = names[i]
    plt.plot(val_accuracy[i], label=name+"_accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title("Vanilla RNN Test Accuracy")

plt.figure()
for i in range(len(names)):
    name = names[i]
    plt.plot(loss[i], label=name+"_loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')
plt.title("Vanilla RNN Training Loss")

states = [20 ,50, 100, 200, 500]
models = ["rnn_mean"]
loss, accuracy, val_loss, val_accuracy, names = compute(models, states)
plt.figure()
for i in range(len(names)):
    name = names[i]
    plt.plot(val_accuracy[i], label=name+"_accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title("Vanilla RNN With Mean Pooling Test Accuracy")

plt.figure()
for i in range(len(names)):
    name = names[i]
    plt.plot(loss[i], label=name+"_loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')
plt.title("Vanilla RNN With Mean Pooling Training Loss")


states = [100]
models = ["rnn", "rnn_mean", "lstm", "lstm_mean"]
loss, accuracy, val_loss, val_accuracy, names = compute(models, states)
plt.figure()
for i in range(len(names)):
    name = names[i]
    plt.plot(val_accuracy[i], label=name+"_accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title("All Model Test Accuracy With State = 100")

plt.figure()
for i in range(len(names)):
    name = names[i]
    plt.plot(loss[i], label=name+"_loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')
plt.title("All Model Training Loss With State = 100")

plt.show()

