import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.autograd import Variable

np.random.seed(1337)
torch.manual_seed(1337)

DATA_PATH = "/home/jeremy/Documents/isepAI/data"
USECASE = "temperature"  # humidity or temperature
LEN_SIZE = 3
EMBEDDING_SIZE = 64
BATCH_SIZE = 32
EPOCH = 100
if USECASE == "temperature":
    CLASS_SIZE = 2
elif USECASE == "humidity":
    CLASS_SIZE = 3
else:
    raise ValueError("Usecase not set")
SAVE_MODEL = True


def batch(tensor, batch_size):
    tensor_list = []
    length = tensor.shape[0]
    i = 0
    while True:
        if (i+1) * batch_size >= length:
            tensor_list.append(tensor[i * batch_size: length])
            return tensor_list
        tensor_list.append(tensor[i * batch_size: (i+1) * batch_size])
        i += 1


class Estimator(object):

    def __init__(self, model):
        self.model = model

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss_f = loss

    def _fit(self, X_list, y_list):
        """
        train one epoch
        """
        loss_list = []
        acc_list = []
        for X, y in zip(X_list, y_list):
            X_v = Variable(torch.from_numpy(np.swapaxes(X, 0, 1)).float())
            y_v = Variable(torch.from_numpy(y).long(), requires_grad=False)

            self.optimizer.zero_grad()
            y_pred = self.model(X_v, self.model.initHidden(X_v.size()[1]))
            loss = self.loss_f(y_pred, y_v)
            loss.backward()
            self.optimizer.step()

            # for log
            loss_list.append(loss.data)
            classes = torch.topk(y_pred, 1)[1].data.numpy().flatten()
            acc = self._accuracy(classes, y)
            acc_list.append(acc)

        return sum(loss_list) / len(loss_list), sum(acc_list) / len(acc_list)

    def fit(self, X, y, batch_size=32, nb_epoch=10, validation_data=()):
        X_list = batch(X, batch_size)
        y_list = batch(y, batch_size)

        for t in range(1, nb_epoch + 1):
            loss, acc = self._fit(X_list, y_list)
            val_log = ''
            if validation_data:
                val_loss, val_acc = self.evaluate(validation_data[0], validation_data[1],
                                                  batch_size)
                val_log = "- val_loss: %06.4f - val_acc: %06.4f" % (val_loss, val_acc)
            print("Epoch %s/%s loss: %06.4f - acc: %06.4f %s" % (t, nb_epoch, loss, acc, val_log))

    def evaluate(self, X, y, batch_size=32):
        y_pred = self.predict(X)

        y_v = Variable(torch.from_numpy(y).long(), requires_grad=False)
        loss = self.loss_f(y_pred, y_v)

        classes = torch.topk(y_pred, 1)[1].data.numpy().flatten()
        acc = self._accuracy(classes, y)
        return loss.data, acc

    def _accuracy(self, y_pred, y):
        return sum(y_pred == y) / y.shape[0]

    def predict(self, X):
        X = Variable(torch.from_numpy(np.swapaxes(X, 0, 1)).float())
        y_pred = self.model(X, self.model.initHidden(X.size()[1]))
        return y_pred

    def predict_classes(self, X):
        return torch.topk(self.predict(X), 1)[1].data.numpy().flatten()


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()

        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        _, hn = self.gru(input, hidden)
        # from (1, N, hidden) to (N, hidden)
        rearranged = hn.view(hn.size()[1], hn.size(2))
        out1 = self.linear(rearranged)
        return out1

    def initHidden(self, N):
        return Variable(torch.randn(1, N, self.hidden_size))


def load_data():
    df = pd.read_json(DATA_PATH + "/output/Observations/Observations.json", lines=True)
    class_weights = []
    if USECASE == "humidity":
        df["target"] = np.where((df['humidity'] < 40), 1, 0)
        df["target"] = np.where((df["humidity"] > 85), 2, df["target"])
        class_weights = [1.25, 5., 5.]
    elif USECASE == "temperature":
        df["target"] = np.where((df["temperature"] < 7), 1, 0)
        class_weights = [0.1, 0.9]
    else:
        raise ValueError("This usecase is not implemented yet")
    X = df[["humidity", "temperature"]].to_numpy()
    y = df[["target"]].to_numpy()
    return X, y, class_weights


def create_sequences(X, y):
    pop = np.zeros((X.shape[0] - LEN_SIZE, LEN_SIZE, X.shape[1]))
    for i in range(X.shape[0] - LEN_SIZE):
        pop[i, :, :] = X[i:i+LEN_SIZE]
    y_final = y[LEN_SIZE:, ]
    return pop, y_final


def freq_classes(a):
    y = np.bincount(a).astype(float)
    ii = np.nonzero(y)[0]
    out = np.vstack((ii, y[ii])).T
    out[:, 1] = out[:, 1] / len(a)
    return out


def main():
    X, y, class_weights = load_data()
    X, y = create_sequences(X, y)
    y = y.reshape(y.shape[0])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    model = GRU(X.shape[2], EMBEDDING_SIZE, CLASS_SIZE)
    clf = Estimator(model)
    clf.compile(optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
                loss=nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights)))
    clf.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=EPOCH,
            validation_data=(X_test, y_test))
    score, acc = clf.evaluate(X_test, y_test)
    print('Test score:', score)
    print('Test accuracy:', acc)

    clf.evaluate(X_test, y_test)
    a = clf.predict_classes(X_test)
    print(freq_classes(a))
    print(freq_classes(y_test))

    if SAVE_MODEL:
        torch.save(model, 'models/classification_' + USECASE + '.pt')
    return clf


if __name__ == '__main__':
    main()
