import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.autograd import Variable
import logging

logger = logging.getLogger(__name__)
np.random.seed(1337)
torch.manual_seed(1337)

DATA_PATH = "data/"
LEN_SIZE = 20
EMBEDDING_SIZE = 64
BATCH_SIZE = 128
EPOCH = 100
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
        for X, y in zip(X_list, y_list):
            X_v = Variable(torch.from_numpy(np.swapaxes(X, 0, 1)).float())
            y_v = Variable(torch.from_numpy(y).float(), requires_grad=False)

            self.optimizer.zero_grad()
            y_pred = self.model(X_v, self.model.initHidden(X_v.size()[1]))
            loss = torch.sqrt(self.loss_f(y_pred, y_v))
            loss.backward()
            self.optimizer.step()

            # for log
            loss_list.append(loss.data)

        return sum(loss_list) / len(loss_list)

    def fit(self, X, y, batch_size=32, nb_epoch=10, validation_data=()):
        X_list = batch(X, batch_size)
        y_list = batch(y, batch_size)

        for t in range(1, nb_epoch + 1):
            loss = self._fit(X_list, y_list)
            val_log = ''
            if validation_data:
                val_loss = self.evaluate(validation_data[0], validation_data[1], batch_size)
                val_log = "- val_loss: %06.4f" % (val_loss)

            print("Epoch %s/%s loss: %06.4f %s" % (t, nb_epoch, loss, val_log))
            if t % 10 == 0:
                logger.info("Epoch %s/%s loss: %06.4f %s" % (t, nb_epoch, loss, val_log))

    def evaluate(self, X, y, batch_size=32):
        y_pred = self.predict(X)

        y_v = Variable(torch.from_numpy(y).float(), requires_grad=False)
        loss = torch.sqrt(self.loss_f(y_pred, y_v))
        return loss.data

    def predict(self, X):
        X = Variable(torch.from_numpy(np.swapaxes(X, 0, 1)).float())
        y_pred = self.model(X, self.model.initHidden(X.size()[1]))
        return y_pred


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


def load_data(USECASE):
    df = pd.read_json(DATA_PATH + "output/Observations/Observations.json", lines=True)
    if USECASE == "humidity":
        df["target"] = df["humidity"]
    elif USECASE == "temperature":
        df["target"] = df["temperature"]
        df[df["target"] == -0.1] = 0.1
    else:
        raise ValueError("This usecase is not implemented yet")
    df["target"] = np.log(df["target"])
    X = df[["target"]].to_numpy()
    y = df[["target"]].to_numpy()
    return X, y


def create_sequences(X, y):
    pop = np.zeros((X.shape[0] - LEN_SIZE, LEN_SIZE, X.shape[1]))
    for i in range(X.shape[0] - LEN_SIZE):
        pop[i, :, :] = X[i:i+LEN_SIZE]
    y_final = y[LEN_SIZE:, ]
    return pop, y_final


def main(USECASE):
    logger.info(f"REGRESSION FOR {USECASE}")
    logger.info("Loading the data")
    X, y = load_data(USECASE)
    logger.info("Creating the sequences for the GRU model")
    X, y = create_sequences(X, y)
    y = y.reshape(y.shape[0])
    logger.info("Splitting the dataset into train and test")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    logger.info("Initialising the GRU model")
    model = GRU(X.shape[2], EMBEDDING_SIZE, 1)
    clf = Estimator(model)
    clf.compile(optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
                loss=nn.MSELoss())
    logger.info("Starting to fit the model")
    clf.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=EPOCH,
            validation_data=(X_test, y_test))
    size = len(y_test)
    a = np.exp(y_test).reshape(size)
    b = np.exp(clf.predict(X_test).detach().numpy()).reshape(size)
    rmse = np.sqrt(((a - b)**2).mean())
    avg_err = (a - b).mean()
    logger.info(f"Our {USECASE} GRU Model has a RMSE, on the test set, of: {rmse}")
    logger.info(f"The average error per prediction for {USECASE}, on the test set, is: {avg_err}")
    if SAVE_MODEL:
        logger.info("Saving the regression model")
        torch.save(model, 'models/regression.pt')
    return clf


if __name__ == '__main__':
    main("temperature")
