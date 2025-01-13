LEARNING_RATE = 0.011
MINI_BATCH_SIZE = 64
EPOCHS = 2

import _thread
import socket
import queue
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import pickle


HOST = 'localhost'
TARGET = 'localhost'
SERVER_PORT = 6000


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        # Create a linear transformation to the incoming data
        self.linear = nn.Linear(input_size, 1)

    # Define how the model is going to be run, from input to output
    def forward(self, x):
        # Apply linear transformation
        output = self.linear(x)
        return output.reshape(-1)

class Client:
    def __init__(self, client_id, port, opt_method):
        self.client_id = client_id
        self.id = ord(client_id[-1]) - ord('0')
        self.port = port
        self.opt_method = opt_method
        self.param_q = queue.Queue()
        self.iter = 0
        self.training = False

        self.X_train, self.y_train, self.X_test, self.y_test, self.train_samples, self.test_samples = self.get_data(client_id)
        self.model = LinearRegressionModel(len(self.X_train[0]))

        self.train_data = [(x, y) for x, y in zip(self.X_train, self.y_train)]
        self.test_data = [(x, y) for x, y in zip(self.X_test, self.y_test)]

        # if mini batch, set batch size accordingly
        self.batch_size = MINI_BATCH_SIZE if self.opt_method else self.train_samples

        # Define dataloader for iterable sample over a dataset
        self.trainloader = DataLoader(self.train_data, batch_size = self.batch_size)
        self.testloader = DataLoader(self.test_data, batch_size = self.test_samples)

        # Define the Mean Square Error Loss
        self.loss = nn.MSELoss()

        # Define the Gradient Descent optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE)

       
    def get_data(self, client_id):
        #train
        df_train = pd.read_csv(f"./FLData/calhousing_train_{client_id}.csv")
        input_features_train = df_train[['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude']]
        output_train = df_train['MedHouseVal']
        input_features_train_arr = input_features_train.values
        output_train_arr = output_train.values
        input_train_tensor = torch.tensor(input_features_train_arr).type(torch.float32)
        output_train_tensor = torch.tensor(output_train_arr).type(torch.float32)
        #test
        df_test = pd.read_csv(f"./FLData/calhousing_test_{client_id}.csv")
        input_features_test = df_test[['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude']]
        output_test = df_test['MedHouseVal']
        input_features_test_arr = input_features_test.values
        output_test_arr = output_test.values
        input_test_tensor = torch.tensor(input_features_test_arr).type(torch.float32)
        output_test_tensor = torch.tensor(output_test_arr).type(torch.float32)
        
        return input_train_tensor, output_train_tensor, input_test_tensor, output_test_tensor, len(output_train_tensor), len(output_test_tensor)


    def train(self, epochs):
        for _ in range(1, epochs + 1):

            self.model.train()
            for x, y in self.trainloader:

                self.optimizer.zero_grad()

                output = self.model(x)
                loss = self.loss(output, y)                
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()

        return loss.data

    def test(self):
        self.model.eval()

        x, y = list(self.testloader)[0]

        y_pred = self.model(x)
        return self.loss(y_pred, y)
    
    def set_parameters(self, model):
        for old_param, new_param in zip(self.model.parameters(), model):
            old_param.data = new_param.data.clone()



    def listen(self):  # server
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((HOST, self.port))
                s.listen(5)
                while True:
                    client_socket, addr = s.accept()
                    data = client_socket.recv(1024)
                    if not data:
                        continue

                    client_socket.sendall(b"received")

                    if data == b"finito":
                        self.param_q.put("finito")
                        return

                    model = pickle.loads(data)
                    if not self.training:
                        self.iter = model[0]
                        self.param_q.put(model[1:])
                        self.training = True

                    client_socket.close()
        
        except:
            print("Could not connect to server socket")


    def send_handshake(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((TARGET, SERVER_PORT))

                packet = pickle.dumps((0, self.id, self.train_samples))
                s.send(packet)

                data = s.recv(1024)
                if data and data.decode() == "received":
                    return
            
        except Exception as e:
            print("Failed to make handshake with server")
            print(e.with_traceback(None))
    
    def send_model(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((TARGET, SERVER_PORT))
                packet = pickle.dumps((self.iter, self.id, list(self.model.parameters())))
                s.sendall(packet)
        
        except Exception as e:
            print("Failed to send model to server")
            print(e.with_traceback(None))


    def run(self):
        # Listening thread
        try:
            _thread.start_new_thread(self.listen, ())
        except Exception as e:
            print("Failed to start listening thread")
            print(e.with_traceback(None))

        self.send_handshake()

        round = 0
        testing_MSEs = []
        training_MSEs = []


        while True:
            round += 1

            model = self.param_q.get()
            if model == 'finito':
                break
            
            # if queue not empty at this time something's not right
            print(f"I am client {self.id}")

            self.set_parameters(model)
            print("Received new global model")

            testing_MSEs.append(self.test())
            print(f"Testing MSE: {testing_MSEs[-1]}")

            print("Local training...")
            training_MSEs.append(self.train(EPOCHS))
            print(f"Training MSE: {training_MSEs[-1]}")

            print("Sending new local model")
            self.send_model()
            self.training = False

            print()
                

        with open(f"{self.client_id}_log.txt", "w") as f:
            if self.opt_method:
                print(f"Learning rate: {LEARNING_RATE}, Methodology: Mini-batch of size {MINI_BATCH_SIZE}, Epochs per round: {EPOCHS}", file=f)
            else:
                print(f"Learning rate: {LEARNING_RATE}, Methodology: Gradient Descent, Epochs per round: {EPOCHS}", file=f)

            for round, (testing_mse, training_mse) in enumerate(zip(testing_MSEs, training_MSEs)):
                print(f"Round {round + 1}: Testing MSE = {testing_mse}, Training MSE = {training_mse}", file=f)




def validate_args(client_id, port, opt_method):
    if client_id[:-1] != "client" or not client_id[-1].isdigit():
        print("Invalid client id. Should be of the format clientX, where X can be 1 to 5.")
        return False
    
    id = int(client_id[-1])
    if id < 1 or id > 5:
        print("Invalid client id. Should be of the format clientX, where X can be 1 to 5.")
        return False
    
    if not port.isdigit() or int(port) != SERVER_PORT + id:
        print("Invalid port. Should be 6000 + client id no.")
        return False
    
    if not opt_method.isdigit() or int(opt_method) not in [0, 1]:
        print("Invalid optimisation number, should be either '0' for GD or '1' for mini-batch GD.")
        return False

    return True

if len(sys.argv) != 4:
    print("Incorrect number of arguments.\n\
          Usage: python client.py <Client-id> <Port-Client> <Opt-Method>")

else:
    _, client_id, port, opt_method = sys.argv
    
    if validate_args(client_id, port, opt_method):
        client = Client(client_id, int(port), int(opt_method))
        client.run()