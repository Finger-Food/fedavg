ITERATIONS = 277
EPOCHS = 2

import torch
import torch.nn as nn
import pickle
from threading import Thread
import _thread
import socket
import queue
import time
from sys import argv
import random

HOST = 'localhost'
TARGET = 'localhost'
MAX_CLIENTS = 5
SERVER_PORT = 6000
START_BUFFER = 5

PORT = 0
MODEL = 1
NUM_TRAIN_SAMPLES = 2
ACTIVE = 3

torch.manual_seed(42)

class LinearRegressionModel(nn.Module):
    def __init__(self, input_size = 8):
        super(LinearRegressionModel, self).__init__()
        # Create a linear transformation to the incoming data
        self.linear = nn.Linear(input_size, 1)

    # Define how the model is going to be run, from input to output
    def forward(self, x):
        # Apply linear transformation
        output = self.linear(x)
        return output.reshape(-1)



class Server:
    def __init__(self, port, sub_client):
        self.sub_client = sub_client
        self.port = port

        self.clients = [None] * (MAX_CLIENTS + 1)
        self.active_clients = 0

        self.model = LinearRegressionModel()
        self.client_q = queue.Queue()
        


    def listen(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((HOST, self.port))
                s.listen(5)
                while True:
                    client_socket, addr = s.accept()
                    try:
                        _thread.start_new_thread(self.receive_msg, (client_socket, addr))
                    except Exception as e:
                        print("Failed to listening client socket thread")
                        print(e.with_traceback(None))
                        pass
        
        except:
            print("Could not connect to server socket")

    def receive_msg(self, socket, address):  # client socket to handle client
        # print("recevied connection from", address)
        data = socket.recv(1024)

        if not data:
            # print("Error receiving data from", address)
            return
        
        socket.sendall(b"received")

        item = pickle.loads(data)
        if item[0] == 0 or item[0] == self.glob_iter:
            self.client_q.put(item[1:])

        socket.close()



    def broadcast_model(self):
        param_packet = pickle.dumps([self.glob_iter] + list(self.model.parameters()))
        sending_threads = list()
        for client_entry in self.clients:
            if client_entry == None:
                continue  ## user with this id has not yet handshaked

            try:
                t = Thread(target=self.send_parameters, args=(param_packet, client_entry))
                t.start()
                sending_threads.append(t)

            except:
                print("could not start parameter packet sending thread")
        
        for t in sending_threads:
            t.join()

    def send_parameters(self, packet, client_entry):
        port = client_entry[PORT]
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((TARGET, port))
                s.sendall(packet)

                data = s.recv(1024)

                if data and data.decode() == "received":
                    # print("success")
                    client_entry[ACTIVE] = True
                    client_entry[MODEL] = None
                    return
                
        except Exception as e:
            print(f"Failed to send model to client{client_entry[PORT] - 6000}")
            # print(e.with_traceback(None))

        client_entry[ACTIVE] = False

    def terminate_clients(self):
        for client_entry in self.clients:
            if client_entry == None:
                continue

            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect((TARGET, client_entry[PORT]))
                    s.sendall(b"finito")
                    
            except Exception as e:
                pass
                # print("failed to terminate", self.port - 6000)
                # print(e.with_traceback(None))


    def get_handshakes(self, is_first):
        if is_first:
            id, sample_size = self.client_q.get()
            self.clients[id] = [self.port + id, None, sample_size, False]
        
        else:
            while not self.client_q.empty():
                id, sample_size = self.client_q.get()
                self.clients[id] = [self.port + id, None, sample_size, False]

    def count_clients(self):
        count = 0
        for client in self.clients:
            if client is not None and client[ACTIVE]:
                count += 1

        return count
    
    def get_models(self):
        remaining_users = self.active_clients
        while remaining_users or not self.client_q.empty():
            try:
                id, data = self.client_q.get(timeout=2*EPOCHS)
            except:
                id = None

            # for dropouts--models not received after a given time
            if id == None:
                for id, client_data in enumerate(self.clients):
                    if client_data is not None and client_data[MODEL] == None:
                        client_data[ACTIVE] = False
                        self.active_clients -= 1
                        print(f"Model not received in time from client{id}")
                    
                return


            if self.clients[id] is not None:
                if self.clients[id][ACTIVE]:
                    print(f"Getting local model from client {id}")
                    self.clients[id][MODEL] = data
                    remaining_users -= 1

            else:
                self.clients[id] = [self.port + id, None, data, False]  # item is the number of samples in the data set
        


    def aggregate_parameters(self):
        # Clear global model before aggregation
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)

        # get all active users
        idxes = []
        for i in range(len(self.clients)):
            if self.clients[i] is not None and self.clients[i][ACTIVE]:
                idxes.append(i)

        # sub sample if necessary
        if self.sub_client > 0 and self.sub_client < len(idxes):
            idxes = random.sample(idxes, self.sub_client)

        # get total training samples for the sub sampled group
        total_train_samples = 0
        for idx in idxes:
            total_train_samples += self.clients[idx][NUM_TRAIN_SAMPLES]

        # aggregate new parameters
        for idx in idxes:
            for server_param, user_param in zip(self.model.parameters(), self.clients[idx][MODEL]):
                server_param.data = server_param.data + user_param.data.clone() * self.clients[idx][NUM_TRAIN_SAMPLES] / total_train_samples


    def run(self):
        # Listening thread
        try:
            _thread.start_new_thread(self.listen, ())
        except Exception as e:
            print("Failed to start listening thread")
            print(e.with_traceback(None))
        
        self.get_handshakes(True)

        time.sleep(START_BUFFER)

        self.get_handshakes(False)

        for self.glob_iter in range(1, ITERATIONS + 1):
            print(f"Global iteration {self.glob_iter}:")

            print("Broadcasting new global model")
            self.broadcast_model()

            # Get local models from clients
            self.active_clients = self.count_clients()
            print(f"Total number of clients: {self.active_clients}")
            self.get_models()            

            if self.active_clients == 0:
                print("No models to aggregate")
            else:
                print("Aggregating new global model")
                self.aggregate_parameters()
            
            print()

        self.terminate_clients()


def validate_args(port, sub_client):
    if not port.isdigit() or port != SERVER_PORT:
        print("Invalid port. Should be set to 6000.")
        return False

    if not sub_client.isdigit() or int(sub_client) < 0 or int(sub_client) > 5:
        print("Invalid sub client sample number. Should be between 0 to 5.")
        return False

    return True


if len(argv) != 3:
    print("Incorrect number of arguments.\n"
          "Usage: python server.py <Port-Server> <Sub-Client>")

else:
    _, port, sub_client = argv
    if validate_args(port, sub_client):
        server = Server(int(port), int(sub_client))
        server.run()
    