# Federated Machine Learning System (FedAVG)

## Project Description

A university assignment implementing a distributed machine learning system utilising the Federated Averaging algorithm (FedAVG). The system consists of one server and up to $5$ clients, where each client possesses its own private dataset. A round of learning involves the server broadcasting a model to each of the clients, each of clients training their local model, then a contribution of the respective local models back to the server to allow it to build a new global model.

This program simulates a FedAVG system on a California Housing Dataset to predict median house price.

### Key Features

- **Multi-threaded TCP Server**: Server can handle multiple connections simultaneously.
- **Clients can join anytime**: The server can handle up to a total of $5$ clients joining the system at any time through the training process.
- **Handles Client Dropout**: The system can handle client dropouts, as well as extended delays in a client returning its trained model.
- **Client Subsampling**: The server takes as input an integer, $M$, determining whether sub-sampling is enabled. A value of $M$ tells the server to randomly aggregate models form only $M$ out of the pool of clients.

---

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/Finger-Food/fedavg
   cd fedavg
   ```

2. Install dependencies if necessary:
   ```bash
   pip install pandas pickle pytorch
   ```

---

## Running the Federated Learning System

1. Start the server with:
    ```bash
    python3 server.py <Port-Server> <Sub-Client>
    ```
    `Port-Server` is the port the server is hosted on and must be set to `6000` for the purposes of this program.  
    `<Sub-Client>` is an integer from $0$ to $4$ that sets the client sub-sampling.

2. Once the server is active, clients can be added to the system from a different terminal with:
    ```bash
    python client.py <Client-id> <Port-Client> <Opt-Method>'
    ```
    `<Client-id>` indicates the client number, i.e. 'client1', 'client2'... 'client5'  
    `<Port-Client>` indicates the port number of this specific client and must be set to $6000 + id$  
    `<Opt-Method>` refers to the method of optimisation desired: $0$ for Gradient Descent and $1$ for mini batch gradient descent.

    Example:
    ```bash
    python client.py client1 6001 1
    ```

3. Details about each training round of the clients can be found in `<Client-id>_log.txt`, and the learning rate, number of rounds trained, epochs per round, etc., are stored as global variables at the top of the source files for easy modification.