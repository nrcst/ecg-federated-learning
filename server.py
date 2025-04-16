import socket
import threading
import pickle
import torch
from collections import OrderedDict
from Crypto.Cipher import AES
from model.models import EcgResNet34
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import logging  # changed code
logging.basicConfig(level=logging.INFO)  # changed code

HOST = '127.0.0.1'
PORT = 65431
KEY = b'0123456789abcdef'
NUM_CLIENTS = 2

client_counter_lock = threading.Lock()
client_counter = 0
client_connections = [None] * NUM_CLIENTS
client_models = [None] * NUM_CLIENTS
client_weights = [0.5, 0.5]

sync_barrier = threading.Barrier(NUM_CLIENTS)

def recvall(conn, n):
    """
    Receives exactly 'n' bytes from the given connection (conn).
    Returns the received data or None if the connection is closed.
    """
    data = b''
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def recv_msg(conn):
    """
    Receives a message framed by a 4-byte length header from the connection.
    Returns the message data or None if incomplete.
    """
    raw_msglen = recvall(conn, 4)
    if not raw_msglen:
        return None
    msglen = int.from_bytes(raw_msglen, byteorder='big')
    return recvall(conn, msglen)

def send_msg(conn, data):
    """
    Sends 'data' to the connection, framing it with a 4-byte length header.
    """
    msg_length = len(data).to_bytes(4, byteorder='big')
    conn.sendall(msg_length)
    conn.sendall(data)

def encrypt(data, key):
    """
    Encrypts 'data' using the provided 'key' with AES EAX mode.
    Returns the nonce, ciphertext, and tag.
    """
    cipher = AES.new(key, AES.MODE_EAX)
    nonce = cipher.nonce
    ciphertext, tag = cipher.encrypt_and_digest(data)
    return nonce, ciphertext, tag

def decrypt(nonce, ciphertext, tag, key):
    """
    Decrypts the 'ciphertext' using the provided 'key' with AES EAX mode.
    Raises an exception if verification fails.
    """
    cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
    return cipher.decrypt_and_verify(ciphertext, tag)

def federated_averaging(global_model, client_models, client_weights):
    """
    Performs federated averaging on the given client models
    to update the weights of 'global_model'.
    """
    with torch.no_grad():
        global_state = global_model.state_dict()
        new_state = {}
        total_weight = sum(client_weights)
        for key in global_state.keys():
            accum = 0
            for client_model, weight in zip(client_models, client_weights):
                accum += client_model.state_dict()[key] * weight
            new_state[key] = accum / total_weight
        global_model.load_state_dict(new_state)
    return global_model

def handle_client(conn, addr, client_index, global_model):
    """
    Handles a single client connection by receiving its model,
    performing federated averaging, and sending the updated global model back.
    """
    print(f"Connected by {addr} as client index {client_index}")
    client_connections[client_index] = conn
    try:
        nonce = recv_msg(conn)
        ciphertext = recv_msg(conn)
        tag = recv_msg(conn)
        if nonce is None or ciphertext is None or tag is None:
            raise ValueError("Incomplete data received from client.")

        decrypted_data = decrypt(nonce, ciphertext, tag, KEY)
        client_model   = pickle.loads(decrypted_data)
        client_models[client_index] = client_model
        
        barrier_id = sync_barrier.wait()
        if barrier_id == 0:
            federated_averaging(global_model, client_models, client_weights)
            print("Global model updated via federated averaging.")
            for i in range(NUM_CLIENTS):
                client_models[i] = None

        sync_barrier.wait()

        model_bytes = pickle.dumps(global_model)
        nonce_send, ciphertext_send, tag_send = encrypt(model_bytes, KEY)
        send_msg(conn, nonce_send)
        send_msg(conn, ciphertext_send)
        send_msg(conn, tag_send)

        del client_model
        del decrypted_data
        del model_bytes

    except Exception as e:
        print(f"Error handling client {addr}: {e}")
    finally:
        conn.close()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    global_model = EcgResNet34(num_classes=8)
    global_model.to(torch.device('cpu'))  # Move model to CPU or GPU as needed

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # changed code
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {HOST}:{PORT}")
        while True:
            conn, addr = s.accept()
            with client_counter_lock:
                idx = client_counter % NUM_CLIENTS
                client_counter += 1
            threading.Thread(
                target=handle_client,
                args=(conn, addr, idx, global_model)
            ).start()