import socket
import threading
import pickle
import torch
import copy
from collections import OrderedDict
from Crypto.Cipher import AES
from model.models import EcgResNet34
import numpy as np
import logging

# --- Config ---
HOST = '127.0.0.1'
PORT = 65431
KEY = b'0123456789abcdef'
NUM_CLIENTS = 2

# --- Logging ---
logging.basicConfig(level=logging.INFO)

# --- Global Variables ---
client_counter_lock = threading.Lock()
client_counter = 0
client_connections = [None] * NUM_CLIENTS
client_weights = [None] * NUM_CLIENTS
client_data_sizes = [None] * NUM_CLIENTS

sync_barrier = threading.Barrier(NUM_CLIENTS)

# --- Network Utilities ---
def recvall(conn, n):
    data = b''
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def recv_msg(conn):
    raw_msglen = recvall(conn, 4)
    if not raw_msglen:
        return None
    msglen = int.from_bytes(raw_msglen, byteorder='big')
    return recvall(conn, msglen)

def send_msg(conn, data):
    msg_length = len(data).to_bytes(4, byteorder='big')
    conn.sendall(msg_length)
    conn.sendall(data)

# --- Encryption Utilities ---
def encrypt(data, key):
    cipher = AES.new(key, AES.MODE_EAX)
    nonce = cipher.nonce
    ciphertext, tag = cipher.encrypt_and_digest(data)
    return nonce, ciphertext, tag

def decrypt(nonce, ciphertext, tag, key):
    cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
    return cipher.decrypt_and_verify(ciphertext, tag)

# --- Federated Averaging ---
def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def weighted_average_weights(w_list, data_sizes):
    total_data = sum(data_sizes)
    w_avg = copy.deepcopy(w_list[0])
    for key in w_avg.keys():
        w_avg[key] = w_list[0][key] * (data_sizes[0] / total_data)
        for i in range(1, len(w_list)):
            w_avg[key] += w_list[i][key] * (data_sizes[i] / total_data)
    return w_avg

# --- Client Handler ---
def handle_client(conn, addr, client_index, global_model):
    print(f"Connected by {addr} as client index {client_index}")
    client_connections[client_index] = conn

    try:
        # --- Receive Encrypted Model ---
        nonce = recv_msg(conn)
        ciphertext = recv_msg(conn)
        tag = recv_msg(conn)
        if nonce is None or ciphertext is None or tag is None:
            raise ValueError("Incomplete data received from client.")

        decrypted_data = decrypt(nonce, ciphertext, tag, KEY)
        client_model = pickle.loads(decrypted_data)

        # --- Store client weights ---
        client_weights[client_index] = copy.deepcopy(client_model.state_dict())
        logging.info(f"Client {client_index} model received.")

        # --- Receive and log real class distribution ---
        label_dist_bytes = recv_msg(conn)
        if label_dist_bytes:
            try:
                class_dist = pickle.loads(label_dist_bytes)
                logging.info(f"Real class distribution for client {client_index}: {class_dist}")
            except Exception as e:
                logging.warning(f"Could not parse label distribution from client {client_index}: {e}")
        else:
            logging.warning(f"No label distribution received from client {client_index}.")

        # --- Receive number of samples for weighting ---
        num_samples_bytes = recv_msg(conn)
        if num_samples_bytes:
            try:
                num_samples = pickle.loads(num_samples_bytes)
                client_data_sizes[client_index] = num_samples
                logging.info(f"Client {client_index} sample size: {num_samples}")
            except Exception as e:
                logging.warning(f"Could not parse sample size from client {client_index}: {e}")
        else:
            logging.warning(f"No sample size received from client {client_index}.")

        # --- Synchronize ---
        barrier_id = sync_barrier.wait()

        # --- Perform Averaging ---
        if barrier_id == 0:
            if any(w is None for w in client_weights):
                logging.warning("Missing client weights. Skipping aggregation.")
            else:
                global_state_dict = weighted_average_weights(client_weights, client_data_sizes)
                global_model.load_state_dict(global_state_dict)
                logging.info("Global model updated via federated averaging.")
                for i in range(NUM_CLIENTS):
                    client_weights[i] = None

        sync_barrier.wait()

        # --- Send Updated Global Model ---
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

# --- Main Server Loop ---
if __name__ == '__main__':
    global_model = EcgResNet34(num_classes=8)
    global_model.to(torch.device('cpu'))

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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