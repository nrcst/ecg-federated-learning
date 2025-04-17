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

HOST = '127.0.0.1'
PORT = 65431
KEY = b'0123456789abcdef'
NUM_CLIENTS = 2

logging.basicConfig(level=logging.INFO)

client_counter_lock = threading.Lock()
client_counter = 0
client_connections = [None] * NUM_CLIENTS
client_weights = [None] * NUM_CLIENTS

sync_barrier = threading.Barrier(NUM_CLIENTS)

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

def encrypt(data, key):
    cipher = AES.new(key, AES.MODE_EAX)
    nonce = cipher.nonce
    ciphertext, tag = cipher.encrypt_and_digest(data)
    return nonce, ciphertext, tag

def decrypt(nonce, ciphertext, tag, key):
    cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
    return cipher.decrypt_and_verify(ciphertext, tag)

def weighted_average(state_dicts, sample_counts):
    total = sum(sample_counts)
    w_avg = copy.deepcopy(state_dicts[0])
    for k in w_avg.keys():
        w_avg[k] = state_dicts[0][k] * sample_counts[0]
        for i in range(1, len(state_dicts)):
            w_avg[k] += state_dicts[i][k] * sample_counts[i]
        w_avg[k] /= total
    return w_avg

def handle_client(conn, addr, client_index, global_model):
    print(f"Connected by {addr} as client index {client_index}")
    client_connections[client_index] = conn

    try:
        nonce = recv_msg(conn)
        ciphertext = recv_msg(conn)
        tag = recv_msg(conn)
        if nonce is None or ciphertext is None or tag is None:
            raise ValueError("Incomplete data received from client.")

        decrypted_data = decrypt(nonce, ciphertext, tag, KEY)
        client_model = pickle.loads(decrypted_data)

        client_weights[client_index] = copy.deepcopy(client_model.state_dict())
        logging.info(f"Client {client_index} model received.")

        label_dist_bytes = recv_msg(conn)
        if label_dist_bytes:
            try:
                class_dist = pickle.loads(label_dist_bytes)
                logging.info(f"Real class distribution for client {client_index}: {class_dist}")
            except Exception as e:
                logging.warning(f"Could not parse label distribution from client {client_index}: {e}")
        else:
            logging.warning(f"No label distribution received from client {client_index}.")

        barrier_id = sync_barrier.wait()

        if barrier_id == 0:
            if any(w is None for w in client_weights):
                logging.warning("Missing client weights. Skipping aggregation.")
            else:
                global_state_dict = weighted_average(client_weights, [1]*len(client_weights))
                global_model.load_state_dict(global_state_dict)
                logging.info("Global model updated via federated averaging.")
                for i in range(NUM_CLIENTS):
                    client_weights[i] = None

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
