import socket
import torch
import pickle
from Crypto.Cipher import AES
import os
import numpy as np
import collections
import copy
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

from collections import Counter

from model.models import EcgResNet34

HOST = '127.0.0.1'
PORT = 65431
KEY = b'0123456789abcdef'

def recvall(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def recv_msg(sock):
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = int.from_bytes(raw_msglen, byteorder='big')
    return recvall(sock, msglen)

def send_msg(sock, data):
    msg_length = len(data).to_bytes(4, byteorder='big')
    sock.sendall(msg_length)
    sock.sendall(data)

def encrypt(data, key):
    cipher = AES.new(key, AES.MODE_EAX)
    nonce = cipher.nonce
    ciphertext, tag = cipher.encrypt_and_digest(data)
    return nonce, ciphertext, tag

def decrypt(nonce, ciphertext, tag, key):
    cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
    data = cipher.decrypt_and_verify(ciphertext, tag)
    return data

def load_preprocessed_data(client_id):
    file_path = f"./client_{client_id}_data.npz"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Preprocessed file {file_path} not found. Please run prepare_data.py first.")
    data = np.load(file_path)
    return data['features'], data['labels']

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs = inputs.float().unsqueeze(1).to(device)
            labels = labels.long().to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Predicted class distribution:", collections.Counter(all_preds))

    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds,
        average='weighted',
        zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    print("\n==== Client Model Evaluation ====")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("================================\n")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def train_local_model(model, train_loader, device, global_weights=None, mu=0.001, epochs=10, learning_rate=0.001):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        total_batches = len(train_loader)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

        for inputs, labels in pbar:
            inputs = inputs.float().unsqueeze(1).to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Add FedProx proximal term
            if global_weights is not None:
                proximal_term = 0.0
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        global_param = global_weights[name].to(device)
                        proximal_term += ((param - global_param) ** 2).sum()
                loss += (mu / 2) * proximal_term

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{running_loss/(pbar.n+1):.4f}"})

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/total_batches:.4f}")

    return model

def add_noise(model, sensitivity, epsilon):
    print("Adding differential privacy noise...")
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn_like(param) * (sensitivity / epsilon) * 0.1
            param.add_(noise)
    return model

if __name__ == '__main__':
    client_id = 1
    print(f"Client {client_id} starting...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading preprocessed data...")
    features, labels = load_preprocessed_data(client_id)

    split_idx = int(0.8 * len(features))
    X_train, X_test = features[:split_idx], features[split_idx:]
    y_train, y_test = labels[:split_idx], labels[split_idx:]

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              pin_memory=True if device.type == 'cuda' else False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             pin_memory=True if device.type == 'cuda' else False)

    local_model = EcgResNet34(num_classes=8).to(device)
    print("Model initialized.")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((HOST, PORT))
            print(f"Connected to server at {HOST}:{PORT}")

            print("\n--- Starting Local Training ---")
            trained_model = train_local_model(local_model, train_loader, device, epochs=1)
            print("--- Local Training Complete ---")

            print("--- Evaluating Local Model ---")
            local_metrics = evaluate_model(trained_model, test_loader, device)

            trained_model = trained_model.to('cpu')
            trained_model = add_noise(trained_model, sensitivity=0.01, epsilon=1.0)

            print("--- Sending model to server ---")
            model_data = pickle.dumps(trained_model)
            nonce, ciphertext, tag = encrypt(model_data, KEY)
            send_msg(s, nonce)
            send_msg(s, ciphertext)
            send_msg(s, tag)

            label_counts = dict(Counter(y_train.tolist()))
            label_data = pickle.dumps(label_counts)
            send_msg(s, label_data)

            num_samples = len(y_train)
            send_msg(s, pickle.dumps(num_samples))

            print("--- Waiting for global model ---")
            nonce_recv = recv_msg(s)
            ciphertext_recv = recv_msg(s)
            tag_recv = recv_msg(s)

            if nonce_recv is None or ciphertext_recv is None or tag_recv is None:
                raise ValueError("Incomplete data received from server.")

            decrypted_data = decrypt(nonce_recv, ciphertext_recv, tag_recv, KEY)
            updated_global_model = pickle.loads(decrypted_data)
            updated_global_model = updated_global_model.to(device)
            global_weights = copy.deepcopy(updated_global_model.state_dict())
            trained_global_model = train_local_model(updated_global_model, train_loader, device, global_weights=global_weights, mu=0.001, epochs=1)

            print("--- Evaluating Updated Global Model ---")
            global_metrics = evaluate_model(trained_global_model, test_loader, device)

            print("\n--- Model Comparison ---")
            print(f"Local Model Accuracy: {local_metrics['accuracy']:.2f}%")
            print(f"Global Model Accuracy: {global_metrics['accuracy']:.2f}%")
            acc_diff = global_metrics['accuracy'] - local_metrics['accuracy']
            print(f"Accuracy Difference: {acc_diff:.2f}% ({'improved' if acc_diff > 0 else 'decreased'})")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            s.close()

    print(f"Client {client_id} finished.")
