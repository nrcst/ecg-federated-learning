# Federated ECG Classification

This repository implements federated learning setup for ECG signal classification using a ResNet‑34 1D model.

## Prerequisites

- Python 3.7.9  
- Conda (optional, but recommended)

## 1. Create & Activate Conda Environment (optional)

```bash
conda create -n fl-ecg_env python=3.7.9
conda activate fl-ecg_env
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Download & Prepare Dataset

1. Download MIT‑BIH dataset (https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip):
   ```bash
   curl -O <DATASET_URL>.zip
   ```
2. Unzip & move files:
   ```bash
   unzip <DATASET_URL>.zip -d mit-bih
   ```
3. Prepare per-client data:
   ```bash
   python data/prepare_data.py --client_id 1 --total_clients 2
   python data/prepare_data.py --client_id 2 --total_clients 2
   ```

This will generate `client_1_mitbih_morphology.npz` and `client_2_mitbih_morphology.npz` in the `mit-bih/` folder.

## 4. Configure Server & Clients

- Edit `server.py` HOST, PORT, and `NUM_CLIENTS` as needed.
- Edit `client.py` (or `client-2.py`) HOST, PORT, and `client_id`.

## 5. Run Federated Learning

1. Start the server:
   ```bash
   python server.py
   ```
2. In separate terminals, start each client:
   ```bash
   python client.py        # client_id = 1
   python client-2.py      # client_id = 2
   ```

Clients will train locally, send encrypted weights to the server, and receive the updated global model.

## 6. Results

After each round, clients print:
- Local model metrics
- Global model metrics
- Accuracy difference

Enjoy experimenting with federated ECG classification!
```