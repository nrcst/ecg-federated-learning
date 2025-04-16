import wfdb
import numpy as np
import os
import argparse

# Define a mapping from annotation symbols to integer labels.
# Adjust this mapping as needed.
BEAT_LABELS = {
    'N': 0,  # Normal beat
    'L': 1,  # Left bundle branch block beat
    'R': 2,  # Right bundle branch block beat
    'A': 3,  # Aberrated atrial premature beat
    'V': 4,  # Premature ventricular contraction
    'F': 5,  # Fusion beat
    'j': 6,  # Nodal (junctional) escape beat
    'e': 7   # Ventricular escape beat
}

def extract_segments(record, ann, window_size=128):
    """
    Given a record and its annotation, extract signal segments
    centered on each annotation sample provided there is enough
    signal data (i.e. avoid boundary issues).
    """
    half_window = window_size // 2
    features = []
    labels = []
    signal = record[:, 0]  # Assume processing lead 1 only
    for sample, symbol in zip(ann.sample, ann.symbol):
        # Only process if we have enough samples before and after the R-peak
        if sample - half_window < 0 or sample + half_window > len(signal):
            continue
        segment = signal[sample - half_window: sample + half_window]
        # Normalize segment (optional)
        segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-6)
        # Map the beat symbol to a numerical label if available.
        if symbol in BEAT_LABELS:
            label = BEAT_LABELS[symbol]
        else:
            continue  # Skip beats that are not mapped
        features.append(segment)
        labels.append(label)
    return np.array(features), np.array(labels)

def preprocess_mitbih(records, client_id, total_clients, window_size=128):
    """
    Preprocess a selection of MIT-BIH records. Distribute records among clients
    using a round-robin assignment.
    """
    all_features = []
    all_labels = []
    for idx, rec_name in enumerate(records):
        # Round-robin assignment: each record is processed by a single client.
        if idx % total_clients != (client_id - 1):  # client_id starts at 1
            continue
        print(f"Processing record {rec_name} for client {client_id}...")
        # Read record and its annotation using WFDB.
        try:
            record = wfdb.rdsamp(rec_name)
            ann = wfdb.rdann(rec_name, 'atr')
        except Exception as e:
            print(f"Error reading record {rec_name}: {e}")
            continue
        features, labels = extract_segments(record[0], ann, window_size)
        all_features.append(features)
        all_labels.append(labels)
    if all_features:
        X = np.concatenate(all_features, axis=0)
        y = np.concatenate(all_labels, axis=0)
    else:
        X, y = np.array([]), np.array([])  # in case no records assigned
    return X, y

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess MIT-BIH data for a client.')
    parser.add_argument('--client_id', type=int, required=True, help='Client ID (starting from 1)')
    parser.add_argument('--total_clients', type=int, required=True, help='Total number of clients')
    parser.add_argument('--data_dir', type=str, default='.', help='Directory containing MIT-BIH records')
    args = parser.parse_args()

    # List of MIT-BIH record names (without file extension).
    # These files should be available in args.data_dir.
    records = [str(i) for i in range(100, 235)]
    
    # Change the working directory to where the MIT-BIH records are located
    os.chdir(args.data_dir)

    X, y = preprocess_mitbih(records, args.client_id, args.total_clients, window_size=128)
    if X.size == 0:
        print("No features extracted. Please check that the records and annotations are available.")
    else:
        output_file = f"client_{args.client_id}_mitbih_morphology.npz"
        np.savez(output_file, features=X, labels=y)
        print(f"Saved preprocessed data for client {args.client_id} to {output_file}")