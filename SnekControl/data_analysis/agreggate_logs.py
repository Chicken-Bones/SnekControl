import re
import os
import csv

with open('aggregate.csv', 'w', newline='') as out_file:
    csvw = csv.writer(out_file)
    csvw.writerow(["timestamp", "model", "hidden_dim", "seq_len", "epochs", "train", "test"])

    for dir_name in os.listdir('.'):
        log_path = os.path.join(dir_name, 'log.txt')
        if not os.path.exists(log_path):
            continue

        with open(log_path, 'r') as file:
            text = file.read()

        m = re.search("hidden_dim=(\d+), model='(\w+)', seq_len=(\d+)", text)
        if m is None:
            continue

        (hidden_dim, model, seq_len) = m.groups()

        for m in re.finditer("Epoch (\d+) MSE Train: ([\d.]+), Test ([\d.]+)", text):
            pass

        (epochs, train, test) = m.groups()

        csvw.writerow([dir_name[3:], model, hidden_dim, seq_len, epochs, train, test])


