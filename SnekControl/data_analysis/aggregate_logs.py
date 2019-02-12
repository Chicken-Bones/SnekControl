import re
import os
import csv

with open('aggregate.csv', 'w', newline='') as out_file:
    csvw = csv.writer(out_file)
    csvw.writerow(["timestamp", "model", "hidden_dim", "seq_len", "epochs", "duration", "train", "test"])

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

        prev_seconds = None
        elapsed = 0
        loss = []
        for m in re.finditer("\[([\d:]+)\] Epoch (\d+) MSE Train: ([\d.]+), Test ([\d.]+)", text):
            (timestamp, epochs, train, test) = m.groups()

            loss.append((epochs, train, test))
            seconds = sum(int(x) * 60 ** i for i, x in enumerate(reversed(timestamp.split(":"))))
            if prev_seconds is not None:
                delta = seconds - prev_seconds
                if delta < 0: delta += 24*60*60
                elapsed += delta

            prev_seconds = seconds

        if prev_seconds is None:
            continue

        m, s = divmod(elapsed, 60)
        h, m = divmod(m, 60)
        duration = "%02d:%02d:%02d" % (h, m, s)
        csvw.writerow([dir_name[3:], model, hidden_dim, seq_len, epochs, duration, train, test])

        with open(os.path.join('.', dir_name, 'loss.csv'), 'w', newline='') as loss_file:
            csvw2 = csv.writer(loss_file)
            csvw2.writerow(["epoch", "train", "test"])
            for epoch, train, test in loss:
                csvw2.writerow([epoch, train, test])


