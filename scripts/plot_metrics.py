"""
This script plots the training state (train/vall loss, lr, gradient norm) from a logs.json.
"""

import matplotlib.pyplot as plt
import numpy as np
import json

log_file = 'weights/logs.json'
output_image = 'results/metrics_plot.png'

with open(log_file, 'r') as file:
    data = json.load(file)

steps = [entry['step'] for entry in data]
tokens = [entry['tokens'] for entry in data]
train_loss = [entry['train_loss'] for entry in data]
val_loss = [entry['val_loss'] if entry['val_loss'] is not None else float('nan') for entry in data]
val_loss_mask = np.isfinite(val_loss)
learning_rate = [entry['learning_rate'] for entry in data]
gradient_norm = [entry['gradient_norm'] for entry in data]

# plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
plt.title('Metrics Over Training Steps')

# first plot: losses = f(tokens)
ax1.set_xlabel('Tokens')
ax1.set_ylabel('Loss')
ax1.plot(tokens, train_loss, 'b-', label='Train Loss', alpha=.7)
ax1.plot(np.array(tokens)[val_loss_mask], np.array(val_loss)[val_loss_mask], 'r-', label='Validation Loss')
ax1.set_title('Loss Metrics')
ax1.legend()

# second plot: lr and gradient_norm = f(tokens)
ax2.set_xlabel('Tokens')
ax2.set_ylabel('Learning Rate', color='g')
line1, = ax2.plot(tokens, learning_rate, 'g-.', label='Learning Rate', markersize=8)
ax2_twin = ax2.twinx()
ax2_twin.set_ylabel('Gradient Norm', color='k')
line2, = ax2_twin.plot(tokens, gradient_norm, 'k--', label='Gradient Norm')
lines, labels = [], []
lines.extend([line1, line2])
labels.extend(['Learning Rate', 'Gradient Norm'])
ax2.legend(lines, labels, loc='upper right')
ax2.set_title('Learning Rate and Gradient Norm')
ax2_twin.grid(False)  

plt.tight_layout()

plt.savefig(output_image)