import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV
run = 9
print(f"run: {run}")
train = 15
df = pd.read_csv(f"run_{run}/train/results.csv")

# Create a figure
plt.figure(figsize=(12, 6))

# Plot training and validation metrics
plt.plot(df["metrics/mAP50(B)"], label="mAP50", color="blue")
plt.plot(df["metrics/mAP50-95(B)"], label="mAP50-95(B)", color="purple")
plt.plot(df["metrics/recall(B)"], label="metrics/recall(B)", color="green")
plt.plot(df["metrics/precision(B)"], label="precision", color="black")


# Add labels and title

plt.xlabel("Epoch"), plt.ylabel("Value")
plt.title(f"run {run} metric")
plt.legend()

# Save the plot instead of showing it
plt.savefig(f"run_{run}_metric.png", dpi=300, bbox_inches='tight')
plt.close()  # Clean up memory