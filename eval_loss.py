import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV
run = "best"
print(f"run: {run}")
train = 14
df = pd.read_csv(f"best/train/results.csv")

# Create a figure
plt.figure(figsize=(12, 6))

plt.plot(df["val/box_loss"], label="Val Box Loss", color="red", linestyle="--")
plt.plot(df["val/cls_loss"], label="val cls_loss", color="green", linestyle="--")
plt.plot(df["val/dfl_loss"], label="val dfl_loss", color="blue", linestyle="--")

plt.plot(df["train/box_loss"], label="train Box Loss", color="red")
plt.plot(df["train/cls_loss"], label="train cls_loss", color="green")
plt.plot(df["train/dfl_loss"], label="train dfl_loss", color="blue")

# Add labels and title
plt.xlabel("Epoch"), plt.ylabel("Value")
plt.title(f"run_best loss comparsion")
plt.legend()

# Save the plot instead of showing it
plt.savefig(f"run_best_loss.png", dpi=300, bbox_inches='tight')
plt.close()  # Clean up memory