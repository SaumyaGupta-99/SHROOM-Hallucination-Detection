import matplotlib.pyplot as plt

# Data for different clustering methods
clustering_methods = ["KMeans", "Hierarchical", "Spectral", "Birch", "Mean", "Gaussian Mixture Model", "ScoreWise clustering", "Best ScoreWise clustering"]

# F1 score data
f1_scores = {
    "Machine Translation": [0.3096, 0.4165, 0.4049, 0.5954, 0.5508, 0.4049, 0.625, 0.3096],
    "Definition Modelling": [0.4278, 0.5979, 0.2308, 0.5583, 0.6449, 0.5420, 0.6766, 0.5420],
    "Paraphrasing": [0.3226, 0.2547, 0.2547, 0.3694, 0.2785, 0.2846, 0.4339, 0.2846]
}

# Accuracy data
accuracies = {
    "Machine Translation": [0.5009, 0.2735, 0.3943, 0.6234, 0.3801, 0.3943, 0.6909, 0.5009],
    "Definition Modelling": [0.4288, 0.5836, 0.5374, 0.4680, 0.4751, 0.5249, 0.6139, 0.5249],
    "Paraphrasing": [0.6640, 0.4693, 0.4693, 0.7360, 0.5227, 0.5307, 0.7147, 0.5307]
}

# Plotting F1 scores
plt.figure(figsize=(12, 6))
for task in f1_scores.keys():
    plt.plot(clustering_methods, f1_scores[task], label=task)
plt.title("F1 Score for Different Clustering Methods")
plt.xlabel("Clustering Methods")
plt.ylabel("F1 Score")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("f1_score_plot.png")

# Plotting accuracies
plt.figure(figsize=(12, 6))
for task in accuracies.keys():
    plt.plot(clustering_methods, accuracies[task], label=task)
plt.title("Accuracy for Different Clustering Methods")
plt.xlabel("Clustering Methods")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_plot.png")
