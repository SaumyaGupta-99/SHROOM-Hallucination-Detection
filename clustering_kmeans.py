import shroom_clustering
import pandas as pd
import numpy as np


def clustering(filename, trial_filename, test_filename, task):
    smaller_sample = shroom_clustering.get_data(trial_filename, task)
    larger_data = shroom_clustering.get_data(filename, task)
    test_data = shroom_clustering.get_data(test_filename, task)
    combined_data = np.concatenate((smaller_sample, larger_data), axis=0)
    kmeans, _ = shroom_clustering.kmeans_clustering(combined_data)


    trial = pd.DataFrame(shroom_clustering.read_file('trial-v1.json'))
    task_labels = trial[trial['task'] == task]['label']
    predicted_trial_clusters = kmeans.fit_predict(smaller_sample)

    hallucination_label, non_hallucination_label = shroom_clustering.get_labels(predicted_trial_clusters, task_labels)
    test = pd.DataFrame(shroom_clustering.read_file('test.model-aware.json'))
    test_labels = test[test['task'] == task]['label']
    test_predictions = kmeans.fit_predict(test_data)
    correct = 0
    for cluster, label in zip(test_predictions, test_labels):
        if hallucination_label == cluster and label == 'Hallucination':
            correct += 1
        elif non_hallucination_label == cluster and label == 'Not Hallucination':
            correct += 1
    accuracy = (correct / len(test_labels))
    print("Best Accuracy is:", accuracy)


print("For Machine Translation:")
clustering('shroom-mt-features.csv', 'shroom-mt-features-trial.csv', 'shroom-mt-features-test.csv', "MT")
print("For Definition Modelling:")
clustering('shroom-dm-features.csv', 'shroom-dm-features-trial.csv', 'shroom-dm-features-test.csv', "DM")
print("For Paraphrasing:")
clustering('shroom-pg-features.csv', 'shroom-pg-features-trial.csv', 'shroom-pg-features-test.csv', "PG")