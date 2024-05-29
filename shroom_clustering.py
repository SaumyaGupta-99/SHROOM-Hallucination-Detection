import pandas as pd
import json
from sklearn.cluster import KMeans
import warnings
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from collections import Counter
from sklearn.cluster import Birch
from sklearn.cluster import MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import numpy as np
import pickle
import os

warnings.filterwarnings("ignore")


def kmeans_clustering(X):
    best_kmeans = None
    best_inertia = float('inf')
    for i in range(10):
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        if score < best_inertia:
            best_kmeans = kmeans
            best_inertia = score

    return best_kmeans, best_kmeans.labels_


def mean_clustering(X):
    # Initialize KMeans object
    mean_model = MeanShift()
    mean_model.fit(X)
    return mean_model, mean_model.labels_


def gmm_clustering(X):
    gmm = GaussianMixture(n_components=2)
    labels = gmm.fit(X)
    return gmm, labels


def birch_clustering(X):
    clustering = Birch(threshold=0.03, n_clusters=2)
    # train the model
    clustering.fit(X)
    return clustering, clustering.labels_


def hierarchical_clustering(X):
    clustering = AgglomerativeClustering(n_clusters=2)  # You can change the number of clusters as needed

    # Fit the model to your data
    clustering.fit(X)

    return clustering, clustering.labels_


def spectral_clustering(X):
    # Perform Spectral Clustering
    spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', n_neighbors=5)

    # Fit the model to your data
    spectral.fit(X)

    return spectral, spectral.labels_


def read_file(filename):
    f = open(current_working_directory + '/data/' + filename)
    data = json.load(f)
    return data


def view_results(test_results, task, hallucination_label):
    true_labels = list(test[test['task'] == task]['label'])
    hyp = list(test[test['task'] == task]['hyp'])
    tgt = list(test[test['task'] == task]['tgt'])
    src = list(test[test['task'] == task]['src'])
    labels = [hallucination if test_results[i] == hallucination_label else not_hallucination for i in
              range(len(test_results))]
    print("For task:" + task)
    if task != "PG":
        [print("Source is: " + src[i] + "\n Expected Target Is: " + tgt[i] + "\n Hypothesis is: " + hyp[
            i] + "\n true_label: " + true_labels[i] + "\n model_output: " + labels[i]) for i in range(20, 25)]
    else:
        [print("Source is: " + src[i] + "\n Hypothesis is: " + hyp[
            i] + "\n true_label: " + true_labels[i] + "\n model_output: " + labels[i]) for i in range(20, 25)]


def process_rouge(data, task):
    if task != 'PG':
        rouge_scores_r = []
        rouge_scores_p = []
        rouge_scores_f = []
        for rouge in data['rouge_score_with_tgt']:
            # rouge = eval(rouge)
            sum_r = rouge['rouge-1']['r'] + rouge['rouge-2']['r'] + rouge['rouge-l']['r']
            sum_p = rouge['rouge-1']['p'] + rouge['rouge-2']['p'] + rouge['rouge-l']['p']
            sum_f = rouge['rouge-1']['f'] + rouge['rouge-2']['f'] + rouge['rouge-l']['f']
            rouge_scores_r.append(sum_r / 3)
            rouge_scores_p.append(sum_p / 3)
            rouge_scores_f.append(sum_f / 3)

        data['rouge_scores_r'] = rouge_scores_r
        data['rouge_scores_p'] = rouge_scores_p
        data['rouge_scores_f'] = rouge_scores_f
        data.drop("rouge_score_with_tgt", axis=1, inplace=True)
    return data


def get_data(filename, task):
    data = pd.read_csv("data/" + filename)
    return process_rouge(data, task)


def get_best_predictions(preds1, preds2, preds3):
    all_predictions = [preds1, preds2, preds3]
    preds = []
    for i in range(len(preds1)):
        predictions_at_index = [preds[i] for preds in all_predictions]
        prediction_counter = Counter(predictions_at_index)
        most_common_prediction = prediction_counter.most_common(1)[0][0]
        if most_common_prediction == "hallucination":
            preds.append(0)
        else:
            preds.append(1)
    return preds


def get_labels(trial_preds, task_labels):
    count_dict = {'H0': 0, 'H1': 0, 'NH0': 0, 'NH1': 0}
    for cluster, label in zip(trial_preds, task_labels):
        if cluster == 0 and label == 'Hallucination':
            count_dict['H0'] += 1
        elif cluster == 1 and label == 'Hallucination':
            count_dict['H1'] += 1
        elif cluster == 0 and label == 'Not Hallucination':
            count_dict['NH0'] += 1
        else:
            count_dict['NH1'] += 1
    # print(count_dict)
    if count_dict['H0'] > count_dict['H1']:
        hallucination_label = 0
        non_hallucination_label = 1
    else:
        hallucination_label = 1
        non_hallucination_label = 0
    # print(hallucination_label, non_hallucination_label)
    return hallucination_label, non_hallucination_label


def get_accuracy(task, test_preds, hallucination_label, non_hallucination_label):
    task_labels = test[test['task'] == task]['label']

    # Initialize variables for TP, FP, TN, FN
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    correct = 0
    for cluster, label in zip(test_preds, task_labels):
        if cluster == hallucination_label and label == hallucination:
            TP += 1
            correct += 1
        elif cluster == hallucination_label and label == not_hallucination:
            FP += 1
        elif cluster == non_hallucination_label and label == hallucination:
            FN += 1
        elif cluster == non_hallucination_label and label == not_hallucination:
            TN += 1
            correct += 1

    # Calculate precision
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    # Calculate recall
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Example usage:
    print("Precision:", precision, "Recall:", recall, "F1 Score:", f1_score)
    print("Accuracy is:", (correct / len(task_labels)))


def score_wise_clustering(filename, trial_filename, test_filename, task, save_model):
    data = get_data(filename, task)
    trial_data = get_data(trial_filename, task)
    test_data = get_data(test_filename, task)
    combined_data = np.concatenate((data, trial_data), axis=0)
    kmeans_models = {}
    for i in range(combined_data.shape[1]):
        column_data = combined_data[:, i].reshape(-1, 1)
        clustering, _ = kmeans_clustering(column_data)
        kmeans_models[i] = clustering
    task_labels = trial[trial['task'] == task]['label']
    test_votes = []
    test_preds = []

    for _ in range(len(test_data)):
        test_votes.append([0, 0])
    for column, kmeans in kmeans_models.items():
        label = kmeans.predict(trial_data.iloc[:, column].values.reshape(-1, 1))
        hallucination_label, non_hallucination_label = get_labels(label, task_labels)
        test_label = kmeans.predict(test_data.iloc[:, column].values.reshape(-1, 1))
        i = 0
        for prediction in test_label:
            if prediction == hallucination_label:
                test_votes[i][0] += 1
            else:
                test_votes[i][1] += 1
            i += 1

    if save_model:
        with open(f"{task}_ensemble_kmeans_models.pkl", "wb") as file:
            pickle.dump(kmeans_models, file)
            print("Models saved successfully.")

    for votes in test_votes:
        if votes[0] >= votes[1]:
            test_preds.append(0)
        else:
            test_preds.append(1)

    get_accuracy(task, test_preds, 0, 1)
    view_results(test_preds, task, 0)


def best_score_wise_clustering(filename, trial_filename, test_filename, task):
    data = get_data(filename, task)
    trial_data = get_data(trial_filename, task)
    test_data = get_data(test_filename, task)
    kmeans_models = {}
    combined_data = np.concatenate((data, trial_data), axis=0)
    for i in range(combined_data.shape[1]):
        column_data = combined_data[:, i].reshape(-1, 1)
        clustering, _ = kmeans_clustering(column_data)
        kmeans_models[i] = clustering

    best_kmeans = None
    best_inertia = float('inf')
    best_column = None
    for column, model in kmeans_models.items():
        score = model.inertia_
        if score < best_inertia:
            best_kmeans = model
            best_inertia = score
            best_column = column
    # print(best_column)
    task_labels = trial[trial['task'] == task]['label']
    trial_preds = best_kmeans.predict(trial_data.iloc[:, best_column].values.reshape(-1, 1))
    test_preds = best_kmeans.predict(test_data.iloc[:, best_column].values.reshape(-1, 1))
    hallucination_label, non_hallucination_label = get_labels(trial_preds, task_labels)
    get_accuracy(task, test_preds, hallucination_label, non_hallucination_label)


def replace_labels(label_list, non_hallucination_label, hallucination_label):
    # Define labels based on provided information
    # Replace 0 with non hallucination and 1 with hallucination
    replaced_list = ["hallucination" if label == hallucination_label else non_hallucination_label for label in
                     label_list]

    return replaced_list


def clustering_scores(filename, trial_filename, test_filename, task, type):
    data = get_data(filename, task)
    trial_data = get_data(trial_filename, task)
    test_data = get_data(test_filename, task)
    X = np.concatenate((data, trial_data), axis=0)
    if type == "kmeans":
        clustering, labels = kmeans_clustering(X)
        test_preds = clustering.predict(test_data)

    elif type == "birch":
        clustering, labels = birch_clustering(X)
        test_preds = clustering.predict(test_data)
    elif type == "hierarchical":
        clustering, labels = hierarchical_clustering(X)
        test_preds = clustering.fit_predict(test_data)
    elif type == "mean":
        clustering, labels = mean_clustering(X)
        test_preds = clustering.fit_predict(test_data)
    elif type == "spectral":
        clustering, labels = spectral_clustering(X)
        test_preds = clustering.fit_predict(test_data)
    elif type == "gmm":
        clustering, labels = gmm_clustering(X)
        test_preds = clustering.predict(test_data)
    elif type == "mean":
        clustering, labels = mean_clustering(X)
        test_preds = clustering.fit_predict(test_data)
    else:
        return

    task_labels = test[test['task'] == task]['label']
    hallucination_label, non_hallucination_label = get_labels(test_preds, task_labels)
    get_accuracy(task, test_preds, hallucination_label, non_hallucination_label)

    # view_results(test_preds, task, hallucination_label)


current_working_directory = os.getcwd()
trial = pd.DataFrame(read_file('trial-v1.json'))
test = pd.DataFrame(read_file('test.model-aware.json'))
hallucination = 'Hallucination'
not_hallucination = 'Not Hallucination'
