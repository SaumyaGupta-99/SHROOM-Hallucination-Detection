import scores
import warnings
import pandas as pd
import numpy as np
import shroom_clustering
import pickle

warnings.filterwarnings("ignore")


def shroom_predict(src, tgt, hyp, task):
    try:
        # Load saved models
        with open(f"{task}_ensemble_kmeans_models.pkl", "rb") as file:
            kmeans_models = pickle.load(file)
            print("Models loaded successfully.")
    except FileNotFoundError:
        print("No saved models found.")
        return

    data = {
        'src': [src],
        'tgt': [tgt],
        'hyp': [hyp],
        'task': [task]
    }
    data_scores = shroom_clustering.process_rouge(scores.calculate_scores(pd.DataFrame(data), task, None), task)
    test_preds = []
    for column, kmeans in kmeans_models.items():
        test_label = kmeans.predict(data_scores.iloc[:, column].values.reshape(-1, 1).astype('float64'))
        test_preds.append(test_label)

    # Ensemble voting
    test_preds = np.array(test_preds)
    predicted_labels = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=test_preds)
    if predicted_labels[0] == 0:
        return "Hallucination"
    return "Not Hallucination"


print(shroom_predict("It was what had attracted me to her in the first place",
                     "To use clues or enticements to lead someone in the desired direction.", "To trick or deceive.",
                     "DM"))
print(
    shroom_predict("I do not need to remind you of the decisive role that Parliament played in the mad cow crisis.", "",
                   "Parliament played an important role in the mad cow crisis.", "PG"))
print(shroom_predict("\u0174acali kuwonawona kakuliro ka ngozi kweniso na umo caro capasi ciikhwaskikirenge.",
                     "They are still trying to determine just how large the crash was and how the Earth will be affected.",
                     "We are still seeing the increasing danger and the impact this planet will have.", "MT"))
