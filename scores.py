import json
import os
import pandas as pd
import numpy as np
import torch
import nltk
from transformers import pipeline
from comet import download_model, load_from_checkpoint
from tqdm import tqdm
from summac.model_summac import SummaCZS, SummaCConv
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
import collections
import math
from bert_score import score as bert_score


def get_training_data(filename):
    current_working_directory = os.getcwd()
    f = open(current_working_directory + '/data/' + filename)

    model_data = json.load(f)
    parsed_data = []
    for i in model_data:
        datum = {}
        for key in i:
            datum[key] = i[key]
        parsed_data.append(datum)
    f.close()

    df_aware = pd.DataFrame(model_data)
    print(df_aware.head())
    return df_aware


def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],  # Deletion
                                   dp[i][j - 1],  # Insertion
                                   dp[i - 1][j - 1])  # Substitution

    return dp[m][n]


def get_sentiment(text):
    # Load pre-trained multilingual sentiment analysis model, as we have multilungual sentences for MT task
    sentiment_analyzer = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    # Perform sentiment analysis
    result = sentiment_analyzer(text)
    return result[0]['label'].split()[0]


def calculate_ter(hypothesis, references):
    min_distance = float('inf')

    for reference in references:
        distance = levenshtein_distance(hypothesis.split(), reference.split())
        min_distance = min(min_distance, distance)

    ter_score = min_distance / len(hypothesis.split())
    return ter_score


def calculate_entropy(sentence):
    # Count the occurrences of each word
    word_counts = collections.Counter(sentence.split())
    total_words = len(sentence.split())

    # Calculate the entropy
    entropy = -sum((count / total_words) * math.log2(count / total_words) for count in word_counts.values())
    return entropy


def jaccard_similarity(sentence1, sentence2):
    set1 = set(sentence1.split())
    set2 = set(sentence2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    similarity = intersection / union
    return similarity


def dice_similarity(sentence1, sentence2):
    set1 = set(sentence1.split())
    set2 = set(sentence2.split())
    intersection = len(set1.intersection(set2))
    dice_sim = (2 * intersection) / (len(set1) + len(set2))
    return dice_sim


def bert_score_calculate(sentence, reference):
    if reference == '':
        return 0
    _, _, bert_scores = bert_score([sentence], [reference], lang='en', model_type='bert-base-uncased')
    return bert_scores.mean().item()


def calculate_scores(data, task, data_type):
    data = data[data['task'] == task]

    # Sentence Entropy
    sentence_entropy = []
    for line in tqdm(data['hyp']):
        sentence_entropy.append(calculate_entropy(line))

    # Sentiment Analysis wrt hyp and tgt
    sentiment_analysis_with_src = []
    # Sentiment Analysis wrt hyp and tgt
    sentiment_analysis_with_tgt = []

    # Bert Score

    bert_score = []
    for hyp, tgt, src in tqdm(zip(data['hyp'], data['tgt'], data['src'])):
        if task != 'PG':
            bert_score.append(bert_score_calculate(hyp, tgt))
        else:
            bert_score.append(bert_score_calculate(hyp, src))
        tone_hyp = get_sentiment(hyp)
        tone_tgt = get_sentiment(tgt)
        tone_src = get_sentiment(src)
        if tone_hyp == tone_tgt:
            sentiment_analysis_with_src.append(1)
        else:
            sentiment_analysis_with_src.append(0)

        if tone_hyp == tone_src:
            sentiment_analysis_with_tgt.append(1)
        else:
            sentiment_analysis_with_tgt.append(0)

    model = SentenceTransformer('xlm-r-distilroberta-base-paraphrase-v1')

    semnatic_similarity_with_src = []
    for sentence, other_sentence in tqdm(zip(data['src'], data['hyp'])):
        english_embedding = model.encode(sentence, convert_to_tensor=True)
        unicode_embedding = model.encode(other_sentence, convert_to_tensor=True)
        semnatic_similarity_with_src.append(util.pytorch_cos_sim(english_embedding, unicode_embedding).item())

    sentence_embedding = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    rouge = Rouge()
    semnatic_similarity_with_tgt = []
    rouge_score = []
    bleu_score = []
    jaccard_similarity_score = []
    dice_similarity_score = []

    if task != "PG":
        for sentence, other_sentence in tqdm(zip(data['tgt'], data['hyp'])):
            vector = sentence_embedding.encode([sentence])[0]
            other_vector = sentence_embedding.encode([other_sentence])[0]
            # Calculate the distance between two sentences
            distance = cosine_similarity([vector], [other_vector])[0][0]
            semnatic_similarity_with_tgt.append(distance)
            rouge_score.append(rouge.get_scores(other_sentence, sentence)[0])
            bleu_score.append(sentence_bleu([sentence], other_sentence))
            jaccard_similarity_score.append(jaccard_similarity(sentence, other_sentence))
            dice_similarity_score.append(dice_similarity(sentence, other_sentence))

    # Summary Scores
    conv_scores = []
    if task != "MT":
        nltk.download('punkt')
        model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e",
                                device="cpu", start_file="default", agg="mean")
        conv_score = []
        for document, summary1 in tqdm(zip(data['src'], data['hyp'])):
            conv_score.append(model_conv.score([document], [summary1]))
        for i in range(len(conv_score)):
            conv_scores.append(conv_score[i]['scores'][0])

    if task == "DM":
        dm_data = {
            'semnatic_similarity_with_src': semnatic_similarity_with_src,
            'sentiment_analysis_with_src': sentiment_analysis_with_src,
            'sentiment_analysis_with_tgt': sentiment_analysis_with_tgt,
            'sentence_entropy': sentence_entropy,
            'semnatic_similarity_with_tgt': semnatic_similarity_with_tgt,
            'rouge_score_with_tgt': rouge_score,
            'bleu_score_with_tgt': bleu_score,
            'jaccard_similarity_score_with_tgt': jaccard_similarity_score,
            'dice_similarity_score_with_tgt': dice_similarity_score,
            'bert-score': bert_score,
            'summary_score': conv_scores
        }
        df = pd.DataFrame(dm_data)


    elif task == "MT":
        # COMET SCORE
        # Download and load the COMET model
        model_path = download_model("wmt20-comet-da")
        model = load_from_checkpoint(model_path)
        # Prepare the data for the model
        comet_data = [{"src": src, "mt": mt, "ref": ref} for src, mt, ref in
                      zip(data['src'], data['hyp'], data['tgt'])]
        # Predict scores
        comet_scores = model.predict(comet_data, batch_size=8, gpus=0)
        # TER SCORE
        ter_scores = []
        for hyp, reference in zip(data['hyp'], data['tgt']):
            ter_score = calculate_ter(hyp, [reference])
            ter_scores.append(ter_score)

        mt_data = {
            'comet': comet_scores[0],
            'ter': ter_scores,
            'semnatic_similarity_with_src': semnatic_similarity_with_src,
            'sentiment_analysis_with_src': sentiment_analysis_with_src,
            'sentiment_analysis_with_tgt': sentiment_analysis_with_tgt,
            'sentence_entropy': sentence_entropy,
            'semnatic_similarity_with_tgt': semnatic_similarity_with_tgt,
            'rouge_score_with_tgt': rouge_score,
            'bleu_score_with_tgt': bleu_score,
            'jaccard_similarity_score_with_tgt': jaccard_similarity_score,
            'dice_similarity_score_with_tgt': dice_similarity_score,
            'bert-score': bert_score
        }

        df = pd.DataFrame(mt_data)

    else:
        pg_data = {
            'semnatic_similarity_with_src': semnatic_similarity_with_src,
            'sentiment_analysis_with_src': sentiment_analysis_with_src,
            'sentence_entropy': sentence_entropy,
            'bert-score': bert_score,
            'summary_score': conv_scores
        }
        df = pd.DataFrame(pg_data)

    if data_type is not None:
        current_working_directory = os.getcwd()
        filename = 'shroom-' + task + '-features-' + data_type + '-bert.csv'
        excel_file_path = current_working_directory + '/data/' + filename
        df.to_csv(excel_file_path, index=False)
        print(f'Data saved to {excel_file_path}')
    else:
        return df
