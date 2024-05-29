import shroom_clustering

print("**********************Kmeans Results**************************************************")
print("For Machine Translation:")
shroom_clustering.clustering_scores('shroom-mt-features.csv', 'shroom-mt-features-trial.csv', 'shroom-mt-features-test.csv', "MT","kmeans")
print("For Definition Modelling:")
shroom_clustering.clustering_scores('shroom-dm-features.csv', 'shroom-dm-features-trial.csv', 'shroom-dm-features-test.csv', "DM",
                   "kmeans")
print("For Paraphrasing:")
shroom_clustering.clustering_scores('shroom-pg-features.csv', 'shroom-pg-features-trial.csv', 'shroom-pg-features-test.csv', "PG",
                  "kmeans")
print("**********************Hierarchical Results**************************************************")
print("For Machine Translation:")
shroom_clustering.clustering_scores('shroom-mt-features.csv', 'shroom-mt-features-trial.csv', 'shroom-mt-features-test.csv', "MT",
                  "hierarchical")
print("For Definition Modelling:")
shroom_clustering.clustering_scores('shroom-dm-features.csv', 'shroom-dm-features-trial.csv', 'shroom-dm-features-test.csv', "DM",
                  "hierarchical")
print("For Paraphrasing:")
shroom_clustering.clustering_scores('shroom-pg-features.csv', 'shroom-pg-features-trial.csv', 'shroom-pg-features-test.csv', "PG",
                  "hierarchical")
print("**********************Spectral Results**************************************************")
print("For Machine Translation:")
shroom_clustering.clustering_scores('shroom-mt-features.csv', 'shroom-mt-features-trial.csv', 'shroom-mt-features-test.csv', "MT",
                  "spectral")
print("For Definition Modelling:")
shroom_clustering.clustering_scores('shroom-dm-features.csv', 'shroom-dm-features-trial.csv', 'shroom-dm-features-test.csv', "DM",
                  "spectral")
print("For Paraphrasing:")
shroom_clustering.clustering_scores('shroom-pg-features.csv', 'shroom-pg-features-trial.csv', 'shroom-pg-features-test.csv', "PG",
                  "spectral")
print("**********************Birch Results**************************************************")
print("For Machine Translation:")
shroom_clustering.clustering_scores('shroom-mt-features.csv', 'shroom-mt-features-trial.csv', 'shroom-mt-features-test.csv', "MT",
                  "birch")
print("For Definition Modelling:")
shroom_clustering.clustering_scores('shroom-dm-features.csv', 'shroom-dm-features-trial.csv', 'shroom-dm-features-test.csv', "DM",
                  "birch")
print("For Paraphrasing:")
shroom_clustering.clustering_scores('shroom-pg-features.csv', 'shroom-pg-features-trial.csv', 'shroom-pg-features-test.csv', "PG",
                  "birch")
print("**********************Mean Results**************************************************")
print("For Machine Translation:")
shroom_clustering.clustering_scores('shroom-mt-features.csv', 'shroom-mt-features-trial.csv', 'shroom-mt-features-test.csv', "MT",
                  "mean")
print("For Definition Modelling:")
shroom_clustering.clustering_scores('shroom-dm-features.csv', 'shroom-dm-features-trial.csv', 'shroom-dm-features-test.csv', "DM",
                  "mean")
print("For Paraphrasing:")
shroom_clustering.clustering_scores('shroom-pg-features.csv', 'shroom-pg-features-trial.csv', 'shroom-pg-features-test.csv', "PG",
                  "mean")
print("**********************Gaussian Mixture Model Results**************************************************")
print("For Machine Translation:")
shroom_clustering.clustering_scores('shroom-mt-features.csv', 'shroom-mt-features-trial.csv', 'shroom-mt-features-test.csv', "MT",
                  "gmm")
print("For Definition Modelling:")
shroom_clustering.clustering_scores('shroom-dm-features.csv', 'shroom-dm-features-trial.csv', 'shroom-dm-features-test.csv', "DM",
                  "gmm")
print("For Paraphrasing:")
shroom_clustering.clustering_scores('shroom-pg-features.csv', 'shroom-pg-features-trial.csv', 'shroom-pg-features-test.csv', "PG",
                  "gmm")
print("**********************ScoreWise clustering Model Results**************************************************")
print("For Machine Translation:")
shroom_clustering.score_wise_clustering('shroom-mt-features.csv', 'shroom-mt-features-trial.csv', 'shroom-mt-features-test.csv', "MT",False)
print("For Definition Modelling:")
shroom_clustering.score_wise_clustering('shroom-dm-features.csv', 'shroom-dm-features-trial.csv', 'shroom-dm-features-test.csv', "DM",False)
print("For Paraphrasing:")
shroom_clustering.score_wise_clustering('shroom-pg-features.csv', 'shroom-pg-features-trial.csv', 'shroom-pg-features-test.csv', "PG",False)
print("**********************Best ScoreWise clustering Model Results**************************************************")
print("For Machine Translation:")
shroom_clustering.best_score_wise_clustering('shroom-mt-features.csv', 'shroom-mt-features-trial.csv', 'shroom-mt-features-test.csv',
                           "MT")
print("For Definition Modelling:")
shroom_clustering.best_score_wise_clustering('shroom-dm-features.csv', 'shroom-dm-features-trial.csv', 'shroom-dm-features-test.csv',
                           "DM")
print("For Paraphrasing:")
shroom_clustering.best_score_wise_clustering('shroom-pg-features.csv', 'shroom-pg-features-trial.csv', 'shroom-pg-features-test.csv',
                           "PG")