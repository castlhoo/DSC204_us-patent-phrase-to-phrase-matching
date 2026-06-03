import zipfile, io
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import r2_score
import scipy.stats as stats

DATA_PATH = "data/us-patent-phrase-to-phrase-matching.zip"

z     = zipfile.ZipFile(DATA_PATH)
train = pd.read_csv(io.BytesIO(z.read("train.csv")))
test  = pd.read_csv(io.BytesIO(z.read("test.csv")))
sub   = pd.read_csv(io.BytesIO(z.read("sample_submission.csv")))

print(f"Train: {train.shape}, Test: {test.shape}")

# Prepend the context code to both anchor and target text
def make_text(df):
    return (df["context"] + " " + df["anchor"]).tolist(), \
           (df["context"] + " " + df["target"]).tolist()

tr_anchor, tr_target = make_text(train)
te_anchor, te_target = make_text(test)

# TF-IDF: fit vocabulary on anchor + target combined
all_texts = tr_anchor + tr_target + te_anchor + te_target
tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=50000)
tfidf.fit(all_texts)

# Compute cosine similarity
def get_scores(anchors, targets):
    vec_a = tfidf.transform(anchors)
    vec_t = tfidf.transform(targets)
    scores = np.array([
        cosine_similarity(vec_a[i], vec_t[i])[0][0]
        for i in range(len(anchors))
    ])
    return scores

print("Computing train similarity...")
train_scores = get_scores(tr_anchor, tr_target)
pearson = stats.pearsonr(train["score"].values, train_scores)[0]
print(f"Train Pearson: {pearson:.4f}")

print("Computing test similarity...")
test_scores = get_scores(te_anchor, te_target)

sub["score"] = test_scores
sub.to_csv("predictions.csv", index=False)
print("predictions.csv saved!")
print(sub.head())
