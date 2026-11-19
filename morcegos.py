"""
Reconhecimento de padrões - identificação de morcegos (Chiroptera) em 8426 genomas mitocondriais.
Estratégia: self-training / aprendizado semissupervisionado com curadoria humana por rodadas.
"""
from sklearn.linear_model import LogisticRegression
import re, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

DIR = "./data/"
SEED = 42
rng = np.random.default_rng(SEED)

# 1. Carga
names = [l.strip() for l in open(DIR + "Names.txt", encoding="latin-1")]
X = np.loadtxt(DIR + "wpc50.dat")
genus = np.array([n.split("_")[0] for n in names])
Xs = StandardScaler().fit_transform(X)

# 2. Supervisão inicial
y = np.full(len(names), -1)               # -1 = não rotulado
y[np.isin(genus, list(SEED_POS))] = 1
y[np.isin(genus, list(SEED_NEG))] = 0

# 3. SMOTE simples + SVM
def fit_model(Xl, yl):
    pos = Xl[yl == 1]
    n_new = max(0, (yl == 0).sum() - len(pos))
    Xa = np.vstack([Xl, smote(pos, n_new)]); ya = np.r_[yl, np.ones(n_new, int)]
    return LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(Xa, ya)

# 4. Rodadas de self-training (CURADORIA define o que foi confirmado/rejeitado)
for rodada in range(1, 7):
    model = fit_model(Xs[y != -1], y[y != -1])
    p = model.predict_proba(Xs[y == -1])[:, 1]
    ...  # lista candidatos -> curadoria -> atualiza y

# 5. Saídas e 6. Avaliação (validação cruzada 10-fold e por gênero)