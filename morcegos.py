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
    return SVC(kernel="rbf", C=10, gamma="scale", probability=True,
               class_weight="balanced", random_state=SEED).fit(Xa, ya)

# 4. Rodadas de self-training (CURADORIA define o que foi confirmado/rejeitado)
for rodada in range(1, 7):
    model = fit_model(Xs[y != -1], y[y != -1])
    p = model.predict_proba(Xs[y == -1])[:, 1]
    ...  # lista candidatos -> curadoria -> atualiza y

# 5. Saídas e 6. Avaliação (validação cruzada 10-fold e por gênero)