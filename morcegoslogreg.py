"""
Reconhecimento de padrões - identificação de morcegos (Chiroptera) em 8426 genomas mitocondriais.
Versão: Regressão Logística
Estratégia: self-training / aprendizado semissupervisionado com curadoria humana por rodadas.

Uso:  python3 morcegos_logreg.py [pasta_dos_dados]      (padrão: ./data, com Names.txt e wpc50.dat)
Saída: pasta saida_logreg/ (classes.txt, indices_morcegos.txt, classificacao_final.csv, modelo.joblib)
"""
import os, sys, warnings, numpy as np, pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
warnings.filterwarnings("ignore")

DIR = (sys.argv[1] if len(sys.argv) > 1 else "./data") + "/"   # pasta com Names.txt e wpc50.dat
OUT = "saida_logreg/"; os.makedirs(OUT, exist_ok=True)
SEED = 42
rng = np.random.default_rng(SEED)

# ---------- 1. Carga ----------
names = [l.strip() for l in open(DIR + "Names.txt", encoding="latin-1")]
X = np.loadtxt(DIR + "wpc50.dat")
genus = np.array([n.split("_")[0] for n in names])
assert X.shape[0] == len(names) == 8426
scaler = StandardScaler().fit(X)
Xs = scaler.transform(X)

# ---------- 2. Supervisão inicial (pesquisa prévia) ----------
# Positivos: gêneros de Chiroptera confirmados (NCBI Taxonomy / Mammal Diversity Database)
SEED_POS = {"Myotis", "Rhinolophus", "Pteropus", "Pipistrellus", "Eptesicus", "Artibeus",
            "Desmodus", "Tadarida", "Plecotus", "Hipposideros"}
# Negativos: gêneros seguramente não-morcego (outros mamíferos próximos + grupos distantes)
SEED_NEG = {"Homo", "Mus", "Rattus", "Bos", "Canis", "Felis", "Sus", "Equus", "Ovis", "Capra",
            "Pan", "Gorilla", "Macaca", "Sorex", "Erinaceus", "Talpa", "Panthera", "Ursus",
            "Candida", "Anopheles", "Epinephelus", "Takifugu", "Acipenser", "Cherax", "Labeo",
            "Anguilla", "Microcebus", "Lepilemur", "Nasutitermes", "Acraea", "Triplophysa"}

y = np.full(len(names), -1)           # -1 = não rotulado
y[np.isin(genus, list(SEED_POS))] = 1
y[np.isin(genus, list(SEED_NEG))] = 0
print(f"Semente: {np.sum(y==1)} positivos, {np.sum(y==0)} negativos, {np.sum(y==-1)} não rotulados")

# ---------- 3. Enriquecimento: SMOTE simples (interpolação entre positivos vizinhos) ----------
def smote(Xp, n_new, k=5):
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(Xp))).fit(Xp)
    idx = nn.kneighbors(Xp, return_distance=False)[:, 1:]
    out = []
    for _ in range(n_new):
        i = rng.integers(len(Xp)); j = rng.choice(idx[i])
        out.append(Xp[i] + rng.random() * (Xp[j] - Xp[i]))
    return np.array(out)

# Classificador (único trecho que muda entre as três versões)
def novo_classificador():
    return LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")

def fit_model(Xl, yl):
    pos = Xl[yl == 1]
    n_new = max(0, (yl == 0).sum() - len(pos))
    Xa = np.vstack([Xl, smote(pos, n_new)]); ya = np.r_[yl, np.ones(n_new, int)]
    return novo_classificador().fit(Xa, ya)

# ---------- 4. Self-training em rodadas com curadoria humana ----------
# thr = limiar de probabilidade para listar candidatos;
# ok  = gêneros candidatos confirmados como Chiroptera após pesquisa (NCBI Taxonomy / literatura).
# Todo candidato listado e NÃO confirmado em "ok" é considerado rejeitado (negativo).
CURADORIA = {
    1: dict(thr=0.5, ok={"Anoura", "Brachyphylla", "Carollia", "Chalinolobus", "Corynorhinus", "Cynopterus", "Epomophorus", "Hypsugo", "Lasiurus", "Lophostoma", "Micronycteris", "Murina", "Mystacina", "Nyctalus", "Pteronotus", "Rhinophylla", "Rousettus", "Saccopteryx", "Sturnira", "Tonatia", "Vampyrum", "Vespertilio"}),
    2: dict(thr=0.1, ok=set()),
}

for rodada, cfg in CURADORIA.items():
    lab = y != -1
    model = fit_model(Xs[lab], y[lab])
    unl = np.where(y == -1)[0]
    p = model.predict_proba(Xs[unl])[:, 1]
    cand = unl[p >= cfg["thr"]]
    resumo = pd.Series(genus[cand]).value_counts()
    print(f"\n=== Rodada {rodada}: {lab.sum()} rotulados (pos={np.sum(y==1)}); {len(cand)} candidatos p>={cfg['thr']} ===")
    print(", ".join(f"{g}({n})" for g, n in resumo.head(40).items()) + (" ..." if len(resumo) > 40 else ""))
    y[unl[p < 0.01]] = 0                                   # negativos confiantes
    y[np.isin(genus, list(cfg["ok"])) & (y == -1)] = 1     # positivos curados
    rejeitados = cand[y[cand] == -1]                       # candidatos não confirmados
    y[rejeitados] = 0                                      # falsos positivos curados
    print(f"-> confirmados: {len(cfg['ok'])} gêneros | rejeitados: {len(set(genus[rejeitados]))} gêneros")

# ---------- 5. Rótulo final ----------
y_final = np.where(y == 1, 1, 0)         # restantes não rotulados = 0
idx_pos = np.where(y_final == 1)[0] + 1  # números de ordem (base 1)
print(f"\nTotal de morcegos: {y_final.sum()} de {len(y_final)}")
pd.DataFrame({"indice": np.arange(1, len(names)+1), "nome": names, "classe": y_final})
np.savetxt(OUT + "classes.txt", y_final, fmt="%d")

#BINDA INDICE COM ESPÉCIE E PRINTA OS DOIS JUNTOS
df_morcegos = pd.DataFrame({
    "indice": idx_pos,
    "especie": np.array(names)[y_final == 1]
})
df_morcegos.to_csv(OUT + "indices_morcegos.txt", sep="\t", index=False, header=False)
print("\nMorcegos identificados (índice  espécie):")
print(df_morcegos.to_string(index=False, header=False))

# ---------- 6. Desempenho ----------
def relatorio(titulo, yt, yp):
    tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
    print(f"{titulo}: acc={accuracy_score(yt,yp):.4f} sens={recall_score(yt,yp):.4f} "
          f"prec={precision_score(yt,yp):.4f} F1={f1_score(yt,yp):.4f}  [TP={tp} FP={fp} FN={fn} TN={tn}]")

NOME = "Regressão Logística"
print("\n--- Validação cruzada estratificada 10-fold no conjunto final ---")
skf = StratifiedKFold(10, shuffle=True, random_state=SEED)
relatorio(NOME, y_final, cross_val_predict(novo_classificador(), Xs, y_final, cv=skf))

print("\n--- Validação por GÊNERO (GroupKFold): gênero nunca visto no treino ---")
relatorio(NOME + " (grupo=gênero)", y_final,
          cross_val_predict(novo_classificador(), Xs, y_final, cv=GroupKFold(10), groups=genus))

# ---------- 7. Modelo final salvo para consultas ----------
joblib.dump({"model": fit_model(Xs, y_final), "scaler": scaler}, OUT + "modelo.joblib")
print("\nModelo salvo em", OUT + "modelo.joblib")
