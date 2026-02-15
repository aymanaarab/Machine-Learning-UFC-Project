# 🥊 Machine Learning UFC Project - Clustering des Styles de Combat

Un projet d'apprentissage automatique complet qui catégorise les fighters UFC en groupes distincts en fonction de leurs styles de combat à l'aide de techniques de clustering avancées.

**Auteur** : Ayman Aarab | **Repository** : [GitHub](https://github.com/aymanaarab/Machine-Learning-UFC-Project)

---

## 📋 Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Structure du projet](#structure-du-projet)
- [Pipeline de données](#pipeline-de-données)
- [Technologies utilisées](#technologies-utilisées)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Résultats du clustering](#résultats-du-clustering)
- [Insights clés](#insights-clés)

---

## 🎯 Vue d'ensemble

Ce projet analyse **8 482 combats UFC** impliquant **4 448 fighters** pour identifier et catégoriser les styles de combat distincts. Utilisant des techniques modernes de machine learning (Feature Engineering, Scaling, PCA, K-Means), le projet classe **835 fighters** en **5 archetypes de combat** basés sur 17 caractéristiques de performance.

### Objectifs principaux
1. **Feature Engineering** : Calculer des métriques de style de combat significatives à partir des statistiques brutes
2. **Prétraitement** : Normaliser les données et réduire la dimensionnalité
3. **Clustering** : Identifier les groupes de fighters avec des styles similaires
4. **Analyse** : Interpréter les clusters et valider les résultats

---

## 📁 Structure du projet

```
Machine Learning UFC Project/
│
├── 📊 data/
│   ├── raw/
│   │   ├── raw_fighters.csv              (4,448 fighters + attributs)
│   │   └── raw_fights_detailed.csv       (8,482 combats détaillés)
│   └── processed/
│       ├── fighter_features.csv          (Features engineered: 6,053 × 22)
│       ├── fighter_features_scaled.csv   (Features normalisées: 835 × 22)
│       ├── fighter_features_pca.csv      (PCA réduit: 835 × 12)
│       └── fighter_clustered.csv         (Résultat final avec labels)
│
├── 📓 notebooks/
│   ├── setup_discovering_data.ipynb      (1. Exploration & EDA)
│   ├── Feature_Engeneering.ipynb         (2. Calcul des features)
│   ├── Preprocessing_scaling.ipynb       (3. Scaling & PCA)
│   └── Clustering.ipynb                  (4. Clustering & analyse)
│
├── 📈 mlruns/                            (Tracking MLflow)
├── mlflow.db                             (Base de données MLflow)
├── robust_scaler.pkl                     (Scaler sauvegardé)
└── pca_transformer.pkl                   (Transformateur PCA sauvegardé)
```

---

## 🔄 Pipeline de données

### Phase 1️⃣ : Exploration des données (`setup_discovering_data.ipynb`)

**Objectif** : Comprendre la structure et la qualité des données

**Points clés**:
- Analyse de **4,448 fighters** avec attributs démographiques (taille, poids, allure)
- Analyse de **8,482 combats** avec statistiques détaillées
- Identification des problèmes de données :
  - 44% de données manquantes pour la portée
  - 7% manquantes pour la taille
  - 19% manquantes pour l'allure
- Découverte de patterns : spectrum clair "Frappeur vs Lutteur"
- Distribution des résultats : 62% Décision, 33% KO/TKO, 20% Soumission

**Sorties** : Visualisations EDA, insights initiaux

---

### Phase 2️⃣ : Feature Engineering (`Feature_Engeneering.ipynb`)

**Objectif** : Transformer les statistiques brutes en features significatives

**Processus**:
1. **Parsing des données** :
   - Format "X of Y" → frappés/tentés
   - "MM:SS" → secondes

2. **Agrégation par fighter** :
   - Combinaison de tous les combats d'un fighter
   - Calcul de totaux et moyennes

3. **Calcul des 17 features de style** :

**Métriques de volume** :
- `sig_str_pm` : Frappés significatifs par minute
- `td_per_fight` : Projections par combat
- `sub_per_fight` : Soumissions par combat
- `kd_per_fight` : Knockdowns par combat
- `ctrl_sec_per_fight` : Contrôle au sol (secondes) par combat

**Métriques de précision** :
- `sig_str_accuracy` : Précision des frappés
- `td_accuracy` : Précision des projections

**Ratios de localisation des frappés** :
- `head_ratio` : % des frappés à la tête
- `body_ratio` : % des frappés au corps
- `leg_ratio` : % des frappés aux jambes

**Ratios de distance des frappés** :
- `dist_ratio` : % en distance
- `clinch_ratio` : % en clinch
- `ground_ratio` : % au sol

**Tendances de victoire** :
- `ko_rate` : % de victoires par KO/TKO
- `sub_rate` : % de victoires par soumission
- `dec_rate` : % de victoires par décision
- `win_rate` : Taux de victoire global

**Résultat final** : Matrice de features 6,053 × 22 colonnes

---

### Phase 3️⃣ : Prétraitement & Scaling (`Preprocessing_scaling.ipynb`)

**Stratégie de normalisation** : **RobustScaler**

**Pourquoi RobustScaler au lieu de StandardScaler** ?
- Les données UFC contiennent des outliers extrêmes (ex: 51.4 sig str/min)
- RobustScaler utilise la médiane et l'écart interquartile (robuste aux outliers)
- StandardScaler aurait compressé la majorité des fighters dans une plage très étroite

**Réduction dimensionnelle** : **PCA (Analyse en Composantes Principales)**
- 7 composantes retiennent **80.7% de la variance**
- Réduit de 17 à 7 features
- Utile pour la visualisation et l'accélération du clustering

**Datasets générés** :
- `fighter_features_scaled.csv` : 17 features complètes, normalisées
- `fighter_features_pca.csv` : 7 composantes PCA
- `robust_scaler.pkl` : Transformateur sauvegardé (pour new data)
- `pca_transformer.pkl` : PCA sauvegardé (pour new data)

---

### Phase 4️⃣ : Clustering (`Clustering.ipynb`)

**Comparaison des algorithmes** :

| Algorithme | K | Silhouette | Davies-Bouldin | Statut |
|-----------|---|-----------|-----------------|--------|
| **K-Means (full)** | **5** | **0.0908** | **2.223** | ✅ **SÉLECTIONNÉ** |
| K-Means (full) | 2 | 0.1557 | 2.211 | Trop simplifié |
| K-Means (PCA) | 5 | 0.0846 | 2.186 | Légèrement moins bon |
| Hierarchique | 5 | 0.0839 | 2.156 | Similaire à K-Means |
| DBSCAN | - | 0.1599 | - | Seulement 7.7% d'outliers |

**Modèle final : K-Means avec K=5**
- **Algorithme** : K-Means
- **Features** : 17 features complètes (normalisées)
- **K** : 5 clusters
- **Score Silhouette** : 0.0908 (normal pour un mix de styles MMA)
- **Random state** : 42

**Distribution des fighters par cluster** :
- Cluster 0 : 126 fighters (15.1%)
- Cluster 1 : 130 fighters (15.6%)
- Cluster 2 : 190 fighters (22.8%)
- Cluster 3 : 180 fighters (21.6%)
- Cluster 4 : 209 fighters (25.0%)

---

## 🥊 Résultats du clustering

### **Cluster 0 : Les Chasseurs de Soumission** 🎯

**Taille** : 126 fighters (~15%)

**Caractéristiques principales** :
- Tentatives de soumission très élevées : **0.587 par combat**
- Ratio au sol très haut : **33.1%**
- Style peu orienté frappés : ratios de frappés faibles
- Taux d'appuis exceptionnels

**Fighters représentatifs** : Charles Oliveira, Frank Mir, Randy Couture

**Style de combat** : Lutteurs de soumission qui privilégient les finitions par soumission plutôt que le volume global

---

### **Cluster 1 : Les Lutteurs / Contrôle** 🤼

**Taille** : 130 fighters (~16%)

**Caractéristiques principales** :
- Taux de projections le plus haut : **2.802 par combat**
- Temps de contrôle très élevé : **306 secondes par combat**
- Excellent taux de décision : **52.9%**
- Style équilibré avec domination au sol

**Fighters représentatifs** : Georges St-Pierre, Clay Guida, Demian Maia, Tito Ortiz

**Style de combat** : Lutteurs dominateurs qui gagnent par contrôle au sol et projections

---

### **Cluster 2 : Les Frappeurs Agressifs** 💥

**Taille** : 190 fighters (~23%)

**Caractéristiques principales** :
- Volume de frappés le plus élevé : **4.413 frappés significatifs/min**
- Taux de KO/TKO le plus haut : **50.7%**
- Focus en distance : **79.5% des frappés en distance**
- Style ultra-offensif

**Fighters représentatifs** : Max Holloway, Dustin Poirier, Donald Cerrone, Anderson Silva

**Style de combat** : Artistes du KO avec output de frappe ultra-rapide et volume élevé

---

### **Cluster 3 : Les Combattants Équilibrés** ⚖️

**Taille** : 180 fighters (~22%)

**Caractéristiques principales** :
- Équilibré dans tous les domaines (frappés, projections, soumissions)
- Taux de décision le plus haut : **59%**
- Fighters polyvalents avec capacités mixtes
- Style bien-arrondi

**Fighters représentatifs** : Jon Jones, Jeremy Stephens, Lyoto Machida

**Style de combat** : Combattants versés qui gagnent par points avec mix de techniques

---

### **Cluster 4 : Les Frappeurs de Volume** 🌊

**Taille** : 209 fighters (~25%)

**Caractéristiques principales** :
- Volume de frappés modéré : **3.093 frappés/min**
- Bonnes capacités de soumission : **0.538 par combat**
- Focus sur les points : **53.8% de taux de décision**
- Style actif avec mix

**Fighters représentatifs** : Jim Miller, Nate Diaz, Urijah Faber

**Style de combat** : Frappeurs actifs qui mixent soumissions et points pour les victoires

---

## 🛠️ Technologies utilisées

### Langages & Frameworks
- **Python 3.x** - Langage principal
- **Jupyter Notebook** - Environnement de développement interactif
- **Google Colab** - Plateforme de développement utilisée

### Bibliothèques Data & ML
- **pandas** - Manipulation de données
- **numpy** - Opérations numériques
- **scikit-learn** - Machine Learning :
  - `KMeans` - Clustering principal
  - `DBSCAN` - Clustering alternatif
  - `AgglomerativeClustering` - Clustering hiérarchique
  - `PCA` - Réduction dimensionnelle
  - `StandardScaler`, `RobustScaler` - Normalisation
  - Métriques : Silhouette, Davies-Bouldin, Calinski-Harabasz

### Visualisation & Analyse
- **matplotlib** - Graphiques matplotlib
- **seaborn** - Visualisations statistiques avancées

### Tracking d'expériences
- **MLflow** - Suivi des expériences ML :
  - Logging des paramètres
  - Tracking des métriques
  - Versioning des artifacts
  - Base de données `mlflow.db`

### Préservation de modèles
- **pickle** - Sérialisation :
  - `robust_scaler.pkl`
  - `pca_transformer.pkl`

---

## 📦 Installation

### Prérequis
- Python 3.7+
- pip ou conda

### Étapes d'installation

1. **Clonez le repository** :
```bash
git clone https://github.com/aymanaarab/Machine-Learning-UFC-Project.git
cd Machine\ Learning\ UFC\ Project
```

2. **Créez un environnement virtuel** (optionnel mais recommandé) :
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. **Installez les dépendances** :
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter mlflow
```

4. **Lancez Jupyter Notebook** :
```bash
jupyter notebook
```

---

## 🚀 Utilisation

### Exécution du pipeline complet

1. **Setup & Exploration** (`setup_discovering_data.ipynb`) :
   - Lance l'exploration initiale des données
   - Découvre les patterns et qualité des données
   - Génère des visualisations EDA

2. **Feature Engineering** (`Feature_Engeneering.ipynb`) :
   - Calcule les 17 features de style
   - Traite les formats de données brutes
   - Génère `fighter_features.csv`

3. **Prétraitement** (`Preprocessing_scaling.ipynb`) :
   - Applique RobustScaler
   - Effectue PCA (7 composantes)
   - Sauvegarde les transformateurs

4. **Clustering** (`Clustering.ipynb`) :
   - Compare les algorithmes de clustering
   - Entraîne le modèle K-Means optimal
   - Génère `fighter_clustered.csv` avec labels

### Utilisation du modèle entraîné

Pour appliquer le scaler et PCA sur de nouveaux fighters :

```python
import pickle
import pandas as pd

# Charger les transformateurs
with open('robust_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('pca_transformer.pkl', 'rb') as f:
    pca = pickle.load(f)

# Charger les données
new_fighters = pd.read_csv('new_fighter_features.csv')

# Appliquer les transformations
scaled_data = scaler.transform(new_fighters)
pca_data = pca.transform(scaled_data)
```

### Accès aux résultats

Fichier de sortie principal : `data/processed/fighter_clustered.csv`

Colonnes :
- Fighter metadata (ID, nom, nombre de combats, poids, allure)
- 17 features de style
- `cluster` - Label du cluster (0-4)

---

## 📊 Insights clés

### Découvertes principales

1. **Diversité des styles** : Les 5 clusters capturent distinctement les archétypes de combat :
   - Lutteurs vs Frappeurs vs Équilibrés

2. **Outliers naturels** : Une silhouette basse (0.0908) est normale en MMA car les fighters innovent et se chevauchent souvent

3. **PCA efficace** : 7 composantes capturent 80.7% de variance, permettant une visualisation claire

4. **RobustScaler crucial** : Les outliers de frappe sont réels et significatifs (Max Holloway = 51.4 sig str/min)

5. **K=5 optimal** : Mieux que K=2 (trop simplifié) et K>5 (fragmentation excessive)

### Implications pratiques

- **Scouting** : Identifier le style dominant et les faiblesses
- **Matchmaking** : Appareiller les combattants en fonction des styles
- **Analyse tactique** : Adapter les stratégies basées sur le cluster
- **Prédiction** : Prédire les résultats via comparaison de styles

---

## 📈 Statistiques de dataset

### Données brutes
```
Total de fighters:           4,448
Combats enregistrés:         8,482
Fighters dans les 2 datasets: 2,637
Fighters avec ≥5 combats:    1,222+
Dataset final:               835 fighters (après nettoyage)

Taux de victoire:      Moyenne 51.5%, Étendue 0-100%
Combats par fighter:   Moyenne 18 combats
Taille:                Moyenne 70.1 pouces (5'10")
Poids:                 Moyenne 164.6 lbs, Étendue 115-300 lbs

Précision des frappés:     45.6% en moyenne
Précision des projections: 27.1% en moyenne
```

### Distribution des features

| Feature | Min | Max | Remarques |
|---------|-----|-----|-----------|
| sig_str_pm | 0.1 | 51.4 | Très asymétrique, outliers importants |
| td_per_fight | 0 | 12 | Variable selon le style |
| sub_per_fight | 0 | 7 | Différenciateur clé |
| ko_rate | 0% | 100% | Binaire selon le fighter |
| dec_rate | 0% | 100% | Complément du taux de KO |

---

## 🔬 Validation & Qualité

### Métriques d'évaluation

1. **Silhouette Score** (0.0908) :
   - Mesure de cohésion intra-cluster et séparation inter-cluster
   - Score bas normal pour données mixtes (MMA styles overlap)

2. **Davies-Bouldin Index** (2.223) :
   - Ratio de similarité intra/inter-cluster
   - Plus bas est mieux

3. **Calinski-Harabasz Score** :
   - Ratio de densité entre clusters
   - Score plus élevé préféré

### Validation du domaine

- ✅ Top fighters (Holloway, GSP, Oliveira) correctement classifiés
- ✅ Fighters polyvalents (Jon Jones) dans le cluster équilibré
- ✅ Corellation observable avec les statistiques UFC officielles
- ✅ Patterns alignés avec la connaissance du domaine

---

## 📝 Tracking d'expériences (MLflow)

Le projet utilise **MLflow** pour tracer tous les runs ML :

#Dans le repertoire du projet 
```bash
mlflow ui
```

Accédez à : `http://localhost:5000`

### Runs enregistrés
1. **feature_engineering_v1** : Calcul initial des features
2. **preprocessing_v1** : Scaling et PCA
3. **clustering_kmeans_k5_full** : Clustering final

### Données loggées
- Paramètres : algorithme, K, ensemble de features
- Métriques : Silhouette, Davies-Bouldin, scores de qualité
- Artifacts : CSV traitées, transformateurs pickle

---



---

