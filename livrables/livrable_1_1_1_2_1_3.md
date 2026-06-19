# Livrable 1.1 — 1.3 : Cas d'usage, Construction et Analyse Critique du Dataset TTM

> Dataset : `ttm_corpus/ttm_dataset_instruction_500.json` — 500 exemples mesurés

---

## 1.1 Présentation du cas d'usage

### Domaine choisi

Le domaine retenu est le **service client (SAV) d'un site e-commerce de vêtements et d'équipements de sport**, fictif mais réaliste, nommé **TTM (ToTheMax)**.

L'objectif est de construire un **assistant conversationnel** capable de répondre aux demandes des clients en s'appuyant sur la base documentaire de l'entreprise :

| Source | Contenu |
|--------|----------|
| FAQ | 310 entrées question/réponse |
| CGV / CGU | Documents contractuels complets |
| Politiques internes | Retours, remboursements, livraison, confidentialité |
| Catalogue produits | 50 SKU avec prix, tailles, matières, stock |

### Problématique métier

Le SAV d'un e-commerce traite un volume élevé de demandes **répétitives et standardisées** :
- suivi de commande
- retours et remboursements
- gestion de compte
- informations produit

Ces demandes :
- mobilisent du temps humain sur des réponses à **faible valeur ajoutée** ;
- doivent rester **conformes** aux CGV et politiques officielles (risque juridique et image de marque) ;
- doivent être traitées **rapidement et de manière homogène**, quel que soit le canal.

### Objectif technique

Fine-tuner un **LLM (Mistral 7B Instruct via QLoRA)** pour automatiser ces réponses tout en :

- restant **fidèle aux sources officielles** (CGV, FAQ, politiques)
- **refusant d'inventer** une information absente de la documentation (**anti-hallucination**)
- gérant les **émotions client** (colère, frustration, inquiétude)
- supportant les **négations**, le **bruit** (fautes, abréviations) et les **conversations multi-tours**

---

## 1.2 Construction du dataset

### Source des données

Le dataset est dérivé d'un **corpus métier synthétique TTM** construit spécifiquement pour le projet :

```
ttm_corpus/
├── 01_FAQ_complete.md              → 310 items Q/R
├── 02_CGV_completes.md             → Conditions Générales de Vente
├── 03_CGU_completes.md             → Conditions Générales d'Utilisation
├── 04_politique_livraison.md
├── 05_politique_retour.md
├── 06_politique_remboursement.md
├── 07_politique_confidentialite.md
└── 08_catalogue_50_produits.json   → 50 SKU
```

### Méthode de collecte

Les données ont été **générées de façon synthétique et structurée** (pas de vrais logs clients), puis enrichies pour se rapprocher de conditions réelles :

| Enrichissement | Description |
|----------------|-------------|
| 🔡 Bruit réaliste | Fautes de frappe, abréviations, ponctuation manquante |
| 😤 Émotions | Variation de l'état émotionnel du client |
| 🔄 Négations | Formulations négatives et doubles négations |
| ❓ Cas ambigus | Questions hors-périmètre, demandes impossibles |
| 💬 Multi-tours | Dialogues en plusieurs échanges |

Chaque exemple est **traçable** : il référence les sources documentaires exactes utilisées (`faq`, `cgv`, `catalogue`, `policy`), ce qui permet de vérifier la cohérence et de limiter les hallucinations.

### Format des données

Chaque exemple suit un **schéma JSON strict** :

```json
{
  "id": "...",
  "category": "SAV | Produits | Compte | Livraison | Aberrations",
  "instruction": "question du client",
  "conversation": [
    { "role": "user",      "content": "..." },
    { "role": "assistant", "content": "..." }
  ],
  "response": "réponse de référence",
  "sources": {
    "faq":       [...],
    "cgv":       [...],
    "catalogue": [...],
    "policy":    [...]
  },
  "metadata": {
    "emotion":   "neutre | colère | frustration | inquiétude | satisfaction",
    "noise":     true,
    "negation":  false,
    "multiturn": true
  }
}
```

### Formats d'export

Le dataset est converti en deux formats d'instruction-tuning :

**Format Alpaca**
```json
{
  "instruction": "...",
  "input": "...",
  "output": "..."
}
```

**Format Mistral**
```
<s>[INST] ... [/INST] ... </s>
```

Puis exporté en **JSONL** avec un **split stratifié 80/20** (train / validation).

### Répartition des exemples

**Total : 500 exemples**

| Catégorie | Nombre | Part |
|-----------|--------|------|
| SAV | 195 | 39 % |
| Produits | 150 | 30 % |
| Compte | 70 | 14 % |
| Livraison | 70 | 14 % |
| Aberrations | 15 | 3 % |
| **Total** | **500** | **100 %** |

**Couverture des cas difficiles (métadonnées)**

| Type | Nombre |
|------|--------|
| Bruit | 92 |
| Émotion non neutre | 80 |
| — dont colère | 18 |
| — dont frustration | 17 |
| — dont inquiétude | 16 |
| — dont satisfaction | 29 |
| Émotion neutre | 420 |
| Négation | 80 |
| Multi-tours | 150 |
| Mono-tour | 350 |

---

## 1.3 Analyse critique des données

### Métriques de qualité mesurées

> Chiffres calculés par script Python sur le fichier `ttm_dataset_instruction_500.json`

| Métrique | Valeur |
|----------|--------|
| Total d'exemples | 500 |
| Instructions uniques | 382 |
| **Instructions dupliquées** | **118 (24 %)** |
| Réponses uniques | 143 |
| **Réponses dupliquées** | **357 (71 %)** |
| Triplets (instruction+conv+réponse) uniques | 460 |
| **Triplets dupliquées** | **40 (8 %)** |
| Conversations vides (`[]`) | 350 (mono-tour) |

---

### Biais possibles

#### 1. Biais de standardisation des réponses ⚠️

Sur 500 exemples, seulement **143 réponses uniques** → taux de duplication ≈ **71 %**.

C'est **volontaire** (réponses-types SAV, gabarits anti-hallucination), mais cela :
- biaise le modèle vers un **style répétitif**
- peut nuire à la **variété perçue** par les clients
- risque de produire des réponses "robotiques"

#### 2. Déséquilibre des catégories ⚠️

```
SAV        ████████████████████████  39 %
Produits   ███████████████           30 %
Compte     ███████                   14 %
Livraison  ███████                   14 %
Aberrations█                          3 %  ← sous-représenté
```

Le modèle risque de **sous-apprendre** la gestion des cas hors-périmètre.

#### 3. Biais émotionnel ⚠️

```
Neutre      ████████████████████████████████████████████  84 %
Non-neutre  ██████                                        16 %
```

**84 % des exemples sont neutres.** Les émotions fortes (colère, frustration) restent minoritaires, ce qui **limite l'apprentissage du ton empathique**.

#### 4. Biais de génération synthétique ⚠️

Les données étant **100 % synthétiques**, elles reflètent :
- le style et les hypothèses du générateur
- un vocabulaire restreint et homogène
- l'absence de cas réels rares mais critiques

---

### Données manquantes

- ❌ Pas de **vrais logs clients** ni de vraie distribution des demandes réelles
- ❌ Absence de **données opérationnelles réelles** (numéros de commande, délais réels, ruptures de stock, litiges complexes, remboursements partiels)
- ❌ **Diversité linguistique limitée** : peu de variété régionale, multilingue, ou de registres très familiers
- ❌ Pas de **retours négatifs réels** ni d'historique d'escalade vers un humain
- ❌ Absence de **validation humaine** des réponses de référence

---

### Cas ambigus

| Type | Nombre | Risque |
|------|--------|--------|
| Aberrations (hors-périmètre) | 15 | Trop peu pour apprendre à refuser |
| Négations | 80 | Traitement de référence gabarisé |
| Multi-tours | 150 | Contexte parfois artificiel |
| Triplets dupliqués | 40 | Frontière entre diversité et redondance |

---

### Limites du dataset

| Limite | Impact |
|--------|--------|
| 100 % synthétique | Représentativité partielle de la réalité terrain |
| Volume modeste (500 ex.) | Insuffisant pour un usage production sur 5 catégories |
| Faible variété lexicale côté réponses | Risque de sur-apprentissage stylistique |
| Pas de validation humaine | Qualité des réponses non certifiée |
| Pas de mesure de satisfaction réelle | Impossible d'évaluer l'impact terrain |

---

### Réponses aux questions critiques

#### ❓ Les données représentent-elles correctement le problème ?

**Partiellement.**

La **structure** du problème SAV e-commerce est bien couverte : catégories, sources traçables, émotions, bruit, négations, multi-tours.

En revanche, la **distribution** ne reflète pas un trafic réel :
- déséquilibre des catégories
- sur-représentation du neutre (84 %)
- réponses très standardisées (143 uniques sur 500)

> Le dataset est un **bon support pédagogique et un prototype crédible**, mais pas un échantillon statistiquement représentatif de la production.

---

#### ❓ Quels biais peuvent être introduits ?

1. **Biais de standardisation** → le modèle apprend des gabarits → réponses peu variées
2. **Biais de déséquilibre** → SAV sur-représenté, Aberrations sous-représentées
3. **Biais émotionnel** → majorité neutre → empathie sous-apprise
4. **Biais de génération synthétique** → style homogène, absence de cas réels rares

---

#### ❓ Que manque-t-il pour un usage professionnel ?

- [ ] De **vraies données terrain** (logs anonymisés) avec la vraie distribution des demandes
- [ ] Un **volume plus important** (idéalement 5 000–10 000 exemples) et un meilleur équilibrage
- [ ] Une **validation humaine** (annotation qualité, revue juridique des réponses CGV)
- [ ] Une **conformité RGPD réelle** (anonymisation, consentement, gestion des données personnelles)
- [ ] Des **garde-fous d'évaluation** :
  - taux d'hallucination
  - fidélité aux sources
  - taux d'escalade vers un humain
- [ ] Un dispositif de **mise à jour continue** quand la FAQ ou les CGV évoluent

---

## Synthèse

```
✅ Points forts
   - Structure JSON riche et traçable
   - Couverture des cas difficiles (bruit, négations, multi-tours)
   - Sources documentaires référencées (anti-hallucination)
   - Formats d'export compatibles Alpaca et Mistral

⚠️ Points d'amélioration
   - 71% de réponses dupliquées → diversifier les formulations
   - 84% neutre → augmenter les cas émotionnels
   - 3% d'aberrations → augmenter à ~10%
   - 0% de données réelles → enrichir avec de vrais logs anonymisés
```

---

*Document généré dans le cadre du cours LLM — Fine-tuning Mistral 7B sur le dataset TTM (ToTheMax)*
