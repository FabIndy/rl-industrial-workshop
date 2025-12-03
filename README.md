# Industrial Workshop Optimization with Q-Learning  
*A reinforcement-learning environment modeling batch production under real-world constraints.*

This project implements a complete Q-Learning workflow for a simulated industrial workshop that must manage stock limits, production batches, time-dependent actions, and storage penalties.  
It is the fourth and most advanced iteration of the model, integrating all previously added constraints (stock, costs, time, penalties).

---

## 1. Environment Rules

The workshop operates with two bounded stock variables:

- **stock_raw** — raw material (0 to 10)  
- **stock_sell** — finished products (0 to 10)

The agent can perform **22 discrete actions**:

| Action | Description                                      |
|--------|--------------------------------------------------|
| 0      | Wait                                             |
| 1..10  | Produce k units of Product 1 (P1)                |
| 11..20 | Produce k units of Product 2 (P2)                |
| 21     | Order +5 units of raw material                   |

**Hard constraints:** both stock levels must remain within `[0, 10]`.

---

## 2. Costs and Rewards

### Product 1 (P1)
- Raw material cost: **1 unit per product**
- Profit: **+2 per unit**
- Duration: **1 time unit per product**

### Product 2 (P2)
- Raw material cost: **2 units per product**
- Profit: **+20 per unit**
- Duration: **3 time units per product**

### Raw Material Order (+5 MP)
- Reward: **–5**
- Duration: **1**

### Wait
- If `stock_raw = 0` → **–1**
- Otherwise → 0  
- Duration: **1**

### Storage Penalty
Applied after each action:

```
reward -= 0.5 * stock_sell
```

---

## 3. Time Management

Time directly affects rewards through action duration:

| Action        | Duration |
|---------------|----------|
| Wait          | 1        |
| Order         | 1        |
| Produce P1    | k        |
| Produce P2    | 3k       |

Each episode lasts **50 time units max**.

---

## 4. Q-Learning Training and Update Equation

### Q-Learning Update Equation

The Q-table is updated using the Bellman rule:

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \, \max_{a'} Q(s',a') - Q(s,a) \right]
\]

Where:

- \(s\) = current state  
- \(a\) = chosen action  
- \(r\) = reward received  
- \(s'\) = next state  
- \(a'\) = possible next actions  
- \(\alpha\) = learning rate  
- \(\gamma\) = discount factor  

This rule allows the agent to estimate long-term value by integrating both immediate rewards and future opportunities.

### Hyperparameters

- `alpha = 0.1`  
- `gamma = 0.95`  
- `epsilon_decay = 0.995`  
- `n_actions = 22`

The Q-table is defined as:

```
Q[stock_raw][stock_sell][action]
```

---

## 5. Optimal Policy (Learned Behavior)

- P2 is preferred when `stock_sell` is low  
- P1 regulates penalties when `stock_sell` is high  
- Ordering raw material occurs **only when stock_raw = 0**  
- Waiting appears in high-penalty or low-time-remaining states  

---

## 6. Business-Oriented Synthesis

### 1. Ordering Policy
The agent orders raw material **only when stock_raw = 0**.

### 2. Product 2 (P2) — High-Profit Engine
When `stock_sell ≤ 3`:
- P2 dominates  
- Long batches maximize early-cycle profitability  

### 3. Product 1 (P1) — Fine Regulation
When `stock_sell ≥ 4`:
- P1 prevents excessive storage penalties  

### 4. Waiting
Used when:
- penalties would rise excessively  
- remaining time is too short  

### 5. Strategic Overview
- **Use P2 aggressively** early  
- **Use P1** to stabilize penalties  
- **Order** only with zero raw stock  
- **Wait** when necessary  

---

## 7. Source Notebook

Located in:

- `notebook/modelisation4.ipynb`

---

# 🇫🇷 Version Française — Modélisation 4 : Atelier Industriel avec Q-Learning

Ce projet implémente un environnement complet de Reinforcement Learning simulant un atelier industriel soumis à des contraintes réelles : limites de stock, production en lots, pénalités de stockage, durée variable des actions et arbitrages économiques.

---

## 1. Règles de l’environnement

Deux stocks bornés :

- **stock_raw** : matière première (0 à 10)  
- **stock_sell** : produits finis (0 à 10)

Actions possibles (**22 actions**) :

| Action | Description                                      |
|--------|--------------------------------------------------|
| 0      | Attendre                                         |
| 1..10  | Produire k unités de Produit 1 (P1)              |
| 11..20 | Produire k unités de Produit 2 (P2)              |
| 21     | Commander +5 unités de MP                        |

---

## 2. Coûts et récompenses

### Produit 1 (P1)
- Coût MP : **1**  
- Marge : **+2**  
- Durée : **1**

### Produit 2 (P2)
- Coût MP : **2**  
- Marge : **+20**  
- Durée : **3**

### Commande
- Récompense : **–5**  
- Durée : **1**

### Attente
- Si `stock_raw = 0` → **–1**, sinon 0  

### Pénalité de stockage
```
reward -= 0.5 * stock_sell
```

---

## 3. Gestion du temps

| Action      | Durée |
|-------------|-------|
| Attendre    | 1     |
| Commander   | 1     |
| Produire P1 | k     |
| Produire P2 | 3k    |

Un épisode dure **50 unités de temps**.

---

## 4. Entraînement Q-Learning et Équation de Mise à Jour

### Équation de Bellman

La Q-table est mise à jour selon :

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \, \max_{a'} Q(s',a') - Q(s,a) \right]
\]

Avec :

- \(s\) : état courant  
- \(a\) : action effectuée  
- \(r\) : récompense reçue  
- \(s'\) : nouvel état  
- \(a'\) : actions possibles  
- \(\alpha\) : taux d’apprentissage  
- \(\gamma\) : facteur de discount  

---

## 5. Politique optimale

- P2 privilégié lorsque `stock_sell` est bas  
- P1 en régulation lorsque `stock_sell` est élevé  
- Commande uniquement avec **stock_raw = 0**  
- Attente dans les zones à forte pénalité  

---

## 6. Synthèse métier

- **P2** maximise le profit initial  
- **P1** stabilise les pénalités  
- **Commande** uniquement en cas de pénurie  
- **Attente** lorsque produire serait néfaste  

---

## 7. Notebook

Disponible dans :  
`notebook/modelisation4.ipynb`
