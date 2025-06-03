---

layout: default
title: Predicting League of Legends Match Outcomes
--------------------------------------------------

# 🧠 Predicting League of Legends Match Outcomes

**Author**: Kalkidan Gebrekirstos
**Course**: DSC 80 - Spring 2025
**Project**: Final Project

---

![Banner](assets/league_banner.jpg)

## 🎮 Background

League of Legends is a competitive team-based game where two sides — blue and red — compete to destroy the enemy base. Despite its complexity, matches are often influenced by early-game events like securing the first tower or dragon.

In this project, I investigate:

1. Whether early objectives like the **first tower** significantly impact win rates.
2. If we can use these early features to **predict match outcomes**.
3. Whether the model shows **bias depending on team side** (blue vs. red).

---

## 📊 Exploratory Data Analysis

I began with an overview of 178,000+ match-team rows. Key patterns emerged:

* Teams securing the **first tower** win \~66% of the time.
* **First dragon** and **first baron** similarly increase win probability.
* Blue-side teams win **53%** of matches; red-side wins **47%**.

These findings suggested strong predictive value in early objectives.

![Win Rate by Side](assets/win_rate_by_side.png)

---

## 🧪 Hypothesis Testing

Using a **Z-test for proportions**, I tested whether securing the first tower statistically improves win rate.

**Results:**

* Teams with first tower: **66.3%** win rate
* Teams without: **31.4%** win rate
* Z-statistic: 146.952
* P-value: < 0.001  ✅ significant

Similar results were found for **first dragon** and **first baron**.

---

## 🤖 Modeling

Using early-game features (`firsttower`, `firstdragon`, `firstbaron`, `teamkills`, `deaths`, `assists`), I framed a binary classification problem.

### Model: `LogisticRegression`

* Accuracy: **95.4%**
* AUC Score: **0.99**
* Balanced precision/recall on both classes

The model is highly accurate and interpretable — a great outcome for such a simple feature set.

---

## ⚖️ Fairness Analysis

Grouping results by `side` revealed:

* **Blue**: 53.1% win rate
* **Red**: 46.6%

This \~6.5% gap could bias model predictions. I chose to **exclude `side`** from the model to avoid unfair advantage.

![Win Rate by Side](assets/win_rate_by_side.png)

---

## ✅ Final Summary & Lessons Learned

### 🔍 What I Did:

* Performed EDA and hypothesis tests
* Trained high-performing predictive models
* Audited side-based bias for fairness

### 📊 Key Takeaways:

* First objectives strongly predict outcome
* Side imbalance is real — fairness matters
* Simple models + good data = powerful results

### 🧠 What I Learned:

* How to apply data science to real-world decisions
* How to assess both performance and ethical impact
* How to explain complex results simply and clearly

---

📓 [Notebook](./template.ipynb)
📁 [GitHub Repo](https://github.com/Kalberhe/DSC80_FINAL_PROJECT)
📷 [Sample Plot](assets/win_rate_by_side.png)

