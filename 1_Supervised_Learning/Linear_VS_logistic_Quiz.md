# Machine Learning Practice Questions & Answers

This document compiles the exact 9 questions and answers reviewed during our practice session.

---

### Question 1
**User:** Linear regression is suitable for predicting email is spam or not? True or false?
**Answer:** False.

---

### Question 2
**User:** Classification problems can have more than two classes? True or false?
**Answer:** True.

---

### Question 3
**User:** You are given a small dataset of university students with the goal of predicting (Final_Grade). Before any modeling, which combination keeps informative features and removes noise? [Plus options A, B, C, D]
| Study_Hours   |   Student_ID | Major   |   Attendance |   Project_Score |   Final_Grade | Unnamed   |
|:--------------|-------------:|:--------|-------------:|----------------:|--------------:|:----------|
| 15.5          |           22 | AI      |         78.4 |              75 |            22 | misc      |
| 10.3          |           23 | AI      |         81.5 |              85 |            26 | misc      |
| 13.8          |           19 | ME      |         82.2 |              73 |            39 | misc      |
| nan           |           16 | BIO     |         80.9 |              72 |            43 | misc      |
| 9.4           |            9 | AI      |         88.7 |              95 |            42 | misc      |
| 20.6          |            8 | BIO     |         90.4 |              82 |            21 | misc      |
| 14.1          |            6 | ME      |         88.7 |              70 |            47 | misc      |
| 15            |           12 | AI      |         85.7 |             120 |            43 | misc      |
| 8.6           |            5 | CS      |         88.7 |              91 |            49 | misc      |
| 15.1          |            4 | BIO     |         87.2 |              82 |            49 | misc      |
| 14.1          |            8 | ME      |         68.2 |             105 |            47 | misc      |
| 12.5          |           13 | ME      |         91.2 |              73 |            54 | misc      |
| 15.9          |           17 | AI      |         70.4 |              91 |            40 | misc      |
| 8.7           |           10 | CS      |         86.4 |              74 |            46 | misc      |
| 14.5          |           15 | CS      |         74.6 |             130 |            37 | misc      |
| 14            |           21 | AI      |         72   |              87 |            60 | misc      |
| 13.2          |            1 | AI      |         76.6 |              79 |            48 | misc      |
| 4.2           |            2 | AI      |         72.2 |              68 |            38 | misc      |
| Eighteen      |           11 | AI      |         92.1 |              64 |            40 | misc      |
| 11.2          |           20 | CS      |         83.5 |              72 |            32 | misc      |
| 14.3          |           18 | ME      |         89.9 |              72 |            29 | misc      |
| 16.9          |           24 | AI      |         84.8 |              67 |            40 | misc      |
| 7.2           |            3 | BIO     |        nan   |              78 |            42 | misc      |
| Ten           |           14 | ME      |        nan   |             nan |            43 | misc      |
| 12.9          |          nan | BIO     |        nan   |             nan |            27 | misc      |
**Answer:** 
```python
data = data.drop('Unnamed', axis=1)
data = data.join(pd.get_dummies(data['Major']).astype(int)).drop('Major', axis=1)
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
data = data.drop(columns=['Student_ID'])
```

```python
data = data.drop('Unnamed', axis=1)
data['Study_Hours'] = pd.to_numeric(data['Study_Hours'], errors='coerce')
data.drop('Final_Grade', axis=1, inplace=True)
data = data.drop(columns=['Student_ID'])
data = data.join(pd.get_dummies(data['Major']).astype(int)).drop('Major', axis=1)
```

```python
data = data.drop('Unnamed', axis=1)
data = data.drop('Major', axis=1)
data['Study_Hours'] = data['Study_Hours'].astype(str)
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
```

```python
data['Unnamed'] = data['Unnamed'].fillna('misc')
data.drop_duplicates(inplace=True)
data['Major'] = data['Major'].astype(bool)
data.dropna(inplace=True)
data = data.drop(columns=['Student_ID'])
```

---

### Question 4
**User:** Which of the following is true about Linear Regression?
A) It predicts discrete values.
B) It predicts continuous values.
C) It is mainly used for classification.
D) It can only handle binary outputs.
**Answer:** B) It predicts continuous values.

---

### Question 5
**User:** What is the primary difference between classification and regression problems in machine learning? In short phrases describe them in points
**Answer:** 
* **Classification:**
  * Predicts **discrete categories** or classes.
  * Outputs are categorical labels (e.g., yes/no, spam/not spam, cat/dog/bird).
* **Regression:**
  * Predicts **continuous numerical values**.
  * Outputs are real numbers (e.g., house prices, temperatures, exam scores).

---

### Question 6
**User:** A company wants to predict how much profit they will make next quarter based on several factors such as marketing spend, number of employees, total sales, and production costs. Which model would you suggest they use? To which category of machine learning problems does this belong? Give me one/one line answers to each
**Answer:**
* **Model:** Multiple Linear Regression
* **Category:** Regression (Supervised Learning)

---

### Question 7
**User:** What kind of output do we expect from a classification model?
**Answer:** We expect the discrete outcome like classes or categorical labels.

---

### Question 8
**User:** Which of the following best describes the purpose of a regression model?
A) To classify data into categories.
B) To predict continuous outcomes.
C) To minimize classification error.
D) To maximize accuracy.
**Answer:** To predict continuous outcomes.

---

### Question 9
**User:** What is the primary similarity between classification and regression problems in machine learning?
**Answer:** Both of them are supervised learning as they both use labeled data.
