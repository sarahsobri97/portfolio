---
title: "Project 1: Credit Card Fraud Detection Under Extreme Imbalanced Data"
layout: post
---

![title](https://raw.githubusercontent.com/sarahsobri97/sarahportfolio.github.io/master/assets/credit%20card%20fraud%20picture.jpeg)

#### Due to the ever-increasing volume of data, it is becoming increasingly difficult for a human expert to discover meaningful patterns from transaction data. It involves deciding which transactions are false among millions of daily transactions. Data-driven initiatives are increasingly being utilised to help with online transactions, user authentication, credit card details verification, and detecting and blocking fraudulent transactions. As a response, banks must engage in innovative technology such as real-time checking, machine learning, and behavioral biometrics to combat fraud.








## Imbalanced data

![title](https://raw.githubusercontent.com/sarahsobri97/sarahportfolio.github.io/master/assets/Imbalanced%20dataset.png)

#### The imbalanced learning challenge refers to the efficiency of machine learning models in the situation of unbalanced data and severe class distribution skews. Learning from imbalanced data sets needs innovative solutions, algorithms, and applications for quickly translating large volumes of raw data into information and knowledge discovery due to their inherent complexity. The majority of cases in such scenarios are divided into one of two classes, with the latter being the more crucial of the two. To date, no specific resampling approach has been proclaimed the best strategy for preventing credit card fraud (Makki et al., 2019a). 

## Gaps in Literature

#### Determining which resampling strategy is the most effective in overcoming this problem remains a mystery to be solved. That being said, many researchers have recently adopted the SMOTE methodology as their ideal way of resampling. It is said to be incredibly efficient and has exceeded many other resampling approaches numerous times. Nevertheless, (Hordri et al., 2018) found that this approach was still vulnerable to oversampling on both minority and majority data, which is one of its known flaws. Thus, our research sees an opportunity to tackle this problem by implementing various resampling strategies to close the research gaps about the current class imbalance problem. Furthermore, this study proposes the use of another resampling technique, the Modified Synthetic Minority Oversampling Technique, which has never been utilised before to balance the fraud dataset.

## Aim and Objective

1. To discover and construct numerous resampling strategies that can be used to address the credit card fraud detection class imbalanced challenge.
2. To assess the implementation of the proposed resampling technique in combination with machine learning algorithms in predicting credit card fraud transactions using various resampling techniques.

## Scope of Research

#### This research proposes the use of the Modified SMOTE (MSMOTE) approach to address the SMOTE constraint of oversampling and latent noise in the data. By comparing each resampling technique with each other, this research will demonstrate the advantages and drawbacks of each. Furthermore, we will discuss how the proposed resampling strategy might increase the performance of classifier algorithms. Eventually, we will present the most optimum technique of the resampling technique and the machine learning algorithm for dealing with credit card fraud.

## Significance of Research

#### In light of modern digitalization, using credit cards to conduct financial transactions at banks or other financial institutions is a common practice. Switching from a manual process to a completely automated system, such as those seen in smart cities, is not without its risk. Credit card fraud is still a problem in the financial business, therefore it must be handled as soon as possible. The study's findings will significantly benefit the community in terms of avoiding fraudulent credit card transactions and thereby decreasing damages by offering new knowledge about the class imbalance concern that credit card fraud detection system developers must solve. This research will also be used to design more effective credit card fraud detection systems in the future.

## Methodology for Handling Imbalanced Domains

![title](https://raw.githubusercontent.com/sarahsobri97/sarahportfolio.github.io/master/assets/Methodology.png)

## Data Source and Type

#### This project makes use of a standard dataset that is freely available on Kaggle, where it comprises over 284,807 records of credit card transactions of European cardholders that occurred within two days in September 2013. The dataset is highly imbalanced where it displays only 492 fraudulent transactions out of 284,807 transactions. The class variable is divided into two, where 1 is a positive class, which is fraud and 0 is the negative class, which is non-fraud. Missing values are not found in the dataset. Visit [kaggle][kaggle] to view the dataset.

## Proposed Machine Learning Models

1. Random Under Sampling
2. Random Over Sampling
3. Synthetic Minority Oversampling Technique (SMTOE)
4. Modified Synthetic Minority Oversampling Technique (MSMOTE)

## Machine Learning Models

1. Logistic Regression
2. Random Forest
3. XGBoost

## Evaluation Metrics For Model Validation

1. Confusion Matrix
2. Accuracy
3. Sensitivity
4. Specificity
5. Precision
6. F1- Score
7. Precision- Recall (PR) curve

## Solved End-to-End Credit Card Fraud Detection Source Code

### Importing Libraries Used

![title](https://raw.githubusercontent.com/sarahsobri97/sarahportfolio.github.io/master/assets/EDA.png)

#### To import the data into a Dataframe object, we have to utilise the pandas package. Numpy, on the other hand, is a general-purpose array-preprocessing package that includes a high-performance multidimensional array object as well as features for managing them. Plotting was accomplished using the matplotlib and seaborn libraries. Some data preprocessing, model development and model evaluation was done with the sklearn library. Finally, when working with classification involving imbalanced classes, the imblearn library was implemented.

### Performing Exploratory Data Analysis
Source code can be included by fencing the code with three backticks. Syntax highlighting works automatically when specifying the language after the backticks.

![title](https://raw.githubusercontent.com/sarahsobri97/sarahportfolio.github.io/master/assets/Descriptive%20Analysis.png)

#### Skimpy is a lightweight programme that also offers summary statistics for datagram variables. We will begin by reviewing the data before beginning EDA operations. There are 284807 rows and 31 columns, as we can see, with no missing data.

```javascript
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/CreditCardFraud/creditcard.csv')
```

![title](https://raw.githubusercontent.com/sarahsobri97/sarahportfolio.github.io/master/assets/first%2020%20rows.png)

#### We can see that the features 'Time' and 'Amount' differ significantly from the rest, thus we'll need to standardise them before generating the model during data pre-processing.

### Remove Dupicates

```javascript
df.duplicated(keep='first).sum()
df.drop_duplicates(keep='first', inplace=True).sum()
```

The dataset contains no null values by default. However, it contained 1081 duplicate values. Using the skim function to remove the duplicated values, we discovered that the dataset now has 283726 rows. We also discovered that the non-fraud class now has 283253 rows while the fraud class had 473 rows.

![title](https://raw.githubusercontent.com/sarahsobri97/sarahportfolio.github.io/master/assets/Data%20summary%20after%20removing%20duplicates.png)

### Standardizaton

#### Rescaling the range of data so that the mean of observed data is 0 and the standard deviation is 1 is the process of standardising a dataset. Because their values differed substantially from the others, we just need to scale the 'Time' and 'Amount' variables.

```javascript
sc = StandardScaler()
amt = df['Amount'].values
df['Amount'] = sc.fit_transform(amt.reshape(-1,1))
df['Amount']
```

```javascript
sc = StandardScaler()
time = df['Time'].values
df['Time'] = sc.fit_transform(amt.reshape(-1,1))
df['Time']
```

### Data Splitting

#### We have given x the task of carrying all the independent variables and y the task of carrying only the target variable, 'Class.' The data was then split into x train, x test, y train, and y test using the train_test_split function, with the training set receiving 80% of the data and the test set receiving the remainder.

```javascript
y = df['Class']
x = df.drop(['Class'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=100, test_size=0.2)
```

## Result Summary of all the models

![title](https://raw.githubusercontent.com/sarahsobri97/sarahportfolio.github.io/master/assets/Final%20Result.png)

#### After analysing all the models, we can conclude that the model that outperforms the rest is the random forest with SMOTE. In real-time scenarios, we cannot have both precision and recall high. If we increase the precision, it will simultaneously reduce recall and vice versa. This is known as the precision-recall trade-off.From table 2, we can see that the random forest model after resampling with SMOTE has the best-balanced trade-off with a precision of 88% and recall of approximately 85%. Precision refers to the percentage of our results being positive, and recall refers to the percentage of total positive results correctly classified by the model. For this work, we did not consider the tuned models as the outputs for precision and recall were compromised when tested for our top three models.

[kaggle]: https://www.kaggle.com/mlg-ulb/creditcardfraud/version/3
