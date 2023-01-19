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

````
```javascript
function foo () {
    return "bar";
}
```
````

This would be rendered as:

```javascript
function foo () {
    return "bar";
}
```


Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: http://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
[kaggle]: https://www.kaggle.com/mlg-ulb/creditcardfraud/version/3
