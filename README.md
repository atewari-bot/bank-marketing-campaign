# Assignment 17.1 - Bank Products Marketing Campaign

**Contents**

  * [Introduction](#Introduction)
  * [Business Understanding](#Business-Understanding)
  * [Data Understanding](#Data-Understanding)
  * [Data Preparation](#Data-Preparation)
  * [Baseline Model Comparison](#Baseline-Model-Comparison)
  * [Model Comparisons](#Model-Comparisons)
  * [Improving the Model](#Improving-the-Model)
  * [Next steps and Recommendations](#Next-steps-and-Recommendations)


## Introduction

Goal is to predict if the client will subscribe (yes/no) to a bank product (long term deposit) based on tele marketing campaign.

[Link to Bank Marketing Campaign Dataset](https://github.com/atewari-bot/bank-marketing-campaign/blob/main/data/bank-additional-full.csv)

[Link to Jupyter Notebook](https://github.com/atewari-bot/bank-marketing-campaign/blob/main/bank_marketing_campaign.ipynb)

## Business Understanding

There are two main approaches for enterprises to promote products and/or services: through mass campaigns, targeting general indiscriminate public or directed marketing, targeting a specific set of contacts. Data is from a Portuguese bank that used its own contact-center to do directed marketing campaigns.

                            [Start Campaign]
                                  |
                                  v
                            [Customer List (Database)]
                                  |
                                  v
                            [Call Customer]
                                  |
                                  v
                            [Is Contact Successful?] -- No --> [Log Attempt, End]
                                  |
                                  Yes
                                  |
                                  v
                            [Ask About Long-Term Deposit]
                                  |
                                  v
                            [Customer Response]
                              |           |
                              No          Yes
                              |            |
                            [Log Refusal]  [Proceed to Offer Details]
                              |            |
                              v            v
                            [End]     [Accept Terms?] -- No --> [Log Rejection, End]
                                            |
                                            Yes
                                            |
                                            v
                                  [Create Deposit Account]
                                            |
                                            v
                                      [Log Success, End]

## Data Understanding

The dataset collected is related to 17 campaigns that occurred between May 2008 and November 2010, corresponding to a total of 79354 contacts. During these phone campaigns, an attractive long-term deposit application, with good interest rates, was offered. For each contact, a large number of attributes was store and if there was a success (the target variable). For the whole database considered, there were 6499 successes (8% success rate).

### Data Preparation

This is the first step of Exploratory Data Analysis (EDA)

* Dataset size is <b>41188 X 21</b>.
* There were no missing values.
* <b><i>y</i> column: </b> Rename to <b>Deposit</b>.

### Understanding Data via visualization

<h>Distribution of Target (Deposit) by Category Features<h>
![Image](/images/pie_chart_for_category_distribution.png)

**Key Takeaways:** 
* 52.38% of people with house loan accepted long term deposit.
* 82.43% of people with personal loan accepted long term deposit.
* May month of most successful month for long term deposit acceptance with 33.43% success.
* Thursday and Monday were most successful days of the week for deposit acceptance with 20.94% and 20.67% success rate.

<h>Distribution by Target (Deposit)<h>
![Image](/images/class_distribution.png)

**Key Takeaways:** 
* Distribution of campaign outcome is greatly imbalanced.
* ~36k have unsuccessful outcome for depsoit acceptance and only ~4.6k were postive outcomes.

<h>Top 20 Features Correlation with Target (Deposit)<h>
![Image](/images/feature_correlation_with_deposit.png)

**Key Takeaways:** 
* Top highly correlated features with target (deposit) are duration, poutcome_success, contact_cellular and month of march, september & october.

<h>Distribution comparision for top correlated features<h>
![Image](/images/violin_chart_by_coef.png)

<h>Top 20 Features Heatmap with Target (Deposit)<h>
![Image](/images/heatmap_top20_coef.png)

**Key Takeaways:**
* Duration feature is most highly positively correlated and nr.employed feature is most highly negatively correlated.

## Baseline Model Comparison

<h>DummyClassifier as baseline model performance metrics<h>
![Image](/images/dummy_metrics.png)

**Key Takeaways:**
| **Metric**   | **Value** | **Interpretation**                                                 |
|--------------|-----------|--------------------------------------------------------------------|
| Accuracy     | 0.80      | High, but likely due to predicting the majority class (not useful) |
| Precision    | 0.12      | Very low — most positive predictions are incorrect                 |
| Recall       | 0.12      | Very low — misses nearly all actual positive cases                 |

## Model Comparisons

| Model               | Train Time | Train Accuracy | Test Accuracy  | Precision Score  | Recall Score  | F1 Score |
|---------------------|------------|----------------|----------------|------------------|---------------|----------|
| DummyClassifier     | 0.135791   | 0.800243       | 0.803714       | 0.121844         | 0.119612      | 0.120718 |
| LogisticRegression  | 0.488620   | 0.910106       | 0.915999       | 0.708481         | 0.432112      | 0.536814 |
| DecisionTree        | 0.271728   | 1.000000       | 0.894999       | 0.533475         | 0.540948      | 0.537186 |
| KNN                 | 0.130004   | 0.921608       | 0.899612       | 0.595825         | 0.338362      | 0.431615 |
| SVM                 | 64.057475  | 0.897329       | 0.897791       | 0.660448         | 0.190733      | 0.295987 |

<h>LogisticRegression model performance metrics<h>
![Image](/images/lr_metrics.png)

**Key Takeaways:**
| **Metric**   | **Value** | **Interpretation**                                       |
|--------------|-----------|----------------------------------------------------------|
| Accuracy     | 0.92      | High — most predictions are correct overall              |
| Precision    | 0.71      | High — when it says “positive,” it’s usually right       |
| Recall       | 0.43      | Moderate — misses more than half of positives            |

<h>DecisionTreeClassifier model performance metrics<h>
![Image](/images/dt_metrics.png)

**Key Takeaways:**
| **Metric**   | **Value** | **Interpretation**                                         |
|--------------|-----------|------------------------------------------------------------|
| Accuracy     | 0.89      | High — most predictions are correct overall                |
| Precision    | 0.53      | Moderate — half of predicted positives are correct         |
| Recall       | 0.54      | Moderate — detects just over half of actual positives      |

<h>KNeighborsClassifier model performance metrics<h>
![Image](/images/knn_metrics.png)

**Key Takeaways:**
| **Metric**   | **Value** | **Interpretation**                                              |
|--------------|-----------|-----------------------------------------------------------------|
| Accuracy     | 0.90      | High — model predicts most outcomes correctly overall           |
| Precision    | 0.60      | Moderate to high — most predicted positives are correct         |
| Recall       | 0.34      | Low — misses most of the actual positive cases                  |

<h>Support Vectors Classifier model performance metrics<h>
![Image](/images/svm_metrics.png)

**Key Takeaways:**
| **Metric**   | **Value** | **Interpretation**                                               |
|--------------|-----------|------------------------------------------------------------------|
| Accuracy     | 0.90      | High — model correctly predicts most outcomes overall            |
| Precision    | 0.66      | High — most predicted positives are correct                      |
| Recall       | 0.19      | Very low — misses the vast majority of actual positive cases     |

## Improving the Model

| Model              | Train Time | Train Accuracy | Test Accuracy | Precision Score | Recall Score | F1 Score | Best Score | Best Params |
|--------------------|------------|----------------|----------------|------------------|--------------|----------|------------|-------------|
| LogisticRegression | 6.99       | 0.9103         | 0.9154         | 0.7052           | 0.4278       | 0.5325   | 0.9092     | `LogisticRegression(C=0.1, max_iter=1000, random_state=42, solver='liblinear')` |
| DecisionTree       | 35.96      | 0.9160         | 0.9184         | 0.6773           | 0.5269       | 0.5927   | 0.9119     | `DecisionTreeClassifier(max_depth=5, min_samples_leaf=4, random_state=42)` |
| KNN                | 183.35     | 1.0000         | 0.9056         | 0.6829           | 0.3017       | 0.4185   | 0.8986     | `KNeighborsClassifier(metric='euclidean', n_neighbors=19, weights='distance')` |
| SVM                | 456.12     | 0.9149         | 0.8753         | 0.4692           | 0.8114       | 0.5946   | 0.8691     | `SVC(C=5, class_weight='balanced', gamma='auto', probability=True)` |

<h>Improved LogisticRegression model performance metrics<h>
![Image](/images/lr_grid_metrics.png)

**Key Takeaways:**
* No improvement observed in model performance after performing cross-validation using GridSearchCV.

| **Metric**   | **Value** | **Interpretation**                                       |
|--------------|-----------|----------------------------------------------------------|
| Accuracy     | 0.92      | High — most predictions are correct overall              |
| Precision    | 0.71      | High — when it says “positive,” it’s usually right       |
| Recall       | 0.43      | Moderate — misses more than half of positives            |

<h>Improved DecisionTreeClassifier model performance metrics<h>
![Image](/images/dt_grid_metrics.png)

**Key Takeaways:**
* The improved model shows a notable increase in precision, meaning it makes fewer false positive errors, which is valuable when the cost of false alarms is high. 
* The accuracy also improved slightly, indicating better overall correctness. 
* Recall remained roughly the same, so the model’s ability to identify actual positives hasn’t changed much. If improving recall is important, further tuning or alternative approaches may be needed.

| **Metric**   | **Previous Score** | **Improved Score** | **Interpretation of Improvement**                                             |
|--------------|--------------------|--------------------|-------------------------------------------------------------------------------|
| Accuracy     | 0.89               | 0.92               | Accuracy increased — model predicts more outcomes correctly overall           |
| Recall       | 0.54               | 0.53               | Recall stayed about the same — still detects just over half of positives      |
| Precision    | 0.53               | 0.68               | Precision improved significantly — positive predictions are now more reliable |

<h>Improved KNeighborsClassifier model performance metrics<h>
![Image](/images/knn_grid_metrics.png)

**Key Takeaways:**
* The improved model shows a higher precision, meaning it makes fewer false positive errors and is more confident when predicting positives. 
* This comes at the cost of a small decrease in recall, so the model misses more actual positives than before. 
* The overall accuracy increased slightly, reflecting better general correctness. 
* This trade-off is common: increasing precision often reduces recall. Depending on your application, you may want to tune the model or threshold to better balance these metrics.

| **Metric**   | **Previous Score** | **Improved Score** | **Interpretation of Change**                                                      |
|--------------|--------------------|--------------------|-----------------------------------------------------------------------------------|
| Accuracy     | 0.90               | 0.91               | Slight improvement — the model predicts outcomes slightly more accurately overall |
| Recall       | 0.34               | 0.30               | Slight decrease — the model detects fewer actual positives                        |
| Precision    | 0.60               | 0.68               | Noticeable improvement — positive predictions are more reliable now               |

<h>Improved Support Vectors Classifier model performance metrics<h>
![Image](/images/svm_grid_metrics.png)

**Key Takeaways:**
The updated model represents a dramatic shift in strategy:
* Recall surged from 0.19 to 0.92 — a major improvement in detecting nearly all positive cases.
* Precision dropped — the model now predicts more false positives.
* Accuracy decreased slightly, likely due to more false positives affecting the overall correct prediction rate.

| **Metric**   | **Previous Score** | **Improved Score** | **Interpretation of Change**                                |
|--------------|--------------------|--------------------|-------------------------------------------------------------|
| Accuracy     | 0.90               | 0.88               | Slightly Decreased — fewer overall correct predictions      |
| Recall       | 0.19               | 0.81               | Significant improvement — now detects most actual positives |
| Precision    | 0.66               | 0.47               | Decreased — more false positives among predicted positives  |


## Next steps and Recommendations

</b>Based on analysis and model metrcis, we learned that imbalanced dataset which is heavily weighted towards the unsuccessful marketing campaigns could not be used effectively to determine features which could provide best model performance. So, it raises below questions:

![Image](/images/contact_barplot_for_acceptance.png)

* Was the marketing campaign not executed effectively to have a balanced dataset?
* There was a high score amongst the "Yes" for customers contacted via Cellular. So did Bank adopted other mode of customer reachout like text messages or Whatsapp messages?
* We observed high score for customer with longer duration of contact. Did bank employed sufficient resources to improve the changes of succesful outcome?
</b>