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
* May month was most successful month for long term deposit acceptance with 33.43% success.
* Thursday and Monday were most successful days of the week for deposit acceptance with 20.94% and 20.67% success rate.

<h>Distribution by Target (Deposit)<h>
![Image](/images/class_distribution.png)

**Key Takeaways:** 
* Distribution of campaign outcome is greatly imbalanced.
* ~36k have unsuccessful outcome for depsoit acceptance and only ~4.6k were postive outcomes.

<h>Top 20 Features Correlation with Target (Deposit)<h>
![Image](/images/feature_correlation_with_deposit.png)

**Key Takeaways:** 
* Top highly correlated features with target (deposit) are duration, poutcome_success, contact_cellular and months of march, september & october.

<h>Distribution comparision for top correlated features<h>
![Image](/images/violin_chart_by_coef.png)

**Key Takeaways:**

* <b>Violin Chart - Duration:</b> This violin plot shows the distribution of the duration of the last contact in seconds.

  * Shape of the Violin: The shape of the violin indicates the density of data points at different duration values. A wider section of the violin means there are more data points with durations in that range.
  * Interpretation: You'll likely see a dense concentration of contacts at shorter durations, with the distribution tapering off as the duration increases. This suggests that most marketing calls were relatively short. Observing the shape can tell you if the duration data is skewed, has multiple peaks (modes), or is roughly symmetrical.

* <b>Violin Chart - Previous Contact Outcome (poutcome_success):</b> This violin plot visualizes the distribution of the poutcome_success feature. This feature indicates whether the outcome of the previous marketing campaign for a client was a success (represented by a value, 1, after one-hot encoding).

  * Shape of the Violin: Since this is a binary variable, the violin plot will show densities around the values representing "success" (i.e. 1) and other outcome (i.e. 0 for failure/nonexistent).
  * Interpretation: The width of the violin at different points will show the proportion of clients who had a successful outcome in the previous campaign compared to others. A wider section at the "failure" value would indicate a higher frequency of unsuccessful previous outcomes in the dataset.

* <b>Violin Chart - Previous Number of Contacts Count (previous):</b> This violin plot shows the distribution of the previous feature, which represents the number of contacts performed before the current campaign for a client.

  * Shape of the Violin: This plot will show the density of clients based on how many times they were contacted in previous campaigns. You'll see a large density at a low number of previous contacts, with the density decreasing as the number of previous contacts increases.
  * Interpretation: This plot helps understand how frequently clients in this dataset were targeted in prior campaigns. A heavy concentration at low values suggests that many clients were either new to the campaign or were not contacted extensively before.

* <b>Violin Chart - Contact Type (contact_cellular):</b> This violin plot visualizes the distribution of the contact_cellular feature, which indicates if the last contact was made via cellular phone.

  * Shape of the Violin: Similar to poutcome_success, this will show the density around the values representing "cellular" contact (i.e. 1) and "telephone" contact (i.e. 0).
  * Interpretation: The relative width of the violin at the "cellular" value compared to the "telephone" value will directly illustrate the proportion of contacts made via cellular versus telephone in the dataset. A wider section at "cellular" means cellular contact was more frequent and successful in driving the positive outcome of the campaign.

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

**Summary:**
* <b>ROC Curve:</b> The curve for the DummyClassifier likely hugs the diagonal line (the line of no-discrimination). This indicates that the model is performing no better than random chance at distinguishing between the positive and negative classes based on the True Positive Rate and False Positive Rate. The area under the curve (AUC) would be close to 0.5.
* <b>Precision-Recall Curve:</b> The Precision-Recall curve for the DummyClassifier would likely be a horizontal line at a precision equal to the proportion of the positive class in the dataset. This is because a dummy classifier that predicts based on class distribution won't effectively trade off precision and recall; its precision will remain constant regardless of the recall level.
* <b>Overall:</b> Both curves for the DummyClassifier demonstrate its inability to build a meaningful model and highlight that any performance metrics (like accuracy) achieved are simply a result of the class distribution in the data, not the model's ability to learn patterns.

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

**Summary:**
* <b>ROC Curve indicates decent overall discrimination:</b> The ROC curve for the 'Yes' class is above the diagonal line, suggesting the model can distinguish between positive and negative classes better than random chance. The AUC score of 0.94 confirms this, although there's still room for improvement towards a perfect score of 1.0.
* <b>Precision-Recall Curve suggests moderate performance for the positive class:</b> The Precision-Recall curve for the 'Yes' class shows a trade-off between precision and recall. While the curve is above the baseline (diagonal green line), the shape suggests that achieving high recall might come at the cost of lower precision, which is common with imbalanced datasets. The legend position being in the 'lower right' suggests that the curve is likely closer to the bottom and right of the plot area, indicating that at higher recall values, the precision might drop significantly.

<h>DecisionTreeClassifier model performance metrics<h>
![Image](/images/dt_metrics.png)

**Key Takeaways:**
| **Metric**   | **Value** | **Interpretation**                                         |
|--------------|-----------|------------------------------------------------------------|
| Accuracy     | 0.89      | High — most predictions are correct overall                |
| Precision    | 0.53      | Moderate — half of predicted positives are correct         |
| Recall       | 0.54      | Moderate — detects just over half of actual positives      |

**Summary:**
* <b>ROC Curve:</b> The ROC curve shows the model's ability to distinguish between the two classes. A curve that bows towards the top-left corner indicates better performance. The closer the curve is to the top-left, the higher the True Positive Rate is for a given False Positive Rate, indicating a good balance between correctly identifying positive cases and minimizing false alarms.
* <b>Precision-Recall Curve:</b> The Precision-Recall curve is particularly useful for imbalanced datasets. A curve closer to the top-right corner indicates better performance, signifying high precision and high recall. This means the model is good at both correctly identifying positive instances (high recall) and having a low rate of false positives (high precision).

<h>KNeighborsClassifier model performance metrics<h>
![Image](/images/knn_metrics.png)

**Key Takeaways:**
| **Metric**   | **Value** | **Interpretation**                                              |
|--------------|-----------|-----------------------------------------------------------------|
| Accuracy     | 0.90      | High — model predicts most outcomes correctly overall           |
| Precision    | 0.60      | Moderate to high — most predicted positives are correct         |
| Recall       | 0.34      | Low — misses most of the actual positive cases                  |

**Summary:**
* <b>ROC Curve and AUC:</b> The ROC curve for the "Yes" class is positioned towards the upper-left corner and has an AUC score of approximately 0.74. This indicates that the KNN model has a reasonably good ability to distinguish between the positive class (subscribed to a deposit) and the negative class (did not subscribe), performing better than random chance (AUC of 0.5).
* <b>Precision-Recall Curve and AUC:</b> The Precision-Recall curve shows the trade-off between precision and recall for the "Yes" class. Its position and shape suggest that achieving high recall (correctly identifying most positive cases) might come at the cost of lower precision (more false positives), which is common in imbalanced datasets like this one.
* <b>Overall Performance:</b> Considering both curves, the KNeighborsClassifier shows moderate performance in predicting the positive class. The ROC AUC is decent, but the Precision-Recall curve suggests that improving the balance between correctly identifying positive cases and minimizing false positives is important, potentially through different hyperparameters or addressing the class imbalance.

<h>Support Vectors Classifier model performance metrics<h>
![Image](/images/svm_metrics.png)

**Key Takeaways:**
| **Metric**   | **Value** | **Interpretation**                                               |
|--------------|-----------|------------------------------------------------------------------|
| Accuracy     | 0.90      | High — model correctly predicts most outcomes overall            |
| Precision    | 0.66      | High — most predicted positives are correct                      |
| Recall       | 0.19      | Very low — misses the vast majority of actual positive cases     |

**Summary:**
* <b>ROC Curve (AUC):</b> The ROC curve shows the trade-off between the True Positive Rate (sensitivity) and the False Positive Rate (1-specificity) at various threshold settings. An AUC (Area Under the Curve) score closer to 1 indicates a better performing classifier that can distinguish between positive and negative classes well.
* <b>Precision-Recall Curve (AUC):</b> The Precision-Recall curve plots the precision against the recall for different thresholds. It is particularly useful for evaluating models on imbalanced datasets, where the ROC curve can be misleading. A higher AUC for the Precision-Recall curve suggests that the model has a good balance between precision and recall.

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

**Summary:**
* The ROC curve for the fine-tuned Logistic Regression model is positioned higher and further to the left than the baseline model's ROC curve, indicating a better trade-off between the True Positive Rate and False Positive Rate.
* The Precision-Recall curve for the fine-tuned model shows improved performance, staying higher for longer across different recall levels compared to the baseline, which suggests better precision (fewer false positives) at various recall levels.
* Comparing metrics, the fine-tuned Logistic Regression model achieved a better AUC score (Area Under the ROC Curve), indicating overall improved discriminative ability between the positive and negative classes compared to the initial Logistic Regression model.

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

**Summary:**
* The ROC Curve for the "Yes" class is closer to the top-left corner than the "No" class, indicating better performance in distinguishing the positive class ("Yes") compared to the negative class ("No"). The ROC AUC score of 0.93 is above 0.5, suggesting the model has a reasonable ability to discriminate between the two classes. A higher AUC generally means a better model for this type of task.
* The Precision-Recall Curve provides insight into the trade-off between precision and recall specifically for the positive class ("Yes"). This curve is particularly useful for imbalanced datasets, like this one, as it focuses on the performance on the minority class. A higher area under the Precision-Recall curve generally indicates better performance, especially when correctly identifying positive instances is crucial.
* Comparing these curves with the baseline model's curves helps understand the extent of improvement achieved by the Decision Tree model. While the ROC AUC suggests some discriminatory power, the shape of the Precision-Recall curve will reveal how well the model balances precision and recall for predicting term deposit subscriptions.

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

**Summary:**
* The ROC curve being closer to the top-left corner with an AUC of 0.89 indicates that the KNeighborsClassifier has a moderate ability to distinguish between the positive (deposit 'Yes') and negative (deposit 'No') classes, outperforming a random guess (AUC of 0.5).
* The Precision-Recall curve shows the trade-off between the precision and recall for the positive class. The shape of this curve provides insights into the model's performance specifically on the minority class ('Yes'), which is important given the imbalanced dataset. A higher area under this curve generally indicates better performance on the positive class.
* Analyzing both curves together is crucial, especially with an imbalanced dataset. While the ROC curve shows overall discrimination, the Precision-Recall curve offers a more informative view of the model's effectiveness in identifying positive cases.

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

**Summary:**
* <b>ROC Curve:</b> The ROC curve being closer to the top-left corner indicates that the SVC model generally performs well in distinguishing between the "Yes" (subscribed) and "No" (did not subscribe) classes, with a good balance between True Positive Rate and False Positive Rate. The AUC score provides a single metric for this overall discriminatory power.
* <b>Precision-Recall Curve:</b> The Precision-Recall curve provides insight into the trade-off between precision and recall for the "Yes" class. A curve that is higher and further to the right indicates better performance, particularly when dealing with imbalanced datasets, showing that the model maintains relatively high precision even as recall increases.
* <b>Overall:</b> By examining both curves together, we get a more complete picture of the SVC model's performance, understanding both its general ability to separate classes (ROC) and its effectiveness in correctly identifying positive cases while minimizing false positives (Precision-Recall), which is crucial for a marketing campaign objective.

## Next steps and Recommendations

</b>Based on analysis and model metrcis, we learned that imbalanced dataset which is heavily weighted towards the unsuccessful marketing campaigns could not be used effectively to determine features which could provide best model performance. So, it raises below questions:

![Image](/images/contact_barplot_for_acceptance.png)

* Was the marketing campaign not executed effectively to have a balanced dataset?
* There was a high score amongst the "Yes" for customers contacted via Cellular. So did Bank adopted other mode of customer reachout like text messages or Whatsapp messages?
* We observed high score for customer with longer duration of contact. Did bank employed sufficient resources to improve the changes of succesful outcome?
</b>