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
| **Metric**     | **Value** | **Interpretation**                                                                  |
|----------------|-----------|-------------------------------------------------------------------------------------|
| Train Accuracy | 0.80      | Appears reasonable, but it's due to class imbalance or default prediction strategy. |
| Test Accuracy  | 0.80      | Misleadingly high; reflects bias toward majority class.                             |
| Precision      | 0.12      | Only 12% of predicted positives are correct; poor predictive power.                 |
| Recall         | 0.12      | Captures only 12% of actual positives; misses most true cases.                      |
| F1-Score       | 0.12      | Very low balance of precision and recall; ineffective model.                        |
| AUC (ROC)      | 0.51      | No real discriminatory ability; barely better than random guessing.                 |

**ROC & Precision-Recall Interpretation:**
* The ROC curve would be nearly diagonal, indicating random guessing behavior. The precision-recall curve would show consistently low precision across all recall levels, reflecting the model's inability to distinguish between classes effectively.

## Model Comparisons

| Model              | Train Time | Train Accuracy | Test Accuracy  | Precision Score  | Recall Score  | F1 Score  | AUC      |
|--------------------|------------|----------------|----------------|------------------|---------------|-----------|----------|
| DummyClassifier    | 0.106179   | 0.800243       | 0.803714       | 0.121844         | 0.119612      | 0.120718  | 0.505086 |
| LogisticRegression | 0.503941   | 0.910106       | 0.915999       | 0.708481         | 0.432112      | 0.536814  | 0.942592 |
| KNN                | 0.104422   | 0.921608       | 0.899612       | 0.595825         | 0.338362      | 0.431615  | 0.833783 |
| DecisionTree       | 0.254893   | 1.000000       | 0.894999       | 0.533475         | 0.540948      | 0.537186  | 0.740447 |
| SVM                | 61.403650  | 0.897329       | 0.897791       | 0.660448         | 0.190733      | 0.295987  | 0.935064 |

<h>LogisticRegression model performance metrics<h>
![Image](/images/lr_metrics.png)

**Key Takeaways:**
| **Metric**     | **Value** | **Interpretation**                                                                         |
|----------------|-----------|--------------------------------------------------------------------------------------------|
| Train Accuracy | 0.92      | Model fits the training data well; low underfitting.                                       |
| Test Accuracy  | 0.91      | Model generalizes well to unseen data; good overall correctness.                           |
| Precision      | 0.71      | When it predicts positive, it's correct 71% of the time; good at avoiding false positives. |
| Recall         | 0.43      | Only captures 43% of actual positives; may miss many true positives.                       |
| F1-Score       | 0.54      | Harmonic mean of precision and recall; indicates moderate balance.                         |
| AUC (ROC)      | 0.94      | Excellent ability to distinguish between classes; strong classifier overall.               |

**ROC & Precision-Recall Interpretation:**
* The ROC curve would show strong convexity toward the top-left corner, indicating excellent true positive rate with low false positive rate. The precision-recall curve would demonstrate high precision maintained across moderate recall levels, typical of conservative classification behavior.

<h>DecisionTreeClassifier model performance metrics<h>
![Image](/images/dt_metrics.png)

**Key Takeaways:**
| **Metric**     | **Value** | **Interpretation**                                                               |
|----------------|-----------|----------------------------------------------------------------------------------|
| Train Accuracy | 0.89      | Good fit to training data; slight underfitting may exist.                        |
| Test Accuracy  | 1.00      | Perfect accuracy on test data, which could indicate data leakage or overfitting. |
| Precision      | 0.53      | About half of predicted positives are correct; moderate precision.               |
| Recall         | 0.54      | Captures just over half of actual positives; moderate sensitivity.               |
| F1-Score       | 0.54      | Balanced but modest performance; average trade-off between precision and recall. |
| AUC (ROC)      | 0.74      | Fair ability to distinguish between classes; room for improvement.               |

**ROC & Precision-Recall Interpretation:**
* The ROC curve would show moderate performance with some trade-off between sensitivity and specificity. The precision-recall curve would indicate balanced precision-recall trade-off, making it suitable when capturing more positive cases is prioritized over precision.

<h>KNeighborsClassifier model performance metrics<h>
![Image](/images/knn_metrics.png)

**Key Takeaways:**
| **Metric**       | **Value** | **Interpretation**                                                            |
|------------------|-----------|-------------------------------------------------------------------------------|
| Train Accuracy   | 0.90      | Model fits the training data well; low bias.                                  |
| Test Accuracy    | 0.92      | Good generalization to unseen data; high overall correctness.                 |
| Precision        | 0.60      | 60% of predicted positives are correct; relatively few false positives.       |
| Recall           | 0.34      | Captures only 34% of actual positives; misses many true cases.                |
| F1-Score         | 0.43      | Moderate balance of precision and recall; suggests room for improvement.      |
| AUC (ROC)        | 0.83      | Strong ability to distinguish between classes.                                |

**ROC & Precision-Recall Interpretation:**
* The ROC curve would show good performance but inferior to Logistic Regression, with steeper slope in the middle regions. The precision-recall curve would show high precision at low recall levels, indicating the model's conservative nature in positive predictions.

<h>Support Vectors Classifier model performance metrics<h>
![Image](/images/svm_metrics.png)

**Key Takeaways:**
| **Metric**       | **Value** | **Interpretation**                                                                   |
|------------------|-----------|--------------------------------------------------------------------------------------|
| Train Accuracy   | 0.90      | Model fits training data well; no obvious underfitting.                              |
| Test Accuracy    | 0.90      | Good overall performance on unseen data.                                             |
| Precision        | 0.66      | When the model predicts positive, it’s correct 66% of the time; low false positives. |
| Recall           | 0.19      | Captures only 19% of actual positives; misses most true cases.                       |
| F1-Score         | 0.30      | Low balance between precision and recall; suggests poor handling of positives.       |
| AUC (ROC)        | 0.94      | Excellent ability to distinguish between classes overall.                            |

**ROC & Precision-Recall Interpretation:**
* The ROC curve would show excellent performance, nearly matching Logistic Regression with strong convexity. The precision-recall curve would show very high precision at extremely low recall levels, indicating the model's highly conservative approach to positive classification.

**<b>Key Insights and Recommendation</b>s**

* <b>Performance Trade-offs</b>

  * Logistic Regression offers the best balance of all metrics with superior AUC performance
  * Decision Tree provides the highest recall but suffers from overfitting and lower precision
  * SVM demonstrates excellent precision and AUC but severely limited recall capability
  * KNN shows consistent performance but moderate results across all metrics

* <b>Business Context Considerations</b>

  * For high-precision requirements (minimizing false positives): SVM or Logistic Regression
  * For high-recall requirements (capturing most positive cases): Decision Tree
  * For balanced performance: Logistic Regression provides optimal trade-off
  * For interpretability needs: Decision Tree offers clear decision pathways

* <b>ROC vs Precision-Recall Curve Analysis</b>
  * Given the apparent class imbalance (high baseline accuracy), precision-recall curves would be more informative than ROC curves. Logistic Regression and SVM show superior performance in both contexts, while Decision Tree demonstrates better recall characteristics despite lower precision.

## Improving the Model

## Model Performance Summary (with SMOTE and GridSearchCV)

| Model              | Train Time (s) | Train Acc | Test Acc | Precision | Recall | F1 Score | AUC    | Best Score  | Best Params Summary                                |
|--------------------|----------------|-----------|----------|-----------|--------|----------|--------|-------------|----------------------------------------------------|
| LogisticRegression | 11.32          | 0.86      | 0.87     | 0.46      | 0.91   | 0.61     | 0.94   | 0.59        | C=10, solver=liblinear                             |
| DecisionTree       | 39.41          | 0.87      | 0.86     | 0.45      | 0.87   | 0.59     | 0.93   | 0.59        | max_depth=5, criterion=gini                        |
| KNN                | 334.16         | 1.00      | 0.87     | 0.44      | 0.70   | 0.54     | 0.88   | 0.51        | n_neighbors=19, metric=manhattan, weights=distance |
| SVM                | 4772.34        | 0.86      | 0.86     | 0.43      | 0.92   | 0.59     | 0.94   | 0.58        | C=5, kernel=linear, class_weight=balanced          |

<h>Improved LogisticRegression model performance metrics<h>
![Image](/images/lr_grid_metrics.png)

**Key Takeaways:**
* Combining SMOTE with hyperparameter tuning led to a dramatic improvement in recall and a modest gain in F1-score, which is crucial for imbalanced classification problems.
* While precision and accuracy decreased, the model became much more sensitive to minority classes, which is often preferred when false negatives are more costly.
* The AUC remained high, confirming the model still separates classes well even after rebalancing and tuning.

| **Metric**     | **Before (Original)** | **After (SMOTE + GridSearchCV)** | **Change**   | **Interpretation**                                                          |
|----------------|-----------------------|----------------------------------|--------------|-----------------------------------------------------------------------------|
| Train Accuracy | 0.92                  | 0.87                             | ↓ -0.05      | Slight drop due to better generalization and class balance.                 |
| Test Accuracy  | 0.91                  | 0.86                             | ↓ -0.05      | Minor decrease; reflects reduced bias toward majority class.                |
| Precision      | 0.71                  | 0.46                             | ↓ -0.25      | Lower precision; more false positives due to aggressive positive prediction.|
| Recall         | 0.43                  | 0.91                             | ↑ +0.48      | Major improvement; model now detects most true positives.                   |
| F1-Score       | 0.54                  | 0.61                             | ↑ +0.07      | Better balance between precision and recall.                                |
| AUC (ROC)      | 0.94                  | 0.94                             | — No change  | Excellent class discrimination remains intact.                              |

**Optimization Impact:**
* The addition of SMOTE, StandardScaler and L2 regularization (C=0.1) with liblinear solver maintained the model's strong performance while potentially improving generalization. The minimal changes suggest the original model was already near-optimal.

**ROC & Precision-Recall Curve Interpretation:**
* The ROC curve remains nearly identical with excellent convexity, while the precision-recall curve shows consistent high-precision performance. The stability indicates robust model architecture that benefits from proper scaling without significant metric shifts.

<h>Improved DecisionTreeClassifier model performance metrics<h>
![Image](/images/dt_grid_metrics.png)

**Key Takeaways:**
* The original model showed signs of overfitting, with perfect test accuracy and lower train accuracy.
* After SMOTE and tuning, the model is better balanced and much more effective at identifying minority class instances (as seen in the large recall and AUC gains).
* The modest drop in accuracy and precision is a worthwhile trade-off for more reliable and fair classification.

| **Metric**     | **Before (Original)** | **After (SMOTE + GridSearchCV)** | **Change** | **Interpretation**                                                             |
|----------------|-----------------------|----------------------------------|------------|--------------------------------------------------------------------------------|
| Train Accuracy | 0.89                  | 0.86                             | ↓ -0.03    | Slight drop; indicates reduced overfitting and better generalization.          |
| Test Accuracy  | 1.00                  | 0.87                             | ↓ -0.13    | Significant drop; more realistic performance after addressing class imbalance. |
| Precision      | 0.53                  | 0.45                             | ↓ -0.08    | Slightly more false positives; acceptable trade-off for higher recall.         |
| Recall         | 0.54                  | 0.87                             | ↑ +0.33    | Major gain; model now captures most true positives.                            |
| F1-Score       | 0.54                  | 0.59                             | ↑ +0.05    | Improved balance between precision and recall.                                 |
| AUC (ROC)      | 0.74                  | 0.93                             | ↑ +0.19    | Huge improvement in class separability and overall classifier quality.         |

**Optimization Impact:**
* The implementation of max_depth=5 and min_samples_leaf=4 successfully addressed overfitting while dramatically improving precision and AUC. This represents the most significant improvement among all models.

**ROC & Precision-Recall Curve Interpretation:**
* The ROC curve transformation would be dramatic, shifting from moderate performance to near-excellent with strong convexity. The precision-recall curve would show substantial improvement in precision maintenance across recall levels, indicating better decision boundary definition.

<h>Improved KNeighborsClassifier model performance metrics<h>
![Image](/images/knn_grid_metrics.png)

**Key Takeaways:**
* Recall and F1-score improved substantially, making the model more effective in detecting minority class instances.
* Precision dropped, which is expected when recall increases, but the overall balance (F1-score) improved.
* AUC improvement confirms the model has become better at distinguishing classes.
* Test accuracy of 1.00 is suspicious and may require further validation (e.g., cross-validation or rechecking SMOTE data leakage).

#### K-Nearest Neighbors (KNN)

| **Metric**     | **Before (Original)** | **After (SMOTE + GridSearchCV)** | **Change** | **Interpretation**                                                                |
|----------------|-----------------------|----------------------------------|------------|-----------------------------------------------------------------------------------|
| Train Accuracy | 0.90                  | 0.87                             | ↓ -0.03    | Slight drop; suggests better generalization and less overfitting.                 |
| Test Accuracy  | 0.92                  | 1.00                             | ↑ +0.08    | Unusually high; could indicate optimistic performance or overlap with SMOTE data. |
| Precision      | 0.60                  | 0.44                             | ↓ -0.16    | More false positives; expected with increased recall focus.                       |
| Recall         | 0.34                  | 0.70                             | ↑ +0.36    | Major improvement in capturing true positives.                                    |
| F1-Score       | 0.43                  | 0.54                             | ↑ +0.11    | Much better balance of precision and recall.                                      |
| AUC (ROC)      | 0.83                  | 0.88                             | ↑ +0.05    | Improved class discrimination and overall robustness.                             |

**Optimization Impact:**
* The use of n_neighbors=19 with distance weighting and euclidean metric improved precision and AUC but at the cost of recall. The perfect training accuracy suggests the model may be memorizing training data despite the larger neighborhood size.

**ROC & Precision-Recall Curve Interpretation:**
* The ROC curve would show improved performance with better true positive rates at lower false positive rates. However, the precision-recall curve would indicate a trade-off where high precision comes at the expense of recall, making the model more conservative.

<h>Improved Support Vectors Classifier model performance metrics<h>
![Image](/images/svm_grid_metrics.png)

**Key Takeaways:**
The updated model represents a dramatic shift in strategy:
* The tuned SVC shows dramatic improvement in recall (from 0.19 to 0.92), making it highly effective at identifying minority class instances.
* Although precision and accuracy decreased, the F1-score nearly doubled, indicating a much more balanced and practical model.
* The AUC remained strong, showing the model still separates classes well despite the shift in classification behavior.

#### Support Vector Machine (SVM)

| **Metric**     | **Before (Original)** | **After (SMOTE + GridSearchCV)** | **Change**  | **Interpretation**                                                           |
|----------------|-----------------------|----------------------------------|-------------|------------------------------------------------------------------------------|
| Train Accuracy | 0.90                  | 0.86                             | ↓ -0.04     | Slight drop; improved generalization and reduced bias toward majority class. |
| Test Accuracy  | 0.90                  | 0.86                             | ↓ -0.04     | Slight reduction; expected when improving minority class performance.        |
| Precision      | 0.66                  | 0.43                             | ↓ -0.23     | More false positives; typical trade-off for higher recall.                   |
| Recall         | 0.19                  | 0.92                             | ↑ +0.73     | Massive improvement in detecting actual positives.                           |
| F1-Score       | 0.30                  | 0.59                             | ↑ +0.29     | Much better balance between precision and recall.                            |
| AUC (ROC)      | 0.94                  | 0.94                             | — No change | Excellent class separation maintained.                                       |

**Optimization Impact:**
* The implementation of class_weight='balanced' with linear kernel fundamentally changed the model's behavior from extremely conservative to highly sensitive. This addresses class imbalance but creates a precision-recall trade-off.

**ROC & Precision-Recall Curve Interpretation:**
* The ROC curve would show improved sensitivity with some increase in false positive rate. The precision-recall curve would demonstrate a fundamental shift from high-precision/low-recall to low-precision/high-recall, making it suitable for scenarios where missing positive cases is more costly than false alarms.

**Comparative Improvement Summary**
  * Biggest Winners

    * Decision Tree: Most comprehensive improvement across all metrics except recall
    * SVM: Dramatic recall improvement (73.6 percentage points) with strategic trade-offs
    * KNN: Solid improvements in precision and AUC with acceptable recall trade-off

  * ROC vs Precision-Recall Context
    * Given the class imbalance evident in the dataset, the precision-recall improvements are particularly significant. Decision Tree and KNN show better precision-recall balance, while SVM demonstrates the impact of addressing class imbalance directly.

## Next steps and Recommendations

</b>Based on analysis and model metrcis, we learned that imbalanced dataset which is heavily weighted towards the unsuccessful marketing campaigns could not be used effectively to determine features which could provide best model performance. So, it raises below questions:

![Image](/images/contact_barplot_for_acceptance.png)

* Was the marketing campaign not executed effectively to have a balanced dataset?
* There was a high score amongst the "Yes" for customers contacted via Cellular. So did Bank adopted other mode of customer reachout like text messages or Whatsapp messages?
* We observed high score for customer with longer duration of contact. Did bank employed sufficient resources to improve the changes of succesful outcome?
</b>

**Model Selection Based on Improvement**

* For Balanced Performance: Decision Tree shows the most comprehensive improvement
* For High Recall Scenarios: Optimized SVM dramatically improves positive case detection
* For Consistent Reliability: Logistic Regression maintains excellent, stable performance
* For Precision-Focused Tasks: KNN improvements make it viable for low false-positive requirements