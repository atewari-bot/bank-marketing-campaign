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

**Performance Overview:**
* Accuracy: High baseline accuracy (80.4%) suggests class imbalance in the dataset
* Precision: Very low (12.2%) indicating poor positive class identification capability
* Recall: Extremely low (12.0%) showing inability to capture true positive cases
* AUC: Near random performance (0.505) confirming no discriminative ability

**ROC & Precision-Recall Interpretation:**
* The ROC curve would be nearly diagonal, indicating random guessing behavior. The precision-recall curve would show consistently low precision across all recall levels, reflecting the model's inability to distinguish between classes effectively.

## Model Comparisons

| Model              | Train Time | Train Accuracy | Test Accuracy | Precision Score  | Recall Score | F1 Score | AUC      |
|--------------------|------------|----------------|---------------|------------------|--------------|----------|----------|
| DummyClassifier    | 0.110381   | 0.800243       | 0.803714      | 0.121844         | 0.119612     | 0.120718 | 0.505086 |
| LogisticRegression | 0.473958   | 0.910106       | 0.915999      | 0.708481         | 0.432112     | 0.536814 | 0.942592 |
| DecisionTree       | 0.254311   | 1.000000       | 0.894999      | 0.533475         | 0.540948     | 0.537186 | 0.740447 |
| KNN                | 0.110113   | 0.921608       | 0.899612      | 0.595825         | 0.338362     | 0.431615 | 0.833783 |
| SVM                | 65.218307  | 0.897329       | 0.897791      | 0.660448         | 0.190733     | 0.295987 | 0.935064 |

<h>LogisticRegression model performance metrics<h>
![Image](/images/lr_metrics.png)

**Key Takeaways:**
| **Metric**   | **Value** | **Interpretation**                                       |
|--------------|-----------|----------------------------------------------------------|
| Accuracy     | 0.92      | High — most predictions are correct overall              |
| Precision    | 0.71      | High — when it says “positive,” it’s usually right       |
| Recall       | 0.43      | Moderate — misses more than half of positives            |

**Performance Overview:**
* Accuracy: Excellent performance (91.6%) with good generalization from training to test
* Precision: Strong precision (70.8%) indicating reliable positive predictions with moderate false positive rate
* Recall: Moderate recall (43.2%) suggesting the model misses a significant portion of positive cases
* AUC: Outstanding discrimination ability (0.943) showing excellent class separation

**ROC & Precision-Recall Interpretation:**
* The ROC curve would show strong convexity toward the top-left corner, indicating excellent true positive rate with low false positive rate. The precision-recall curve would demonstrate high precision maintained across moderate recall levels, typical of conservative classification behavior.

<h>DecisionTreeClassifier model performance metrics<h>
![Image](/images/dt_metrics.png)

**Key Takeaways:**
| **Metric**   | **Value** | **Interpretation**                                         |
|--------------|-----------|------------------------------------------------------------|
| Accuracy     | 0.89      | High — most predictions are correct overall                |
| Precision    | 0.53      | Moderate — half of predicted positives are correct         |
| Recall       | 0.54      | Moderate — detects just over half of actual positives      |

**Performance Overview:**
* Accuracy: Perfect training accuracy (100%) indicates overfitting, with test accuracy dropping to 89.5%
* Precision: Moderate precision (53.3%) showing higher false positive rate compared to Logistic Regression
* Recall: Highest recall (54.1%) among all models, effectively capturing more positive cases
* AUC: Moderate discrimination (0.740) suggesting reasonable but not optimal class separation

**ROC & Precision-Recall Interpretation:**
* The ROC curve would show moderate performance with some trade-off between sensitivity and specificity. The precision-recall curve would indicate balanced precision-recall trade-off, making it suitable when capturing more positive cases is prioritized over precision.

<h>KNeighborsClassifier model performance metrics<h>
![Image](/images/knn_metrics.png)

**Key Takeaways:**
| **Metric**   | **Value** | **Interpretation**                                              |
|--------------|-----------|-----------------------------------------------------------------|
| Accuracy     | 0.90      | High — model predicts most outcomes correctly overall           |
| Precision    | 0.60      | Moderate to high — most predicted positives are correct         |
| Recall       | 0.34      | Low — misses most of the actual positive cases                  |

**Performance Overview:**
* Accuracy: High accuracy (90.0%) with slight overfitting tendency (training: 92.2%)
* Precision: Good precision (59.6%) showing reasonable positive prediction reliability
* Recall: Low recall (33.8%) indicating conservative classification with many missed positive cases
* AUC: Good discrimination ability (0.834) demonstrating solid class separation capability

**ROC & Precision-Recall Interpretation:**
* The ROC curve would show good performance but inferior to Logistic Regression, with steeper slope in the middle regions. The precision-recall curve would show high precision at low recall levels, indicating the model's conservative nature in positive predictions.

<h>Support Vectors Classifier model performance metrics<h>
![Image](/images/svm_metrics.png)

**Key Takeaways:**
| **Metric**   | **Value** | **Interpretation**                                               |
|--------------|-----------|------------------------------------------------------------------|
| Accuracy     | 0.90      | High — model correctly predicts most outcomes overall            |
| Precision    | 0.66      | High — most predicted positives are correct                      |
| Recall       | 0.19      | Very low — misses the vast majority of actual positive cases     |

**Performance Overview:**
* Accuracy: High accuracy (89.8%) with excellent training-test consistency indicating good generalization
* Precision: Strong precision (66.0%) showing reliable positive predictions
* Recall: Lowest recall (19.1%) indicating extremely conservative classification behavior
* AUC: Excellent discrimination (0.935) demonstrating superior class boundary definition

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

| Model              | Train Time | Train Accuracy | Test Accuracy | Precision Score | Recall Score | F1 Score | AUC      | Best Score | Best Params Summary |
|--------------------|------------|----------------|---------------|------------------|--------------|----------|----------|------------|----------------------|
| LogisticRegression | 7.34       | 0.910          | 0.915         | 0.705            | 0.428        | 0.533    | 0.942    | 0.909      | C=0.1, solver='liblinear', scaled |
| DecisionTree       | 30.62      | 0.916          | 0.918         | 0.677            | 0.527        | 0.593    | 0.933    | 0.912      | max_depth=5, min_samples_leaf=4, scaled |
| KNN                | 291.98     | 1.000          | 0.906         | 0.683            | 0.302        | 0.419    | 0.890    | 0.899      | n_neighbors=19, metric='euclidean', weights='distance', scaled |
| SVM                | 677.49     | 0.847          | 0.849         | 0.422            | 0.927        | 0.580    | 0.943    | 0.847      | C=1, kernel='linear', class_weight='balanced', scaled |

<h>Improved LogisticRegression model performance metrics<h>
![Image](/images/lr_grid_metrics.png)

**Key Takeaways:**
* No real improvement observed in model performance after performing cross-validation using GridSearchCV.

| **Metric**      | **Previous** | **Improved** | **Change**     | **Interpretation of Improvement**                      |
|-----------------|--------------|--------------|----------------|--------------------------------------------------------|
| Test Accuracy   | 0.916        | 0.915        | ≈ No Change    | Performance remained stable with tuning.               |
| Precision Score | 0.708        | 0.705        | -0.003         | Slight dip; still performs well on positive prediction.|
| Recall Score    | 0.432        | 0.428        | -0.004         | Negligible decrease.                                   |
| AUC             | 0.943        | 0.942        | ≈ No Change    | No real change in ranking capability.                  |

**Performance Changes:**
* Accuracy: Minimal change (91.6% → 91.5%), showing the model was already well-tuned
* Precision: Slight decrease (70.8% → 70.5%), indicating minor trade-off for overall performance
* Recall: Marginal improvement (43.2% → 42.8%), demonstrating consistent positive case detection
* AUC: Maintained excellence (0.942 → 0.942), confirming robust discriminative ability

  **Optimization Impact:**
  * The addition of StandardScaler and L2 regularization (C=0.1) with liblinear solver maintained the model's strong performance while potentially improving generalization. The minimal changes suggest the original model was already near-optimal.
  
  **ROC & Precision-Recall Curve Interpretation:**
  * The ROC curve remains nearly identical with excellent convexity, while the precision-recall curve shows consistent high-precision performance. The stability indicates robust model architecture that benefits from proper scaling without significant metric shifts.

<h>Improved DecisionTreeClassifier model performance metrics<h>
![Image](/images/dt_grid_metrics.png)

**Key Takeaways:**
* The improved model shows a notable increase in precision, meaning it makes fewer false positive errors, which is valuable when the cost of false alarms is high. 
* The accuracy also improved slightly, indicating better overall correctness. 
* Recall remained roughly the same, so the model’s ability to identify actual positives hasn’t changed much. If improving recall is important, further tuning or alternative approaches may be needed.

| **Metric**          | **Previous** | **Improved** | **Change** | **Interpretation of Improvement**                           |
|---------------------|--------------|--------------|------------|-------------------------------------------------------------|
| Test Accuracy       | 0.895        | 0.918        | +0.023     | Notable improvement in overall accuracy.                    |
| Precision Score     | 0.533        | 0.677        | +0.144     | Large gain; better at predicting true positives.            |
| Recall Score        | 0.541        | 0.527        | -0.014     | Slight decrease, but still balanced.                        |
| AUC                 | 0.740        | 0.933        | +0.193     | Major boost in distinguishing positive vs. negative cases.  |

**Performance Changes:**
* Accuracy: Significant improvement (89.5% → 91.8%), with reduced overfitting (100% → 91.6% train accuracy)
* Precision: Substantial enhancement (53.3% → 67.7%), reducing false positive rate dramatically
* Recall: Slight decrease (54.1% → 52.7%), maintaining reasonable positive case capture
* AUC: Major improvement (0.740 → 0.933), showing vastly enhanced class discrimination

  **Optimization Impact:**
  * The implementation of max_depth=5 and min_samples_leaf=4 successfully addressed overfitting while dramatically improving precision and AUC. This represents the most significant improvement among all models.
  
  **ROC & Precision-Recall Curve Interpretation:**
  * The ROC curve transformation would be dramatic, shifting from moderate performance to near-excellent with strong convexity. The precision-recall curve would show substantial improvement in precision maintenance across recall levels, indicating better decision boundary definition.

<h>Improved KNeighborsClassifier model performance metrics<h>
![Image](/images/knn_grid_metrics.png)

**Key Takeaways:**
* The improved model shows a higher precision, meaning it makes fewer false positive errors and is more confident when predicting positives. 
* This comes at the cost of a small decrease in recall, so the model misses more actual positives than before. 
* The overall accuracy increased slightly, reflecting better general correctness. 
* This trade-off is common: increasing precision often reduces recall. Depending on your application, you may want to tune the model or threshold to better balance these metrics.

#### K-Nearest Neighbors (KNN)

| **Metric**          | **Previous** | **Improved** | **Change**     | **Interpretation of Improvement**                       |
|---------------------|--------------|--------------|----------------|---------------------------------------------------------|
| Test Accuracy       | 0.900        | 0.906        | +0.006         | Modest gain in accuracy.                                |
| Precision Score     | 0.596        | 0.683        | +0.087         | Significant improvement; more confident positive calls. |
| Recall Score        | 0.338        | 0.302        | -0.036         | Slight loss; missing a few more actual positives.       |
| AUC                 | 0.834        | 0.890        | +0.056         | Good gain in model's ranking ability.                   |

**Performance Changes:**
* Accuracy: Moderate improvement (90.0% → 90.6%), with perfect training accuracy indicating potential overfitting
* Precision: Significant enhancement (59.6% → 68.3%), improving positive prediction reliability
* Recall: Concerning decrease (33.8% → 30.2%), reducing positive case detection capability
* AUC: Substantial improvement (0.834 → 0.890), enhancing class separation ability

  **Optimization Impact:**
  * The use of n_neighbors=19 with distance weighting and euclidean metric improved precision and AUC but at the cost of recall. The perfect training accuracy suggests the model may be memorizing training data despite the larger neighborhood size.
  
  **ROC & Precision-Recall Curve Interpretation:**
  * The ROC curve would show improved performance with better true positive rates at lower false positive rates. However, the precision-recall curve would indicate a trade-off where high precision comes at the expense of recall, making the model more conservative.

<h>Improved Support Vectors Classifier model performance metrics<h>
![Image](/images/svm_grid_metrics.png)

**Key Takeaways:**
The updated model represents a dramatic shift in strategy:
* Recall surged from 0.19 to 0.92 — a major improvement in detecting nearly all positive cases.
* Precision dropped — the model now predicts more false positives.
* Accuracy decreased slightly, likely due to more false positives affecting the overall correct prediction rate.

#### Support Vector Machine (SVM)

| **Metric**          | **Previous** | **Improved** | **Change**     | **Interpretation of Improvement**                               |
|---------------------|--------------|--------------|----------------|-----------------------------------------------------------------|
| Test Accuracy       | 0.898        | 0.849        | -0.049         | Accuracy declined after tuning.                                 |
| Precision Score     | 0.660        | 0.422        | -0.238         | Large precision drop; more false positives.                     |
| Recall Score        | 0.191        | 0.927        | +0.736         | Huge recall boost; nearly all positives correctly identified.   |
| AUC                 | 0.935        | 0.943        | +0.008         | Slight AUC gain; more balanced decision threshold after tuning. |

**Performance Changes:**
* Accuracy: Significant decrease (89.8% → 84.9%), showing reduced overall performance
* Precision: Dramatic decline (66.0% → 42.2%), indicating increased false positive rate
* Recall: Massive improvement (19.1% → 92.7%), transforming from conservative to aggressive classification
* AUC: Marginal improvement (0.935 → 0.943), maintaining excellent discrimination despite accuracy drop

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