# Assignment 17.1 - Bank Products Marketing Campaign

**Contents**

  * [Introduction](#Introduction)
  * [Business Understanding](#Business-Understanding)
  * [Data Understanding](#Data-Understanding)
  * [Data Preparation](#Data-Preparation)
  * [Baseline Model Comparison](#Baseline-Model-Comparison)
  * [Model Comparisons](#Model-Comparisons)
  * [Improving the Model](#Improving-the-Model)
  * [Findings](#Findings)
  * [Next steps and Recommendations](#Next-steps-and-Recommendations)



## Introduction

Goal is to predict if the client will subscribe (yes/no) to a bank product (long term deposit) based on tele marketing campaign.

[Link to Bank Marketing Campaign Dataset](https://github.com/atewari-bot/bank-marketing-campaign/blob/main/data/bank-additional-full.csv)

[Link to Jupyter Notebook](https://github.com/atewari-bot/bank-marketing-campaign/blob/main/bank_marketing_campaign.ipynb)

## Business Understanding

There are two main approaches for enterprises to promote products and/or services: through mass campaigns, targeting general indiscriminate public or directed marketing, targeting a specific set of contacts. Data from a Portuguese bank that used its own contact-center to do directed marketing campaigns.

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

![Image](/images/pie_chart_for_category_distribution.png)

**Key Takeaways:** 
* 52.38% of people with house loan accepted long term deposit.
* 82.43% of people with personal loan accepted long term deposit.
* May month of most successful month for long term deposit acceptance with 33.43% success.
* Thursday and Monday were most successful days of the week for deposit acceptance with 20.94% and 20.67% success rate.

![Image](/images/class_distribution.png)

**Key Takeaways:** 
* Distribution of campaign outcome is greatly imbalanced.
* ~36k have unsuccessful outcome for depsoit acceptance and only ~4.6k were postive outcomes.


![Image](/images/feature_correlation_with_deposit.png)

**Key Takeaways:** 
* Top highly correlated features with target (deposit) are duration, poutcome_success, contact_cellular and month of march, september & october.

![Image](/images/violin_chart_by_coef.png)

![Image](/images/heatmap_top20_coef.png)

## Baseline Model Comparison

## Model Comparisons

## Improving the Model

## Findings

## Next steps and Recommendations