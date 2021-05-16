# Payment's seer
## _Could an intelligent seer foretell the destiny of next month cash flow?_

In some countries' local real estate market, most developers have to handle a portfolio of seller financing contracts from the buyers' mortgages. In this sense, predictability over the payments and contract status over the following months are critical for their business to survive. The proposal was to implement an LSTM architecture over the Home Credit Kaggle dataset to tackle this issue. If successful, it would be possible to derive the probability of payment and the contract status at month t. If time allows, experiments will be conducted with the recent adaptions of the Transformer architecture to time-series problems.

A robust approach to this issue would be an implementation along the line of the 2016 paper [Deep Learning for Mortgage Risk](https://arxiv.org/abs/1607.02470). Although, for contexts where there is not a comprehensive and complete variety of different time-series, the understanding of how much the recurrent dynamics of payments pattern impact the cash flow predictability could prove valuable for business decision-making in the real estate market.

## Summary

### Question:
- Could the payments dates enable the prediction of a contract next month's status?

### Dataset:
- [Home Credit Kaggle dataset](https://www.kaggle.com/c/home-credit-default-risk/data)

### Data Preparation:
- For each contract in the dataset, input was built by generating N sequences with the days past due for each installment. The output/target variable was derived from classifying the following month's number of days past due. e.g: [0, 0, 0] --> ["Payment on schedule"]

### Proposed architecture:
- LSTM

### Results:

### Conclusion:

### Limitations and next steps:
- Experiment with Transformers architecture. Moreover, try to get more data from different sources beyond the own payment dynamics of the contract. Finally, if possible, compound the dataset with different variables just as in the aforementioned paper.