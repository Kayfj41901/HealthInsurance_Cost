#### Personal Project By: Kayla Brock | Codeup | Jemison Cohort | June 28, 2022

### "_I want to drive down health care costs_" -Chris Gibson

#### A project using clustering and linear regression models to predict the annual health expenditure of health insurance customers

<hr style="border-top: 1px groove LightCyan ; margin-top: 1px; margin-bottom: 1px"></hr>

# I. Project Overview & EXECUTIVE SUMMARY

#### _Project Goal_ 

The goal of this project is to create a machine learning model that accurately estimates the annual medical expenditure of health insurance customers. It is believed that, with better predictive capabilities, the insurance company will offer new plans that will both retain current customers as well as attract new customers.

#### _Description_

This project aims to create a model that can accurately predict the annual medical expenditure of a customer based on a variety of factors. I chose this project because I am interested and deeply invested in this topic. At one point in my teaching career, because of exorbitant health insurance premiums, I had to leave my classroom job to pursue a teaching job online. It is my hope through this analysis I can make sense of why health insurance premiums are so costly. 

#### _Initial Thoughts & Hypotheses_

I believe older customers will have a significantly higher annual medical expenditure than younger customers. I believe females (in childbearing years) will have a significantly higher annual medical expenditure than males. I believe customers with a bmi heavier than a 'healthy range' will have a higher annual medical expenditure than customers within a healthy bmi range. I believe per child annual medical expenditure will be significantly less than per adult annual medical expenditure. I believe smoker annual medical expenditure will be significantly higher than non smoker annual medical expenditure. I believe, all things being equal, the annual medical expenditure should be equal from region to region.

#### _Initial Questions_

- Do older customers have higher annual medical expenditures than younger customers?
- Do females (in childbearing years) have a significantly higher annual medical expenditure than males? 
- Do customers with a bmi heavier than a 'healthy range'  have a higher annual medical expenditure than customers within a healthy bmi range? 
- Is the annual medical expenditure of smokers significantly higher than non smokers?
- All things being equal, Is the annual medical expenditure equal from region to region? 

#### _Key Findings_

While the models did not perform as well as I had hoped, there is still valuable insight to be gained from this report. First, the reason the models performed poorly is due to the inclusion of outliers. Although it is common knowledge outliers negatively impact any regression model, for the purpose of business analysis, it was critical to include them.

The question moving forward should be, 'how should we handle the outliers?' From my perspective, I see two clear options. Option 1, offer 'cluster rate' health insurance premiums that have a maximum payout of the largest non-outlier maximum value. Keeping in mind this number will not be adequate for some customers, offer a different package (at a higher rate) for those that 'elect' to choose a different type of coverage. The second option is to do further research on the outliers to determine what _legal qualifying factor_ could place them in a separate category. Because preexisting conditions are protected by law, the research team needs to focus on any characteristic that the outlier group has in common that is not covered by law. (i.e. smokers vs non-smokers). 

This report should not be seen as the final word but rather the first step in making more informed decisions moving forward. I believe strongly, whether the board chooses to offer cluster rate insurance premiums as they are or proceed in further research, there will be many opportunities for data analysis in the future. I am confident small changes can make big differences in health insurance affordability as well as company profitability.

#### _Deliverables_

* README file - overview of project as well as steps to reproduce 
* imports.py - contains code for imports
* prepare.py - contains code to prepare and split the data 
* visuals.py - contains code for visuals 
* Final Report Jupyter Notebook - contains final presentation

# II. Project Data

<hr style="border-top: 1px groove LightCyan ; margin-top: 1px; margin-bottom: 1px"></hr>

#### _Data Dictionary_

The final DataFrame used to explore the data for this project contains the following variables (columns). The variables, along with their data types, are defined below: 


```python
from collections import OrderedDict
import pandas as pd
features = OrderedDict([ ('feature', ['age', 'sex', 'bmi', 'children', 'smoker',
       'region', 'charges' ]), ('description', ['age of customer', 'customer gender', 'customer bmi', 'number of children',
                                               'smoker status: yes or no', 'region customer resides', 'total annual medical charges'
                                                ])])                           

df = pd.DataFrame.from_dict(features)
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>age</td>
      <td>age of customer</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sex</td>
      <td>customer gender</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bmi</td>
      <td>customer bmi</td>
    </tr>
    <tr>
      <th>3</th>
      <td>children</td>
      <td>number of children</td>
    </tr>
    <tr>
      <th>4</th>
      <td>smoker</td>
      <td>smoker status: yes or no</td>
    </tr>
    <tr>
      <th>5</th>
      <td>region</td>
      <td>region customer resides</td>
    </tr>
    <tr>
      <th>6</th>
      <td>charges</td>
      <td>total annual medical charges</td>
    </tr>
  </tbody>
</table>
</div>



# III. Project PLAN 

<hr style="border-top: 1px groove LightCyan ; margin-top: 1px; margin-bottom: 1px"></hr>

#### The following outlines the process taken through the data science pipeline to complete this project

#### _Plan_

In the planning stage I: read the project expectations, create a project outline, write a project goal to include how I would measure success or failure, review the overview of the dataset, document all initial thoughts, questions, and hypotheses, create a plan for completing the project, create a data dictionary to define features, and create a local folder and github repository.

#### _Acquire_

In the acquire stage I: create a .gitignore, obtain health insurance data from Kaggle, and save the data as a csv in my local folder. 

#### _Prepare_

In the Prepare stage I: review the dataset to see if there are any missing values(no null values), split the data into train, validate, test, and create a prepare.py file with functions to prepare data.  

#### _Explore_

In the Explore stage I: identified relationships between the target variable and features through univariate and bivariate exploration, performed six statistical tests (4 T-test and 1 ANOVA test) to determine the significance of the relationship between the target variable and the features. All tests are supported by visuals and takeaways. 

#### _Model AND Evaluate_

In the model and evaluate stage I: established baseline accuracy, trained and fit multiple models with varying algorithms and hyperparameters, compared evaluation metrics across models, evaluated best performing models using validate set, test final model on out-of-sample testing dataset, and summarized performance

#### _Deliver_

In the final stage I: prepared final notebook in Jupyter Notebook. I: wrote out my project description, introduction to include goals, created an executive summary which included all my key findings and recommendations, created headers and dividers to organize the flow of the notebook, asked and answered all questions, added summaries and supplementary markdown to guide the reader through the notebook.

# IV. Supplementary Files 

<hr style="border-top: 1px groove LightCyan ; margin-top: 1px; margin-bottom: 1px"></hr>

- prepare.py - provides code to prepare, split, and encode the data 
- visuals.py - provides code for all visuals found in the notebook 
- imports.py - provides imports 

# V. Steps to Reproduce

<hr style="border-top: 1px groove LightCyan ; margin-top: 1px; margin-bottom: 1px"></hr>

- Create a Kaggle profile, find and download 'Health Insurance' dataset
- Clone this repo (including prepare.py, imports.py, and visuals.py)
- Run Final Report Jupyter notebook to view the final product


```python

```
