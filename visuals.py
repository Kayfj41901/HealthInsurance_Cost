from imports import *
df = pd.read_csv('insurance.csv')
train, validate, test = prepare.split_insurance_data(df)

def distribution_of_charges():
    fig = px.histogram(train, 
                   x='charges', 
                   marginal='box', 
                   color_discrete_sequence=['orange'], 
                   title='Distribution of Charges')
    fig.update_layout(bargap=0.1)
    fig.show()

def distribution_of_age():
    fig = px.histogram(train, 
                   x='age', 
                   marginal='box', 
                   color_discrete_sequence=['dark blue'], 
                   title='Distribution of Age')
    fig.update_layout(bargap=0.1)
    fig.show()

def distribution_of_bmi():
    fig = px.histogram(train, 
                   x='bmi', 
                   marginal='box', 
                   color_discrete_sequence=['orange'], 
                   title='Distribution of BMI')
    fig.update_layout(bargap=0.1)
    fig.show()

def distribution_of_children():
    fig = px.histogram(train, 
                   x='children', 
                   marginal='box', 
                   color_discrete_sequence=['dark blue'], 
                   title='Distribution of Children')
    fig.update_layout(bargap=0.1)
    fig.show()

def KDE_older_vs_younger():
    plt.figure(figsize = (12, 7))
    #define variable for visualization
    train_older_customers = train[train['age'] > 39]
    train_younger_customers = train[train['age'] < 40]
    sns.kdeplot(train_older_customers['charges'], label = 'Customers Over 39', shade = True, color = 'r')
    sns.kdeplot(train_younger_customers['charges'], label = 'Customers Under 40', shade = True, color = 'b')

    plt.xlabel('Expenditure')


def KDE_male_vs_female_under40():
    plt.figure(figsize = (12, 7))
    #define variables for visualization
    train_older_customers = train[train['age'] > 39]
    train_younger_customers = train[train['age'] < 40]
    female_customers = train_younger_customers[train_younger_customers['sex'] == 'female']
    male_customers = train_younger_customers[train_younger_customers['sex'] == 'male']

    sns.kdeplot(female_customers['charges'], label = 'Female Customers Under 40', shade = True, color = 'r')
    sns.kdeplot(male_customers['charges'], label = 'Male Customers Under 40', shade = True, color = 'b')

    plt.xlabel('Expenditure')


def KDE_obese_vs_notobese():
    plt.figure(figsize = (12, 7))
    #create variable for visualization
    train_not_obese = df[df['bmi'] < 30]
    train_obese = df[df['bmi'] > 30.00001]
    sns.kdeplot(train_not_obese['charges'], label = 'Customers Not Obese', shade = True, color = 'r')
    sns.kdeplot(train_obese['charges'], label = 'Customers Obese', shade = True, color = 'b')

    plt.xlabel('Expenditure')


def KDE_smokers_vs_nonsmokers():
    plt.figure(figsize = (12, 7))
    #define smokers for visualization
    smokers = df[df['smoker'] == 'yes']
    non_smokers = df[df['smoker']== 'no']
    sns.kdeplot(smokers['charges'], label = 'Smokers', shade = True, color = 'r')
    sns.kdeplot(non_smokers['charges'], label = 'Non Smokers', shade = True, color = 'b')

    plt.xlabel('Expenditure')