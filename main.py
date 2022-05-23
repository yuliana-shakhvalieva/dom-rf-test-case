import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import binom

# File reading
df_main = pd.read_excel("default of credit card clients.xls", header=1)

# Separation of the sample into test and training
factors = ['LIMIT_BAL', 'EDUCATION', 'SEX', 'AGE', 'PAY_0', 'PAY_AMT1']
X_train, X_test, y_train, y_test = train_test_split(df_main[factors], df_main['default payment next month'],
                                                    test_size=0.3,
                                                    random_state=42,
                                                    stratify=df_main['default payment next month'])

# Logistic regression constructing
logistic_model = sm.Logit(y_train, X_train).fit()
print(f'Результаты логистической регрессии: \n{logistic_model.summary2()}')


# GINI
def calculate_gini(X, y):
    y_predict = logistic_model.predict(X)
    roc_auc_score_train = roc_auc_score(y, y_predict)
    return 2 * roc_auc_score_train - 1


gini_train = calculate_gini(X_train, y_train)
gini_test = calculate_gini(X_test, y_test)
print(f'Абсолютное значение предсказательной способности модели  \n'
      f'Метрика GINI на тренировочной выборке: {gini_train} \n'
      f'Метрика GINI на тестовой выборке: {gini_test} \n')

# Relative change of GINI
rel_change_gini = (gini_test - gini_train) / gini_train
print(f'Относительное изменение метрики GINI на тренировочной/тестовой выборках: {rel_change_gini}\n')


# Bootstrap for GINI
def confidence_interval(X, y, alpha, tests_number):
    gini_bootstrap = []
    origin_sample = X.join(y)
    for i in range(tests_number):
        sample = origin_sample.sample(n=len(origin_sample), replace=True)
        y_predict = logistic_model.predict(sample[factors])
        logistic_roc_auc_score = roc_auc_score(sample['default payment next month'], y_predict)
        gini_bootstrap.append(2 * logistic_roc_auc_score - 1)
    percentile = alpha / 2
    conf_interval = (np.percentile(gini_bootstrap, percentile), np.percentile(gini_bootstrap, 100 - percentile))
    return conf_interval


alpha = 0.05
tests_number = 1000
print(f'95% доверительный интервал для метрики GINI \n'
      f'на тренировочной выборке: {confidence_interval(X_train, y_train, alpha, tests_number)}\n'
      f'на тестовой выборке: {confidence_interval(X_test, y_test, alpha, tests_number)}\n')

# Matrix of paired correlations of model factors
corr_matrix = X_train.corr()
print(f'Матрица парных корреляций факторов модели: \n{corr_matrix}\n')
sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu")
# thread-blocking call. close pyplot window to continue program flow
plt.show()

# VIF
VIF_table = pd.DataFrame([variance_inflation_factor(X_train, i) for i in range(X_train.shape[1])],
                         index=factors,
                         columns=['VIF'])
print(f'Фактор инфляции дисперсии: \n{VIF_table}\n')


# PSI
def resolve_interval(intervals, value, accuracy=0):
    selector = {intervals[i] <= value < intervals[i + 1]:
                (round(intervals[i], accuracy), round(intervals[i + 1], accuracy)) for i in range(len(intervals) - 1)}
    return selector.get(True, 0)


quantitative_factors = ['LIMIT_BAL', 'AGE', 'PAY_AMT1']
dfs = [X_train, X_test]
for quantitative_factor in quantitative_factors:
    intervals = [min(X_train[quantitative_factor].min(), X_test[quantitative_factor].min())] + \
                [np.percentile(X_train[quantitative_factor], percent) for percent in range(20, 100, 20)] + \
                [max(X_train[quantitative_factor].max(), X_test[quantitative_factor].max()) + 1]
    for df in dfs:
        df[quantitative_factor + '_GROUP'] = [resolve_interval(intervals, value) for value in df[quantitative_factor]]


def calculate_psi(factor_group, factor):
    psi_train = X_train.groupby(factor_group).agg({factor_group: 'count'}) \
        .rename(columns={factor_group: factor + '_train'})
    psi_test = X_test.groupby(factor_group).agg({factor_group: 'count'}) \
        .rename(columns={factor_group: factor + '_test'})
    psi_df = psi_train.join(psi_test).fillna(0.0)
    column_train = psi_df[psi_df.columns[0]]
    column_test = psi_df[psi_df.columns[1]]
    psi_df['PSI'] = ((column_train / column_train.sum() - column_test / column_test.sum()) *
                     np.log((column_train / column_train.sum()) / (column_test / column_test.sum())))
    return psi_df['PSI'].sum()


factors_group = ['LIMIT_BAL_GROUP', 'EDUCATION', 'SEX', 'AGE_GROUP', 'PAY_0', 'PAY_AMT1_GROUP']
psi_list = [calculate_psi(factor_group, factor) for factor_group, factor in zip(factors_group, factors)]

PSI_table = pd.DataFrame([psi for psi in psi_list],
                         index=factors,
                         columns=['PSI'])
print(f'Тест PSI на уровне факторов модели (сравнение тренировочной и тестовой выборок): \n{PSI_table}\n')


# Binomial test
def binomial_test(PD, obs, default):
    return binom.ppf(0.95, obs, PD) >= default


df_main['PD'] = logistic_model.predict(df_main[factors])
intervals_for_pd = [np.percentile(df_main['PD'], percent) for percent in range(0, 100, 10)] + [1]

df_main['PD_GROUP'] = [resolve_interval(intervals_for_pd, value, 4) for value in df_main['PD']]

binomial_df = df_main.groupby(df_main['PD_GROUP']).agg({'PD': 'mean',
                                                        'ID': 'count',
                                                        'default payment next month': 'sum'})\
                                                  .rename(columns={'PD': 'Average PD',
                                                                   'ID': 'Observation',
                                                                   'default payment next month': 'Default fact'})

binomial_table = pd.DataFrame([binomial_test(row['Average PD'], row['Observation'], row['Default fact']) for i, row in binomial_df.iterrows()],
                              index=binomial_df.index.array,
                              columns=['Test passed'])

print(f'Биномиальный тест: \n{binomial_table}\n')
