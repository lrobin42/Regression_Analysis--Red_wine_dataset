from random import randint
from scipy.stats import zscore
from skimpy import skim
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import OLS
from statsmodels.regression.linear_model import WLS
from statsmodels.stats.api import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import RFE
from statsmodels.stats.stattools import jarque_bera
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
import statsmodels.api as sm
import statsmodels.stats.api as sms
wine_df = pd.read_csv("winequality-red.csv")

# Change column names to use underscores instead of spaces
wine_df.columns = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
]

def duplicate_checker(df, subset=None):
    """Isolate out duplicate rows for df given

    Arguments:
        df {dataframe}: dataframe to check for duplicate rows
        subset {list}: list of strings corresponding dataframe columns

    Returns:
        duplicate_rows {dataframe}: dataframe of duplicated rows from input df
    """
    duplicate_rows = df.duplicated(subset=subset)
    duplicate_rows = df.loc[duplicate_rows.index]

    if duplicate_rows.empty:
        print(f"{df}: No duplicates found.")

    else:
        return duplicate_rows

def null_value_printer(df):
    """Prints either that df has no null values, or null values by column

        Arguments:
            df {dataframe}: Input dataframe to check
    Returns:
        output {integer}: count of null values within df
        prints {string}: "Zero null values" if no null values found
    """
    count = df.isnull().sum().sum()

    if count == 0:
        print(f"Zero null values")
    else:
        output = podcasts_df.isnull().sum()[podcasts_df.isnull().sum() > 0]
        return output

def dataset_splitter(X, Y, test_size=0.3, random_state=0):
    """Returns X_train, X_test, y_train, y_test for X and Y specified

    Returns:
        x_train, x_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=0.3,
        shuffle=True,
        random_state=2,
    )
    return X_train, X_test, y_train, y_test

def define_x_and_y(df,y_string, scale_X=False):
    """Function defines x matrix and y series for backward elimination.

    Args:
        y_string (string): Takes in y-variable as a string


    Returns:
        X (dataframe): df of all possible dependent variables
        Y (dataframe): df of independent variable
    """


    Y = df[[y_string]]
    X = df.drop(columns=[y_string])

    if scale_X==True:
        scaler = StandardScaler()
        model = scaler.fit(X)
        X = pd.DataFrame(model.transform(X), columns=X.columns) 
    return X, Y

def find_max_p_value(ols):
    """Function intakes ols model instance from statsmodels linear regression. From there it creates a dataframe with the features and p_values from the ols model, returning the max p_value and corresponding instance.

    Args:
        ols (statsmodels.regression.linear_model.RegressionResultsWrapper): OLS is instance of linear regression from statsmodels

    Returns:
        index (integer): index of the max p_value within the OLS summary
        p_value (float): the max p-value
    """

    p_value_series = ols.pvalues
    p_df = np.transpose(pd.DataFrame([p_value_series.index, p_value_series]))
    p_df.columns = ["features", "p_values"]
    index = p_df[p_df.p_values == p_df.p_values.max()].index
    index = index[0]
    p_value = p_df.p_values.max()

    return index, p_value

def run_OLS(x_train, y_train):
    """Runs statsmodels multi-linear regression for a specific x and y training set, returning the ols instance and printing out the summary.

    Args:
        x_opt (dataframe): features being used as dependent variables in this iteration of backward elimination
        y_train (dataframe): dependent variable dataframe

    Returns:
        ols: returns statsmodels ols instance 
    """

    y_train = y_train.values.reshape(-1, 1)
    ols = sm.OLS(endog=y_train, exog=x_train).fit()
    return ols

def calculate_expected_value(array):
    """Calculates the expected value of a given array

    Args:
        array (np.array): Array of values

    Returns:
        Expected value (float): Expected value of the array
    """
    # Set the probability that each residual will have
    probability = 1 / len(array)

    # Initialize E=0

    Expected_value = 0

    # Calculate E[r] for all residuals
    for i in range(0, len(array)):
        Expected_value += array.iloc[i] * probability
    Expected_value

    return Expected_value

def plot_residuals(resid):
    """Function plots residuals against observations for a given regression model

    Args:
        resid (_type_): _description_
    """
    sns.scatterplot(x=range(0,len(resid)), y=resid)
    plt.axhline(y=0, color='k', linestyle='--')
    plt.title("Residuals vs. observations")
    plt.xlabel("Observations")
    plt.ylabel("Residuals")
    plt.show()

def backward_elimination(y, significance_level=0.05):
    """Conducts backward elimination after taking in y variable as a string.

    Args:
        y (str): string denoting y variable within an overall dataset
        significance_level (float, optional): User can specify alpha value for statistical significance. Defaults to 0.05.

    Returns:
        regression: returns ols model as a statsmodel regression object
    """

    # Define x and y
    x_all_features, dependent_variable = define_x_and_y(y)

    # Now that our X matrix is defined, let's split x and y into training and test sets
    x_train, x_test, y_train, y_test = dataset_splitter(
        X=x_all_features, Y=dependent_variable
    )

    # Make sure that our regression contains a y-intercept
    x_train = sm.add_constant(x_train)

    # initialize x_columns as all features in X. We'll eliminate columns from here iteratively
    x_columns = list(range(0, np.shape(x_train)[1]))
    x_opt = x_train.iloc[:, x_columns].copy()

    # initialize max_p to arbitrarily high number
    max_p = 100

    while max_p > significance_level:

        # Run the OLS regression
        ols = run_OLS(x_opt, y_train)

        # Isolate max p value
        pmax_index, max_p = find_max_p_value(ols)

        # Drop the column corresponding to pmax_index

        del x_columns[int(pmax_index)]
        x_opt = x_train.iloc[:, x_columns]

        # rerun the multilinear regression
        ols = run_OLS(x_opt, y_train)
        if max_p < significance_level:
            break

    return ols

def scaled_backward_elimination(scaled_x,unscaled_y):
    """scaled_x, unscaled_y are dataframes"""
    """Generates scaled ols backward elimination model

    Returns:
        _type_: _description_
    """

    # Now that our X matrix is defined, let's split x and y into training and test sets
    x_train, x_test, y_train, y_test = dataset_splitter(
        X=scaled_x, Y=unscaled_y
    )

    # Make sure that our regression contains a y-intercept
    x_train = sm.add_constant(x_train)

    # initialize x_columns as all features in X. We'll eliminate columns from here iteratively
    x_columns = list(range(0, np.shape(x_train)[1]))
    x_opt = x_train.iloc[:, x_columns].copy()
    # initialize max_p to arbitrarily high number
    max_p = 100

    while max_p > significance_level:

        # Run the OLS regression
        ols = run_OLS(x_opt, y_train)

        # Isolate max p value
        pmax_index, max_p = find_max_p_value(ols)

        # Drop the column corresponding to pmax_index

        del x_columns[int(pmax_index)]
        x_opt = x_train.iloc[:, x_columns]

        # rerun the multilinear regression
        ols = run_OLS(x_opt, y_train)
        if max_p < significance_level:
            break

    return ols

def breusch_pagan_test(results, significance_level=0.05):
    """Conducts breusch pagan test for heteroscedasticity

    Args:
        results (statsmodel regression): statsmodel regression results
        significance_level (float, optional): significance level for hypothesis test. Defaults to 0.05.
    """
    name = ["Lagrange multiplier statistic", "p-value", "F-value", "F_test p-value"]
    test = het_breuschpagan(results.resid, results.model.exog)

    df = pd.DataFrame(data={"statistic": name, "value": test}).sort_values(
        "value", ascending=True
    )

    # Reject the null hypothesis if needed
    if test[1] < significance_level:
        verdict = "\nReject the null hypothesis, heteroscedastic errors."
    else:
        verdict = "Fail to reject the null hypothesis, homoscedastic errors."
    print(df, "\n\n",verdict,'\n')

def plot_cooks_distance(results, BE_instance):
    """plots Cook's Distance for given regression model

    Args:
        results (statsmodels): statsmodels OLS results
        BE_instance (back_elimination): custom back_elimination instance 
    """
    influence=results.get_influence()
    (cooks,_)=influence.cooks_distance

    #Define cook's distance rejection threshold
    cooks_threshold=4/(n-k-1)

    #stemplot for cooks distance
    plt.subplots(figsize=(18,4))
    plt.xlabel('Observations_Index')
    plt.ylabel('Cooks Distance')

    plt.stem(np.arange(len(BE_instance.x_train)),cooks)

    plt.axhline(y = cooks_threshold, color = 'k', linestyle = '-.') 
    plt.title("Cook's distance for training set observations");
    plt.show()

def create_outlier_dataframe(results):
    """Assembles dataframe comprising influence, Cook's D, and studentized residuals for a given OLS model

    Args:
        results (statsmodels): statsmodels OLS results instance

    Returns:
        outliers_df: dataframe of influence, Cook's D, and studentized residuals for inputted OLS model 
    """
    influence=results.get_influence()
    (cooks,_)=influence.cooks_distance

    residuals=results.resid

    #Let's also create a dataframe of outliers with their influence values
    outliers_df=pd.DataFrame(data={'residual': residuals, "cooks_distance":cooks})

    # add studentized residuals to our dataframe
    outliers_df["studentized_residuals"] = influence.resid_studentized

    outliers_df['influence']=influence.influence
    outliers_df['leverage']=influence.hat_matrix_diag

    return outliers_df

def plot_influence_studentized(df):
    """Plots studentized residuals against leverage for given outliers_df

    Args:
        df (dataframe): dataframe of outliers
    """
    plt.scatter(data=df, x='leverage',y='studentized_residuals');
    plt.xlabel('Leverage');
    plt.title('Influence and leverage of residuals');
    plt.ylabel('Studentized residuals');

def jarque_bera_test(residuals):
    """Intakes residuals to output JB statistics for a given regression model

    Args:
        residuals (pandas series): residuals of a statsmodel regression instance
    """
    # Jarque-Bera test for normality
    statistics = jarque_bera(residuals, axis=0)

    name = ["Jarque-Bera", "p-value", "skewness", "kurtosis"]
    print(pd.DataFrame(data={"statistics": name, "values": statistics}))

def calculate_VIF(BE):
    """Calculates variance inflation factor for backward elimination class instance

    Args:
        BE (back_elimination): backward elimination model instance

    Returns:
        df: dataframe of VIF for all variables in BE model
    """
    VIF=[]
    variables=[]
    for idx, element in enumerate(BE.x_train.columns):
        VIF.append(variance_inflation_factor(BE.x_train, idx))
        variables.append(element)
    df=pd.DataFrame(data={'variable':variables,'VIF':VIF}).sort_values('VIF',ascending=True)[:-1]
    #Drop row for constant
    df = df[df['variable'] != 'const']
    return df

def harvey_collier_test(results):
    """Harvey Collier test for given regression

    Args:
        results (statsmodels): OLS results
    """
    skip = len(results.params)
    rr = sms.recursive_olsresiduals(ols, skip=skip, alpha=0.95, order_by=None)

    #Conduct t-test on standardized recursive residuals
    tval,pval=stats.ttest_1samp(rr[3][skip:], 0)
    if pval < significance_level:
        print(
            f"p-value: {np.round(pval,3)}. \nReject the null hypothesis, model is improperly specified as a linear model"
        )
    else:
        print(
            f"p-value: {np.round(pval,3)}. \nFail to reject the null hypothesis, model is correctly specified as a linear model."
        )

def rainbow_test(results):
    """Conducts rainbow test to specify if model is properly specified as a linear model. 

    Args:
        results (statsmodels): OLS results
    """
    from statsmodels.stats.diagnostic import linear_rainbow

    fstat, pvalue = linear_rainbow(results)

    if pvalue < significance_level:
        print(
            f"p-value: {np.round(pvalue,3)}. \nReject the null hypothesis, model is improperly specified as a linear model"
        )
    else:
        print(
            f"p-value: {np.round(pvalue,3)}. \nFail to reject the null hypothesis, model is correctly specified as a linear model."
        )

def qq_plot_residuals(residuals_set):
    import pylab as py
    sm.qqplot(residuals, line="45")
    py.show()

def calculate_residuals(results):
    residuals = results.resid
    return residuals

def plot_influence(results,title):
    influence=results.get_influence()
    cooks = influence.cooks_distance[0]
    cooks_threshold = np.median(cooks) + np.std(cooks)
    # Plot threshold for high influence points
    plt.figure(figsize=(14,4))
    plt.subplot(1, 2, 1)
    plt.suptitle("Outliers by influence and leverage")
    plt.axhline(y=cooks_threshold, color="r", linestyle="-.")
    plt.scatter(range(0,len(cooks)), cooks,)
    plt.xlabel("Observations")
    plt.ylabel("Cooks Distance")
    plt.title("Influential Points by Cook's Distance")
    plt.suptitle(title)

    plt.subplot(1, 2, 2)
    plt.scatter(
        data=create_outlier_dataframe(results),
        x="leverage",
        y="studentized_residuals",
    )
    plt.xlabel("Leverage")
    plt.title("Influence and leverage of residuals")
    plt.ylabel("Studentized residuals")
    plt.axvline(x=2 * len(results.params.index[1:]) / int(results.nobs), color="r", linestyle="-.")
    plt.show()

def single_mean_test(population,mu,alpha=0.05):
    from scipy import stats
    t_stat, p_value = stats.ttest_1samp(a=population, popmean=mu)
    
    if p_value < alpha:
        print(
            f"p-value: {p_value}\nReject the null hypothesis. We have sufficient evidence that the average alcohol content is not {mu} percent by volume."
    )
    else:
        print(
        f"p-value: {p_value}\nFail to reject the null hypothesis. We have sufficient evidence that the null hypothesis is correct, and the average alcohol content is {mu} percent by volume."
    )

def compare_regressions(results_1, results_2):
    # print a dataframe of r-squared, F-probabilities, and F-stats of each regression

    stats_1 = [np.round(results_1.rsquared_adj,3), results_1.f_pvalue, results_1.fvalue]
    stats_2 = [np.round(results_2.rsquared_adj,3), results_2.f_pvalue, results_2.fvalue]
    d = {"current model": stats_1, "new model": stats_2}

    df = pd.DataFrame(data=d, index=["R-squared-adj", "F-probabilities", "F-statistic"])
    return df

def scale_dataframe(df):
    """Applies Standard Scaling to dataframe

    Args:
        df (dataframe): dataset

    Returns:
        df: scaled dataframe
    """
    scaler = StandardScaler()
    model = scaler.fit(df)
    df = pd.DataFrame(model.transform(df), columns=df.columns) 
    return df

class back_elimination:
    """Class creates backward elimination model. Functions within the class act to both wrap around statsmodels regression class and consolidate variables centrally for model validation and testing. 
    """
    def __init__(self):
        self.regression_counter = 0

    def backward_elimination(self, x, y, regression='ols',significance_level=0.05):
        """x and y are strings"""

        # Now that our X matrix is defined, let's split x and y into training and test sets
        x_train, self.x_test, self.y_train, self.y_test = dataset_splitter(X=x, Y=y)

        # Make sure that our regression contains a y-intercept
        x_train = sm.add_constant(x_train)

        # initialize x_columns as all features in X. We'll eliminate columns from here iteratively
        x_columns = list(range(0, np.shape(x_train)[1]))
        x_opt = x_train.iloc[:, x_columns].copy()

        # initialize max_p to arbitrarily high number
        max_p = 100

        while max_p > significance_level:

            # Run the OLS regression
            ols = run_OLS(x_opt, self.y_train)
            self.regression_counter += 1
            # Isolate max p value
            pmax_index, max_p = find_max_p_value(ols)

            # Drop the column corresponding to pmax_index

            del x_columns[int(pmax_index)]
            x_opt = x_train.iloc[:, x_columns]

            if regression=='ols':
                # rerun the multilinear regression
                ols = run_OLS(x_opt, self.y_train)
                self.regression_counter + 1
                if max_p < significance_level:
                    # Save the final x_train as a class attribute
                    self.x_train = x_train.iloc[:, x_columns]
                    self.ols = ols
                    break
            if regression=='wls':
                wls =  WLS(self.y_train, self.x_train, weights=1)
                test = wls_model.fit()
                test.summary()

                
                self.regression_counter + 1
                if max_p < significance_level:
                    # Save the final x_train as a class attribute
                    self.x_train = x_train.iloc[:, x_columns]
                    self.ols = ols
                    break


        # Now let's resize x_test to the match our final x_opt subset
        training_columns = self.x_train.columns[1:]

        # ensure the columns are the same, and in the same order
        self.x_test = self.x_test[training_columns]

        # add column of constants
        self.x_test = sm.add_constant(self.x_test)
        return self.ols

def RFE_performance(x_train, y_train, x_test, y_test, k):
    """Function computes accuracy, precision, recall, and f1_score for RFE with k-features specified

    Args:
        x_train (df): training x
        y_train (df): training y
        k (int): number of features desired in RFE model

    Returns:
        _type_: _description_
    """

    # Create a logistic regression model
    model = LogisticRegression()

    # Use RFE to select the top k features
    rfe = RFE(model, n_features_to_select=k)
    rfe.fit(x_train, y_train)

    y_pred = rfe.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)


    values = [accuracy, precision, recall, f1]

    return values

def RFE_performance_training(x_train, y_train, k):
    """Function computes accuracy, precision, recall, and f1_score for RFE with k-features specified

    Args:
        x_train (df): training x
        y_train (df): training y
        k (int): number of features desired in RFE model

    Returns:
        _type_: _description_
    """

    # Create a logistic regression model
    model = LogisticRegression()

    # Use RFE to select the top k features
    rfe = RFE(model, n_features_to_select=k)
    rfe.fit(x_train, y_train)

    y_pred = rfe.predict(x_train)

    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred)
    recall = recall_score(y_train, y_pred)
    f1 = f1_score(y_train, y_pred)


    values = [accuracy, precision, recall, f1]

    return values