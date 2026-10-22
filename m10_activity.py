# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.24.2",
#     "matplotlib==3.11.2",
#     "pandas==3.0.5",
#     "scikit-learn>=1.5",
#     "seaborn==0.13.2",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    # GitHub repository: https://github.com/edjavi120/m10-activity
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
    from sklearn.model_selection import train_test_split

    return (
        LinearRegression,
        mean_absolute_error,
        mo,
        pd,
        plt,
        r2_score,
        root_mean_squared_error,
        sns,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Lung Function

    A respiratory clinic ran routine spirometry on 500 healthy adult men who have never smoked (a simulated dataset, so some patterns are stronger than they would be in real clinic data). Each row is one participant, with these columns:
    - standing height, in centimetres
    - age, in years
    - forced expiratory volume in one second (FEV1), in litres, a measure of lung function where a higher value means more air is exhaled quickly
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Lung Function Data

    **Instructions:** Load `lung_function.csv` into a DataFrame and display it.
    """)
    return


@app.cell
def _(pd):
    # Load the dataset
    df = pd.read_csv("lung_function.csv")
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Height vs. FEV1: Scatter Plot

    **Instructions:** Create a scatter plot of height and FEV1.
    """)
    return


@app.cell
def _(df, plt, sns):
    # Scatter plot of height against FEV1
    sns.scatterplot(df, x="height_cm", y="fev1_l")
    plt.title("Height vs. FEV1")
    plt.xlabel("Height (cm)")
    plt.ylabel("FEV1 (L)")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Age vs. FEV1: Scatter Plot

    **Instructions:** Create a scatter plot of age and FEV1.
    """)
    return


@app.cell
def _(df, plt, sns):
    # Scatter plot of age against FEV1
    sns.scatterplot(df, x="age_years", y="fev1_l")
    plt.title("Age vs. FEV1")
    plt.xlabel("Age (years)")
    plt.ylabel("FEV1 (L)")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Height vs. FEV1: Correlation

    **Instructions:** Calculate the correlation between height and FEV1.
    """)
    return


@app.cell
def _(df):
    # Correlation of height with FEV1
    df["height_cm"].corr(df["fev1_l"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Age vs. FEV1: Correlation

    **Instructions:** Calculate the correlation between age and FEV1.
    """)
    return


@app.cell
def _(df):
    # Correlation of age with FEV1
    df["age_years"].corr(df["fev1_l"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Predictors

    **Instructions:** Store height and age as the predictors, `X`.
    """)
    return


@app.cell
def _(df):
    # Predictors (X): height and age
    X = df[["height_cm", "age_years"]]
    return (X,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Response

    **Instructions:** Store FEV1 as the response, `y`.
    """)
    return


@app.cell
def _(df):
    # Response (y): FEV1
    y = df["fev1_l"]
    return (y,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Training and Test Sets

    **Instructions:** Split `X` and `y` into training and test sets, with 20% of the rows in the test set. Name them `X_train`, `X_test`, `y_train` and `y_test`, and display the number of rows in the training and test sets.
    """)
    return


@app.cell
def _(X, train_test_split, y):
    # Split into training and test sets, then show the number of rows in each
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    len(X_train), len(X_test)
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Regression Model

    **Instructions:** Create a linear regression model named `model` and fit it to the training set. Display the slopes and the intercept of the fitted model.
    """)
    return


@app.cell
def _(LinearRegression, X_train, y_train):
    # Create the model and fit it to the training set, then show the slopes (height first, then age) and the intercept
    model = LinearRegression()
    model.fit(X_train, y_train)
    model.coef_, model.intercept_
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Training Predictions

    **Instructions:** Predict FEV1 for the training set and store the predictions as `train_pred`.
    """)
    return


@app.cell
def _(X_train, model):
    # Predict FEV1 for the training set
    train_pred = model.predict(X_train)
    return (train_pred,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Residual Plot

    **Instructions:** Create a scatter plot of the predicted FEV1 and the residual (actual minus predicted) for the training set.
    """)
    return


@app.cell
def _(plt, sns, train_pred, y_train):
    # Residual plot: actual minus predicted, against predicted
    sns.scatterplot(x=train_pred, y=y_train - train_pred)
    plt.axhline(0, color="red")
    plt.title("Training Set Residuals")
    plt.xlabel("Predicted FEV1 (L)")
    plt.ylabel("Residual (actual - predicted)")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Test Predictions

    **Instructions:** Predict FEV1 for the test set and store the predictions as `y_pred`.
    """)
    return


@app.cell
def _(X_test, model):
    # Predict FEV1 for the test set
    y_pred = model.predict(X_test)
    return (y_pred,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Test Score Residuals

    **Instructions:** Create a scatter plot of the predicted FEV1 and the residual (actual minus predicted) for the test set.
    """)
    return


@app.cell
def _(plt, sns, y_pred, y_test):
    # Residual plot for the test set: actual minus predicted, against predicted
    sns.scatterplot(x=y_pred, y=y_test - y_pred)
    plt.axhline(0, color="red")
    plt.title("Test Set Residuals")
    plt.xlabel("Predicted FEV1 (L)")
    plt.ylabel("Residual (actual - predicted)")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Test R Squared

    **Instructions:** Calculate the R squared of the model on the test set.
    """)
    return


@app.cell
def _(r2_score, y_pred, y_test):
    # Test R squared
    r2_score(y_test, y_pred)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Test MAE

    **Instructions:** Calculate the MAE (mean absolute error) of the model on the test set.
    """)
    return


@app.cell
def _(mean_absolute_error, y_pred, y_test):
    # Test MAE
    mean_absolute_error(y_test, y_pred)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Test RMSE

    **Instructions:** Calculate the RMSE (root mean squared error) of the model on the test set.
    """)
    return


@app.cell
def _(root_mean_squared_error, y_pred, y_test):
    # Test RMSE
    root_mean_squared_error(y_test, y_pred)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Predict for New Patients

    **Instructions:** Four new participants are every combination of two heights, 165 cm and 185 cm, and two ages, 30 and 60 years. Predict the FEV1 of each one.
    """)
    return


@app.cell
def _(model, pd):
    # Predict FEV1 for each new participant (every combination of height and age)
    model.predict(
        pd.DataFrame({"height_cm": [165, 165, 185, 185], "age_years": [30, 60, 30, 60]})
    )
    return


if __name__ == "__main__":
    app.run()
