from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

class ModelsTrainer:
    """
    A class to train and evaluate machine learning models.

    This class is designed to train different models on a given dataset, evaluate their performance,
    and plot the importance of features for the models. It uses scikit-learn's model selection and
    metrics modules for model training and evaluation.

    Attributes
    ----------
    trained_models : dict
        A dictionary to store trained models.
    X : DataFrame
        Feature data used for training models.
    y : Series
        Target variable.

    Methods
    -------
    calculate_model_metrics(name, model, folds=5):
        Evaluates a model's performance using cross-validation.

    train_model_with_split(model, name):
        Trains a model with a train-test split and calculates performance metrics.

    train_final_model(model, name):
        Trains a model on the entire dataset.

    plot_feature_importance(model_name, top_n=15):
        Plots the top n feature importances of a trained model.
    """

    trained_models = {}
    
    def __init__(self, column_to_predict, dataset):
        """
        Initializes the ModelsTrainer with the dataset and the column to be predicted.

        Parameters
        ----------
        column_to_predict : str
            The name of the column to be predicted.
        dataset : DataFrame
            The dataset containing the features and target variable.
        """
        self.X = dataset.drop([column_to_predict],axis=1)
        self.y = dataset[column_to_predict]
        
    def calculate_model_metrics(self,name,model,folds=5):
        """
        Evaluate a model using cross-validation and print the performance metrics.

        This method calculates the mean squared error (MSE) and R-squared (R2) for the model
        using cross-validation. The performance metrics for each fold and their averages are printed.

        Parameters
        ----------
        name : str
            The name of the model.
        model : estimator object
            The model to be evaluated.
        folds : int, optional
            The number of folds to use for cross-validation (default is 5).
        """
        print("Training model:",name)
        scores = cross_validate(model, self.X, self.y, scoring=['neg_mean_squared_error', 'r2'], cv=folds, return_train_score=False)
        scores['test_neg_mean_squared_error'] = -scores['test_neg_mean_squared_error']
        
        print("MSE in each fold:",name, scores['test_neg_mean_squared_error'])
        print("Average MSE:",name, scores['test_neg_mean_squared_error'].mean())
        print("R2 in each fold:",name, scores['test_r2'])
        print("Average R2:",name, scores['test_r2'].mean())

    def train_model_with_split(self,model,name):
        """
        Train a model with a train-test split and calculate performance metrics.

        This method splits the data into training and testing sets, trains the model,
        and calculates the mean squared error (MSE) and R-squared (R2) on both sets.

        Parameters
        ----------
        model : estimator object
            The model to be trained.
        name : str
            The name of the model.
        """
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.20, random_state=42)
        model.fit(x_train, y_train.squeeze())
        y_train_preds = model.predict(x_train)
        y_test_preds = model.predict(x_test)

        mse_train = round(mean_squared_error(y_train, y_train_preds),3)
        mse_test = round(mean_squared_error(y_test, y_test_preds),3)
        r2_train = round(r2_score(y_train, y_train_preds),3)
        r2_test = round(r2_score(y_test, y_test_preds),3)
        
        print(f"{name} MSE train {mse_train}, test: {mse_test}")
        print(f"{name} R^2 {r2_train}, test: {r2_test}")

    def get_model(self, name):
        """
        Retrieve a trained model from the trained_models dictionary.

        This method returns the model associated with the given name, if it exists in the
        trained_models dictionary.

        Parameters
        ----------
        name : str
            The name of the model to retrieve.

        Returns
        -------
        estimator object or None
            The trained model if found, otherwise None.
        """
        return self.trained_models.get(name)

    def train_final_model(self, model, name):
        """
        Train a model on the entire dataset and store it in the trained_models dictionary.

        This method trains the model using all the available data and stores the trained model
        in the trained_models dictionary with the given name.

        Parameters
        ----------
        model : estimator object
            The model to be trained.
        name : str
            The name of the model.

        Returns
        -------
        estimator object
            The trained model.
        """
        self.trained_models[name] = model.fit(self.X, self.y)
        return self.trained_models[name]

    def plot_feature_importance(self, model_name, top_n=15):
        """
        Plot the top n feature importances of a trained model.

        This method retrieves the feature importances from the trained model and plots
        the top n features in a horizontal bar chart.

        Parameters
        ----------
        model_name : str
            The name of the model whose feature importance is to be plotted.
        top_n : int, optional
            The number of top features to plot (default is 15).
        """
        model = self.trained_models[model_name]
        importance = model.feature_importances_
        features = self.X.columns
        indices = np.argsort(importance)[-top_n:]
    
        plt.figure(figsize=(10, 8))
        plt.title('Feature importance')
        plt.barh(range(len(indices)), importance[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative importance')
        plt.show()

