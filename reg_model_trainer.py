import pandas as pd
import joblib
import os
import numpy as np
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
# from pycaret.regression import create_model
from pycaret.regression import setup, create_model, finalize_model
import pycaret.utils
from sklearn.linear_model import (
    Ridge, Lasso, ElasticNet,
    BayesianRidge, ARDRegression, HuberRegressor, PassiveAggressiveRegressor, SGDRegressor
)
from sklearn.svm import SVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
)

# List of Regression Models
pycaret_regression_models = [
    'lr', 'ridge', 'lasso', 'en', 'br', 'ard', 'huber', 'par', 'svm', 'knn', 'dt', 'rf', 'et', 'ada', 'gbr', 'mlp'#, 'xgboost'
]

scikit_learn_regression_models = [
     Ridge, Lasso, ElasticNet,
    BayesianRidge, ARDRegression, HuberRegressor, PassiveAggressiveRegressor, SGDRegressor,
    DecisionTreeRegressor, RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, AdaBoostRegressor, 
    KNeighborsRegressor, SVR, NuSVR
]

# Statsmodels Regression Models
statsmodels_regression_models = ['OLS', 'GLS', 'WLS', 'GLSAR', 'RLM']

# Scipy Regression Models
scipy_regression_models = ['curve_fit',  'TheilSen', 'ransac', 'HuberRegressor', 'GammaRegressor',
                           'PoissonRegressor', 'TweedieRegressor', 'GeneralizedLinearRegressor']

# PyTorch Regression Models
pytorch_regression_models = [ 'MLP (Multi-layer Perceptron)', 'RNN (Recurrent Neural Network)',
                             'LSTM (Long Short-Term Memory)', 'GRU (Gated Recurrent Unit)']

# Combined List of Regression Models
all_regression_models = (
        pycaret_regression_models +
        [model.__name__ for model in scikit_learn_regression_models] +
        statsmodels_regression_models +
        scipy_regression_models +
        pytorch_regression_models
)

# Ensure unique models in the list
unique_regression_models = list(set(all_regression_models))
import pickle
from flask import session


class RegModelTrainer:
    DEFAULT_MODELS_FOLDER = 'app_users/default/projects/default/trained_models' 
    def __init__(self, dataset_path, target_column):
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.models = {}
        # Set models_dir based on the session
        if session and 'Session_project_folder' in session:
            self.set_models_dir()
        else:
            self.models_dir = self.DEFAULT_MODELS_FOLDER
        self.model_filename = 'best_model'
        self.best_model = {}
        self.best_rmse = float('inf')
        self.X = {}
        self.y = {}
        self.rmse_results = []
        self.rmse_csv_filename = 'rmse_results.csv'
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        # Load data during initialization
        self.load_data()

    def get_user_models_dir(self):
        # Extract the user's folder from the session or use a default if not available
        user_folder = session.get('Session_project_folder')
        
        if user_folder is None:
            # Use the default system folder if 'Session_project_folder' is not found
            user_folder = 'trained_models'  # You can adjust this default value as needed

        # Create a directory path based on the 'trained_models' folder within the user's folder 
        return os.path.join(user_folder, 'trained_models')

    def set_models_dir(self):
        self.models_dir = self.get_user_models_dir()
          

    def load_data(self):
        if self.dataset_path is not None:
            data = pd.read_csv(self.dataset_path)

            # Clean up column names
            data.columns = data.columns.str.strip()  # Remove leading/trailing whitespaces
            data.columns = data.columns.str.replace('\r\n', '')  # Remove newline characters


            char_cols = data.select_dtypes(include=['object']).columns

            label_mapping = {}

            for c in char_cols:
                data[c], label_mapping[c] = pd.factorize(data[c])

            self.X = data.drop(self.target_column, axis=1)
            self.y = data[self.target_column]

    def train_models(self):
        # timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        rmse_results_filename = f'rmse_results_{self.timestamp}.csv'

        # Set up PyCaret environment
        setup(data=pd.concat([self.X, self.y], axis=1), target=self.y, session_id=42)


        best_model_info = {
            'model_name': None,
            'model_filename': None,
            'headers': [],
        }

        # Check for missing values in the target variable
        missing_values_y = self.y.isnull().sum()
        if missing_values_y > 0:
            print(f"Missing values in target variable (before imputation): {missing_values_y}")
            
            # Impute missing values with the median
            median_y = np.nanmedian(self.y)
            self.y = self.y.fillna(median_y)

            print(f"Missing values in target variable (after imputation): {self.y.isnull().sum()}")


        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.y_test = y_test
        self.X_test = X_test

        for name in unique_regression_models:
            # Check if the model is available in PyCaret
            if name in pycaret_regression_models:
                model = create_model(name)
            else:
                # Check if the model is available in scikit-learn
                if name in [model.__name__ for model in scikit_learn_regression_models]:
                    model = [model for model in scikit_learn_regression_models if model.__name__ == name][0]()
                else:
                    print(f"Model '{name}' not found. Skipping...")
                    continue

            model_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('model', model)
            ])
            model_pipeline.fit(X_train, y_train)
            self.save_model(model_pipeline, name)


            # Save each model with a unique filename including the timestamp
            model_filename = f'{name}_{self.timestamp}'
            model_filename_w_ext = f'{model_filename}.pkl'
            model_path = os.path.join(self.models_dir, model_filename_w_ext)
            with open(model_path, 'wb') as file:
                pickle.dump(model_pipeline, file)
            print(f'Saved model: {model_path}')

            y_pred = model_pipeline.predict(X_test)

            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            # explained_var = explained_variance_score(y_test, y_pred)

            self.rmse_results.append({
                'Model': name,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'Target_Column': self.target_column,
                'Time': self.timestamp,
                'Model_Filename': model_filename
            }) #Need the same for out of sample

            if rmse < self.best_rmse:
                self.best_rmse = rmse
                self.best_model = model_pipeline
                self.model_name = name
                self.model_filename = model_filename

                # Update the best_model_info
                best_model_info['model_name'] = self.model_name
                best_model_info['model_filename'] = self.model_filename
                best_model_info['best_rmse'] = self.best_rmse
                best_model_info['headers'] = self.X.columns.tolist()

        self.save_rmse_results(rmse_results_filename)
        self.save_model(self.best_model, self.model_filename)

        # Finalize the best model and save
        best_model = finalize_model(create_model(self.model_name))
        self.save_model(best_model, self.model_filename)

        return best_model_info

    def save_rmse_results(self, rmse_results_filename):
        rmse_df = pd.DataFrame(self.rmse_results)
        if not os.path.exists(self.rmse_csv_filename):
            rmse_df.to_csv(self.rmse_csv_filename, index=False)
        else:
            existing_df = pd.read_csv(self.rmse_csv_filename)
            updated_df = pd.concat([existing_df, rmse_df], ignore_index=True)
            updated_df.to_csv(self.rmse_csv_filename, index=False)

    def evaluate_rmse(self, model_filename):
        try:
            Session_project_folder = session.get('Session_project_folder')
            Session_model_folder = os.path.join(Session_project_folder,'trained_models')
            Session_model_path = os.path.join(Session_model_folder, f'{model_filename}.pkl')
            session['Session_model_folder'] = Session_model_folder
            session['Session_model_path'] = Session_model_path

            if not os.path.exists(Session_model_path):
                return 'Model not found'

            loaded_model = joblib.load(Session_model_path)

            X_test, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            y_pred = loaded_model.predict(X_test)

            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            return {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }

        except Exception as e:
            return str('Error: ' + str(e))

    def sort_rmse_results(self):
        rmse_df = pd.DataFrame(self.rmse_results)
        sorted_df = rmse_df.sort_values(by='RMSE', ascending=True)
        return sorted_df

    def save_model(self, model, model_filename):
        model_path = os.path.join(self.models_dir, f'{model_filename}.pkl')

        joblib.dump(model, model_path)

    def save_rmse_results(self, rmse_results_filename):
        rmse_df = pd.DataFrame(self.rmse_results)
        if not os.path.exists(self.rmse_csv_filename):
            rmse_df.to_csv(self.rmse_csv_filename, index=False)
        else:
            existing_df = pd.read_csv(self.rmse_csv_filename)
            updated_df = pd.concat([existing_df, rmse_df], ignore_index=True)
            updated_df.to_csv(self.rmse_csv_filename, index=False)

    def predict(self, input_data, model_filename):
        Session_model_path = session.get('Session_model_path')
        username = session.get('_user_id')
        print(f'MT Reg Predict model_filename: {self.model_filename}')
        print(f'MT Reg Predict models_dir: {self.models_dir}')
        print(username)
        try:
            
            model_path = Session_model_path #os.path.join(Session_project_folder, username, 'projects', f'{model_filename}.pkl')
            print(f'MT Reg Predict: Model path: {model_path}')


            if not os.path.exists(model_path):
                return 'Model not found'

            loaded_model = joblib.load(model_path)
            print(f'MT Reg Predict: Loaded model: {loaded_model}')

            if isinstance(input_data, str):
                input_data = pd.read_csv(input_data)

            input_data = input_data.fillna(input_data.mean())

            char_cols = input_data.select_dtypes(include=['object']).columns

            label_mapping = {}

            for c in char_cols:
                input_data[c], label_mapping[c] = pd.factorize(input_data[c])

            predictions = loaded_model.predict(input_data)

            result = {
                'Predictions': predictions.tolist()
            }

            return result

        except Exception as e:
            return str('Error: ' + str(e))


# Example usage:
# dataset_path = 'your_dataset.csv'
# target_column = 'your_target_column'
# model_trainer = ModelTrainer(dataset_path, target_column)
# model_trainer.load_data()
# best_model_info = model_trainer.train_models()
# print(best_model_info)

# Example of how to evaluate the RMSE of a specific model:
# model_name_to_evaluate = 'Linear Regression'  # Replace with the model name you want to evaluate
# rmse_metrics = model_trainer.evaluate_rmse(model_name_to_evaluate)
# print(f'RMSE metrics for {model_name_to_evaluate}: {rmse_metrics}')

# Example of how to sort the RMSE results:
# sorted_rmse_results = model_trainer.sort_rmse_results()
# print(sorted_rmse_results)

# Example of how to predict using a specific model:
# input_data_path = 'your_input_data.csv'  # Replace with the path to your input data
# predictions_result = model_trainer.predict(input_data_path, model_name_to_evaluate)
# print(predictions_result)

   
