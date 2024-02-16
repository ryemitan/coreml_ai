import pandas as pd
from sklearn.metrics import accuracy_score
import joblib
import os
from datetime import datetime
from sklearn.linear_model import (
    LogisticRegression,
    # RidgeClassifier,
    SGDClassifier,
    PassiveAggressiveClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
    BaggingClassifier,
)
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import pickle
from flask import session


# # TimestampManager class to handle timestamps
# class TimestampManager:
#     def __init__(self):
#     #     self.timestamp = None

#     # def set_timestamp(self):
#         self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

#     def get_timestamp(self):
#         # if self.timestamp is None:
#         #     self.set_timestamp()
#         return self.timestamp

class ModelTrainer:
    DEFAULT_MODELS_FOLDER = 'app_users/default/projects/default/trained_models' 
    def __init__(self, dataset_path, target_column):
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.models = {
            'Logistic Regression': LogisticRegression(),
            # 'Ridge Classifier': RidgeClassifier(),
            'SGD Classifier': SGDClassifier(),
            'Passive Aggressive Classifier': PassiveAggressiveClassifier(),
            'Decision Tree Classifier': DecisionTreeClassifier(),
            'Random Forest Classifier': RandomForestClassifier(),
            'Gradient Boosting Classifier': GradientBoostingClassifier(),
            'AdaBoost Classifier': AdaBoostClassifier(),
            'Extra Trees Classifier': ExtraTreesClassifier(),
            'Support Vector Classifier': SVC(kernel='linear', probability=True),
            'Linear SVC': LinearSVC(),
            'NuSVC': NuSVC(probability=True),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Multi-layer Perceptron': MLPClassifier(),
            'Gaussian Naive Bayes': GaussianNB(),
            'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
            'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
            'Voting Classifier': VotingClassifier(estimators=[
                ('rf', RandomForestClassifier()), 
                ('svc', SVC(kernel='linear', probability=True)), 
                ('knn', KNeighborsClassifier())
            ]),
            'Bagging Classifier': BaggingClassifier(
                base_estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42
            ),
            'HistGradient Boosting Classifier': HistGradientBoostingClassifier(),
            # Add more classification models as needed (up to 30)
        }
        
        # C:\ai_core_cl\app_users\default\projects\default\trained_models
        self.default_models_folder = self.DEFAULT_MODELS_FOLDER

        # self.set.models_dir()
        # Set models_dir based on the session
        if session and 'Session_project_folder' in session:
            self.set_models_dir()
        else:
            self.models_dir = self.default_models_folder

        self.model_filename = 'best_model'  # Specify the model filename for the best model
        self.best_model = {}
        self.best_accuracy = 0
        self.X = {}
        self.y = {}
        self.accuracy_results = []  # Store accuracy results in a list
        self.accuracy_csv_filename = 'accuracy_results.csv'
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # self.timestamp_manager = TimestampManager()

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

        # Create a directory path based on the user's folder within the 'trained_models' folder
        return os.path.join(user_folder, 'trained_models')

    def set_models_dir(self):
        self.models_dir = self.get_user_models_dir()
        session['Session_models_folder'] = self.models_dir


    def load_data(self):
        if self.dataset_path is not None:
            data = pd.read_csv(self.dataset_path)

            char_cols = data.select_dtypes(include=['object']).columns

            label_mapping = {}

            for c in char_cols:
                data[c], label_mapping[c] = pd.factorize(data[c])

            self.X = data.drop(self.target_column, axis=1)
            self.y = data[self.target_column]

    def train_models(self):
        # timestamp = self.timestamp_manager.get_timestamp()
        accuracy_results_filename = f'accuracy_results_{self.timestamp}.csv'

        best_model_info = {
            'model_name': None,
            'model_filename': None,
            'headers': [],
        }

        for name, model in self.models.items():
            # model_name_with_timestamp = f'{name}_{timestamp}'
            model_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('model', model)
            ])
            model_pipeline.fit(self.X, self.y)
            self.save_model(model_pipeline, name)
            
            
            # Save each model with a unique filename including the timestamp
            model_filename = f'{name}_{self.timestamp}'
            model_filename_w_ext = f'{model_filename}.pkl'
            model_path = os.path.join(self.models_dir, model_filename_w_ext)
            with open(model_path, 'wb') as file:
                pickle.dump(model_pipeline, file)
            print(f'Saved model: {model_path}')


            y_pred = model_pipeline.predict(self.X)
            accuracy = accuracy_score(self.y, y_pred)
            # roc_auc = roc_auc_score(self.y, model_pipeline.predict_proba(self.X)[:, 1])
            confusion_mat = confusion_matrix(self.y, y_pred)
            classification_rep = classification_report(self.y, y_pred)

            self.accuracy_results.append({
                'Model': name,
                'Accuracy': accuracy,
                # 'ROC_AUC': roc_auc,
                'Confusion_Matrix': confusion_mat.tolist(),
                'Classification_Report': classification_rep,
                'Target_Column': self.target_column,
                'Time': self.timestamp,
                'Model_Filename': model_filename
            })

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model = model_pipeline
                self.model_name = name
                self.model_filename = model_filename
                # print(self.model_filename)

                # Update the best_model_info
                best_model_info['model_name'] = self.model_name
                best_model_info['model_filename'] = self.model_filename
                best_model_info['best_accuracy'] = self.best_accuracy
                best_model_info['headers'] = self.X.columns.tolist()

        # self.trntamp = timestamp
        self.save_accuracy_results(accuracy_results_filename)  # Save accuracy results to CSV
        # self.save_model(self.best_model, self.model_filename)  # Save the best model
        print(self.model_filename)

        # Return the best_model_info as JSON
        return best_model_info

    def evaluate_accuracy(self, model_filename):
        try:

            Session_project_folder = session.get('Session_project_folder')
            Session_model_folder = os.path.join(Session_project_folder,'trained_models')
            Session_model_path = os.path.join(Session_model_folder, f'{self.model_filename}.pkl')
            session['Session_model_folder'] = Session_model_folder
            session['Session_model_path'] = Session_model_path

            # timestamp = self.timestamp_manager.get_timestamp()
            print(self.model_filename)
            # model_path = os.path.join(self.models_dir, f'{self.model_filename}.pkl')
            print(f'Clss Evaluate: Model path: {Session_model_path}')

            if not os.path.exists(Session_model_path):
                return 'Model not found'
            
            loaded_model = joblib.load(Session_model_path)
            print(loaded_model)

            # with open(model_path, 'rb') as file:
            #     loaded_model = pickle.load(file)
            #     print(f'Loaded model: {loaded_model}')

            y_pred = loaded_model.predict(self.X)
            accuracy = accuracy_score(self.y, y_pred)

            return accuracy
        except Exception as e:
            return str('Error: ' + str(e))

    def sort_accuracy_results(self):
        # Convert the accuracy results to a DataFrame and sort it by accuracy
        accuracy_df = pd.DataFrame(self.accuracy_results)
        sorted_df = accuracy_df.sort_values(by='Accuracy', ascending=False)
        return sorted_df

    def save_model(self, model, model_filename):
        # timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # timestamp = self.timestamp_manager.get_timestamp()

        print(f'Save_Model : {model}')
        model_path = os.path.join(self.models_dir, f'{model_filename}')

        # with open(model_path, 'wb') as file:
        #     pickle.dump(model_filename, file)


    
    def save_accuracy_results(self, accuracy_results_filename):
        accuracy_df = pd.DataFrame(self.accuracy_results)
        if not os.path.exists(self.accuracy_csv_filename):
            accuracy_df.to_csv(self.accuracy_csv_filename, index=False)
        else:
            existing_df = pd.read_csv(self.accuracy_csv_filename)
            updated_df = pd.concat([existing_df, accuracy_df], ignore_index=True)
            updated_df.to_csv(self.accuracy_csv_filename, index=False)

    def predict(self, input_data, model_filename):
        # timestamp = self.timestamp_manager.get_timestamp()
        Session_model_path = session.get('Session_model_path')
        print(f'MT Predict Session_model_path: {Session_model_path}')
        
        # print(timestamp)
        try:
           

            # model_path = os.path.join(self.models_dir, f'{self.model_filename}.pkl')
            model_path = Session_model_path
            print(f'MT Predict: Model path: {model_path}')
            
            # model_path = os.path.join(self.models_dir, f'{self.model_filename}.pkl')
            # print(f'Model path: {model_path}')

            if not os.path.exists(model_path):
                return 'Model not found'
            

            loaded_model = joblib.load(model_path)
            print(f'Predict: Loaded model: {loaded_model}')

            # with open(model_path, 'rb') as file:
            #     loaded_model = pickle.load(file)
            #     print(loaded_model)

            if isinstance(input_data, str):
                input_data = pd.read_csv(input_data)

            input_data = input_data.fillna(input_data.mean())

            char_cols = input_data.select_dtypes(include=['object']).columns

            label_mapping = {}

            for c in char_cols:
                input_data[c], label_mapping[c] = pd.factorize(input_data[c])

            print(f'Input data shape: {input_data.shape}')
            predictions = loaded_model.predict(input_data)
            print(f'Predictions: {predictions}')

            probabilities = None
            if hasattr(loaded_model, 'predict_proba'):
                probabilities = loaded_model.predict_proba(input_data)
                print(f'Probabilities: {probabilities}')

            result = {
                'Predictions': predictions.tolist(),
                'Probabilities': [p[1] for p in probabilities] if probabilities is not None else None
            }
            # print(result)

            return result  # Return predictions and probabilities as a JSON object


        except Exception as e:
            print(e)
            return str('Error: ' + str(e))


# Example usage:
# dataset_path = 'your_dataset.csv'
# target_column = 'your_target_column'
# model_trainer = ModelTrainer(dataset_path, target_column)
# model_trainer.load_data()
# best_model_info = model_trainer.train_models()
# print(f'Best Model: {best_model_info["model_name"]}')
# print(model_trainer.sort_accuracy_results())
