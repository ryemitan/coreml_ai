from flask import Flask, request, jsonify, Response, send_file, render_template, session, redirect, url_for, flash
from flask_cors import CORS, cross_origin
from flask_session import Session
import os
import pandas as pd
import threading
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import logging
import numpy as np
import json
import secrets
from flask_login import LoginManager, login_user, current_user, login_required
from user import User  # Import the User class
from traceback import print_exc  # Import the print_exc function
from passlib.hash import scrypt
from datetime import datetime
import joblib 
import pickle
import logging
import uuid

app = Flask(__name__, static_url_path='/static')



# Set the secret key for Flask sessions
app.secret_key = secrets.token_hex(48) # Change this to a secure random key in production
# Generate a UUID for dynamic part of the prefix
dynamic_prefix = str(uuid.uuid4())
# Configure the session to use filesystem (you can use other backends like Redis)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = True
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_KEY_PREFIX'] = 'AICORE_' + dynamic_prefix  # Change this to a unique prefix
app.config['SESSION_FILE_DIR'] = os.path.join(app.root_path, 'app_users','sessions')



print(app.secret_key)
login_manager = LoginManager(app)

# Initialize the Flask-Session extension
Session(app)
CORS(app, support_credentials=True)

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO) 

# Custom middleware to log request information
class RequestLoggerMiddleware:
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        path = environ.get('PATH_INFO', '')
        method = environ.get('REQUEST_METHOD', '')
        logging.info(f"Request: {method} {path}")
        return self.app(environ, start_response)
    
# Register the middleware
app.wsgi_app = RequestLoggerMiddleware(app.wsgi_app)

# Assume you have a user database or some mechanism to handle user authentication
# Load user data from the file
with open('users.json', 'r') as file:
    users = json.load(file)
# print(users)

# 
# 
# 
# ______________________________________________________________________________________________
def generate_new_user():
    # Autogenerate a unique username (you can modify this logic as needed)
    new_username = f"user_{secrets.token_hex(4)}"
    print(f"New user created: {new_username}")

    # Autogenerate a random password (you can modify this logic as needed)
    new_password = secrets.token_hex(16)
    print(f"Password for {new_username}: {new_password}")

    # Hash the password before saving it
    hashed_password = scrypt.hash(new_password)

    # Save the new user to the users dictionary
    users[new_username] = {
        'password': hashed_password,
        'special_pet': new_password,
        'favorite_actor': new_username,
    }

    # Save the updated users dictionary back to users.json
    with open('users.json', 'w') as file:
        json.dump(users, file, indent=2)

    # print(f"New user created: {new_username}")
    # print(f"Password for {new_username}: {new_password}")

# Call the function to generate a new user
generate_new_user()
# 
# 
# 
# ____________________________________________________________________________________________________

@login_manager.user_loader
def load_user(user_id):
    # This function is required by Flask-Login.
    # It should return the user object or None if the user is not found.
    return User(user_id)

# Route to render api.html
@app.route('/api', methods=['GET'])
def render_api_page():
    try:
        # Retrieve query parameters from the request with default values
        api_link = request.args.get('api_link', '')
        username = request.args.get('username', '')
        model_filename = request.args.get('model_filename', '')
        model_folder_path = request.args.get('model_folder_path', '')
        session['api_link'] = api_link

        Session_model_folder = session.get('Session_model_folder')
        API_Required_Fields = session.get('API_Required_Fields')
        user_projects_folder = session.get('user_projects_folder')

        api_data_path = os.path.join(Session_model_folder, 'api.json')

        # Read API data from the JSON file or any other data source
        with open(api_data_path, 'r') as file:
            api_summary = json.load(file)

        api_data = read_api_data()
        api_count = api_data.get('api_count', 0)
        model_files_count = get_files_count(model_folder_path)
        projects_count = get_folders_count(user_projects_folder)

        api_info = {
            'api_count': api_count,
            'model_files_count': model_files_count,
            'folders_count': projects_count,
            'api_link': api_link,
            'username': username,
            'model_filename': model_filename,
            'model_folder_path': model_folder_path,
            'API_Required_Fields': API_Required_Fields
        }
        session['api_info'] = api_info

        return render_template('api.html', api_summary=api_summary, api_info=api_info)

    except Exception as e:
        # Handle exceptions (e.g., file not found, invalid JSON format)
        print(f"Error: {str(e)}")
        return render_template('error.html', error_message=str(e))




@app.route('/sign_in', methods=['POST'])
def sign_in():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        # print(data)
        # print(users)
        # print(username)
        # Check if the username exists and the password matches
        # if username in users and scrypt.verify(password, users[username]['password']):
        if username in users and scrypt.verify(password, users[username]['password']):
            # Get the user object (replace with your logic to get the user object)
            user = User(username)
            # print(user)
            # Log in the user
            login_user(user)
            

            # Redirect to the /train page upon successful sign-in
            return jsonify({'message': 'Sign in successful', 'success': True, 'redirect_url': url_for('dashboard')})
            # return jsonify({'message': 'Sign in successful', 'success': True, 'redirect_url': url_for('train')})
        
        else:
            return jsonify({'error': 'Invalid username or password', 'success': False}), 401
    except Exception as e:
        print(f"Error during sign-in: {str(e)}")
        print_exc()  # Print the full traceback
        return jsonify({'error': 'Internal server error during sign in'}), 500

@app.route('/recover_password', methods=['POST'])
def recover_password():
    data = request.get_json()
    username = data.get('username')
    special_pet = data.get('special_pet')
    favorite_actor = data.get('favorite_actor')
    favorite_car = data.get('favorite_car')
    favorite_location = data.get('favorite_location')

    if username in users and \
       special_pet == users[username]['special_pet'] and \
       favorite_actor == users[username]['favorite_actor'] and \
       favorite_car == users[username]['favorite_car'] and \
       favorite_location == users[username]['favorite_location']:
        recovered_password = users[username]['password']
        return jsonify({'recovered_password': recovered_password})
    else:
        return jsonify({'error': 'Invalid recovery information'}), 401

@app.route('/sign_up', methods=['POST'])
def sign_up():
    try:
        data = request.get_json()
        print(data)
        username = data.get('username')
        password = data.get('password')
        special_pet = data.get('special_pet')
        favorite_actor = data.get('favorite_actor')

        # Check if the username already exists
        if username in users:
            return jsonify({'error': 'Username already exists'}), 400

        # Add the new user to the user database
        users[username] = {
            'password': scrypt.hash(password),  # Hash the password before storing
            'special_pet': special_pet,
            'favorite_actor': favorite_actor
        }

        # Save the updated user database to the JSON file
        with open('users.json', 'w') as file:
            json.dump(users, file, indent=2)  # Save the updated users dictionary

        user_folder = os.path.join('app_users', username)
        os.makedirs(user_folder)

        project_folder = os.path.join(user_folder, 'projects')
        os.makedirs(project_folder)


        return jsonify({'message': 'Sign up successful'})
    except Exception as e:
        print(f"Error in sign_up: {str(e)}")
        return jsonify({'error': 'Internal server error during sign up'}), 500

# Set the path to store uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'csv'}  # Set of allowed file extensions

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


from model_trainer import ModelTrainer
from reg_model_trainer import RegModelTrainer  # Import the RegModelTrainer class
# Create dictionaries to hold trainers for each thread
trainers = {}
reg_trainers = {}

# Initialize the ModelTrainer and RegModelTrainer
trainer = ModelTrainer(None, None)
reg_trainer = RegModelTrainer(None, None)

# Store the evaluation results globally
evaluation_results = {}




def get_or_create_trainer(thread_id, dataset_file, target_column, trainer_class):
    if thread_id not in trainers:
        session['target_column'] = target_column
        if trainer_class == 'Classification':
            trainers[thread_id] = ModelTrainer(dataset_file, target_column)
        elif trainer_class == 'Regression':
            reg_trainers[thread_id] = RegModelTrainer(dataset_file, target_column)
    return trainers.get(thread_id) if trainer_class == 'Classification' else reg_trainers.get(thread_id)


def get_trainer_by_model_name(user_name, model_name):
    # Your logic to retrieve the trainer based on user name and model name
    return trainers.get((user_name, model_name))


def register_model(user_name, model_name, timestamp, api_endpoint):
    # Your logic to register the model details and associated API endpoint
    # You might store this information in a database or other persistent storage
    pass


def register_prediction(user_name, model_name, prediction_result):
    # Your logic to register the prediction result
    # You might store this information in a database or other persistent storage
    pass


@app.route('/')
def home():
    return render_template('FE_App.html')


@app.route('/dashboard')
@login_required
def dashboard():
    username = current_user.get_id()
    return render_template('dashboard.html', username=username)

@app.route('/create_new_model')
@login_required
def create_new_model():
    username = get_current_user_id()
    user_projects_folder = os.path.join("app_users", username, 'projects')
    session['user_projects_folder'] = user_projects_folder
    existing_projects = get_existing_projects(user_projects_folder)

    return render_template('create_new_model.html', existingProjects=existing_projects, username=username)

@app.route('/check_project_uniqueness')
def check_project_uniqueness():
    try:
        username = get_current_user_id()
        project_name = session.get('project_name')
        is_unique = check_project_name_uniqueness(username, project_name)
        return jsonify({'unique': is_unique})

    except Exception as e:
        print(f"Error in check_project_uniqueness: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/create_project', methods=['POST'])
def create_project():
    username = get_current_user_id()

    if not username:
        flash('Error: Unable to retrieve the username.', 'error')
        return redirect(url_for('some_error_route'))

    user_projects_folder = os.path.join("app_users", username, 'projects')
    print(user_projects_folder)
    new_project_name = request.form.get('project_name')
    print(new_project_name)

    if 'existingProject' in request.form:
        # Handle the case when an existing project is selected
        existing_project_name = request.form.get('existingProject')
        print(existing_project_name)
        session['project_name']=existing_project_name

        flash(f'Using existing project "{existing_project_name}".', 'info')
        return redirect(url_for('render_train'))


    if not new_project_name:
        flash('Error: Project name is required.', 'error')
        return redirect(url_for('create_new_model'))

    if user_projects_folder and new_project_name:
        if new_project_name in os.listdir(user_projects_folder):
            flash('Project with that name already exists. Please choose a different name.', 'error')
            return redirect(url_for('create_new_model'))
        else:
            create_new_project(username, user_projects_folder, new_project_name)
            flash(f'Project "{new_project_name}" created successfully. You can proceed to train models.', 'success')
            session['project_name']=new_project_name
            return redirect(url_for('render_train'))
    else:
        flash('Error: Unable to retrieve project name or user folder.', 'error')
        return redirect(url_for('some_error_route'))

def get_current_user_id():
    return current_user.get_id() if current_user.is_authenticated else None

def check_project_name_uniqueness(username, project_name):
    user_projects_folder = os.path.join("app_users", username, 'projects')
    return project_name not in os.listdir(user_projects_folder)

def get_existing_projects(user_projects_folder):
    return [folder for folder in os.listdir(user_projects_folder) if os.path.isdir(os.path.join(user_projects_folder, folder))]

def create_new_project(username, user_projects_folder, new_project_name):
    new_project_folder = os.path.join(user_projects_folder, new_project_name)
    
    print(new_project_folder)
    os.makedirs(new_project_folder)
    session['Session_project_folder'] = new_project_folder

    # Create additional folders within the new_project_folder
    for folder_name in ['trained_models', 'uploads', 'logs']:
        os.makedirs(os.path.join(new_project_folder, folder_name))

    flash(f'Project "{new_project_name}" created successfully. You can proceed to train models.', 'success')


@app.route('/review_models')
def review_models():
    return render_template('review_models.html')

@app.route('/view_create_apis')
def view_create_apis():
    return render_template('view_create_apis.html')

@app.route('/user_configurations')
def user_configurations():
    return render_template('user_configurations.html')




def get_or_create_trainer(thread_id, dataset_file, target_column, trainer_class):
    if thread_id not in trainers:
        if trainer_class == 'Classification':
            trainers[thread_id] = ModelTrainer(dataset_file, target_column)
        elif trainer_class == 'Regression':
            reg_trainers[thread_id] = RegModelTrainer(dataset_file, target_column)
    return trainers.get(thread_id) if trainer_class == 'Classification' else reg_trainers.get(thread_id)


def get_data_type(column_values):
    # Function to detect data type (Numeric or Categorical)
    try:
        # Convert values to floats to check if all are numeric
        float_values = [float(value) for value in column_values]
        return 'Numeric'
    except ValueError:
        return 'Categorical'

def count_missing_data(column_values):
    # Function to count missing data
    return column_values.count('')

@app.route('/process_file', methods=['POST'])
def process_file():
    try:
        # Check if the POST request has the file part
        if 'trainData' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['trainData']

        # Check if the file is empty
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        # Process the file content (example: read CSV data)
        lines = file.read().decode('utf-8').split('\n')
        headers = lines[0].split(',')
        
        # Get data types and missing data counts for each column
        column_data = {}
        for header in headers:
            column_index = headers.index(header)
            column_values = [row.split(',')[column_index].strip() for row in lines[1:]]

            data_type = get_data_type(column_values)
            missing_data_count = count_missing_data(column_values)

            column_data[header] = {
                'data_type': data_type,
                'missing_data_count': missing_data_count
            }

        # Store relevant data in Flask session
        session['headers'] = headers
        session['lines'] = lines
        session['column_data'] = column_data

        return jsonify({'success': 'File processed successfully', 'column_data': column_data})

    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'})


@app.route('/update_trainer_class/<selected_target_column>', methods=['GET'])
def update_trainer_class(selected_target_column):
    try:
        # Retrieve session data
        headers = session.get('headers', [])
        lines = session.get('lines', [])

        # Ensure headers and lines are available
        if not headers or not lines:
            return jsonify({'error': 'Session data not found'})

        # Get data type for the selected target column
        selected_target_column_index = headers.index(selected_target_column)
        selected_target_column_values = [row.split(',')[selected_target_column_index].strip() for row in lines[1:]]
        selected_target_column_data_type = get_data_type(selected_target_column_values)

        # Update Trainer Class based on the data type
        trainer_class = 'Regression' if selected_target_column_data_type == 'Numeric' else 'Classification'

        # Update session data if needed
        session['trainer_class'] = trainer_class

        return jsonify({'trainer_class': trainer_class})

    except Exception as e:
        return jsonify({'error': f'Error updating Trainer Class: {str(e)}'})


# New route to render train.html
@app.route('/train', methods=['GET', 'POST'])
@login_required
def render_train():
    logging.info('Request to /train endpoint received')
    username = get_current_user_id()

    if request.method == 'POST':
        print(request.args)
        print(request.form)
        # Retrieve the project_name from the form data
        project_name = request.form.get('project_name')
        session['project_name'] = project_name
        Session_project_folder = os.path.join("app_users", username, 'projects', project_name)
        session['Session_project_folder'] = Session_project_folder

        print(project_name)
        print(Session_project_folder)
        # Retrieve the logged-in username from the session
        
        return render_template('train_models.html', username=username, project_name=project_name)

    # Handle GET request if needed
    return render_template('train_models.html')

    
trained_models_results = []

@app.route('/train_models', methods=['POST'])
@login_required
@cross_origin(supports_credentials=True)
def train_models():
    try:
        project_name = session.get('project_name')
        
        print("Received train_models request")
        print(project_name)
        Session_project_folder = session.get('user_projects_folder')
        print(f"Session_project_folder: {Session_project_folder}")
        # Get the JSON data from the request body
        # data = request.get_json()
        # Check if the required fields are present
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request.'}), 400
        
        dataset_file = request.files['file']
        # project_name=session.get('project_name')
        # print(request.form)
        print(session)
        username = get_current_user_id()
        # username = current_user.get_id()
        print(f"UserName: {username}")
      

        # Check if the file extension is allowed
        if not allowed_file(dataset_file.filename):
            return jsonify({'error': 'Invalid file format. Please upload a CSV file.'}), 400
        

        # Save the file to the UPLOAD_FOLDER
        filename = secure_filename(dataset_file.filename)
        print(filename)

        # Check if username is not None
        if username is not None:     

            # Check if project_name is not None
            if project_name is not None:
                # Continue with the rest of the code
                # ...
                Session_upload_folder = os.path.join(Session_project_folder, project_name,'uploads')
                print(f"Session_upload_folder: {Session_upload_folder}")

                file_path = os.path.join(Session_upload_folder, filename)
                print(f"file_path: {Session_upload_folder}")

                # ...
            else:
                return jsonify({'error': 'Project name is missing.'}), 400
        else:
            return jsonify({'error': 'Username is missing.'}), 400



        # file_path = os.path.join(Session_project_folder, 'uploads', filename)
        
        dataset_file.save(file_path)


        target_column = request.form.get('target_column')
        trainer_class = request.form.get('trainer_class')
        session['trainer_class'] = trainer_class
        # print(trainer_class)


        if dataset_file is None  :
            return jsonify({'error': 'Missing or invalid payload parameters for Dataset file.'}), 400
        
        if target_column is None  :
            return jsonify({'error': 'Missing or invalid payload parameters for Target Column.'}), 400

        if trainer_class not in ['Classification', 'Regression'] :
            return jsonify({'error': 'Missing or invalid payload parameters for Trainer Class.'}), 400                

        # Get the current thread's identifier
        thread_id = threading.get_ident()
        # print((thread_id, file_path, target_column, trainer_class))

        # Retrieve or create the thread-local trainer
        trainer = get_or_create_trainer(thread_id, file_path, target_column, trainer_class)
        # print(trainer)

        # Load the data
        trainer.load_data()

        # Call your training code
        result = trainer.train_models()
        # print(result)

        if 'model_name' in result and 'model_filename' in result:
            model_name = result['model_name']
            model_filename = result['model_filename']
        else:
            print("Error: Unexpected structure of values returned by trainer.train_models()")

        # You can call separate functions here to evaluate accuracy, return JSON responses, etc.
        # Evaluate accuracy
        if trainer_class == 'Classification':
            accuracy = trainer.evaluate_accuracy(model_name)
            evaluation_result = trainer.sort_accuracy_results().to_dict(orient='records')
        elif trainer_class == 'Regression':
            rmse_metrics = trainer.evaluate_rmse(model_name)
            evaluation_result = trainer.sort_rmse_results().to_dict(orient='records')

        # Get the list of column headers from the loaded dataset
        headers = trainer.X.columns.tolist()
        # print(evaluation_result)

        # Convert evaluation_result to a DataFrame
        evaluation_result_df = pd.DataFrame(evaluation_result)

        
        print("Training successful")
            
         # Include the best model info in the response
        best_model_info = {
            'model_filename': model_filename,
            'model_name': model_name,
            'headers': headers,
        }

        # Store the evaluation results
        global evaluation_results
        evaluation_results = {
            # 'best_model_info': best_model_info,
            'evaluation_result': evaluation_result,
            # 'accuracy': accuracy if trainer_class == 'Classification' else rmse_metrics,
            # 'redirect_url': '/gmodels'
        }

        return jsonify({
            'message': 'Models trained and best model saved.',
            'best_model_info': best_model_info,
            'evaluation_result': evaluation_result,
            'accuracy': accuracy if trainer_class == 'Classification' else rmse_metrics,
            'redirect_url': '/gmodels'
        })

    except Exception as e:
        print(f"Error in train_models: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/gmodels')
@login_required
def render_gmodels():
    username = current_user.get_id()
    return render_template('get_models.html', username = username)

@app.route('/get_models', methods=['GET'])
def get_models():
    try:
        # global trainer
        global trainer
        # Add debug print statements to check the trainer object
        username = current_user.get_id()
        # Session_project_folder = session.get('Session_project_folder')
        Session_project_folder = session.get('user_projects_folder')
        project_name = session.get('project_name')

        # models_dir = 'trained_models'
       
        models_dir = os.path.join(Session_project_folder, project_name,'trained_models')
        print(models_dir)
        session['Session_model_folder'] = models_dir

        # print(models_dir)
        if not os.path.exists(models_dir):
            return jsonify({'models': []})
        
        # print(trainer.accuracy_results.sort(by='Accuracy'))

        model_files = os.listdir(models_dir)
        # print(model_files)
        model_names = [os.path.splitext(filename)[0] for filename in model_files]
        # print(model_names)

        
        # Get the evaluation results from the global variable
        global evaluation_results
        evaluation_result = evaluation_results.get('evaluation_result', {})
        # Convert the list of dictionaries to a DataFrame
        trainer_accuracy_df = pd.DataFrame(evaluation_result)
        print(trainer_accuracy_df)

        trainer_class = session.get('trainer_class')

         # Sort the DataFrame based on trainer class
        if trainer_class == 'Classification':
            # Sort by 'Accuracy' for classification
            sort_columns = ['Accuracy']
            sort_type = False
        elif trainer_class == 'Regression':
            # Sort by 'RMSE' for regression
            sort_columns = ['RMSE']
            sort_type = True
        else:
            # Handle the case when trainer class is neither Classification nor Regression
            print("Error: Invalid trainer class.")
            return jsonify({'error': 'Invalid trainer class'}), 400


        # Sort the DataFrame by the 'Accuracy' column
        try:
            models_with_accuracy = trainer_accuracy_df.sort_values(by=sort_columns, ascending=sort_type)
        except KeyError:
            # If the 'Accuracy' column is not present in the DataFrame
            print("Error: 'Accuracy'or 'RMSE' column not found in the DataFrame.")
            models_with_accuracy = pd.DataFrame()
            # print(models_with_accuracy)

        return jsonify({'models': models_with_accuracy.to_dict(orient='records')})
    

    except Exception as e:
        # Log the exception for debugging
        print(f"Error in get_models: {str(e)}")

        # Return an error response
        return jsonify({'error': 'Internal Server Error'}), 500
    return render_template('evaluate_model.html')

# New route to render predict.html

@app.route('/cre_dict', methods=['GET', 'POST'])
@login_required
def pre_dict():
    model_name = request.form.get('model_name')  # Retrieve model_name from the form data
    model_filename = request.form.get('model_filename') 
    # Other logic as needed...
    username = current_user.get_id()

    return render_template('pre_dict.html', model_name=model_name, model_filename=model_filename,username = username)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Received predict request")

        # Check if the required fields are present
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request.'}), 400

        dataset_file = request.files['file']
        username = current_user.get_id()
        Session_project_folder = session.get('Session_project_folder')
        project_name = session.get('project_name')
        print(f"Session_project_folder: {Session_project_folder}")
        print(f"project_name: {project_name}")
        

        # Check if the file extension is allowed
        if not allowed_file(dataset_file.filename):
            return jsonify({'error': 'Invalid file format. Please upload a CSV file.'}), 400

        # Save the file to the UPLOAD_FOLDER
        filename = secure_filename(dataset_file.filename)
        print(f"filename: {filename}")

        # Check if username is not None
        if username is not None:     

            # Check if project_name is not None
            if project_name is not None:
                # Continue with the rest of the code
                # ...
                Session_upload_folder = os.path.join(Session_project_folder,'uploads')
                print(f"Session_upload_folder: {Session_upload_folder}")

                file_path = os.path.join(Session_upload_folder, filename)
                print(f"file_path: {file_path}")

                # ...
            else:
                return jsonify({'error': 'Project name is missing.'}), 400
        else:
            return jsonify({'error': 'Username is missing.'}), 400




        # file_path = os.path.join(Session_project_folder, 'uploads', filename)
        # print(file_path)
        dataset_file.save(file_path)

        # Retrieve other necessary parameters (e.g., model name, model file path, etc.) from the request or session
        model_name = request.form.get('model_name')  # Assuming the model name is passed as a form field
        print(model_name)
        Session_model_folder = session.get('Session_model_folder')
        # model_path = os.path.join(Session_model_folder, f'{model_name}.pkl')
        # session['Session_model_path'] = model_path
        # print(f'Predict-Route - Loaded model: {model_path}')

        # Load the trained model using the model name and file path
        # prediction_result = trainer.predict(file_path, model_name)
        prediction_result = trainer.predict(file_path, model_name)
        # print(prediction_result)
        # This step will depend on how you manage your trained models. 
        # It could involve loading a serialized model or re-training a model with the same parameters and dataset.

        return jsonify({
            'message': 'Prediction completed.',
            'prediction_result': prediction_result,
            'model_name': model_name,
            'redirect_url': url_for('result', predictions_r=prediction_result, model_name=model_name),  # Redirect URL after prediction
        })

    except Exception as e:
        print(f"Error in predict: {str(e)}")
        return jsonify({'error': str(e)}), 500

# New route to render evaluate_model.html
@app.route('/evaluatemodel')
@login_required
def render_evaluatemodel():
    username = current_user.get_id()
    return render_template('evaluate_models.html', username = username)

import ast

@app.route('/store_session', methods=['POST'])
def store_session():
    if request.method == 'POST':
        data = request.get_json()
        api_required_fields = data.get('apiRequiredFields')
        session['API_Required_Fields'] = api_required_fields
        return {'status': 'success'}
    

@app.route('/result', methods=['GET', 'POST'])
@login_required
def result():
    try:
        username = current_user.get_id()
        print(request.args)
        predictions_r = request.args.get('prediction_r')
        print(predictions_r)

        # If predictions and probabilities are JSON strings, parse them
        prediction_result = json.loads(predictions_r)

        predictions = prediction_result.get('Predictions', [])
        probabilities = prediction_result.get('Probabilities', [])
        model_name = request.args.get('model_name')

        print(predictions)
        print(probabilities)
        print(model_name)

        # Render the result.html template with the parsed data
        return render_template('result.html', 
                               predictions=predictions, 
                               probabilities=probabilities, 
                               model_name=model_name,
                               username=username)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/evaluate_model', methods=['POST'])
def evaluate_model():
    try:
        global trainer
        model_name = request.json.get('model_name')
        accuracy = trainer.evaluate_accuracy(model_name)

        if accuracy is not None:
            return jsonify({'accuracy': accuracy})
        else:
            return jsonify({'error': 'Model not found or evaluation failed.'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_model/<model_name>', methods=['GET'])
def download_model(model_name):
    Session_project_folder = session.get('Session_project_folder')
    try:
        model_path = os.path.join(Session_project_folder,'trained_models', f'{model_name}.pkl')

        if not os.path.exists(model_path):
            return jsonify({'error': 'Model not found.'}), 404

        return send_file(model_path, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_column_analysis', methods=['GET'])
def get_column_analysis():
    try:
        model_name = request.args.get('model_name')

        # Load the dataset used for training the selected model
        dataset_path = f"uploads/{model_name}.csv"
        df = pd.read_csv(dataset_path)

        # Perform column analysis
        column_analysis = []

        for column in df.columns:
            analysis_result = {
                'column': column,
                'row_count': len(df),
                'missing_records': df[column].isnull().sum(),
                # You can include more analysis here based on your requirements
            }
            column_analysis.append(analysis_result)

        return jsonify(column_analysis)

    except Exception as e:
        print(f"Error in get_column_analysis: {str(e)}")
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/deploy_setup', methods=['GET', 'POST'])
@login_required
def d_model():
    model_name = request.form.get('model_name')
    user_name = request.form.get('user_name')
    username = current_user.get_id()
    print(f"Deploy_level Session: {session}")

    return render_template('deploy_setup.html', user_name=user_name, model_name=model_name, username =username )

@app.route('/save_and_predict_api', methods=['POST'])
def save_and_predict_api():
    try:
        # Extract information from the request
        model_name = request.form.get('model_name')
        user_name = request.form.get('user_name')
        
        # Save the model file
        save_model_result = save_model(request.files['model_file'], user_name, model_name)
        
        if save_model_result['error']:
            return jsonify(save_model_result), 400  # Return an error response if saving the model fails

        # Continue with prediction API logic
        prediction_result = perform_prediction(request.files['csv_file'], user_name, model_name)
        

        return jsonify({
            'message': 'Model saved and prediction completed.',
            'prediction_result': prediction_result,
            'model_name': model_name,
            'user_name': user_name,
        })

    except Exception as e:
        print(f"Error in save_and_predict_api: {str(e)}")
        return jsonify({'error': str(e)}), 500

def save_model(model_file, user_name, model_name):
    # Implement the logic to save the model to the production folder
    # Return a dictionary with 'error' key indicating success or failure
    Session_project_folder = session.get('Session_project_folder')
    project_name = session.get('project_name')
    try:
        # Customize this based on your file saving logic
        production_folder_path = f"{Session_project_folder}/'trained_models/{model_name}"
        os.makedirs(production_folder_path, exist_ok=True)
        model_file.save(os.path.join(production_folder_path, secure_filename(model_file.filename)))
        
        return {'error': None}
    except Exception as e:
        return {'error': str(e)}

def perform_prediction(csv_file, user_name, model_name):
    # Implement the logic to use the saved model for prediction
    # Return the prediction result
    try:
        # Customize this based on your prediction logic
        # Load the saved model from the production folder
        model_path = f"production_models/{user_name}/{model_name}/{model_name}.h5"
        loaded_model = load_model(model_path)
        
        # Perform prediction using the loaded model and the CSV file
        prediction_result = make_prediction(loaded_model, csv_file)
        
        return prediction_result
    except Exception as e:
        print(f"Error in perform_prediction: {str(e)}")
        return {'error': str(e)}

#
    #
    #
    #
    #
    #
    #
UPLOAD_FOLDER = 'api_upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'csv'}  # Set the allowed file extensions


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#
#
#
#



@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        print("Received predict API request")

        # Check if the 'file' part is present in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request.'}), 400

        dataset_file = request.files['file']

        # Check if the file is an allowed file type
        if not allowed_file(dataset_file.filename):
            return jsonify({'error': 'Invalid file format. Please upload a CSV file.'}), 400

        # Retrieve other parameters from the form
        user_name = request.form.get('user_name')
        model_filename = request.form.get('model_filename')
        model_folder = request.form.get('model_folder_path')

        # Generate a timestamp for subfolder creation
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

        # Create subfolders based on username, model_filename, and timestamp
        user_folder = os.path.join(app.config['UPLOAD_FOLDER'], user_name)
        model_folder = os.path.join(user_folder, model_filename)
        timestamp_folder = os.path.join(model_folder, timestamp)

        # Create subfolders if they don't exist
        for folder in [user_folder, model_folder, timestamp_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        # Save the file to the timestamped folder
        filename = secure_filename(dataset_file.filename)
        file_path = os.path.join(timestamp_folder, filename)
        dataset_file.save(file_path)

        # Load the pre-trained machine learning model
        model_path = os.path.join(model_folder, model_filename)
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model not found for the specified user.'}), 404

        model = joblib.load(model_path)

        # Perform any necessary preprocessing on the input data
        # Replace the following lines with your actual preprocessing code
        input_data = preprocess_data(file_path)

        # Make predictions using the loaded model
        prediction = model.predict(input_data)

        # Perform any other necessary actions with the prediction result
        # For example, save the prediction or update a database

        return jsonify({
            'message': 'Prediction completed.',
            'prediction_result': prediction.tolist(),
            'user_name': user_name,
            'model_filename': model_filename,
            'timestamp': timestamp,
        })

    except Exception as e:
        print(f"Error in predict_api: {str(e)}")
        return jsonify({'error': str(e)}), 500


def preprocess_data(file_path):
    # Placeholder for data preprocessing logic
    # Replace this with your actual data preprocessing code
    # Here, it's assumed that the CSV file contains three features: feature1, feature2, feature3
    # You may need to adjust this based on your actual data
    data = pd.read_csv(file_path)
    # Assuming three features: feature1, feature2, feature3
    input_data = data
    return input_data


# Dictionary to store deployed models and their endpoints
deployed_models = {}


@app.route('/deploy/<username>/<model_filename>', methods=['POST'])
@login_required
def deploy_model(username, model_filename):
    try:

        # Construct the model_endpoint
        model_endpoint = f"{model_filename}_{username}"
        Session_project_folder = session.get('Session_project_folder')
        # print(model_endpoint)

        # Get the folder path from the previous page
        folder_path = os.path.join(Session_project_folder,'trained_models')
        print(folder_path)

        # Add the ".pkl" extension to the model filename
        model_filename = f"{model_filename}.pkl"

        # Combine folder path and model filename to get the model path
        model_path = os.path.join(folder_path, model_filename)
        # print(model_path)
        print(f'Deploy route: {model_path}')

        # Check if the model file exists
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model file does not exist.'}), 400

        # return jsonify({'message': f'Model deployed at endpoint: /{model_endpoint}'})
        # Load the deployed model using pickle
        with open(model_path, 'rb') as file:
            deployed_models[model_endpoint] = pickle.load(file)

        print("Deployed Models:", deployed_models)
        session['Session_model_endpoint'] = f'{model_endpoint}'


        # Create an api.json file with relevant information
        api_info = {
            'username': username,
            'date_of_deployment': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_trainer_class': session.get('trainer_class'),  # Replace with your actual model trainer class
            'model_name': model_filename,
            'target_column_name': session.get('target_column'),  # Replace with your actual target column name
            'api_required_fields': session.get('API_Required_Fields'),  # Replace with the required fields for your API
            'api_endpoint': session.get('api_link'),  # You can set this based on your conditions
            'api_status': 'Active' if session.get('api_link') is not None else 'Inactive'  #  conditional expression
        }
        
        # Set the path for the user's folder
        Session_model_folder = session.get('Session_model_folder')       
        

        # Create the api.json file or append to existing file
        api_file_path = os.path.join(Session_model_folder, 'api.json')
        if os.path.exists(api_file_path):
            with open(api_file_path, 'r') as existing_file:
                existing_data = json.load(existing_file)
            existing_data.append(api_info)
            with open(api_file_path, 'w') as api_file:
                json.dump(existing_data, api_file, indent=2)
        else:
            with open(api_file_path, 'w') as api_file:
                json.dump([api_info], api_file, indent=2)

        print(f'API information saved at: {api_file_path}')
        print(f"Deploy_level Session: {session}")

        return jsonify({'message': f'{model_endpoint}'})


    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/<model_endpoint>', methods=['POST'])
def predict_with_deployed_model(model_endpoint):
    try:
        # Check if the requested model endpoint exists
        if model_endpoint not in deployed_models:
            return jsonify({'error': f'Model at endpoint /{model_endpoint} not found.'}), 404

        # Get input data from the request
        input_data = request.get_json()

        # Perform any necessary preprocessing on the input data
        # Replace the following lines with your actual preprocessing code
        # For example, convert input_data into the format expected by the model

        # Make predictions using the deployed model
        prediction = deployed_models[model_endpoint].predict([input_data])
        print(f"Deploy_level Session: {session}")

        # Return the prediction as JSON
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# # Define the user-specific folders
# user_projects_folder = session.get('user_projects_folder')
# Session_model_path = session.get('Session_model_path')

# Function to get the count of files in a folder
def get_files_count(folder_path):
    if os.path.exists(folder_path):
        return len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])
    return 0

# Function to get the count of subfolders in a folder
def get_folders_count(folder_path):
    if os.path.exists(folder_path):
        return len([name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))])
    return 0

# Function to read API data from a JSON file
def read_api_data():
    Session_project_folder= session.get('Session_project_folder')
    api_data_path = os.path.join((Session_project_folder), 'api.json')
    if os.path.exists(api_data_path):
        with open(api_data_path, 'r') as file:
            api_data = json.load(file)
            api_count = len(api_data)  # Count the number of rows
            return {'api_count': api_count}
    return {'api_count': 0}




@app.route('/get_trained_models')
def get_trained_models():
    trained_models = []

    # Fetch the list of trained models from the user_trained_models folder
    trained_models_folder = 'user_trained_models'
    if os.path.exists(trained_models_folder):
        for filename in os.listdir(trained_models_folder):
            if filename.endswith(".pkl"):
                model_name = os.path.splitext(filename)[0]
                trained_models.append({"name": model_name, "filename": filename})

    return jsonify({"trainedModels": trained_models})


@app.route('/get_api_info')
def get_api_info():
    try:
        # Fetch API information from your backend or any other data source
        api_info = session.get('api_info')
        return jsonify({"apiInfo": api_info})
    except Exception as e:
        # Handle exceptions
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500





@app.errorhandler(500)
def internal_server_error(error):
    # Log the error or perform any necessary actions
    app.logger.error(f"Internal Server Error: {error}")
    return jsonify({"error": "Internal Server Error"}), 500






# if __name__ == '__main__':
   

#     app.run(debug=True, host='172.20.10.5', port=8100, use_reloader=False)
# if __name__ == '__main__':
#     options = {
#         'bind': '0.0.0.0:8100',  # Specify the IP address and port
#         'workers': 4,  # You can adjust this based on your needs
#     }
#     StandaloneApplication(app, options).run()