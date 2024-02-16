from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS, cross_origin
from model_trainer import ModelTrainer
from reg_model_trainer import RegModelTrainer
import os
import pandas as pd
import threading
from werkzeug.utils import secure_filename
import logging
import json

app = Flask(__name__, static_url_path='/static')
CORS(app, support_credentials=True)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Set the path to store uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create dictionaries to hold trainers for each thread
trainers = {}
reg_trainers = {}

# Initialize the ModelTrainer and RegModelTrainer
trainer = ModelTrainer(None, None)
reg_trainer = RegModelTrainer(None, None)

# Store the evaluation results globally
evaluation_results = {}

ALLOWED_EXTENSIONS = {'csv'}  # Set of allowed file extensions


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_or_create_trainer(thread_id, dataset_file, target_column, trainer_class):
    if thread_id not in trainers:
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


@app.route('/train')
def render_train():
    return render_template('train_models.html')


@app.route('/train_models', methods=['POST'])
@cross_origin(supports_credentials=True)
def train_models():
    
    try:
        print("Received train_models request")
        

        # Get the JSON data from the request body
        # data = request.get_json()

        # Check if the required fields are present
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request.'}), 400
        

        
        dataset_file = request.files['file']
        # print(f"Datafile: {dataset_file}")

        # Check if the file extension is allowed
        if not allowed_file(dataset_file.filename):
            return jsonify({'error': 'Invalid file format. Please upload a CSV file.'}), 400
        

        # Save the file to the UPLOAD_FOLDER
        filename = secure_filename(dataset_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        dataset_file.save(file_path)



        # if dataset_file and allowed_file(dataset_file.filename):  # assuming you have an `allowed_file` function
        #     # Save the file to a specific location
        #     file_path = os.path.join(app.root_path, secure_filename(dataset_file.filename))
        #     dataset_file.save(file_path)
        # print(f"File Path: {file_path}")
        target_column = request.form.get('target_column')
        trainer_class = request.form.get('trainer_class')
        # print(trainer_class)



        # # Get the uploaded file
        # dataset_file = request.files['file']
        # print(dataset_file)

        # # Check if the file is present and allowed
        # if dataset_file:
        #     # Save the file to a specific location
        #     file_path = os.path.join(app.root_path, secure_filename(dataset_file.filename))
        #     dataset_file.save(file_path)
        # print(f"File Path: {file_path}")
        # print(f"File: {dataset_file}")

        # target_column = request.form.get['target_column']
        # trainer_class = request.form.get['trainer_class']

        # print(request.files)
        # print(request.form)
        # print(f"Target Column: {target_column}")
        # print(f"Trainer Class: {trainer_class}")

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

        # Include model filename in the response
        # print(evaluation_result_df.to_dict(orient='records'))
        # print(evaluation_result_df.to_json(orient='records'))
        # print('---------------------------------')
        # print(evaluation_result_df.to_dict(orient='records'))
            
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
        
        

        # return jsonify({
        #     'message': 'Models trained and best model saved.',
        #     'best_model_info': best_model_info,
        #     'evaluation_result': evaluation_result,
        #     'accuracy': accuracy if trainer_class == 'Classification' else rmse_metrics,
        #     'redirect_url': '/gmodels'
        # })

    except Exception as e:
        print(f"Error in train_models: {str(e)}")
        return jsonify({'error': str(e)}), 500


# @app.route('/gmodels')
# def render_gmodels():
#     return render_template('get_models.html')


# @app.route('/get_models', methods=['GET'])
# def get_models():
    # try:
    #     # global trainer
    #     global trainer
    #     # print(trainer)
    #     models_dir = 'trained_models'
    #     # print(models_dir)
    #     if not os.path.exists(models_dir):
    #         return jsonify({'models': []})

    #     model_files = os.listdir(models_dir)
    #     # print(model_files)
    #     model_names = [os.path.splitext(filename)[0] for filename in model_files]

    #     global evaluation_results
    #     print(evaluation_results)
    #     evaluation_result = evaluation_results.get('evaluation_result', {})
    #     print(evaluation_result)
    #     trainer_accuracy_df = pd.DataFrame(evaluation_result)
    #     print(trainer_accuracy_df)

    #     try:
    #         models_with_accuracy = trainer_accuracy_df.sort_values(by='Accuracy', ascending=False)
    #     except KeyError:
    #         print("Error: 'Accuracy' column not found in the DataFrame.")
    #         models_with_accuracy = pd.DataFrame()

    #     return jsonify({'models': models_with_accuracy.to_dict(orient='records')})

    # except Exception as e:
    #     print(f"Error in get_models: {str(e)}")
    #     return jsonify({'error': 'Internal Server Error'}), 500


@app.route('/gmodels')
def render_gmodels():
    return render_template('get_models.html')

@app.route('/get_models', methods=['GET'])
def get_models():
    try:
        # global trainer
        global trainer
        # Add debug print statements to check the trainer object

        models_dir = 'trained_models'
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

        # Sort the DataFrame by the 'Accuracy' column
        try:
            models_with_accuracy = trainer_accuracy_df.sort_values(by='Accuracy', ascending=False)
        except KeyError:
            # If the 'Accuracy' column is not present in the DataFrame
            print("Error: 'Accuracy' column not found in the DataFrame.")
            models_with_accuracy = pd.DataFrame()

        return jsonify({'models': models_with_accuracy.to_dict(orient='records')})
    

    except Exception as e:
        # Log the exception for debugging
        print(f"Error in get_models: {str(e)}")

        # Return an error response
        return jsonify({'error': 'Internal Server Error'}), 500



    return render_template('evaluate_model.html')



@app.route('/cre_dict', methods=['GET', 'POST'])
def pre_dict():
    model_name = request.form.get('model_name')
    return render_template('pre_dict.html', model_name=model_name)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Received predict request")

        # Check if the required fields are present
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request.'}), 400

        dataset_file = request.files['file']

        # Check if the file extension is allowed
        if not allowed_file(dataset_file.filename):
            return jsonify({'error': 'Invalid file format. Please upload a CSV file.'}), 400

        # Save the file to the UPLOAD_FOLDER
        filename = secure_filename(dataset_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        dataset_file.save(file_path)

        # Retrieve other necessary parameters (e.g., model name, model file path, etc.) from the request or session
        model_name = request.form.get('model_name')  # Assuming the model name is passed as a form field

        # Load the trained model using the model name and file path
        prediction_result = trainer.predict(file_path, model_name)
        # This step will depend on how you manage your trained models. 
        # It could involve loading a serialized model or re-training a model with the same parameters and dataset.

        return jsonify({
            'message': 'Prediction completed.',
            'prediction_result': prediction_result,
            'model_name': model_name,
            'redirect_url': '/results'  # Redirect URL after prediction
        })

    except Exception as e:
        print(f"Error in predict: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict_api', methods=['POST'])
def predict_using_api():
    try:
        print("Received predict request")

        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request.'}), 400

        dataset_file = request.files['file']

        if not allowed_file(dataset_file.filename):
            return jsonify({'error': 'Invalid file format. Please upload a CSV file.'}), 400

        filename = secure_filename(dataset_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        dataset_file.save(file_path)

        user_name = request.form.get('user_name')
        model_name = request.form.get('model_name')
        timestamp = request.form.get('timestamp')

        trainer = get_trainer_by_model_name(user_name, model_name)

        if trainer is None:
            return jsonify({'error': 'Model not found for the specified user.'}), 404

        trainer.load_model_from_timestamp(model_name, timestamp)
        prediction_result = trainer.predict(file_path, model_name)

        register_prediction(user_name, model_name, prediction_result)

        return jsonify({
            'message': 'Prediction completed.',
            'prediction_result': prediction_result,
            'model_name': model_name,
            'user_name': user_name,
            'timestamp': timestamp,
        })

    except Exception as e:
        print(f"Error in predict_using_api: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/result', methods=['GET', 'POST'])
def result():
    try:
        data = request.args.get('prediction_result')
        print(request.url)
        print(request.args)
        print(f"Received prediction_result: {data}")

        if not data:
            return jsonify({'error': 'No prediction result provided.'}), 400

        parsed_data = json.loads(data)

        if 'Predictions' not in parsed_data or 'Probabilities' not in parsed_data:
            return jsonify({'error': 'Invalid prediction result format.'}), 400

        return render_template('result.html',
                               predictions=parsed_data['Predictions'],
                               probabilities=parsed_data['Probabilities'],
                               model_name=parsed_data.get('model_name', 'Not Provided'))

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
    try:
        model_path = os.path.join('trained_models', f'{model_name}.pkl')

        if not os.path.exists(model_path):
            return jsonify({'error': 'Model not found.'}), 404

        return send_file(model_path, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_column_analysis', methods=['GET'])
def get_column_analysis():
    try:
        model_name = request.args.get('model_name')

        dataset_path = f"uploads/{model_name}.csv"
        df = pd.read_csv(dataset_path)

        column_analysis = []

        for column in df.columns:
            analysis_result = {
                'column': column,
                'row_count': len(df),
                'missing_records': df[column].isnull().sum(),
            }
            column_analysis.append(analysis_result)

        return jsonify(column_analysis)

    except Exception as e:
        print(f"Error in get_column_analysis: {str(e)}")
        return jsonify({'error': 'Internal Server Error'}), 500


@app.errorhandler(500)
def internal_server_error(error):
    app.logger.error(f"Internal Server Error: {error}")
    return jsonify({"error": "Internal Server Error"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9100, use_reloader=False)
