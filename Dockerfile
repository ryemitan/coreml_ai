# Use the official Python image as the base image
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Set the environment variable
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Install any dependencies specified in requirements.txt
RUN cat requirements.txt
RUN pip install --no-cache-dir --no-deps -r requirements.txt

# Conditionally install pywin32 only if on Windows
# RUN if [ "$OS" = "Windows_NT" ]; then pip install pywin32; fi

# Copy the content of the local src directory to the /app directory
COPY . /app/

# Expose port 8100 for the Flask app to listen on
EXPOSE 8100

# Define environment variable for Flask to run in production mode
ENV FLASK_ENV=production

# Command to run the application
CMD ["python", "app_app.py"]
