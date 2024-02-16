from flask_login import UserMixin
import json
import secrets
from werkzeug.security import generate_password_hash, check_password_hash
from passlib.hash import scrypt

class User(UserMixin):
    def __init__(self, username):
        self.id = username

    def get_id(self):
        return self.id
    
    def is_authenticated(self):
        return True  # Replace with your authentication logic

    @classmethod
    def hash_password(cls, password):
        """
        Class method to hash a password. Use it when creating a new user.
        """
        return generate_password_hash(password)
    
    def check_password(self, password):
        """
        Check if the provided password matches the stored hashed password.
        """
        # Replace scrypt with the appropriate hashing function if you switch from scrypt to another algorithm
        return check_password_hash(self.get_hashed_password(), password)

    def get_hashed_password(self):
        """
        Retrieve the hashed password from the user object.
        """
        # Replace this with your actual logic to retrieve the hashed password from the user object
        # This might involve accessing a database or another storage mechanism
        # Assuming users are stored in a JSON file
        with open('users.json', 'r') as file:
            users = json.load(file)

        return users.get(self.id, {}).get('password') # Replace with your actual implementation

