from database.db import mongo

import random

def generate_unique_user_id():
    """Generate a unique user_id between 1-100."""
    while True:
        user_id = random.randint(1, 100)
        if not User.find_user_by_id(user_id):  # Ensure it's unique
            return user_id
class User:
    """
    User Schema:
    - user_id (int): Unique identifier for the user
    - name (str): Full name of the user
    - email (str): Email address of the user
    - password (str): Hashed password
    - stitched_images (list): List of stitched images (initially empty)
    """
    
    @staticmethod
    def create_user( name: str, email: str, hashed_password: str):
        user_id = generate_unique_user_id()
        user = {
            "user_id": user_id,
            "name": name,
            "email": email,
            "password": hashed_password,  # Storing hashed password
            "stitched_images": []  # Initially empty
        }
        mongo.db.users.insert_one(user)
    
    @staticmethod
    def find_user_by_email(email: str):
        return mongo.db.users.find_one({"email": email})
    @staticmethod
    def find_user_by_id(user_id: str):
        return mongo.db.users.find_one({"user_id":user_id})

    @staticmethod
    def find_all_users():
        return list(mongo.db.users.find({}, {"_id": 0, "password": 0}))

    @staticmethod
    def get_user_images(email: str):
        user = mongo.db.users.find_one({"email": email}, {"_id": 0, "stitched_images": 1})
        return user["stitched_images"] if user else []
    @staticmethod
    def update_user_images(email: str, stitched_image_path: str):
    # """Update the user's stitched_images list by adding a new image path."""
        mongo.db.users.update_one(
            {"email": email},  # Find user by email
            {"$push": {"stitched_images": stitched_image_path}}  # Append new image path
        )
