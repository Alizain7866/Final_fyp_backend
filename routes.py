from flask import Blueprint, request, jsonify, session,current_app, send_file, send_from_directory
from flask_cors import cross_origin
from flask_bcrypt import Bcrypt
from models import User
from flask_mail import Mail, Message

import random
import os
import shutil
import subprocess
from flask_session import Session

user_static_dictionary = {}

bcrypt = Bcrypt()
routes = Blueprint("routes", __name__)
mail = Mail()
STATIC_FOLDER = os.path.join(os.getcwd(), "static")  # Get absolute path to static/
print(STATIC_FOLDER)
# UPLOAD_FOLDER = "uploads"

# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)  # Create the uploads folder if it doesn't exist

# current_app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}


# def generate_unique_user_id():
#     """Generate a unique user_id between 1-100."""
#     while True:
#         user_id = random.randint(1, 100)
#         if not User.find_user_by_id(user_id):  # Ensure it's unique
#             return user_id



# routes.config["SESSION_TYPE"] = "filesystem"  # Can be 'redis', 'memcached', 'mongodb', etc.
# routes.config["SESSION_PERMANENT"] = False
# routes.config["SESSION_USE_SIGNER"] = True
# routes.config["SESSION_FILE_DIR"] = "./flask_session_data"  # Directory for storing session files
# Session(routes) 


@routes.route("/register", methods=["POST"])
@cross_origin(origins="http://localhost:3000")

def register():
    #   user = {
    #         "user_id": user_id,
    #         "name": name,
    #         "email": email,
    #         "phone-number":phone_number,
    #         "password": hashed_password,  # Storing hashed password
    #         "stitched_images": []  # Initially empty
    #     }
    data = request.json
    print(data)
    if not data.get("email") or not data.get("password") or not data.get("name"):
        return jsonify({"error": "All fields (name, email, password) are required"}), 400

    if User.find_user_by_email(data["email"]):
        return jsonify({"error": "User already exists"}), 400
    
    # user_id = generate_unique_user_id()


    hashed_password = bcrypt.generate_password_hash(data["password"]).decode("utf-8")
    # User.create_user(data["name"], data["email"],data["phone_number"], hashed_password)
    User.create_user(data["name"], data["email"], hashed_password, data["phone_number"])

    
    return jsonify({"message": "User registered successfully"} ), 201

@routes.route("/login", methods=["POST"])
def login():
    data = request.json
    user = User.find_user_by_email(data["email"])
    
    if not user or not bcrypt.check_password_hash(user["password"], data["password"]):
        return jsonify({"error": "Invalid credentials"}), 401

    # session["user"] = {"user_id": user["user_id"], "name": user["name"], "email": user["email"], "stitched_images":user["stitched_images"]}
   
    return jsonify({
        "message": "Login successful",
        "user": {
            "user_id": user["user_id"],
            "name": user["name"],
            "email": user["email"],
            "stitched_images": user["stitched_images"]
        }
    }), 200

@routes.route("/users", methods=["GET"])
def get_all_users():
    users = User.find_all_users()
    return jsonify(users), 200

@routes.route("/user/images", methods=["GET"])
def get_user_images():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    email = session["user"]["email"]
    images = User.get_user_images(email)
    return jsonify(images), 200


# @routes.route("/stitching", methods=["POST"])
# @cross_origin(origins="http://localhost:3000")
# def upload_images():
#     """Endpoint to upload images."""
#     print("Reached here I have")
#     data = request.get_json()
#     email = data.get("email")
#     user= User.find_user_by_email(email)
#     email1 = email.split("@")
#     email2 = email[0]

#     # Access the app's configuration using current_app
#     upload_folder = current_app.config["UPLOAD_FOLDER"]

#     # Ensure the folder exists
#     # if not os.path.exists(upload_folder):
#     #     os.makedirs(upload_folder)

#     # # Clear the folder if not empty
#     # if os.listdir(upload_folder):
#     #     shutil.rmtree(upload_folder)
#     #     os.makedirs(upload_folder)

#     # # Save uploaded images
#     # images = request.files.getlist("images")
#     # for image in images:
#     #     image.save(os.path.join(upload_folder, image.filename))

#     # Execute the Python script to count images
#     result = subprocess.run(["python", "count_images.py", email2], check=True, text=True, capture_output=True)

#     # Empty the folder after running the script
#     # if os.listdir(upload_folder):
#     #     shutil.rmtree(upload_folder)
#     #     os.makedirs(upload_folder)
#     stitched_image_path = result.stdout.strip()

#     # User.update_one(
#     #         {"email": email},
#     #         {"$push": {"stitched_images": stitched_image_path}},  # Append to existing array
#     #         upsert=True  # Create user record if not exists
#     #     )
#     print(stitched_image_path)

#     print("BALALALALALALALall")
#     if os.listdir(upload_folder):
#         shutil.rmtree(upload_folder)
#         os.makedirs(upload_folder)

#     User.update_one(
#     {"email": user.email},  # Use the correct user email from the `user` object
#     {"$push": {"stitched_images": stitched_image_path}},  # Append to existing array
#     upsert=True  # Create user record if it doesn't exist
# )
#     return jsonify({"message": "Images uploaded and processed successfully."}), 200



@routes.route("/stitching", methods=["POST"])
@cross_origin(origins="http://localhost:3000")
def upload_images():
    """Endpoint to upload images."""
    print("Reached here I have")
    
    # Get JSON data from request
    data = request.json
    email = data.get("email")
    
    if not email:
        return jsonify({"error": "Email is required"}), 400

    user = User.find_user_by_email(email)
    
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    email1 = email.split("@")
    email2 = email1[0]  # Corrected email extraction
    
    # Access the app's configuration using current_app
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    
    # Ensure the folder exists
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    # Execute the Python script to count images
    result = subprocess.run(["python", "count_images.py", email2], check=True, text=True, capture_output=True)
    
    stitched_image_path = result.stdout.strip()
    print(stitched_image_path)

    # Ensure the folder is emptied after processing
    if os.listdir(upload_folder):
        shutil.rmtree(upload_folder)
        os.makedirs(upload_folder)

    # Update user record in the database
    User.update_user_images(email, stitched_image_path)  # Correct update method


    return jsonify({"message": "Images uploaded and processed successfully.", "stitched_image": stitched_image_path}), 200


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@routes.route("/uploads", methods=["POST"])
@cross_origin(origins="http://localhost:3000")
def upload_files():
    upload_folder = current_app.config["UPLOAD_FOLDER"]

    if "files" not in request.files:
        return jsonify({"message": "No file part in the request"}), 400

    files = request.files.getlist("files")
    if not files or all(file.filename == "" for file in files):
        return jsonify({"message": "No selected files"}), 400

    # Check if upload folder is not empty, then clear it
    if os.path.exists(upload_folder) and os.listdir(upload_folder):
        for existing_file in os.listdir(upload_folder):
            existing_file_path = os.path.join(upload_folder, existing_file)
            if os.path.isfile(existing_file_path):
                os.remove(existing_file_path)

    saved_files = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)
            saved_files.append(filename)
        else:
            return jsonify({"message": f"File {file.filename} is not allowed"}), 400

    return jsonify({"message": "Files uploaded successfully", "uploaded_files": saved_files})





@routes.route("/see-uploads", methods=["GET"])
@cross_origin(origins="http://localhost:3000")
def list_uploaded_images():
    try:
        upload_folder = current_app.config["UPLOAD_FOLDER"]
        base_url = request.host_url  # Gets http://localhost:5000/

        images = [
            {"name": filename, "url": f"{base_url}uploads/{filename}"}
            for filename in os.listdir(upload_folder)
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".gif"))
        ]

        return jsonify({"images": images}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@routes.route("/get-final-image", methods=["GET"])
@cross_origin(origins="http://localhost:3000")
def getimageoutput():
    name = request.args.get("email")
    if not name:
        return jsonify({"error": "Email is required"}), 400

    name = name.split("@")[0]  # Extract the username
    print(f"Processed username: {name}")

    base_url = request.host_url  # Gets http://localhost:5000/

    # Read the counter value
    try:
        with open("counter.txt", "r") as file:
            counter = file.read().strip()
    except Exception as e:
        return jsonify({"error": f"Failed to read counter file: {str(e)}"}), 500

    final_image_name = f"final_stitched_output_{counter}_{name}.jpg"
    final_image_path = os.path.join(current_app.static_folder, final_image_name)
    print(f"Final image path: {final_image_path}")

    if not os.path.exists(final_image_path):
        return jsonify({"error": f"Image {final_image_name} not found"}), 404

    final_image_url = f"{base_url}static/{final_image_name}"
    print(f"Returning Image URL: {final_image_url}")  # Debugging

    return jsonify({"imageUrl": final_image_url})



@routes.route("/download-final-image", methods=["GET"])
@cross_origin(origins="http://localhost:3000")
def download_final_image():
    name = request.args.get("email")
    if not name:
        return jsonify({"error": "Email parameter is required"}), 400

    name = name.split("@")[0]  # Extract name from email
    print(name)

    try:
        with open("counter.txt", "r") as file:
            counter = file.read().strip()
    except Exception as e:
        return jsonify({"error": f"Failed to read counter file: {str(e)}"}), 500
    
    STATIC_FOLDER = os.path.join(os.getcwd(), "static")  # Get absolute path to static/

    final_image_path = os.path.join(STATIC_FOLDER, f"final_stitched_output_{counter}_{name}.jpg")


    try:
        return send_file(final_image_path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500



#latest update
@routes.route("/api/get-images", methods=["GET"])
@cross_origin(origins="http://localhost:3000", supports_credentials=True)
def get_images():
    try:
        email = request.args.get("email")

        if not email:
            return jsonify({"error": "Email is required"}), 400

        user = User.find_images({"email": email})

        if not user:
            return jsonify({"images": []})  # Return empty list if user is not found

        return jsonify({"images": user.get("stitched_images", [])})  # Return user's images

    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500
    





@routes.route('/send-email', methods=['POST'])
def send_email():
    data = request.get_json()
    email = data.get('email')  # Get the email from frontend input
    subject = data.get('subject', 'Default Subject')
    message = data.get('message', 'Hello! This is a test email.')

    if not email:
        return jsonify({'error': 'Email address is required'}), 400

    try:
        msg = Message(subject, recipients=[email])
        msg.body = message
        mail.send(msg)
        return jsonify({'message': f'Email sent successfully to {email}'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@routes.route('/static/<filename>')
def serve_image(filename):
    current_app.config['STATIC_FOLDER'] = STATIC_FOLDER
    return send_from_directory(current_app.config['STATIC_FOLDER'], filename)