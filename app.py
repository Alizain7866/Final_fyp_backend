from flask import Flask
from config import Config
from flask_session import Session
from config import Config
from database.db import init_db
from flask_mail import Mail, Message
from routes import routes
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import os
app = Flask(__name__, static_folder='static')
# app.config["UPLOAD_FOLDER"] = os.path.join(os.getcwd(), "upload_folder")

app.config.from_object(Config)  # Load configurations

Session(app)


#session

# app.config["SESSION_TYPE"] = "filesystem"  # Can be 'redis', 'memcached', 'mongodb', etc.
# app.config["SESSION_PERMANENT"] = True
# app.config["SESSION_USE_SIGNER"] = True
# app.config["SESSION_FILE_DIR"] = "./flask_session_data"  # Directory for storing session files
# Session(app) 




mail = Mail(app)

#session








app.config['SECRET_KEY'] = 'dev_secret_key'  # Or use a more secure key
print(app.config['SECRET_KEY'])  # Make sure it prints the correct key

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)  # Create the uploads folder if it doesn't exist

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

CORS(app, resources={r"/uploads": {"origins": "http://localhost:3000"},
                     r"/stitching": {"origins": "http://localhost:3000"},
                     r"/login": {"origins": "http://localhost:3000"},
                     r"/get-final-image": {"origins": "http://localhost:3000"},
                     r"/download-final-image": {"origins": "http://localhost:3000"},
                     r"/send-email":{"origins":"http://localhost:3000"},
                     r"/register":{"origins":"htpp://localhost:3000"},
                     r"/delete-final-image":{"origins":"htpp://localhost:3000"}


                     }, supports_credentials=True)   

Session(app)

db = init_db(app)


app.register_blueprint(routes)

if __name__ == "__main__":
    app.run(debug=True)
