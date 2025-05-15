import io
import os
from dotenv import load_dotenv
from flask import Flask, request, make_response
from flask_cors import CORS
from langchain_core.load import loads
from werkzeug.utils import secure_filename
import boto3
from graphrag import chunk_and_store, generate_response
from s3_utils import get_s3_filenames

# Define global variables and load environment variables
app = Flask(__name__)
CORS(app)
load_dotenv()
s3 = boto3.client('s3',
                  aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
                  aws_secret_access_key=os.getenv('AWS_SECRET_KEY'))

# Only allow PDF files
ALLOWED_EXTENSIONS = {'pdf'}


# Only allow files of allowed type to be uploaded
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Deals with HTTP OPTIONS requests
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        res = make_response()
        res.headers['X-Content-Type-Options'] = '*'
        return res


# Backend API for uploading files to AWS S3 storage
@app.route("/upload", methods=['POST'])
def uploadPDF():
    if request.method == 'POST':
        # Check if the post request has a file part
        if len(request.files) == 0:
            return 'No file part!', 400
        uploaded_files = request.files.getlist("files[]")

        for file in uploaded_files:
            # Convert to bytes object for upload
            file_bytes = io.BytesIO(file.read())

            # Ensure the file has a valid and safe filename
            if file.filename == '':
                return 'No selected file', 400
            if not allowed_file(file.filename):
                return 'File type not permitted', 400
            filename = secure_filename(file.filename)

            # Use the boto3 API to upload the file to the bucket, under the pdf folder
            try:
                s3.upload_fileobj(
                    file,
                    os.getenv("AWS_BUCKET_NAME"),
                    "pdf/" + filename
                )
                print("File upload success for ", filename)
            except Exception as e:
                print("An error occurred during file upload: ", e)
                return e

            # Call the chunking algorithm on the file and store the chunks in a separate folder
            try:
                chunk_and_store(file_bytes, filename, s3)
            except Exception as e:
                print("An error occurred during chunking: ", e)
                return e
        return 'File upload success', 200


# Backend API for handling chat requests to the LLM
@app.route("/chat", methods=['POST'])
def chatbot():
    if request.method == 'POST':
        try:
            # Get the list of all chunks we have available for context
            filenames = get_s3_filenames(s3, prefix='chunks/', file_type='.json')
            chunks_contextual = []

            # For each chunk, retrieve the data and decode into usable format
            for chunk in filenames:
                txt_obj = s3.get_object(Bucket=os.getenv("AWS_BUCKET_NAME"), Key='chunks/' + chunk)
                content = txt_obj['Body'].read().decode('utf-8')
                chunks_contextual.append(loads(content))

            question = request.form['text']  # Get the user's question
            response = generate_response(question, chunks_contextual)  # Generate a response with the RAG LLM system
            return response, 200
        except Exception as e:
            print("An error occured during response generation: ", e)
            return e


# Backend API for handling displaying the files stored on AWS
@app.route("/files", methods=['GET'])
def files():
    if request.method == 'GET':
        # Use the boto3 API to retrieve all pdf filenames in the pdf folder
        try:
            filenames = get_s3_filenames(s3, prefix='pdf/', file_type='.pdf')
            return {"files": filenames}, 200
        except Exception as e:
            print("An error occurred: ", e)
            return e


# Test route, ignore
@app.route("/")
def test():
    return "<h1>Test!</h1>"


if __name__ == "__main__":
    app.run(debug=False)
