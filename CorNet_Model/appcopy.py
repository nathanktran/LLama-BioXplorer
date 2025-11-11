from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess


app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
CORS(app, resources={r"/*": {"origins": "*"}})


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


@app.route('/')
def hello():
    return "Hello, World!"


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204

    data = request.json
    abstract_text = data.get('abstract_text', '').replace('\n', ' ').replace('\r', ' ')
    title = data.get('title', '').replace('\n', ' ').replace('\r', ' ')
   
    try:
        process = subprocess.Popen(
            ['python', 'sneha_pre3.py', abstract_text, title],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        stdout, stderr = process.communicate()
       
        if process.returncode != 0:
            return jsonify({'error': stderr}), 500
       
        # Try to parse JSON output, fallback to original format
        try:
            import json
            result = json.loads(stdout.strip())
            return jsonify(result)
        except json.JSONDecodeError:
            # Fallback to original behavior if JSON parsing fails
            return jsonify({'prediction': stdout.strip()})
   
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)