from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess

app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}) 
# CORS(app, resources={r"/Name": {"origins": "http://localhost:3000"}})
# CORS(app, resources={r"/*": {"origins": "*"}})
# CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://localhost:5000", "*"]}})
#CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://localhost:5001"]}})
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "http://localhost:3000"}})


@app.route('/')
def hello():
    return "Hello, World!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    abstract_text = data.get('abstract_text', '')
    title = data.get('title', '')
    
    try:
        # Use Popen with universal_newlines for compatibility with older Python versions
        process = subprocess.Popen(
            ['python', 'sneha_pre2.py', abstract_text, title],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True  # Use universal_newlines for text mode
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            return jsonify({'error': stderr}), 500
        
        return jsonify({'prediction': stdout.strip()})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


