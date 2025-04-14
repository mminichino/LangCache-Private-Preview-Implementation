import os
import json
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# Get GitHub token from environment variable
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', 'YOUR_GITHUB_TOKEN_HERE')
AZURE_ENDPOINT = "https://models.inference.ai.azure.com"
MODEL_NAME = "text-embedding-3-small"

@app.route('/v1/embeddings', methods=['POST'])
def generate_embeddings():
    """Generate embeddings using Azure-hosted OpenAI embeddings service with GitHub token"""
    try:
        # Get input text from request
        data = request.json
        input_text = data.get('input', '')
        
        # Ensure input is in the correct format
        if isinstance(input_text, str):
            input_text = [input_text]
        
        # Call Azure-hosted OpenAI embeddings service
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {GITHUB_TOKEN}'
        }
        
        payload = {
            'input': input_text,
            'model': MODEL_NAME
        }
        
        response = requests.post(
            f"{AZURE_ENDPOINT}/embeddings",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            print(f"Error from Azure OpenAI API: {response.status_code} - {response.text}")
            return jsonify({
                'error': {
                    'message': f"Error from Azure OpenAI API: {response.status_code} - {response.text}",
                    'type': 'api_error',
                    'code': response.status_code
                }
            }), 500
        
        # Extract embeddings from response
        result = response.json()
        
        # Format response to match OpenAI API format
        formatted_response = {
            'data': [
                {
                    'embedding': item['embedding'],
                    'index': item['index'],
                    'object': 'embedding'
                }
                for item in result['data']
            ],
            'model': MODEL_NAME,
            'object': 'list',
            'usage': result['usage']
        }
        
        return jsonify(formatted_response)
    
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return jsonify({
            'error': {
                'message': f"Error generating embeddings: {e}",
                'type': 'server_error',
                'code': 500
            }
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
