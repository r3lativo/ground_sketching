# api_clients.py
import requests
from typing import Optional

class DiffuserApiClient:
    def __init__(self, api_url: str):
        "Initializes the client with the server's API endpoint URL."
        self.api_url = api_url

    def generate(self, payload: dict) -> Optional[str]:
        "Sends a payload to the diffuser server to generate an image."
        try:
            print(f"Sending request to {self.api_url}...")
            # Increased timeout for requests that might involve text generation
            response = requests.post(self.api_url, json=payload, timeout=400)
            response.raise_for_status()
            
            response_json = response.json()

            if "prompt" in response_json and "image_b64" in response_json:
                return response_json["prompt"], response_json["image_b64"]
            else:
                print(f"API Error: 'prompt' or 'image_b64' not found in response. Response: {response_json}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Diffuser API Client Error: {e}")
            return None
