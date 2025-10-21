# api_clients.py
import requests
from typing import Optional, Tuple

class DiffuserApiClient:
    """
    A client for interacting with the Diffuser API server.
    It encapsulates the logic for making network requests and handling responses.
    """
    def __init__(self, api_url: str):
        """
        Initializes the client with the server's API endpoint URL.

        Args:
            api_url (str): The full URL of the API endpoint (e.g., "http://localhost:8000/generate").
        """
        self.api_url = api_url

    def generate(self, payload: dict) -> Optional[Tuple[str, str]]:
        """
        Sends a payload to the diffuser server to generate an image.

        Args:
            payload (dict): A dictionary containing all request data, such as dialogue_lines,
                            refine_prompt, initial_image_b64, and parameters.

        Returns:
            A tuple of (prompt, image_b64) on success, or None if any error occurred.
        """
        try:
            print(f"Sending request to {self.api_url}...")
            # Send the HTTP POST request. The payload is automatically encoded as JSON.
            # A long timeout is set because the server process (refinement + diffusion) can take time.
            response = requests.post(self.api_url, json=payload, timeout=400)
            
            # This is a crucial check: it will raise an HTTPError if the server responded
            # with an error status code (e.g., 404 Not Found, 500 Internal Server Error).
            response.raise_for_status()
            
            # Parse the JSON response from the server into a Python dictionary.
            response_json = response.json()

            # Validate that the response contains the data we expect.
            if "prompt" in response_json and "image_b64" in response_json:
                # If valid, return the prompt and the image data as a tuple.
                return response_json["prompt"], response_json["image_b64"]
            else:
                # If the response is malformed, log an error and return None.
                print(f"API Error: 'prompt' or 'image_b64' not found in response. Response: {response_json}")
                return None
        
        # This block catches any network-related errors (e.g., connection refused, timeout).
        except requests.exceptions.RequestException as e:
            print(f"Diffuser API Client Error: {e}")
            return None
