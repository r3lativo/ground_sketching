# generators.py
import requests
import base64
import cairosvg
from typing import Optional
import re

def extract_svg(raw_output: str) -> str:
    match = re.search(r"<svg.*?</svg>", raw_output, re.DOTALL)
    if match:
        return match.group(0)
    return ""

class DiffuserGeneratorClient:
    def __init__(self, api_url: str):
        self.api_url = api_url
        print(f"Diffuser client pointing to: {api_url}")

    def generate(self, payload: dict) -> Optional[str]:
        try:
            response = requests.post(self.api_url, json=payload, timeout=300)
            response.raise_for_status()
            return response.json().get("image_b64")
        except requests.exceptions.RequestException as e:
            print(f"API Error: {e}")
            return None

class SgpGeneratorClient:
    def __init__(self, api_url: str):
        self.api_url = api_url
        print(f"SGP client pointing to: {api_url}")

    def generate(self, payload: dict) -> Optional[str]:
        sgp_payload = {"prompt": payload.get("prompt")}
        try:
            response = requests.post(self.api_url, json=sgp_payload, timeout=300)
            response.raise_for_status()
            
            # 1. Get the raw text from the model
            raw_svg_text = response.json().get("svg_text", "")
            
            # 2. Clean the raw text to get pure SVG
            clean_svg_text = extract_svg(raw_svg_text)
            
            if not clean_svg_text:
                print("API Error: Model did not return valid SVG content.")
                return None

            # 3. Render the clean SVG to PNG
            png_bytes = cairosvg.svg2png(bytestring=clean_svg_text.encode('utf-8'))
            
            return base64.b64encode(png_bytes).decode('utf-8')
        except (requests.exceptions.RequestException, Exception) as e:
            print(f"API or SVG Rendering Error: {e}")
            return None
