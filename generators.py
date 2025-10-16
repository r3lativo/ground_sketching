# generators.py
import requests
import base64
import cairosvg
import re
from typing import Optional, List

def extract_svg(raw_output: str) -> str:
    match = re.search(r"<svg.*?</svg>", raw_output, re.DOTALL)
    if match:
        return match.group(0)
    return ""

def extract_answer(raw_output: str) -> str:
    match = re.search(r"<answer>*?</answer>", raw_output, re.DOTALL)
    if match:
        return match.group(0)
    return ""

class RefinerClient:
    def __init__(self, api_url: str):
        self.api_url = api_url

    def refine(self, dialogue_lines: List[str], prompt_template: str) -> Optional[str]:
        payload = {"dialogue_lines": dialogue_lines, "prompt_template": prompt_template}
        try:
            response = requests.post(self.api_url, json=payload, timeout=180)
            response.raise_for_status()
            # return response.json().get("description")
            raw_description = response.json().get("description")
            return extract_answer(raw_description)
        except requests.exceptions.RequestException as e:
            print(f"Refiner API Error: {e}")
            return None

class DiffuserGeneratorClient:
    def __init__(self, api_url: str):
        self.api_url = api_url

    def generate(self, payload: dict) -> Optional[str]:
        # print(payload)
        try:
            response = requests.post(self.api_url, json=payload, timeout=300)
            response.raise_for_status()
            return response.json().get("image_b64")
        except requests.exceptions.RequestException as e:
            print(f"Diffuser API Error: {e}")
            return None

class SgpGeneratorClient:
    def __init__(self, api_url: str):
        self.api_url = api_url

    def generate(self, payload: dict) -> Optional[str]:
        sgp_payload = {"prompt": payload.get("prompt")}
        try:
            response = requests.post(self.api_url, json=sgp_payload, timeout=300)
            response.raise_for_status()
            raw_svg_text = response.json().get("svg_text", "")
            clean_svg_text = extract_svg(raw_svg_text)

            if not clean_svg_text:
                print("SGP Error: Model did not return valid SVG content.")
                return None
            
            # Return both the rendered PNG and the raw SVG for state-passing
            png_bytes = cairosvg.svg2png(bytestring=clean_svg_text.encode('utf-8'), output_height=payload.get("canvas_height", 512))
            png_b64 = base64.b64encode(png_bytes).decode('utf-8')
            return {"image_b64": png_b64, "svg_text": clean_svg_text}
            
        except (requests.exceptions.RequestException, Exception) as e:
            print(f"SGP API or SVG Rendering Error: {e}")
            return None
