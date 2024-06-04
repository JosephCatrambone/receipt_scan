import base64
import json
import os
from typing import List

import requests
from fastapi import FastAPI, UploadFile
from PIL import Image
from pydantic import BaseModel, Field
from pydantic_core import from_json
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from guardrails.hub import RegexMatch
from guardrails import Guard


# Self-hosted Models:
SELF_HOSTED = os.environ.get("SELF_HOSTED", "false").lower() == "true"
if SELF_HOSTED:
    global processor
    global model
    MODEL_NAME = "microsoft/trocr-base-printed"
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

# Non-self-hosted Model
API_KEY = os.environ.get("OPENAI_API_KEY", None)
if API_KEY is not None:
    import openai

# Basic Validators:
QUANTITY_REGEX = "\(?\d+x?\)?"
PRICE_REGEX = "\$?\d+(\.\d\d)?"

# Data
class LineItem(BaseModel):
    text: str = Field(default="", description="The text name or description of the line item.")
    quantity: str = Field(default="1", description="The quantity modifier for the given item, i.e. (2x, 3x)", validators=[RegexMatch(regex=QUANTITY_REGEX),])
    price: str = Field(default="$0.00", description="The cost of the item.", validators=[RegexMatch(regex=PRICE_REGEX),])

class Receipt(BaseModel):
    items: List[LineItem] = Field(default_factory=list, description="An array of LineItems.")
    raw_text: str = Field(default="", description="The full extracted text.")

# Guards:
prompt = """Please scan the attached receipt and return a JSON object with the item names, quantities, and costs.  ${gr.complete_json_suffix_v2}"""
guard = Guard.from_pydantic(output_class=Receipt, prompt=prompt)

app = FastAPI()

@app.post("/scan-trocr")
async def scan_trocr(image: UploadFile):
    if not SELF_HOSTED:
        raise Exception("SELF_HOSTED is not set to 'true' in the environment. Models are not loaded.")
    image = Image.open(image.file).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    guard.validate(generated_text)
    return generated_text


@app.post("/scan-openai")
def scan_openai(image: UploadFile) -> Receipt:
    if API_KEY is None:
        raise Exception("OPENAI_API_KEY is not defined in your environment. Please set it.")
    encoded_image = base64.b64encode(image.file.read()).decode("utf-8")
    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {API_KEY}"
    }
    # Hoist all variables so we can replace the llm_call with our own prompt method.
    # We're doing this because we need to push an image and a prompt.
    # For reference: https://platform.openai.com/docs/guides/vision
    def call_openai(prompt: str, *args, **kwargs):
        payload = {
          "model": "gpt-4o",
          "messages": [
            {
              "role": "user",
              "content": [
                {
                  "type": "text",
                  "text": prompt
                },
                {
                  "type": "image_url",
                  "image_url": {
                    "url": f"data:{image.content_type};base64,{encoded_image}"
                  }
                }
              ]
            }
          ],
          "max_tokens": 500,
          "temperature": 0
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response = response.json()
        response = response['choices'][0]['message']['content']
        response = response.strip("```json\n")
        response = response.strip("```")
        return response

    # This is a little redundant: it double-validates the returned structure, but might be useful for a model that returns partial data.
    #validated_output, *rest = guard(llm_api=call_openai,)
    #return Receipt.model_validate(from_json(validated_output, allow_partial=True))
    _raw_model_out, validated_output, *rest = guard(llm_api=call_openai,)
    return Receipt.parse_obj(validated_output)

