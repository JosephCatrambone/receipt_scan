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

import openai
API_KEY = os.environ.get("OPENAI_API_KEY", None)

from guardrails.hub import RegexMatch
from guardrails import Guard


# Models:
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

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
guard = Guard.from_pydantic(Receipt)

app = FastAPI()

@app.post("/scan")
async def scan_trocr(image: UploadFile):
    image = Image.open(image.file).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    guard.validate(generated_text)
    return generated_text


@app.post("/scan-openai")
def scan_openai(image: UploadFile) -> Receipt:
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
          "max_tokens": 500
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        print(f"DEBUG openai response: {response.json()}")
        response = response.json()['choices'][0]['message']['content']
        #response = """```json\n{\n    "items": [\n {\n "text": "Generic Services",\n "quantity": "1x",\n "price": "$30.00"\n },\n {\n "text": "Some other item",\n "quantity": "2x",\n"price": "$5.30"\n }\n    ],\n "raw_text": "Fake Receipt\\nGeneric Services 1x $30.00\\nSome other item 2x $5.30\\nTotal $35.30"\n}\n```"""
        response = response.strip("```json\n")
        response = response.strip("```")
        return response
    prompt = """Please scan the attached receipt and return a JSON object with the item names, quantities, and costs.  ${gr.complete_json_suffix_v2}"""
    guard = Guard.from_pydantic(output_class=Receipt, prompt=prompt)

    validated_output = Receipt(items=[], raw_text="")
    validated_output, *rest = guard(
        llm_api=call_openai,
    )
    print(f"DEBUG validated_output: {validated_output}")
    result = Receipt.model_validate(from_json(validated_output, allow_partial=True))
    return result

