import jsonschema
import json
from configuration_schemas import short_ffn_config_schema
import re


def remove_null_values(data):
    """
    Recursively remove keys with value None or "null" (string) from JSON-like objects.
    """
    if isinstance(data, dict):
        # Use dictionary comprehension to filter out None and "null"
        return {k: remove_null_values(v) for k, v in data.items() if v is not None and v != "null"}
    elif isinstance(data, list):
        # Apply the same function to each item in the list
        return [remove_null_values(item) for item in data]
    else:
        return data


json_txt1 = '{"l":[{"lt":"d","u":50}]}'
json_txt2 = '{"l":[{"u":50}]}'
json_txt3 = ' {"cfg":{"lm":0.01,"bs":32,"ep":50,"lr":0.001,"lf":0.1,"lp":5,"md":0.0001,"cd":2,"mlr":0.00001,"esp":10},"l":[{"lt":"d","u":50,"kr":null,"br":null,"krl":null,"brl":null,"ki":null,"bi":null,"dr":null,"bn":null,"a":"r","r":null}]}'

repsonse_json = json.loads(json_txt3)

cleaned_json = remove_null_values(repsonse_json)

print(cleaned_json)

a = jsonschema.validate(cleaned_json, short_ffn_config_schema)
print(a)




def analyze_response(response):
    pattern = r'SWITCH TO (\d{1,3}) HIDDEN LAYERS'
    match = re.search(pattern, response)
    if match:
        # return the number as integer
        return int(match.group(1))
    # return None if the pattern was not found
    return None

response1 = "SWITCH TO 4 HIDDEN LAYERS"
response2 = "switch TO 4 HIDDEN LAYERS"

print(analyze_response(response2))