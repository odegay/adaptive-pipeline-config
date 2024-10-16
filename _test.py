import jsonschema
import json
from configuration_schemas import short_ffn_config_schema
import re


json_txt1 = '{"l":[{"lt":"d","u":50}]}'
json_txt2 = '{"l":[{"u":50}]}'
repsonse_json = json.loads(json_txt1)
print(repsonse_json)
a = jsonschema.validate(repsonse_json, short_ffn_config_schema)
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