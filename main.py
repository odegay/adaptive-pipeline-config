import base64
import json
import jsonschema
from adpipsvcfuncs import publish_to_pubsub, load_current_pipeline_data, save_current_pipeline_data
from adpipsvcfuncs import fetch_gcp_secret, openAI_request, load_valid_json
from adpipwfwconst import MSG_TYPE
from adpipwfwconst import PIPELINE_TOPICS as TOPICS
import requests 
import re
from promtps import system_prompt, get_first_request_prompt, generate_LLM_prompt
from configuration_schemas import short_ffn_config_schema
import logging

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)  # Capture DEBUG, INFO, WARNING, ERROR, CRITICAL
if not root_logger.handlers:
    # Create console handler and set its log level to DEBUG
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    # Add the handler to the root logger
    root_logger.addHandler(ch)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Capture DEBUG, INFO, WARNING, ERROR, CRITICAL
opeanai_api_key = fetch_gcp_secret('adaptive-pipeline-openai-api-token')

def validate_new_model_config_JSON(response_text: str) -> str:
    repsonse_json = load_valid_json(response_text)
    if (repsonse_json is None):
        logger.error(f"Failed to load a valid JSON from OpenAI response. Response JSON is {repsonse_json}")
        prompt = generate_LLM_prompt()
        prompt += f""" 
        Make sure to provide a valid JSON configuration.
        You previously responded with the text below, it failed to upload as a valid JSON:
        <<<
        {response_text}
        >>>
        """
        return prompt
    else:
        try:
            jsonschema.validate(repsonse_json, short_ffn_config_schema)
            logger.debug(f"Successfully validated the JSON configuration")
            return None
        except Exception as ve:
            logger.error(f"Failed to validate the JSON configuration. Error: {ve}")
            prompt = generate_LLM_prompt()
            prompt += f""" 
            Make sure to provide a valid JSON configuration.
            You previously responded with the text below, it did not pass the schema validation:
            <<<
            {response_text}
            >>>
            """
            return prompt

def validate_message(pubsub_message: dict) -> bool:
    # Validate the message to start a model configuration
    # Log the entire message and its type
    logger.debug(f"Decoded Pub/Sub message: {pubsub_message}")
    if 'status' in pubsub_message:        
        if pubsub_message['status'] == MSG_TYPE.START_MODEL_CONFIGURATION.value:
            if 'pipeline_id' in pubsub_message:
                logger.debug(f"Start model configuration with pipeline_id: {pubsub_message['pipeline_id']} received")
                return True
            else:
                logger.debug("Pipeline ID is missing in the message")
                return False
        else:
            logger.debug(f"Message not intended to start a model configuration {pubsub_message['status']} received")
            return False
    else:
        logger.debug("Message type is missing in the message")
        return False
    
def send_OpenAI_request(prompt: str) -> str:
    response = openAI_request(opeanai_api_key, system_prompt, prompt)
    if response is None:
        logger.error("Failed to get a response from OpenAI")
        return None    
    # Extract the response from OpenAI
    response_text = response.choices[0].message.content
    logger.debug(f"OpenAI response: {response}")
    logger.debug(f"OpenAI response text: {response_text}")
    return response_text

def check_layers_increase(response: str) -> int:
    pattern = r'SWITCH TO (\d{1,3}) HIDDEN LAYERS'
    match = re.search(pattern, response)
    if match:
        # return the number as integer
        return int(match.group(1))
    # return None if the pattern was not found
    return None

def new_layers_configuration(pipeline_data: dict, new_layers: int) -> dict:
    current_hidden_layers_ct = pipeline_data.get('current_hidden_layers_ct')
    pipeline_data['current_hidden_layers_ct'] = new_layers

    if pipeline_data.get('hidden_layers_configs') is not None:
        current_layer = None
        for layer in pipeline_data['hidden_layers_configs']: # iterate over the list to find the matching layer
            if layer.get('hidden_layers_ct') == current_hidden_layers_ct:
                current_layer = layer
                break

        if current_layer is not None:
            current_layer['is_completed'] = True
        else:
            logger.error(f"Failed to find the current layer configuration for hidden_layers_ct: {current_hidden_layers_ct}")
            return None
    else:
        logger.error("Failed to find the hidden_layers_configs in the pipeline data")
        return None
    return pipeline_data

def iterate_LLM_cycle(prompt: str, pipeline_data: dict) -> dict:
    response_text = send_OpenAI_request(prompt)            
    new_layers = check_layers_increase(response_text)
    if new_layers is None:
        for i in range(3):
            prompt = validate_new_model_config_JSON(response_text)
            if prompt is None:
                logger.debug(f"Successfully validated the JSON configuration, JSON: {response_text}")
                pipeline_data['current_configuration'] = response_text
                return pipeline_data
            else:
                response_text = send_OpenAI_request(prompt)
        if i == 2:
            logger.error(f"Failed to load a valid JSON from OpenAI response after 3 attempts. The last response JSON text is {response_text}")
            return None
    else:
        pipeline_data = new_layers_configuration(pipeline_data, new_layers)
        prompt = generate_LLM_prompt(pipeline_data, new_layers)
        logger.debug(f"New hidden layers number request detected: {new_layers}")   
        pipeline_data = iterate_LLM_cycle(prompt, pipeline_data)


def save_model_configuration_and_publish_message(pipeline_data: dict) -> bool:
    response_json = load_valid_json(pipeline_data['current_configuration'])

    if response_json is None:
        logger.error(f"Failed to load a valid JSON from OpenAI response. Response JSON is {response_json}")
        return False  
    #At the first run of the pipeline, the current_hidden_layers_ct is not set, so we set it to 1
    if "current_hidden_layers_ct" not in pipeline_data:
        pipeline_data['current_hidden_layers_ct'] = 1
    pipeline_data['status'] = MSG_TYPE.GENERATE_NEW_MODEL.value
    #pipeline_data['status'] = MSG_TYPE.NEW_MODEL_CONFIGURATION_SUCCESS.value             
    #API call to save the configuration
    save_current_pipeline_data(pipeline_data)    
    pub_message_data = {
    "pipeline_id": pipeline_data['pipeline_id'],
    "status": MSG_TYPE.GENERATE_NEW_MODEL.value,
    #"status": MSG_TYPE.NEW_MODEL_CONFIGURATION_SUCCESS.value,
    "current_configuration": pipeline_data['current_configuration']
    }
    publish_to_pubsub(TOPICS.WORKFLOW_TOPIC.value, pub_message_data)   
    logger.debug(f"Publishing message to topic: {TOPICS.WORKFLOW_TOPIC.value} with data: {pub_message_data}")

#def save_new_layers_ct_and_publish_message():
def adatptive_pipeline_generate_config(event, context):        
    pubsub_message = ""
    if 'data' in event:
        pubsub_message = base64.b64decode(event['data']).decode('utf-8')
        pubsub_message = json.loads(pubsub_message)
        logger.debug(f"Decoded Pub/Sub message: {pubsub_message}")
    else:
        logger.debug("Data is missing in the event")
        return False   
    
    if not validate_message(pubsub_message):
        logger.debug("Skipping message processing due to a message not intended to start a procedure")
        return "Skipping message processing due to a message not intended to start a procedure"
    
    pipeline_id = pubsub_message['pipeline_id']

    pipeline_data = load_current_pipeline_data(pipeline_id)
    if pipeline_data is None:
        logger.error(f"Failed to load the pipeline data for pipeline_id: {pipeline_id}")
        return f"Failed to load the pipeline data for pipeline_id: {pipeline_id}"
    else:
        logger.debug(f"Loaded pipeline data for pipeline_id: {pipeline_id}. Data: {pipeline_data}")

    prompt = generate_LLM_prompt(pipeline_data, 0)
    pipeline_data = iterate_LLM_cycle(prompt, pipeline_data)
    response_text = pipeline_data['current_configuration']
    
    if "current_hidden_layers_ct" not in pipeline_data:
        pipeline_data['current_hidden_layers_ct'] = 1
    
    if save_model_configuration_and_publish_message(pipeline_data) is False:
        logger.error(f"TEST PIPELINE FINALIZATION FAILED to save the model configuration and publish the message. OpenAI response text: {response_text}")
        return f"TEST PIPELINE FINALIZATION Failed to save the model configuration and publish the message"

    logger.debug(f"OpenAI response text: {response_text}")
    logger.debug(f"TEST PIPELINE FINALIZATION SUCCESS. OpenAI response text: {response_text}")
    return f"TEST PIPELINE FINALIZATION SUCCESS. OpenAI response text: {response_text}"        