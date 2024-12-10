import base64
import json
import jsonschema
from adpipsvcfuncs import publish_to_pubsub, load_current_pipeline_data, save_current_pipeline_data
from adpipsvcfuncs import fetch_gcp_secret, load_valid_json
from adpipwfwconst import MSG_TYPE
from adpipwfwconst import PIPELINE_TOPICS as TOPICS
import requests 
import re
from promtps import system_prompt, get_first_request_prompt, generate_LLM_prompt
from configuration_schemas import short_ffn_config_schema
import logging
from openai import OpenAI

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

def remove_null_values(data):
    """
    Recursively remove keys with value None or "null" (string) from JSON-like objects.

    Args:
        data (dict or list): The JSON-like object to be cleaned.

    Returns:
        dict or list: The cleaned JSON-like object with null values removed.
    """
    if isinstance(data, dict):
        # Use dictionary comprehension to filter out None and "null"
        return {k: remove_null_values(v) for k, v in data.items() if v is not None and v != "null"}
    elif isinstance(data, list):
        # Apply the same function to each item in the list
        return [remove_null_values(item) for item in data]
    else:
        return data

def validate_new_model_config_JSON(response_text: str, pipeline_data: dict, isNew: int) -> str:
    """
    Validate the new model configuration JSON received from OpenAI.

    Args:
        response_text (str): The JSON response text from OpenAI.
        pipeline_data (dict): The current pipeline data.
        isNew (int): Flag indicating if it's a new configuration.

    Returns:
        str: The prompt to be sent to OpenAI if validation fails, otherwise None.
    """
    repsonse_json = load_valid_json(response_text)
    repsonse_json = remove_null_values(repsonse_json)
    
    if (repsonse_json is None):
        logger.error(f"Failed to load a valid JSON from OpenAI response. Response JSON is {repsonse_json}")
        prompt = generate_LLM_prompt(pipeline_data, isNew)
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
            prompt = generate_LLM_prompt(pipeline_data, 0)
            prompt += f""" 
            Make sure to provide a valid JSON configuration.
            You previously responded with the text below, it did not pass the schema validation:
            <<<
            {response_text}
            >>>
            """
            return prompt

def validate_message(pubsub_message: dict) -> bool:
    """
    Validate the Pub/Sub message to start a model configuration.

    Args:
        pubsub_message (dict): The Pub/Sub message.

    Returns:
        bool: True if the message is valid, otherwise False.
    """
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
    
def openAI_request(api_key: str, role: str, request: str) -> dict:
    """
    Send a request to OpenAI and get the response.

    Args:
        api_key (str): The OpenAI API key.
        role (str): The role for the OpenAI request.
        request (str): The request to be sent to OpenAI.

    Returns:
        dict: The response from OpenAI.
    """
    client = OpenAI(api_key=api_key)
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": request},
            ],
            temperature=1,
            max_tokens=16383,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0)        
    except Exception as e:
        logger.error(f"Failed to get completion from OpenAI: {str(e)}")
        return None
    return completion

def send_OpenAI_request(prompt: str) -> str:
    """
    Send a request to OpenAI and get the response text.

    Args:
        prompt (str): The prompt to be sent to OpenAI.

    Returns:
        str: The response text from OpenAI.
    """
    logger.debug(f"Sending OpenAI request with system_prompt: {system_prompt} and prompt: {prompt}")
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
    """
    Check if the response from OpenAI suggests increasing the number of hidden layers.

    Args:
        response (str): The response text from OpenAI.

    Returns:
        int: The number of hidden layers to switch to, or None if no switch is suggested.
    """
    pattern = r'SWITCH TO (\d{1,3}) HIDDEN LAYERS'
    match = re.search(pattern, response)
    if match:
        # return the number as integer
        return int(match.group(1))
    # return None if the pattern was not found
    return None

def new_layers_configuration(pipeline_data: dict, new_layers: int) -> dict:
    """
    Update the pipeline data with the new number of hidden layers.

    Args:
        pipeline_data (dict): The current pipeline data.
        new_layers (int): The new number of hidden layers.

    Returns:
        dict: The updated pipeline data.
    """
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
    """
    Iterate through the LLM cycle to get a valid model configuration.

    Args:
        prompt (str): The prompt to be sent to OpenAI.
        pipeline_data (dict): The current pipeline data.

    Returns:
        dict: The updated pipeline data with the valid model configuration.
    """
    response_text = send_OpenAI_request(prompt)            
    new_layers = check_layers_increase(response_text)
    if new_layers is None:
        for i in range(3):
            prompt = validate_new_model_config_JSON(response_text, pipeline_data, 0)
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
    """
    Save the model configuration and publish a message to the Pub/Sub topic.

    Args:
        pipeline_data (dict): The current pipeline data.

    Returns:
        bool: True if the configuration is saved and the message is published, otherwise False.
    """
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

def adatptive_pipeline_generate_config(event, context):        
    """
    Cloud Function to generate a new model configuration for the adaptive pipeline.

    Args:
        event (dict): The event data from Pub/Sub.
        context (google.cloud.functions.Context): The context of the event.

    Returns:
        str: The result of the function execution.
    """
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
