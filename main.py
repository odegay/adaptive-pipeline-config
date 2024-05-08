import base64
import json
import jsonschema
from adpipsvcfuncs import publish_to_pubsub, fetch_gcp_secret, openAI_request, load_valid_json
from adpipwfwconst import MSG_TYPE
from adpipwfwconst import PIPELINE_TOPICS as TOPICS
import logging
import requests 
from promtps import system_prompt, get_first_request_prompt
from configuration_schemas import short_ffn_config_schema

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
api_url = fetch_gcp_secret('adaptive-pipeline-persistence-layer-url')
api_key = fetch_gcp_secret('adaptive-pipeline-API-token')
opeanai_api_key = fetch_gcp_secret('adaptive-pipeline-openai-api-token')

def load_previous_model_configurations():
    #TODO: Implement the logic to load previous model configurations

    json_string_config = """
    {
    "layers": [
        {"layer_type":"dense", "units":96, "kernel_regularizer":"l2", "kernel_initializer":"glorot_uniform", "bias_initializer":"zeros", "activation":"relu", "kernel_regularizer_lambda":0.001},
        {"layer_type":"dense", "units":64, "kernel_regularizer":"l2", "kernel_initializer":"glorot_uniform", "bias_initializer":"zeros", "dropout_rate":0.4, "activation":"relu", "kernel_regularizer_lambda":0.001},
        {"layer_type":"dense", "units":32, "kernel_regularizer":"l2", "kernel_initializer":"glorot_uniform", "bias_initializer":"zeros", "dropout_rate":0.4, "activation":"relu", "kernel_regularizer_lambda":0.001},
        {"layer_type":"dense", "units":16, "kernel_regularizer":"l2", "kernel_initializer":"glorot_uniform", "bias_initializer":"zeros", "activation":"relu", "kernel_regularizer_lambda":0.001}
    ]
    }
    """

    # Load previous model configurations    
    logger.debug("Placeholder for loading previous model configurations")
    return json_string_config

def generate_LLM_prompt():
    # Opens a prompt.txt located in the same folder file and reads the prompt
    prompt = get_first_request_prompt(50, 100)
    # prompt = prompt + " " + load_previous_model_configurations()
    # prompt = prompt + " " + additional_prompt
    return prompt

def validate_message(pubsub_message):
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

def adatptive_pipeline_generate_config(event, context):        

    pubsub_message = ""
    if 'data' in event:
        pubsub_message = base64.b64decode(event['data']).decode('utf-8')
        pubsub_message = json.loads(pubsub_message)
        logger.debug(f"Decoded Pub/Sub message: {pubsub_message}")
    else:
        logger.debug("Data is missing in the event")
        return False    
    
    if (validate_message(pubsub_message) == True):
        prompt = generate_LLM_prompt()
        response_text = send_OpenAI_request(prompt)
        response_json = None
        
        for i in range(3):
            passed = False
            repsonse_json = load_valid_json(response_text)
            if (repsonse_json is None):
                logger.error(f"Failed to load a valid JSON from OpenAI response. Response JSON is {response_json}")
                prompt = generate_LLM_prompt()
                prompt += f""" 
                Make sure to provide a valid JSON configuration.
                You previously responded with the text below, it failed to upload as a valid JSON:
                <<<
                {response_text}
                >>>
                """
            else:
                try:
                    jsonschema.validate(repsonse_json, short_ffn_config_schema)
                    passed = True
                except Exception as ve:
                    passed = False
                    logger.error(f"Failed to validate the JSON configuration. Error: {ve}")
                    prompt = generate_LLM_prompt()
                    prompt += f""" 
                    Make sure to provide a valid JSON configuration.
                    You previously responded with the text below, it did not pass the schema validation:
                    <<<
                    {response_text}
                    >>>
                    """
            if passed:
                logger.debug("Successfully loaded a valid JSON from OpenAI response")
                break
            else:
                response_text = send_OpenAI_request(prompt)

        if i == 2:
            logger.error(f"Failed to load a valid JSON from OpenAI response after 3 attempts. The last response JSON text is {response_json}")
            return f"Failed to load a valid JSON from OpenAI response after 3 attempts. The last response JSON text is {response_json}"

            
        logger.debug(f"OpenAI response text: {response_text}")
        logger.debug(f"TEST PIPELINE FINALIZATION. OpenAI response text: {response_text}")
        return f"TEST PIPELINE FINALIZATION. OpenAI response text: {response_text}"        

        # Construct the message to be published 
        message_data = {
            "pipeline_id": pubsub_message['pipeline_id'],
            "status": MSG_TYPE.REQUEST_LLM_NEW_MODEL_CONFIGURATION.value,
            "prompt": prompt            
        }

        if not api_url:
            logger.error("Failed to fetch the API URL")
            return None
        headers = {
            "Authorization": api_key
        }
        try:
            response = requests.put(f"{api_url}/update/{pubsub_message['pipeline_id']}", json=message_data, headers=headers)
            if response.status_code != 200:
                logger.error(f"Failed to update the pipeline status. Response: {response.text}")                
                return "Failed to update the pipeline status. Error: {response.text}"
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return "Failed to update the pipeline status. Error: {str(e)}"            

        topic_name = "adaptive-pipeline-config-topic"
        publish_to_pubsub(topic_name, message_data)
        logger.debug(f"Published message to topic: {topic_name} with data: {message_data} configuration generated")
        return f"Successfully published a message to {topic_name} with data: {message_data} configuration generated"
    else:
        logger.debug("Skipping message processing due to a message not intended to start a procedure")
        return "Skipping message processing due to a message not intended to start a procedure"