import base64
import os
import json
from google.cloud import pubsub_v1
import requests
from adpipsvcfuncs import publish_to_pubsub
from adpipwfwconst import MSG_TYPE
from adpipwfwconst import PIPELINE_TOPICS as TOPICS
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Capture DEBUG, INFO, WARNING, ERROR, CRITICAL

# Stub helper functions
def helper_function_1():
    # Replace with your helper function's logic
    print("Helper function 1 executed")
def helper_function_2():
    # Replace with your helper function's logic
    print("Helper function 2 executed")
# def publish_to_pubsub(topic_name, project_id, data):
#     """Publishes a message to a Google Cloud Pub/Sub topic."""
#     # Publish the message to Pub/Sub
#       # Publish the message to Pub/Sub (with dynamic project_id)
#     publisher = pubsub_v1.PublisherClient()
#     topic_path = publisher.topic_path(project_id, topic_name)
#     data = json.dumps(data).encode("utf-8")
#     future = publisher.publish(topic_path, data)
#     print(future.result()) 
#     # Data must be a bytestring
#     # Publish the message, the result is a future that provides details on the delivery

def validate_message(event, context):
    """Background Cloud Function to be triggered by Pub/Sub.
    Args:
         event (dict):  The dictionary with data specific to this type of event.
         context (google.cloud.functions.Context): Metadata of triggering event.
    """
    # Decode the PubSub message
    pubsub_message = base64.b64decode(event['data']).decode('utf-8')
    logger.debug(f"Decoded Pub/Sub message: {pubsub_message}")  
    print(pubsub_message)
    # Validate the message
    if 'data' in event:
        if 'start procedure' in pubsub_message:
            print('Starting a new procedure')
            return True
        else:
            print('Not a start procedure message')
            return False
    else:
        return False

def adatptive_pipeline_generate_config(event, context):    
    """Triggered by a change to a Cloud Storage bucket."""
    # Fetch Project ID from Metadata Server
    if (validate_message(event, context) == True):
        metadata_server_url = "http://metadata/computeMetadata/v1/project/project-id"
        headers = {"Metadata-Flavor": "Google"}
        project_id = requests.get(metadata_server_url, headers=headers).text    
        # Call your helper functions
        helper_function_1()
        helper_function_2()
        # Construct the message to be published 
        message_data = {
            "status": "success",  # Example, replace with your relevant data
            # ... add more data if needed 
        }
        topic_name = "adaptive-pipeline-workflow-topic"
        publish_to_pubsub(topic_name, message_data)
        print(f"Successfully published a message to {topic_name} and project_id {project_id}")
        return f"Successfully published a message to {topic_name} and project_id {project_id}"
    else:
        print("Skipping message processing due to a message not intended to start a procedure")
        return "Skipping message processing due to a message not intended to start a procedure"