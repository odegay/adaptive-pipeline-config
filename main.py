import base64
import os
import json
from google.cloud import pubsub_v1
import requests

# Replace with your Google Cloud project ID
project_id = os.environ.get('PROJECT_ID')
# Replace with the desired Pub/Sub topic name 
topic_name = os.environ.get('TOPIC_NAME')

# Stub helper functions
def helper_function_1():
    # Replace with your helper function's logic
    print("Helper function 1 executed")

def helper_function_2():
    # Replace with your helper function's logic
    print("Helper function 2 executed")

def publish_to_pubsub(data):
    """Publishes a message to a Google Cloud Pub/Sub topic."""

    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_name)

    # Data must be a bytestring
    data = json.dumps(data).encode("utf-8")

    # Publish the message, the result is a future that provides details on the delivery
    future = publisher.publish(topic_path, data)
    print(future.result()) 

def adatptive_pipeline_generate_config(event, context):    
    """Triggered by a change to a Cloud Storage bucket."""
    # Fetch Project ID from Metadata Server
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
    # Publish the message to Pub/Sub
      # Publish the message to Pub/Sub (with dynamic project_id)
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_name)
