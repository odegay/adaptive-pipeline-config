import os
import sys
from google.cloud import secretmanager

def access_secret(secret_id, version_id="latest"):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{os.environ['PROJECT_ID']}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("utf-8")

if __name__ == "__main__":
    for secret_name in sys.argv[1:]:
        secret_value = access_secret(secret_name)
        print(f"{secret_name}={secret_value}")  # 