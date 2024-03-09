"""
Update this docstring
"""

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient

def get_secret(vault_name, secret_name):
    """
    Update this docstring
    """
    # Key Vault details
    key_vault_name = vault_name
    key_vault_uri = f"https://{key_vault_name}.vault.azure.net/"

    # Authenticate using DefaultAzureCredential
    # This will use your Azure CLI or Visual Studio/Visual Studio Code credentials if logged in
    credential = DefaultAzureCredential()

    # Create a SecretClient using the credential
    client = SecretClient(vault_url=key_vault_uri, credential=credential)

    # Retrieve a secret
    retrieved_secret = client.get_secret(secret_name)

    # Use the retrieved secret
    #print(f"The secret '{secret_name}' was successfully retrieved")
    return retrieved_secret.value


def get_blob_file_list(connection_string, container_name):
    

    # Azure Storage account connection string
    #connection_string = "<your-azure-storage-connection-string>"

    # Name of the container
    #container_name = "<your-container-name>"

    # Create a BlobServiceClient using the connection string
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Get a reference to the container
    container_client = blob_service_client.get_container_client(container_name)

    # Initialize an empty list to hold the blob names
    blob_list = []

    # List all blobs in the container and add them to the list
    for blob in container_client.list_blobs():
        blob_list.append(blob.name)

    # Now blob_list contains all the blob names
    #print("List of blobs in the container:", blob_list)
    return blob_list

# You can now perform follow-on tasks with the blob_list


