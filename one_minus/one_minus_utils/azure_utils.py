"""
Update this docstring
"""

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

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
