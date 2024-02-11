#import required libraries
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import AmlCompute
from azureml.core import Run, Dataset
from azure.ai.ml import command
from azure.ai.ml import Input


def conn_ml_wksp(subscription_id, resource_group, workspace):
    #connect to the workspace
    ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)

    # get the list of compute instances in the ML Workspace Environment
    computes = ml_client.compute.list()

    # print the list
    for compute in computes:
        print(f"Name: {compute.name}, Type: {compute.type}")

    #  get the list of datasets in the ML Workspace Environment
    datasets = ml_client.data.list()

    # Print the list of datasets
    for dataset in datasets:
        print(f"Name: {dataset.name}, Version: {dataset.version}, Type: {dataset.type}")

    # get the list of environments in the ML Workspace
    environments = ml_client.environments.list()

    for env in environments:
        print(f"Name: {env.name}, Version: {env.version}")
    
    return ml_client


if __name__ == '__main__':
    #Enter details of your Azure Machine Learning workspace
    subscription_id = '8f00e0d7-79c6-4b37-bfea-a3b9362bd229'
    resource_group = 'yolov8-models'
    workspace = 'yolov8-hyp-tuning'

    ml_client = conn_ml_wksp(subscription_id, resource_group, workspace)

    job = command(
        inputs=dict(
            epochs=3, ml_workspace_dataset="minus-iter2-fo-asset"
        ),
        compute="test-32gb-ram",
        environment="yolo_env:1",
        code="/Users/debbiesubocz/GitReps/hydropho-natics/one_minus/hyd_yolo_training/azure_trng_jobs",  # location of source code
        command="python train_model.py \
            --num_epochs ${{inputs.epochs}} \
            --ml_workspace_dataset ${{inputs.ml_workspace_dataset}}",
        experiment_name="yolo-job-exp-test",
        display_name="yolo-job-displayname-test",
    )

    ml_client.jobs.create_or_update(job)