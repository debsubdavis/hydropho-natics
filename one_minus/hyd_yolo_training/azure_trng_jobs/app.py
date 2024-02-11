from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml import command, Input
from azure.ai.ml.constants import AssetTypes

def conn_ml_wksp(subscription_id, resource_group, workspace):
    ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)
    return ml_client

def submit_job(ml_client):
    job = command(
        inputs=dict(
            data_dir=Input(
                type=AssetTypes.URI_FOLDER,
                path="azureml:minus-iter2-fo-asset:1"
            ),
            epochs=100,
            pretrained="yolov8n.pt",
            batch_size=16,
            img_size=640,
            output_dir="azureml:minus-iter2-fo-asset:1",
            exp_name="yolo_exp_v2"
        ),
        compute="test-32gb-ram",
        environment="yolo_env:2",
        code="/Users/debbiesubocz/GitReps/hydropho-natics/one_minus/hyd_yolo_training/azure_trng_jobs",  # Adjust to the actual path of your training script
        command="python train_model.py --data_dir ${{inputs.data_dir}} --pretrained ${{inputs.pretrained}} --batch_size ${{inputs.batch_size}} --epochs ${{inputs.epochs}} --img_size ${{inputs.img_size}} --output_dir ${{inputs.output_dir}} --exp_name ${{inputs.exp_name}}",
        experiment_name="yolo-experiment-v2",
        display_name="Yolo Training Job v2",
    )
    return ml_client.jobs.create_or_update(job)

if __name__ == '__main__':
    # subscription_id = 'your-subscription-id'
    # resource_group = 'your-resource-group'
    # workspace = 'your-workspace-name'
    subscription_id = '8f00e0d7-79c6-4b37-bfea-a3b9362bd229'
    resource_group = 'yolov8-models'
    workspace = 'yolov8-hyp-tuning'

    ml_client = conn_ml_wksp(subscription_id, resource_group, workspace)
    job = submit_job(ml_client)
    print(f"Submitted job: {job.name}")
