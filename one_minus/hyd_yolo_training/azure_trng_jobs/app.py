from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml import command, Input
from azure.ai.ml.constants import AssetTypes

def conn_ml_wksp(subscription_id, resource_group, workspace):
    ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)
    print("ml_client:")
    print(ml_client)
    return ml_client

def submit_job(ml_client, exp_name, disp_name, input_path, compute_instance, ml_studio_env):
    job = command(
        inputs=dict(
            data_dir=Input(
                type=AssetTypes.URI_FOLDER,
                path="azureml:minus_iter_3_data:1"
                #path=input_path
            ),
            # model="yolov8m.pt",
            epochs=200,
            patience=50,
            pretrained="yolov8m.pt",
            # batch_size=8,
            # img_size=640,
            # device="cpu",
            # optimizer="Adam",
            # lr0=0.01,
            # iou=0.2,
            # dropout=0.2,
            # #output_dir="azureml:minus-iter2-fo-asset:1",
            # exp_name=exp_name

        ),
        compute="cluster1",
        #compute=compute_instance,
        environment="yolo_env:1",
        #environment=ml_studio_env,
        code="/Users/debbiesubocz/GitReps/hydropho-natics/one_minus/hyd_yolo_training/azure_trng_jobs",  # Adjust to the actual path of your training script
        # command="python train_model.py --data_dir ${{inputs.data_dir}} --pretrained ${{inputs.pretrained}} --batch_size ${{inputs.batch_size}} --epochs ${{inputs.epochs}} --img_size ${{inputs.img_size}} --exp_name ${{inputs.exp_name}} --optimizer ${{optimizer}} --patience ${{inputs.patience}} --lr0 ${{inputs.lr0}} --iou ${{inputs.iou}} --dropout ${{dropout}}",
        command="python train_model.py --data_dir ${{inputs.data_dir}} --pretrained ${{inputs.pretrained}}  --epochs ${{inputs.epochs}}  --patience ${{inputs.patience}}",
        experiment_name=exp_name,
        display_name=disp_name,
    )
    return ml_client.jobs.create_or_update(job)

if __name__ == '__main__':
    # subscription_id = 'your-subscription-id'
    # resource_group = 'your-resource-group'
    # workspace = 'your-workspace-name'
    subscription_id = '8f00e0d7-79c6-4b37-bfea-a3b9362bd229'
    resource_group = 'yolov8-models'
    workspace = 'testing'

    compute = "cluster2"

    experiment_name = "MINUS-iter2-8m.pt"
    display_name = "MINUS-iter2-8m.pt"
    
    input_path = "azureml:minus_iter_2_data:1"

    docker_env = "yolo_env:1"


    ml_client = conn_ml_wksp(subscription_id, resource_group, workspace)
    job = submit_job(ml_client, experiment_name, display_name, input_path, compute, docker_env)
    print(f"Submitted job: {job.name}")
