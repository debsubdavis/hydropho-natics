from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml import command, Input
from azure.ai.ml.entities import CommandJob
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.sweep import SweepJob, Choice, Objective, SweepJobLimits, GridSamplingAlgorithm

# Function to connect to the Azure ML workspace
def conn_ml_wksp(subscription_id, resource_group, workspace):
    # Create MLClient object using Azure credentials and workspace details
    ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)
    print(ml_client)
    return ml_client

# Function to submit hyperparameter tuning job to Azure ML
def submit_hp_tuning_job(ml_client, exp_name, exp_display_name, compute_name, max_concurrent_trials, mlstd_env):
    # Define the SweepJob for hyperparameter tuning
    sweep_job = SweepJob(
        compute=compute_name,
        experiment_name="SJ-" + exp_name,
        display_name="SJ: " + exp_display_name,
        sampling_algorithm=GridSamplingAlgorithm(),
        objective=Objective(goal="maximize", primary_metric="metrics/mAP50B"),
        limits=SweepJobLimits(max_concurrent_trials=max_concurrent_trials, trial_timeout=14400),
        search_space=dict(
            batch_size=Choice([16]),
            lr0=Choice([0.01, 0.001]),
            optimizer=Choice(["Adam", "SGD"]),
            dropout=Choice([0.2])
        ),
        trial=CommandJob(
            code="/Users/debbiesubocz/GitReps/hydropho-natics/one_minus/hyd_yolo_training/azure_trng_jobs",
            command="python train_model.py --data_dir ${{inputs.data_dir}} --exp_name ${{inputs.exp_name}} --epochs ${{inputs.epochs}} --patience ${{inputs.patience}} --pretrained ${{inputs.pretrained}} --iou ${{inputs.iou}}",
            environment=ml_studio_env,
            inputs=dict(
                data_dir=Input(
                    type=AssetTypes.URI_FOLDER,
                    path="azureml://subscriptions/8f00e0d7-79c6-4b37-bfea-a3b9362bd229/resourcegroups/yolov8-models/workspaces/testing/datastores/minus_iter_2/paths/without-mooring/",
                ),
                pretrained="yolov8m.pt",
                optimizer="Adam",
                iou=0.3,
                exp_name=exp_name,
                epochs=200,
                patience=50
            ),
            experiment_name="CJ-" + exp_name,
            display_name="SJ: " + exp_display_name
        )
    )

    return ml_client.jobs.create_or_update(sweep_job)

if __name__ == '__main__':
    # Azure subscription details
    subscription_id = '8f00e0d7-79c6-4b37-bfea-a3b9362bd229'
    resource_group = 'yolov8-models'
    workspace = 'testing'
    
    # Hyperparameter tuning experiment details
    experiment_name = "MINUS-iter3-hyp-tuning-200"
    experiment_display_name = "MINUS iter3 hyp tuning 200 epochs"
    max_conc_trials = 6
    azureml_compute_name = "cluster2"
    ml_studio_env = "yolo_env:1"

    # Connect to Azure ML workspace
    ml_client = conn_ml_wksp(subscription_id, resource_group, workspace)
    
    # Submit hyperparameter tuning job to Azure ML
    job = submit_hp_tuning_job(ml_client, experiment_name, experiment_display_name, azureml_compute_name, max_conc_trials, ml_studio_env)
    print(f"Submitted hyperdrive job: {job.name}")
