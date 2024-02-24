from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml import command, Input
from azure.ai.ml.entities import CommandJob
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.sweep import SweepJob, Choice, Objective, SweepJobLimits, GridSamplingAlgorithm

def conn_ml_wksp(subscription_id, resource_group, workspace):
    ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)
    print(ml_client)
    return ml_client

def submit_hp_tuning_job(ml_client, exp_name, exp_display_name, compute_name, max_concurrent_trials, mlstd_env):
    sweep_job = SweepJob(
        compute=compute_name, #Standard_E4s_v3
        experiment_name="SJ-" + exp_name,
        display_name="SJ: " + exp_display_name,
        #sampling_algorithm="random",
        sampling_algorithm=GridSamplingAlgorithm(),
        objective=Objective(goal="maximize", primary_metric="metrics/mAP50B"),
        #limits=Limit(max_total_trials=20, max_concurrent_trials=4, trial_timeout_minutes=60),
        #limits=SweepJobLimits(max_total_trials=20, max_concurrent_trials=4, trial_timeout=3600),
        limits=SweepJobLimits(max_concurrent_trials=max_concurrent_trials, trial_timeout=14400),
        search_space=dict(
            batch_size=Choice([8, 16, 32]),
            lr0=Choice([0.01, 0.001]),
            #iou=Choice([0.3, 0.5]),
            #dropout=Choice([0.2, 0.5])
        ),
        trial=CommandJob(
            code="/Users/debbiesubocz/GitReps/hydropho-natics/one_minus/hyd_yolo_training/azure_trng_jobs",  # Adjust to the actual path of your training script
            command="python train_model.py --data_dir ${{inputs.data_dir}} --exp_name ${{inputs.exp_name}} --epochs ${{inputs.epochs}} --patience ${{inputs.patience}} --pretrained ${{inputs.pretrained}} --optimizer ${{inputs.optimizer}} --iou ${{inputs.iou}}",
            environment=ml_studio_env,
            inputs=dict(
                data_dir=Input(
                    type=AssetTypes.URI_FOLDER,
                    path="azureml://subscriptions/8f00e0d7-79c6-4b37-bfea-a3b9362bd229/resourcegroups/yolov8-models/workspaces/testing/datastores/minus_iter_3/paths/minus-iter-3-WITHOUT-miw/"
                    # path="azureml://subscriptions/8f00e0d7-79c6-4b37-bfea-a3b9362bd229/resourcegroups/yolov8-models/workspaces/yolov8-hyp-tuning/datastores/one_iter_1/paths/one-without-mooring/"
                ),
                pretrained=Input(
                    type=AssetTypes.URI_FILE,
                    path="azureml://subscriptions/8f00e0d7-79c6-4b37-bfea-a3b9362bd229/resourcegroups/yolov8-models/workspaces/testing/datastores/minus_iter_2/paths/without-mooring/best.pt"
                ),
                optimizer="SGD",
                iou=0.5,
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
    # Replace these with your Azure subscription details
    subscription_id = '8f00e0d7-79c6-4b37-bfea-a3b9362bd229'
    resource_group = 'yolov8-models'
    workspace = 'testing'
    
    experiment_name = "MINUS-iter3-hyp-tuning-200"
    experiment_display_name = "MINUS iter3 hyp tuning 200 epochs"
    max_conc_trials = 6
    azureml_compute_name = "cluster1"
    ml_studio_env = "yolo_env:1"

    ml_client = conn_ml_wksp(subscription_id, resource_group, workspace)
    job = submit_hp_tuning_job(ml_client, experiment_name, experiment_display_name,azureml_compute_name,max_conc_trials,ml_studio_env)
    print(f"Submitted hyperdrive job: {job.name}")
