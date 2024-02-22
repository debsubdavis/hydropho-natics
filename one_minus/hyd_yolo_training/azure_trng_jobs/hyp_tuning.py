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

def submit_hp_tuning_job(ml_client, exp_name, exp_display_name):
#    from azure.ai.ml.entities import CommandJob
#    from azure.ai.ml.sweep import BayesianSamplingAlgorithm, Objective, SweepJob, SweepJobLimits

#    command_job = CommandJob(
#        inputs=dict(kernel="linear", penalty=1.0),
#        compute=cpu_cluster,
#        environment=f"{job_env.name}:{job_env.version}",
#        code="./scripts",
#        command="python scripts/train.py --kernel $kernel --penalty $penalty",
#        experiment_name="sklearn-iris-flowers",
#    )

#    sweep = SweepJob(
#        sampling_algorithm=BayesianSamplingAlgorithm(),
#        trial=command_job,
#        search_space={"ss": Choice(type="choice", values=[{"space1": True}, {"space2": True}])},
#        inputs={"input1": {"file": "top_level.csv", "mode": "ro_mount"}},
#        compute="top_level",
#        limits=SweepJobLimits(trial_timeout=600),
#        objective=Objective(goal="maximize", primary_metric="accuracy"),
#    )


    sweep_job = SweepJob(
        compute="e4ds-cluster3", #Standard_E4s_v3
        experiment_name="SJ-" + exp_name,
        display_name="SJ: " + exp_display_name,
        #sampling_algorithm="random",
        sampling_algorithm=GridSamplingAlgorithm(),
        objective=Objective(goal="maximize", primary_metric="metrics/mAP50B"),
        #limits=Limit(max_total_trials=20, max_concurrent_trials=4, trial_timeout_minutes=60),
        #limits=SweepJobLimits(max_total_trials=20, max_concurrent_trials=4, trial_timeout=3600),
        limits=SweepJobLimits(max_concurrent_trials=24),
        search_space=dict(
            batch_size=Choice([8, 16, 32]),
            optimizer=Choice(["Adam", "SGD"]),
            lr0=Choice([0.01, 0.001]),
            iou=Choice([0.3, 0.5]),
            dropout=Choice([0.2, 0.5])
        ),
        trial=CommandJob(
            code="/Users/debbiesubocz/GitReps/hydropho-natics/one_minus/hyd_yolo_training/azure_trng_jobs",  # Adjust to the actual path of your training script
            command="python train_model.py --data_dir ${{inputs.data_dir}} --exp_name ${{inputs.exp_name}} --epochs ${{inputs.epochs}} --patience ${{inputs.patience}}",
            environment="yolo_env:2",
            inputs=dict(
                data_dir=Input(
                    type=AssetTypes.URI_FOLDER,
                    path="azureml://subscriptions/8f00e0d7-79c6-4b37-bfea-a3b9362bd229/resourcegroups/yolov8-models/workspaces/yolov8-hyp-tuning/datastores/minus_iter_2_fo/paths/without-mooring/training_dataset/"
                    # path="azureml://subscriptions/8f00e0d7-79c6-4b37-bfea-a3b9362bd229/resourcegroups/yolov8-models/workspaces/yolov8-hyp-tuning/datastores/one_iter_1/paths/one-without-mooring/"
                ),
                exp_name=exp_name,
                epochs=500,
                patience=100
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
    workspace = 'yolov8-hyp-tuning'
    
    experiment_name = "MINUS-iter2-hyp-tuning-500"
    experiment_display_name = "MINUS iter2 hyp tuning 500 epochs"

    ml_client = conn_ml_wksp(subscription_id, resource_group, workspace)
    job = submit_hp_tuning_job(ml_client, experiment_name, experiment_display_name)
    print(f"Submitted hyperdrive job: {job.name}")
