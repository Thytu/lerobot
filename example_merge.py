from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_ids = ["Thytu/so101-object-in-box_v0.1", "Thytu/so101-object-in-box_v0.1_multi-shot"]
new_repo_id = "Thytu/so101-object-in-box_v0.2"

merged_ds = LeRobotDataset.merge(repo_ids, new_repo_id)
merged_ds.push_to_hub()
