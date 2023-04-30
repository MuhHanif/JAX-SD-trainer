from huggingface_hub import login, HfApi
import os

api_token = "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"
model_collection_path = "/root/project/dataset"
repo_id = "MuhammadHanif/stable-diffusion-v1-5-high-res"
path_in_repo = ""

for epoch in range(73, 99):
    model_folder = f"stable-diffusion-v1-5-flax-e1-{epoch}"

    login(api_token)  # enter private token here
    api = HfApi()
    model_path = os.path.join(model_collection_path, model_folder)

    api.upload_folder(
        folder_path=model_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns="**/logs/*.txt",
        commit_message=f"steps {epoch*10000}",
    )
