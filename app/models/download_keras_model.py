from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="keras-io/monocular-depth-estimation",
    local_dir="./monocular_keras",
    cache_dir=None,           # disable the shared cache; force download into our folder
    resume_download=True      # safe to re-run if interrupted
)
