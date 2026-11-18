# Phase 4 — Gradio and Hugging Face Spaces

**Status:** The Gradio application is implemented and smoke-tested locally. Automated deployment is configured but has not been activated. A Hugging Face Space must be created and repository secrets must be added.

## Local launch

```bash
pip install -r requirements.txt
python app.py
```

The app supports five heuristic policies. It also shows the trained DQN policy when `artifacts/dqn_warehouse.pt` exists in the app directory.

## Create the Hugging Face Space

1. Create a new Space at https://huggingface.co/new-space.
2. Select **Gradio** as the SDK.
3. Select a Space name. Record the Space ID as `account-name/space-name`.
4. Do not add a second README or application entry point. The repository README contains the Space metadata.

## Enable deployment from GitHub Actions

In the GitHub repository, open **Settings → Secrets and variables → Actions** and add:

- `HF_TOKEN`: a Hugging Face access token with write permission for the Space.
- `HF_SPACE_ID`: the Space ID in `owner/name` format.

After both secrets exist, pushes to the `main` branch run tests and deploy the app only if the tests pass. Pull requests run tests but do not deploy.

Workflow file: `.github/workflows/test-and-deploy-space.yml`.

## DQN checkpoint

The checkpoint is not committed to this Git repository. To enable the DQN option in the Space:

1. Train a checkpoint with `python -m training.train_dqn`.
2. Upload `artifacts/dqn_warehouse.pt` to the Space repository, or add it through a supported Git LFS workflow.
3. Restart the Space.

Without the checkpoint, the five heuristic policies remain available.

## Local smoke validation

The app import and a complete Gradio callback run are covered by `tests/test_app.py`. The demo reports reshuffles, retrieved items, failures, delay, episode return, and decision count.

## Deployment status

No Space URL is claimed here. Deployment is complete only after the user creates the Space, adds both GitHub Actions secrets, and the workflow reports a successful deploy job.
