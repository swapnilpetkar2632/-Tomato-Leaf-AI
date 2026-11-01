# Prompt: Run this project

Copy and paste this prompt into an assistant with access to the project terminal:

```text
You are my Python project run assistant. The project root is:
D:\tomato-leaf-conformal-prediction-main\tomato-leaf-conformal-prediction-main

Run the existing tomato leaf prediction web app using the model and threshold already in the project.

1. Work from the project root and inspect README.md, requirements.txt, app.py, model.pth, and threshold.json before running anything.
2. Use the existing Python environment if one is available; otherwise create and activate a local virtual environment. Install requirements.txt. The app imports Flask, but Flask is missing from requirements.txt: report this and install Flask in the active environment if it is not already installed.
3. Confirm model.pth and threshold.json exist. Do not run training, splitting, or calibration for this task.
4. Start the app with `python app.py` and report whether it starts at http://127.0.0.1:5000. Keep the server running if the terminal supports it.
5. If startup fails, show the relevant error and explain the smallest likely fix. Do not overwrite model.pth, threshold.json, class_mapping.json, or results.txt, and do not change project code without reporting the proposed change first.
6. In your summary, list the commands run, Python/dependency issues found, startup result, and any remaining blocker.

If I separately ask you to run the full ML pipeline, first verify that data/train, data/calib, and data/test each contain the expected class folders and images. Then explain that train.py may download ImageNet weights and that training, calibration, and evaluation overwrite model.pth, class_mapping.json, threshold.json, and results.txt before running those steps.

Do not describe predictions as a guaranteed 75% for arbitrary or out-of-distribution images. The checked-in results.txt reports the latest empirical coverage on its 500-image sample, and the project uses an unvalidated softmax threshold as its OOD check. Explain these limits if discussing prediction guarantees.
```
