# Prompt: Fix the project issues

Copy and paste the prompt below into an AI coding agent with access to this repository.

```text
You are a senior Python/PyTorch engineer. Fix the tomato leaf conformal prediction project in:
D:\tomato-leaf-conformal-prediction-main\tomato-leaf-conformal-prediction-main

Goal: make the documented workflow runnable, keep the model artifacts compatible, and ensure the UI does not make unsupported conformal prediction claims.

First inspect README.md, PROJECT_PLAN.md, requirements.txt, app.py, predict.py, split_data.py, train.py, calibrate.py, evaluate.py, templates/index.html, threshold.json, class_mapping.json, and results.txt. Do not overwrite model.pth or regenerate model artifacts unless explicitly necessary.

Fix these issues:

1. Add Flask to requirements.txt because app.py imports it.
2. Make split_data.py either implement the documented `--src PATH` option or remove the stale documentation. Prefer implementing it with argparse, preserving the current default data directory behavior. Validate the source path before copying files.
3. Fix the `/api/predict` sample_path branch. If sample_path is supported, safely resolve it only inside DATA_TEST and reject traversal or arbitrary filesystem paths. If it is not needed by the UI, remove the unused branch and validation message.
4. Remove the hardcoded training assumption where possible: derive the number of classes from ImageFolder and use that value for class indices and the model output. Fail clearly if the expected tomato classes are missing. Keep class mapping consistent across training, calibration, evaluation, and inference.
5. Make the conformal set behavior consistent between evaluate.py and predict.py. Do not silently replace an empty conformal set with a heuristic set while calling it a certified set. Either return an explicit empty-set status or document and label any fallback as a non-conformal display fallback. Evaluation and UI must use the same rule.
6. Correct the product language. Do not claim a per-image, per-class, legal, or arbitrary-OOD 75% guarantee. Explain that split conformal coverage is a marginal in-distribution guarantee under its assumptions. Show the calibrated target and empirical evaluation result separately.
7. Treat the current top-softmax >= 0.20 check as a heuristic, not a validated OOD detector. Rename fields or UI text if needed so users are not told that non-leaf images are reliably detected. Preserve the existing response schema where practical.
8. Update README.md and PROJECT_PLAN.md so commands, options, dependencies, class-count behavior, and guarantees match the implementation.

Implementation rules:
- Keep the solution small and readable; avoid new dependencies unless required.
- Preserve existing model architecture and artifact formats where possible.
- Use pathlib and safe path checks for filesystem input.
- Return clear HTTP 4xx errors for invalid uploads or sample paths and avoid exposing internal tracebacks in production responses.
- Do not change the dataset, model.pth, threshold.json, class_mapping.json, or results.txt merely to make a check pass.

Verification:
1. Run a syntax/compile check for all Python files.
2. Run the project’s environment check if dependencies are available.
3. Exercise the Flask app with its test client: GET `/`, POST `/api/predict` with no file, and POST a valid image if one is available. Confirm status codes and response shape.
4. Run split_data.py --help and confirm the documented option behavior.
5. Run a static search for stale claims such as “legal guarantee”, “75% guarantee” for each prediction, and undocumented options.
6. Do not run training or calibration unless a required compatibility check proves it is necessary; if you do, explain which artifacts will be overwritten before running.

Final response format:
- Files changed
- Issues fixed
- Verification commands and results
- Any remaining limitations, especially coverage and OOD limitations
```
