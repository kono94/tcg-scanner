#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from pathlib import Path

import coremltools as ct
import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "training" / "card_recognizer"))

from model import CardModel
from pipeline.experiments import DEFAULT_EXPERIMENTS_ROOT, checkpoint_for_experiment, experiment_dir, load_experiment_config, write_experiment_metadata
from pipeline.paths import resolve_path


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class LogitsOnlyModel(torch.nn.Module):
    def __init__(self, model: CardModel):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        _, logits = self.model(image)
        return logits


def load_labels(labels_path: Path, cards_dir: Path) -> list[dict[str, str | int]]:
    print(f"Loading labels from {labels_path}")
    image_labels = labels_path.read_text(encoding="utf-8").splitlines()
    metadata_by_id: dict[str, dict] = {}

    json_paths = sorted(cards_dir.glob("*/*.json"))
    for json_path in tqdm(json_paths, desc="Loading card metadata", unit="file"):
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        card_id = data.get("id") or json_path.stem
        metadata_by_id[card_id] = data

    labels: list[dict[str, str | int]] = []
    for index, image_label in enumerate(image_labels):
        card_id = Path(image_label).stem
        metadata = metadata_by_id.get(card_id, {})
        labels.append(
            {
                "classIndex": index,
                "cardID": card_id,
                "imageLabel": image_label,
                "name": metadata.get("name") or card_id,
            }
        )

    return labels


def load_model_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    print(f"Loading checkpoint: {checkpoint_path}")
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
        prefix = "model."
        if any(key.startswith(prefix) for key in state):
            state = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
        return {key.replace("embedding_layer.", "feature_layer."): value for key, value in state.items()}
    if isinstance(raw, dict):
        return {key.replace("embedding_layer.", "feature_layer."): value for key, value in raw.items()}
    raise TypeError(f"Unsupported checkpoint format: {checkpoint_path}")


def run_torch_smoke_test(model: CardModel, output_count: int) -> None:
    print("Running PyTorch smoke test...")
    with torch.inference_mode():
        features, logits = model(torch.rand(1, 3, 224, 224))
    if tuple(logits.shape) != (1, output_count):
        raise AssertionError(f"Unexpected logits shape {tuple(logits.shape)}")
    if features.ndim != 2 or features.shape[0] != 1:
        raise AssertionError(f"Unexpected feature shape {tuple(features.shape)}")


def run_coreml_smoke_test(coreml_model, output_count: int) -> None:
    from PIL import Image

    print("Running CoreML smoke test...")
    image = Image.new("RGB", (224, 224), color=(127, 127, 127))
    prediction = coreml_model.predict({"image": image})
    logits = prediction["logits"]
    if logits.shape[-1] != output_count:
        raise AssertionError(f"Unexpected CoreML logits shape {logits.shape}")


def update_app_manifest(manifest_path: Path, experiment_id: str) -> None:
    payload = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    payload["recognizerVersion"] = f"experiment:{experiment_id}"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the PyTorch card recognizer to CoreML.")
    parser.add_argument("--experiment-id", required=True, help="Experiment ID under the experiments root.")
    parser.add_argument("--experiments-root", default=DEFAULT_EXPERIMENTS_ROOT, type=Path)
    parser.add_argument("--checkpoint", default=None, type=Path, help="Lightning checkpoint or plain CardModel state dict.")
    parser.add_argument("--labels", default=None, type=Path)
    parser.add_argument("--cards-dir", default=None, type=Path)
    parser.add_argument(
        "--model-output",
        default="tcg-scanner-app/tcg-scanner-app/Models/card_recognizer.mlpackage",
        type=Path,
    )
    parser.add_argument(
        "--labels-output",
        default="tcg-scanner-app/tcg-scanner-app/Resources/recognizer_labels.json",
        type=Path,
    )
    parser.add_argument(
        "--app-manifest-output",
        default="tcg-scanner-app/tcg-scanner-app/Resources/app_model_manifest.json",
        type=Path,
    )
    parser.add_argument(
        "--install-to-app",
        action="store_true",
        help="Write directly to the app bundle paths and stamp recognizerVersion with the experiment ID.",
    )
    parser.add_argument("--skip-coreml-smoke-test", action="store_true")
    args = parser.parse_args()

    run_dir = experiment_dir(args.experiment_id, root=args.experiments_root)
    cfg = load_experiment_config(args.experiment_id, args.experiments_root)
    checkpoint_path = resolve_path(args.checkpoint) if args.checkpoint else checkpoint_for_experiment(run_dir)
    labels_path = resolve_path(args.labels) if args.labels else resolve_path(cfg["dataset"]["labels_path"])
    cards_dir = resolve_path(args.cards_dir) if args.cards_dir else resolve_path(cfg["dataset"]["cards_dir"])
    print(f"Exporting recognizer for experiment {args.experiment_id}")
    print(f"Experiment directory: {run_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Cards directory: {cards_dir}")
    if not args.install_to_app:
        args.model_output = run_dir / "app_export" / "card_recognizer.mlpackage"
        args.labels_output = run_dir / "app_export" / "recognizer_labels.json"
    print(f"CoreML output: {args.model_output}")
    print(f"Labels output: {args.labels_output}")

    state_dict = load_model_state(checkpoint_path)
    output_count = state_dict["classification_layer.weight"].shape[0]
    feature_dim = state_dict["classification_layer.weight"].shape[1]
    labels = load_labels(labels_path, cards_dir)
    if len(labels) != output_count:
        raise ValueError(f"Label count {len(labels)} does not match model output count {output_count}")

    model = CardModel(num_labels=output_count, feature_dim=feature_dim, pretrained=False)
    model.load_state_dict(state_dict)
    model.eval()
    run_torch_smoke_test(model, output_count)

    wrapped = LogitsOnlyModel(model).eval()
    example_input = torch.rand(1, 3, 224, 224)
    print("Tracing Torch model...")
    traced = torch.jit.trace(wrapped, example_input)

    # CoreML image preprocessing applies a single channel scale, so use the
    # average MobileNetV3 std and keep exact per-channel mean offsets.
    average_std = sum(IMAGENET_STD) / len(IMAGENET_STD)
    print("Converting Torch model to CoreML. This can take a while...")
    coreml_model = ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="image",
                shape=example_input.shape,
                color_layout=ct.colorlayout.RGB,
                scale=1.0 / (255.0 * average_std),
                bias=[-mean / std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD)],
            )
        ],
        outputs=[ct.TensorType(name="logits")],
        minimum_deployment_target=ct.target.iOS16,
        convert_to="mlprogram",
    )
    coreml_model.short_description = "One Piece TCG exact-card recognizer."
    coreml_model.input_description["image"] = "Detected card crop."
    coreml_model.output_description["logits"] = "Unnormalized class scores aligned with recognizer_labels.json."
    coreml_model.user_defined_metadata["label_count"] = str(len(labels))
    coreml_model.user_defined_metadata["source_checkpoint"] = str(checkpoint_path)
    coreml_model.user_defined_metadata["experiment_id"] = args.experiment_id

    if args.model_output.exists():
        shutil.rmtree(args.model_output)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    print("Saving CoreML package...")
    coreml_model.save(args.model_output)

    if not args.skip_coreml_smoke_test:
        try:
            loaded_coreml_model = ct.models.MLModel(args.model_output)
            run_coreml_smoke_test(loaded_coreml_model, len(labels))
            print("CoreML smoke test passed")
        except Exception as exc:
            print(f"CoreML smoke test skipped/failed on this host: {exc}")

    args.labels_output.parent.mkdir(parents=True, exist_ok=True)
    print("Writing recognizer labels...")
    with args.labels_output.open("w", encoding="utf-8") as handle:
        json.dump(labels, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    if args.install_to_app:
        print("Updating app model manifest...")
        update_app_manifest(args.app_manifest_output, args.experiment_id)
    write_experiment_metadata(
        run_dir,
        {
            "coreml_export": str(args.model_output),
            "coreml_labels": str(args.labels_output),
            "installed_to_app": bool(args.install_to_app),
        },
    )

    print(f"Wrote {args.model_output}")
    print(f"Wrote {len(labels)} labels to {args.labels_output}")


if __name__ == "__main__":
    main()
