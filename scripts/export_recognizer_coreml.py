#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from pathlib import Path

import coremltools as ct
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model import CardModel


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
    image_labels = labels_path.read_text(encoding="utf-8").splitlines()
    metadata_by_id: dict[str, dict] = {}

    for json_path in cards_dir.glob("*/*.json"):
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the PyTorch card recognizer to CoreML.")
    parser.add_argument("--state-dict", default="mobile_large_v1_state_dict.pth", type=Path)
    parser.add_argument("--labels", default="names.txt", type=Path)
    parser.add_argument("--cards-dir", default="cards", type=Path)
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
    args = parser.parse_args()

    labels = load_labels(args.labels, args.cards_dir)
    state_dict = torch.load(args.state_dict, map_location="cpu", weights_only=True)
    output_count = state_dict["classification_layer.weight"].shape[0]
    if len(labels) != output_count:
        raise ValueError(f"Label count {len(labels)} does not match model output count {output_count}")

    model = CardModel(num_labels=output_count)
    model.load_state_dict(state_dict)
    model.eval()

    wrapped = LogitsOnlyModel(model).eval()
    example_input = torch.rand(1, 3, 224, 224)
    traced = torch.jit.trace(wrapped, example_input)

    # CoreML image preprocessing applies a single channel scale, so use the
    # average MobileNetV3 std and keep exact per-channel mean offsets.
    average_std = sum(IMAGENET_STD) / len(IMAGENET_STD)
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

    if args.model_output.exists():
        shutil.rmtree(args.model_output)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    coreml_model.save(args.model_output)

    args.labels_output.parent.mkdir(parents=True, exist_ok=True)
    with args.labels_output.open("w", encoding="utf-8") as handle:
        json.dump(labels, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    print(f"Wrote {args.model_output}")
    print(f"Wrote {len(labels)} labels to {args.labels_output}")


if __name__ == "__main__":
    main()
