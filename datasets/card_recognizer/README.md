Custom dataset format in order to train the card recognizer.
Mainly there are subfolders of each OP series with one image as .jpg and the corresponding .json with additional card information, like the original image url.

Actually, only the .jpg and the name of the .jpg is important for training as each card ID is a class to learn for the classifier.

The script-based pipeline now trains on all scraped images under `datasets/card_recognizer/cards/`.
Add real phone-camera validation examples under:

```text
datasets/card_recognizer/manual_eval/<card-id>/
```

For example:

```text
datasets/card_recognizer/manual_eval/OP01-001/IMG_1001.jpg
```

Validation uses this manual folder.

Multiple images for one class are supported. Keep the scraped flat layout:

```text
datasets/card_recognizer/cards/OP01/OP01-001.jpg
```

and add extra samples with a double-underscore suffix:

```text
datasets/card_recognizer/cards/OP01/OP01-001__phone-front.jpg
```

or use a class folder:

```text
datasets/card_recognizer/cards/OP01/OP01-001/phone-front.jpg
```

Only `__` means "same class, extra sample". A card like `OP01-001_p1` is treated as its own class.

## Real-Life Card Photos

Use `scripts/import_real_card_images.py` to import real card photos that you own or have permission to use.
Do not point this at sites that reserve rights against AI training or that block automated scraping.

Create a CSV or JSONL source manifest with `card_id` plus either `image_path` or `image_url`:

```csv
card_id,image_path,source,license,notes
OP03-056,/absolute/path/to/phone-photo.jpg,manual capture,owned,front photo on desk
```

Import raw normalized JPEGs without changing the training set:

```bash
python scripts/import_real_card_images.py \
  --set OP03 \
  --source-manifest sources.csv
```

Stage imported files directly as extra training samples:

```bash
python scripts/import_real_card_images.py \
  --set OP03 \
  --source-manifest sources.csv \
  --promote-to-training
```

Raw imports are written under:

```text
datasets/card_recognizer/real_raw/OP03/<card-id>/
```

Promoted training files use the existing extra-sample convention:

```text
datasets/card_recognizer/cards/OP03/OP03-056__real-<hash>.jpg
```

## Google Lens Sale Image Collection

Use `scripts/crawl_google_lens_sales_images.py` to collect Google Lens result-image candidates for every card in a set. The crawler starts from the official card image URL in each card JSON file, adds the Lens text query `sale` by default, and saves normalized JPEG candidates plus JSON sidecars under `real_raw`.

Install the browser dependency once:

```bash
python -m pip install playwright
python -m playwright install chromium
```

Run a small review crawl first:

```bash
python scripts/crawl_google_lens_sales_images.py \
  --set OP03 \
  --card-id OP03-056 \
  --headed
```

Run a full set crawl:

```bash
python scripts/crawl_google_lens_sales_images.py \
  --set OP03 \
  --max-results 12 \
  --headed
```

Raw Lens imports are written under:

```text
datasets/card_recognizer/real_raw/OP03/<card-id>/
```

After manually reviewing rights, label correctness, and image quality, either re-run with `--promote-to-training` or copy selected images into the training folder using the existing extra-sample convention:

```text
datasets/card_recognizer/cards/OP03/OP03-056__google-lens-<hash>.jpg
```

You can change the Lens text query while still using the official card image as the visual seed:

```bash
python scripts/crawl_google_lens_sales_images.py \
  --set OP03 \
  --query-template "{card_id} {name} sale"
```
