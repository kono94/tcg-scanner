# TCG Scanner

## Setup

Clone the repository without automatically downloading all Git LFS files.

```bash
sudo apt install git-lfs # if not already installed
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/kono94/tcg-scanner.git
cd tcg-scanner
```

Pull only the app model artifacts needed for iOS builds:

```bash
git lfs pull --include "tcg-scanner-app/tcg-scanner-app/Models/**"
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Refresh the bundled app price snapshot:

```bash
python scripts/scrape_prices.py
```

Build the iOS app:

```bash
xcodebuild -project tcg-scanner-app/tcg-scanner-app.xcodeproj -scheme tcg-scanner-app -destination 'generic/platform=iOS' build
```

Export a recognizer CoreML package after training:

```bash
python scripts/export_recognizer_coreml.py
```

https://github.com/user-attachments/assets/992716b5-3d6d-4835-84aa-8eb74fbf4293
