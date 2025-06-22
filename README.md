### Setup
Clone repository but do not automatically download all git-lfs files.

```bash
sudo apt install git-lfs # if not already isntalled
set GIT_LFS_SKIP_SMUDGE=1 && \
git clone https://github.com/kono94/tcg-scanner.git && \ 
cd tcg-scanner
```

Pull the model weights:
```bash
git lfs pull app/weights/*
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

```bash
export PYTHONPATH=$(pwd)
python app/video.py IMG_3374.mp4
```

Start webservice with templating frontend to upload USB camera images
```bash
uvicorn app.main:app --reload
```

Scrape current prices of OP cards:
```bash
python app/scrape_prices.py
```

https://github.com/user-attachments/assets/992716b5-3d6d-4835-84aa-8eb74fbf4293
