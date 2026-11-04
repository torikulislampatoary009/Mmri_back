# Backend: U-Net tumor segmentation API (Flask + MySQL)

## 1. Install
```
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
(For GPU training install the CUDA build of PyTorch from pytorch.org first.)

## 2. Database
```
mysql -u root -p < schema.sql
cp .env.example .env        # then put your MySQL password in .env
```

## 3a. Quickest: use a public pretrained brain-MRI U-Net (no training)
Add `MODEL_SOURCE=hub` to your `.env` and run the server with internet access the first time.
It downloads a U-Net already trained on the same LGG dataset from PyTorch Hub (`mateuszbuda/brain-segmentation-pytorch`).
Works best on FLAIR brain MRI slices like those in that dataset. This path was not tested in the build environment.

## 3b. Or train your own (required for real results otherwise)
No trained weights are included: training needs the dataset and a GPU.
Use the free Kaggle dataset "LGG MRI Segmentation" (mateuszbuda/lgg-mri-segmentation, 110 patients, FLAIR slices + tumor masks).

On a Kaggle notebook with GPU on (about 20 to 40 minutes for 30 epochs):
```
!git clone <or upload this backend folder>   # you need model.py and train.py
!python train.py --data /kaggle/input/lgg-mri-segmentation/kaggle_3m --epochs 30 --out unet_best.pt
```
Locally: `python train.py --data /path/to/kaggle_3m --epochs 30`

Download `unet_best.pt` and place it at `backend/models/unet_best.pt`.
Training prints validation Dice and IoU each epoch and keeps the best checkpoint. A Dice of roughly 0.80 to 0.90 is typical for this dataset.

For cell nuclei (Data Science Bowl 2018): merge each image's mask files into one mask, then arrange as `data/images/<name>.png` and `data/masks/<name>.png` and pass `--data data`.

## 4. Run
```
python app.py        # http://localhost:5000
```
Without `models/unet_best.pt` the server starts in DEMO mode (a crude brightness filter) so you can test the whole flow. The frontend shows a banner when this is the case.

## API
| Method | Path | Purpose |
|---|---|---|
| GET | /api/health | status, demo mode, model metrics |
| POST | /api/predict | form-data: `image` (file), `threshold` (0.05 to 0.95) |
| GET | /api/history | recent predictions |
| GET / DELETE | /api/history/<id> | one prediction / remove it and its files |
| GET | /files/<images|masks|overlays>/<name> | stored images |

If MySQL is down, /api/predict still returns the segmentation (with `saved: false`).
Research prototype only. Not for clinical use.
