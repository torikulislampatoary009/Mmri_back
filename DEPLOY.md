# Deploy: Render (free web service) + Aiven (free MySQL)

## 1. Create the database on Aiven
1. Create a MySQL service on the free plan.
2. When it is running, open its Overview page and copy:
   - the **Service URI** (looks like `mysql://avnadmin:PASSWORD@HOST:PORT/defaultdb?ssl-mode=REQUIRED`)
   - the **CA certificate** (download `ca.pem`, or copy its text)
3. In Aiven's "Allowed IP addresses" (if shown) make sure connections from Render are permitted (the default allows all).

No SQL needed: the app creates its two tables on first connection.

## 2. Push the backend to GitHub
Unzip backend.zip and push the folder contents as the root of a new repository (`app.py`, `render.yaml`, etc. must sit at the top level).
If you train your own model, also commit `models/unet_best.pt` (about 30 MB, fine for Git) and delete the `MODEL_SOURCE` entry in `render.yaml`.

## 3. Create the Render service
Render dashboard > New > **Blueprint** > pick the repo (it reads `render.yaml`). Or New > Web Service with:
- Build command: `pip install -r requirements-render.txt && python download_model.py`
- Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120`
- Environment variables: `PYTHON_VERSION=3.11.9`, `MODEL_SOURCE=hub`, `TORCH_HOME=/opt/render/project/src/torch_cache`

Then add the secret variables:
| Variable | Value |
|---|---|
| `DB_URL` | the Aiven Service URI |
| `DB_SSL_CA_PEM` | the full text of `ca.pem` (paste it as is) |
| `ALLOWED_ORIGIN` | your frontend address, e.g. `https://yourname.netlify.app` (use `*` while testing) |

After the deploy goes live, open `https://YOUR-SERVICE.onrender.com/api/health?db=1`. You want `"db_ok": true`.

## 4. Point the frontend at it
In the frontend's `config.js` replace `https://YOUR-SERVICE-NAME.onrender.com` with your Render URL, then host the frontend folder anywhere static (Render Static Site, Netlify, GitHub Pages).

## Free-tier things to know
- **Cold starts:** Render free services sleep after about 15 minutes idle and take up to a minute to wake. The frontend shows "waking the server" and retries.
- **Memory:** the free instance has 512 MB. PyTorch plus this model should fit for single requests, but it is tight. Images are shrunk to 1024 px on the longest side to help. If Render reports "out of memory", the options are a paid instance or running without the model (demo mode).
- **Disk:** it is wiped on restart, which is why images are stored in MySQL rather than on disk. Keep an eye on Aiven's free storage limit; use the Delete button in the app to clear old scans.
- **Aiven free plan limits** (storage, idle power-off) can change, so check the Aiven console if the database stops responding.
- The `hub` model runs code from the `mateuszbuda/brain-segmentation-pytorch` GitHub repo at build time (`trust_repo=True`). Skip it by training your own model instead.
