# Running the assistant on Kaggle (free GPU)

Your laptop has no GPU, so the AI is slow locally. Kaggle gives a **free GPU**
where it runs in **seconds**. Use this to test it and to demo it to the client.

## Steps
1. Go to **https://www.kaggle.com/code** → **New Notebook**.
2. **File → Import Notebook** → upload `kaggle/run_on_kaggle.ipynb` from this repo.
   (Or **File → Import** from the GitHub URL.)
3. In the right sidebar **Settings**:
   - **Accelerator** → `GPU T4 x2`
   - **Internet** → `On`  (required to download models)
4. Click **Run All**.
5. Wait for the models to download (first run only). The cell labelled **6** prints:
   ```
   OPEN THIS LINK IN YOUR BROWSER: https://something.trycloudflare.com
   ```
6. Open that link → tap the 🎤 mic and speak in **English / Urdu / Arabic**, or type.

## Notes
- The public link works **only while the notebook is running**. It's for testing
  and demos — not the final hosting.
- Want sharper answers? In the notebook, change `qwen2.5:3b` to `qwen2.5:7b`
  (cells 2, 4, 5). The T4 GPU can handle it.
- For the **real client website**, deploy the same project on the client's own
  server and use a permanent domain — see `../INTEGRATION.md`.

## Why Kaggle can't be the final home
Kaggle sessions stop after a few hours and limit you to ~30 GPU hours/week. It's
perfect for testing on a GPU, but the client needs an always-on server for
production.
