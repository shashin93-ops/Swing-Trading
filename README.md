# Swing Breakout Scanner — Setup Guide

A free, self-hosted dashboard that scans Nifty 500 stocks weekly for
breakout / near-breakout setups, with weekly volume confirmation to
reduce false breakouts.

---

## Part 1 — Run it on your own computer first (10 minutes)

### Step 1: Install Python
If you don't already have Python installed:
- Windows/Mac: download from https://www.python.org/downloads/ (get 3.10 or newer)
- During install on Windows, tick **"Add Python to PATH"**

Check it worked by opening a terminal (Command Prompt / Terminal) and running:
```
python --version
```

### Step 2: Put the files in a folder
Create a folder anywhere, e.g. `swing-scanner`, and place these 3 files inside it:
- `app.py`
- `requirements.txt`
- `README.md` (this file)

### Step 3: Install the dependencies
Open a terminal **inside that folder** and run:
```
pip install -r requirements.txt
```
This installs Streamlit, yfinance, pandas, numpy, and requests. Takes 1–2 minutes.

### Step 4: Run the app
```
streamlit run app.py
```
Your browser should auto-open to `http://localhost:8501` — that's your dashboard.

### Step 5: Use it
- In the sidebar, leave "Limit stocks scanned" around **100** for your first test run
  (scanning all 500 takes a few minutes since it downloads real data).
- Click **Run Scan**.
- If it says it couldn't fetch the Nifty 500 list from NSE (this happens — NSE
  often blocks automated requests), download the list yourself from:
  https://www.niftyindices.com/indices/equity/broad-based-indices/nifty-500
  (there's a "Download list of constituents" link), then upload that CSV using
  the sidebar's uploader and click Run Scan again.
- Once it works, move the "Limit stocks scanned" slider to 500 for the full universe.

### Step 6: Tune your thresholds
Every slider in the sidebar maps directly to the rules we discussed:
- Compression band, near-high %, volume expansion multiple, and extension cap
- You have **two independent sets** — Strict (your buy list) and Watchlist (looser, for tracking setups)
- Adjust and re-run until the result counts feel right for your style — same
  debugging approach as we used in Chartink: loosen one at a time if you get zero results.

---

## Part 2 — Put it on the free web (so you get a URL, not just localhost)

### Step 1: Create a free GitHub account
https://github.com/signup (skip if you already have one)

### Step 2: Create a new repository
- Click **New repository** → name it e.g. `swing-scanner` → set to **Public** → Create.
- Upload the same 3 files (`app.py`, `requirements.txt`, `README.md`) using
  GitHub's "Add file → Upload files" button in the browser — no command line needed.

### Step 3: Create a free Streamlit Community Cloud account
https://share.streamlit.io → Sign in with GitHub.

### Step 4: Deploy
- Click **New app**
- Select your `swing-scanner` repository, branch `main`, main file `app.py`
- Click **Deploy**

Streamlit installs your requirements automatically and gives you a public URL
like `https://your-name-swing-scanner.streamlit.app` — bookmark it, open it every
Friday/Saturday, click Run Scan.

### Notes on the hosted version
- Streamlit Cloud free tier apps "sleep" after inactivity — the first load
  after a few idle days takes ~30 seconds to wake up. Normal.
- Since it's public, don't put any private API keys or personal data in the repo.
- If NSE blocks the live fetch from Streamlit's servers (more likely than from
  your home computer), just use the CSV upload option — same as local use.

---

## What to tell me once it's running

Once you've run it for real, send me:
- The result counts for Strict vs Watchlist
- A couple of example rows

...and we'll tune the thresholds together, exactly like we did with the Chartink version.
