import lightkurve as lk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse, os

def download_and_clean(kid: str, quarter: int = 3):
    """Download Kepler light curve and normalize flux."""
    print(f"Downloading {kid} Q{quarter}...")
    search = lk.search_lightcurve(kid, mission='Kepler', quarter=quarter)
    lc = search.download().remove_nans().remove_outliers()
    
    # Normalize
    flux = lc.flux.value
    flux_norm = flux / np.nanmedian(flux)
    time = lc.time.value
    
    return time, flux_norm

def save_csv(time, flux, kid, out_dir="data"):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame({"time": time, "flux": flux})
    path = f"{out_dir}/sample_{kid}.csv"
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return path

def plot_raw(time, flux, kid, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, flux, color='gray', lw=0.5, alpha=0.8)
    ax.set_xlabel("Time (BKJD days)")
    ax.set_ylabel("Normalized Flux")
    ax.set_title(f"Raw Light Curve — {kid}")
    ax.set_facecolor("#f9f9f9")
    plt.tight_layout()
    path = f"{out_dir}/{kid}_raw.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Plot saved: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kid", default="KIC11442793")
    parser.add_argument("--quarter", type=int, default=3)
    args = parser.parse_args()
    
    time, flux = download_and_clean(args.kid, args.quarter)
    save_csv(time, flux, args.kid)
    plot_raw(time, flux, args.kid)
    print("Preprocessing complete.")
