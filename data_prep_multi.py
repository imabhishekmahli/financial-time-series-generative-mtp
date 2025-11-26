import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ========= CONFIG =========
ASSETS = [
    {
        "name": "sp500",
        "ticker": "^GSPC",
        "start": "2010-01-01",
        "end":   "2024-01-01",
    },
    {
        "name": "goog",
        "ticker": "GOOG",
        "start": "2010-01-01",
        "end":   "2024-01-01",
    },
    {
        "name": "corn",
        "ticker": "ZC=F",  # Corn futures (Yahoo Finance)
        "start": "2010-01-01",
        "end":   "2024-01-01",
    },
]

SEQ_LEN = 60  # 60-day windows


def download_close_prices(ticker, start, end):
    print(f"\nDownloading {ticker} from {start} to {end} ...")
    data = yf.download(ticker, start=start, end=end)
    print("Columns:", data.columns)
    if "Close" not in data.columns:
        raise RuntimeError(f"No 'Close' column found for {ticker}")
    close = data["Close"].dropna().to_numpy().astype(float).reshape(-1)
    print(f"Close shape for {ticker}:", close.shape)
    return close


def prices_to_log_returns(prices):
    log_p = np.log(prices)
    rets = np.diff(log_p)
    print("Returns shape:", rets.shape)
    return rets


def make_sequences(returns, seq_len):
    xs = []
    for i in range(len(returns) - seq_len):
        xs.append(returns[i:i+seq_len])
    xs = np.array(xs, dtype=np.float32)
    print("Sequences shape:", xs.shape)
    return xs


def quick_plots(asset_name, returns, seqs):
    # First 300 returns
    plt.figure(figsize=(10,4))
    plt.plot(returns[:300])
    plt.title(f"First 300 log returns ({asset_name})")
    plt.tight_layout()
    fname1 = f"{asset_name}_returns_first300.png"
    plt.savefig(fname1, dpi=150)
    plt.close()
    print("Saved:", fname1)

    # Example sequences
    plt.figure(figsize=(10,6))
    for i in range(min(5, len(seqs))):
        plt.plot(seqs[i], alpha=0.8, label=f"Seq {i+1}")
    plt.title(f"Example {SEQ_LEN}-day windows ({asset_name})")
    plt.legend()
    plt.tight_layout()
    fname2 = f"{asset_name}_example_sequences.png"
    plt.savefig(fname2, dpi=150)
    plt.close()
    print("Saved:", fname2)


if __name__ == "__main__":
    for asset in ASSETS:
        name   = asset["name"]
        ticker = asset["ticker"]
        start  = asset["start"]
        end    = asset["end"]

        print("\n==============================")
        print(f"Processing asset: {name} ({ticker})")
        prices = download_close_prices(ticker, start, end)
        rets   = prices_to_log_returns(prices)
        seqs   = make_sequences(rets, SEQ_LEN)

        out_file = f"{name}_returns_sequences.npy"
        np.save(out_file, seqs)
        print(f"Saved sequences to {out_file}")

        quick_plots(name, rets, seqs)

    print("\nData prep for all assets done.")

