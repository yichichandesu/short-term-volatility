import datetime as dt
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

HOME_PATH = os.environ['HOME']

# put your data in this folder or change forder name accordingly.
DATA_PATH = os.path.join(HOME_PATH, 'data', 'fordham')

# this is where your output get written
OUTPUT_PATH = os.path.join(HOME_PATH, 'output', 'fordham')
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
    print(f"Created directory: {OUTPUT_PATH}")


def get_files(dt_start, dt_end, exchange, instrument_type, symbol):
    span_days = pd.date_range(dt_start, dt_end, freq="D").strftime("%Y%m%d")
    files_for_range = []
    for day in span_days:
        file_path = Path(os.path.join(DATA_PATH, exchange, instrument_type, "level1", day,
                                      symbol + "." + day + ".level1.1min.csv.gz"))

        if not file_path.exists():
            print(f"[FILE NOT FOUND] {file_path}")
            continue
        file_for_day = pd.read_csv(file_path)
        files_for_range.append(file_for_day)
    data = pd.concat(files_for_range)
    return data


def main():
    exchange = "binance"
    instrument_type = "future"

#change dates to cover different periods
    start_date = "20240101"
    end_date = "20240201"

    start_dt = dt.datetime.strptime(start_date, "%Y%m%d")
    end_dt = dt.datetime.strptime(end_date, "%Y%m%d")

    data = get_files(start_dt, end_dt, exchange, instrument_type, "BTCUSDT")
    print("Read ", len(data), " rows")
    print("Keys:", data.keys())

    # Plot midprice over time
    plt.figure(figsize=(20, 5))
    plt.plot(data["ts_end"], data["close_mid"])

    plt.xlabel("ts")
    plt.ylabel("price")
    plt.title("Mid Price")

    file_path = os.path.join(OUTPUT_PATH, 'close_mid.png')
    plt.savefig(file_path)
    print("Plot saved to ", file_path)
    plt.close()

    # calculating mid to mid return
    data["ret"] = ((data["close_mid"] - data["close_mid"].shift(1)) / data["close_mid"].shift(1)).ffill().fillna(0)

    plt.figure(figsize=(10, 10))
    plt.hist(data["ret"], bins=100)

    plt.yscale("log")  # Set y-axis to logarithmic scale
    plt.xlabel("Returns")
    plt.ylabel("Frequency (Log Scale)")
    plt.title("Histogram of Price Returns")

    file_path = os.path.join(OUTPUT_PATH, 'mid_return_hist.png')
    plt.savefig(file_path)
    print("Plot saved to ", file_path)
    plt.close()


if __name__ == "__main__":
    main()
