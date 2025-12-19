import sys
sys.path.insert(0,'./src')

from butterflow import lex, Parser, TypeChecker, Builder, Runtime
import numpy as np 

def main():
  input_code = """
  close : Signal<Float> = data(id="price")
  volume : Signal<Float> = data(id="volume")
  adv20 : Signal<Float> = ts_mean(signal=volume, period=20)
  volume_level : Signal<Float> = divide(dividend=volume, divisor=adv20)

  dynamic_ma(signal: Signal<Float>, lookback: Signal<Float>, multiplier: Signal<Float>) : Signal<Float> = {
    lb : Signal<Float> = multiply ( baseline = lookback, multiplier = multiplier )
    result : Signal<Float> = ts_mean(signal = signal, period = lb)
  }

  lookback : Float = 10.
  result : Signal<Float> = dynamic_ma(signal=close, lookback=lookback, multiplier=volume_level)
  """

  # 1. Lex & Parse
  tokens = lex(input_code)
  parser = Parser(tokens)
  ast = parser.parse()

  # 2. Type Check (Strict validation before graph build)
  checker = TypeChecker()
  checker.check(ast)

  # 3. Build Graph
  builder = Builder()
  graph = builder.build(ast)
  print(f"\nFinal Graph Result:\nresult = {graph}")

  # 4. Computation
  data_close = np.load("data/close.npy")
  data_open = np.load("data/close.npy")
  runtime = Runtime(data = {
      # Arbitrary data for testing
      'data("price")': data_close,
      'data("volume")': (data_close - data_open) / (data_close + data_open) * 2 * 100 + 1 
  })
  result = runtime.run(graph)
  print(f"\nRuntime:\nresult = {result}")

  ## Output Example
  # Type Checking...
  #   [OK] close : Signal<Float>
  #   [OK] volume : Signal<Float>
  #   [OK] adv20 : Signal<Float>
  #   [OK] volume_level : Signal<Float>
  # Type Checking...
  #   [OK] lb : Signal<Float>
  #   [OK] result : Signal<Float>
  #   [OK] lookback : Float
  #   [OK] result : Signal<Float>

  # Building Graph...

  # Final Graph Result:
  # result = ts_mean(data("price"), multiply(10.0, divide(data("volume"), ts_mean(data("volume"), 20))))

  # Runtime:
  # result = [[    nan     nan     nan ...     nan     nan     nan]
  # [    nan     nan     nan ...     nan     nan     nan]
  # [    nan     nan     nan ...     nan     nan     nan]
  # ...
  # [191.016  57.388  90.279 ...  92.434  16.223  23.718]
  # [191.816  57.477  90.484 ...  92.662  16.192  23.625]
  # [192.585  57.742  90.84  ...  92.913  16.194  23.545]]

def download_data():
    import sys

    import numpy as np
    import pandas as pd

    symbol_dict = {
        "TOT": "Total",
        "XOM": "Exxon",
        "CVX": "Chevron",
        "COP": "ConocoPhillips",
        "VLO": "Valero Energy",
        "MSFT": "Microsoft",
        "IBM": "IBM",
        "TWX": "Time Warner",
        "CMCSA": "Comcast",
        "CVC": "Cablevision",
        "YHOO": "Yahoo",
        "DELL": "Dell",
        "HPQ": "HP",
        "AMZN": "Amazon",
        "TM": "Toyota",
        "CAJ": "Canon",
        "SNE": "Sony",
        "F": "Ford",
        "HMC": "Honda",
        "NAV": "Navistar",
        "NOC": "Northrop Grumman",
        "BA": "Boeing",
        "KO": "Coca Cola",
        "MMM": "3M",
        "MCD": "McDonald's",
        "PEP": "Pepsi",
        "K": "Kellogg",
        "UN": "Unilever",
        "MAR": "Marriott",
        "PG": "Procter Gamble",
        "CL": "Colgate-Palmolive",
        "GE": "General Electrics",
        "WFC": "Wells Fargo",
        "JPM": "JPMorgan Chase",
        "AIG": "AIG",
        "AXP": "American express",
        "BAC": "Bank of America",
        "GS": "Goldman Sachs",
        "AAPL": "Apple",
        "SAP": "SAP",
        "CSCO": "Cisco",
        "TXN": "Texas Instruments",
        "XRX": "Xerox",
        "WMT": "Wal-Mart",
        "HD": "Home Depot",
        "GSK": "GlaxoSmithKline",
        "PFE": "Pfizer",
        "SNY": "Sanofi-Aventis",
        "NVS": "Novartis",
        "KMB": "Kimberly-Clark",
        "R": "Ryder",
        "GD": "General Dynamics",
        "RTN": "Raytheon",
        "CVS": "CVS",
        "CAT": "Caterpillar",
        "DD": "DuPont de Nemours",
    }


    symbols, names = np.array(sorted(symbol_dict.items())).T

    quotes = []

    for symbol in symbols:
        print("Fetching quote history for %r" % symbol, file=sys.stderr)
        url = (
            "https://raw.githubusercontent.com/scikit-learn/examples-data/"
            "master/financial-data/{}.csv"
        )
        quotes.append(pd.read_csv(url.format(symbol)))

    records = pd.DataFrame({s:q.set_index("date").stack() for s,q in zip(symbols,quotes)}).swaplevel(1, 0, axis=0).sort_index()

    for field in ["close", "open"]:
        records.loc[field].to_csv(f"data/{field}.csv")
        np.save(f"data/{field}.npy", records.loc[field].values)


if __name__=="__main__":
    main()