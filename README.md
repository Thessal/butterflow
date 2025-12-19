# Butterflow

A data pipeline language

## Features

* Typing
* Function

## Example

```
# Assignments
close : Signal<Float> = data(id="price")
volume : Signal<Float> = data(id="volume")
adv20 : Signal<Float> = ts_mean(signal=volume, period=20)
volume_level : Signal<Float> = divide(dividend=volume, divisor=adv20)

# Function definition
dynamic_ma(signal: Signal<Float>, lookback: Signal<Float>, multiplier: Signal<Float>) : Signal<Float> = {
lb : Signal<Float> = multiply ( baseline = lookback, multiplier = multiplier )
result : Signal<Float> = ts_mean(signal = signal, period = lb)
}

# Scoping
lookback : Float = 10.

# "result" is a special variable where the result is stored
result : Signal<Float> = dynamic_ma(signal=close, lookback=lookback, multiplier=volume_level)
```