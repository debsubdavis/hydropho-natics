import numpy as np
import pandas as pd

window_length = 10
hann_window = 0.5 - (0.5 * np.cos(2 * np.pi / window_length * np.arange(window_length)))
print(hann_window)

df = pd.DataFrame(hann_window)
print(df)
