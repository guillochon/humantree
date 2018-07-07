import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from humantree import HumanTree

ht = HumanTree()

with open(os.path.join('text', 'city_of_cambridge.csv'), 'r') as f:
    df = pd.read_csv(f)

limit = 1000

row_i = np.random.choice(df.shape[0], limit, replace=False)

shade_values = []
for ri, i in enumerate(tqdm(row_i)):
    if ri >= limit:
        break
    row = df.iloc[i]

    address = ' '.join([row['NUMBER'], row['STREET'], 'Cambridge, MA'])

    # Get shade fraction.
    shade = ht.shade_fraction_of_address(address)

    # Get zillow object.
    try:
        zill = ht.get_zillow(address)
    except Exception:
        continue

    # Get property value.
    value = zill.zestimate.amount

    # Get sqft.
    sqft = ht.get_sqft(zill, lot=True)

    shade_values.append([ri, shade, value, sqft])

with open(os.path.join('text', 'shade-values.csv'), 'w') as f:
    for sv in shade_values:
        f.write(','.join(['' if x is None else str(x) for x in sv]) + '\n')
