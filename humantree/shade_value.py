import os

import pandas as pd

from humantree import HumanTree
from tqdm import tqdm

ht = HumanTree()

with open(os.path.join('text', 'city_of_cambridge.csv'), 'r') as f:
    df = pd.read_csv(f)

shade_values = []
for ri, row in tqdm(df.iterrows()):
    if ri >= 10:
        break
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
    sqft = ht.get_sqft(zill)

    shade_values.append([ri, shade, value, sqft])

with open(os.path.join('text', 'shade-values.csv'), 'w') as f:
    for sv in shade_values:
        f.write(','.join([str(x) for x in sv]) + '\n')
