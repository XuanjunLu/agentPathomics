import numpy as np
import pandas as pd
from pathomics.interplay.sptil_extractor.extract_SpaTIL_feature import extract_SpaTIL_feature

coords_0 = pd.read_excel('./coords.xlsx', sheet_name='lympha')
coords_1 = pd.read_excel('./coords.xlsx', sheet_name='notlympha')
coords_0 = np.array(coords_0)
coords_1 = np.array(coords_1)

feat_df, groups = extract_SpaTIL_feature(coords_0, coords_1)
feat_df.to_excel('./result.xlsx', index=True, header=False)

