# import pickle
# from pathlib import Path
# import os
# import sys

# ROOT_DIR = Path(__file__).resolve().parents[2]  # ICU-Deterioration-MLOps
# sys.path.append(str(ROOT_DIR))

# DATA_PATH = Path(os.getenv("DATA_PATH", ROOT_DIR / "data" / "processed" / "set_c_processed.pkl"))


# with open(DATA_PATH, "rb") as f:
#     data = pickle.load(f)

# print(type(data))
# print(data.keys())
# print("######################################")
# for k, v in data.items():
#     print(k, type(v))
#     try:
#         print("  shape:", v.shape)
#     except AttributeError:
#         print("  len:", len(v))
#     print("printing v[0:2]:")
#     print(v[0:2])


import os
import pickle
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]  # ICU-Deterioration-MLOps
DATA_PATH = Path(os.getenv("DATA_PATH", ROOT_DIR / "data" / "processed" / "set_c_processed.pkl"))
with open(DATA_PATH , 'rb') as f:
    data = pickle.load(f)

print("Available RecordIDs:", data['record_ids'][:10])  # First 10
print("Total records:", len(data['record_ids']))
print("Data shape for first record:", data['X'][0].shape)