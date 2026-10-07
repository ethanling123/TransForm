"""
CodecDB-inspired data-driven encoding selection for Parquet columns.

Extracts per-column features (cardinality ratio, sortedness, value type/range)
and selects per-column encoding parameters using rules derived from the
CodecDB paper (Jiang et al., SIGMOD '21).

Reference:
  Hao Jiang, Chunwei Liu, John Paparrizos, Andrew A. Chien, Jihong Ma,
  Aaron J. Elmore. "Good to the Last Bit: Data-Driven Encoding with CodecDB."
  SIGMOD '21. https://doi.org/10.1145/3448016.3457283
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pyarrow as pa

CARDINALITY_THRESHOLD = 0.3


def compute_column_features(table: pa.Table) -> Dict[str, Dict[str, Any]]:
    """
    Extract encoding-relevant features for each column in *table*.

    Per-column features:
      dtype, n_rows, n_unique, cardinality_ratio, is_sorted,
      and type-specific extras (max/min for ints, mean_value_length for strings).
    """
    features: Dict[str, Dict[str, Any]] = {}
    n_rows = table.num_rows

    for name in table.column_names:
        col = table.column(name)
        arr = col.combine_chunks()

        feat: Dict[str, Any] = {
            "dtype": str(arr.type),
            "n_rows": n_rows,
        }

        try:
            n_unique = len(arr.unique())
        except Exception:
            n_unique = n_rows
        feat["n_unique"] = n_unique
        feat["cardinality_ratio"] = n_unique / n_rows if n_rows > 0 else 1.0

        is_numeric = pa.types.is_integer(arr.type) or pa.types.is_floating(arr.type)
        is_string = pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type)

        if is_numeric and n_rows > 1:
            try:
                np_arr = arr.to_numpy(zero_copy_only=False)
                if pa.types.is_floating(arr.type):
                    valid = np_arr[~np.isnan(np_arr)]
                else:
                    valid = np_arr
                if len(valid) > 1:
                    feat["is_sorted"] = bool(np.all(np.diff(valid) >= 0))
                    feat["max_value"] = int(np.max(valid))
                    feat["min_value"] = int(np.min(valid))
                else:
                    feat["is_sorted"] = True
                    feat["max_value"] = int(valid[0]) if len(valid) else 0
                    feat["min_value"] = feat["max_value"]
            except Exception:
                feat["is_sorted"] = False
        elif is_string and n_rows > 1:
            try:
                py_list = arr.to_pylist()
                non_null = [s for s in py_list if s is not None]
                feat["is_sorted"] = all(a <= b for a, b in zip(non_null, non_null[1:]))
                lengths = [len(s) for s in non_null]
                feat["mean_value_length"] = (
                    sum(lengths) / len(lengths) if lengths else 0
                )
            except Exception:
                feat["is_sorted"] = False
        else:
            feat.setdefault("is_sorted", n_rows <= 1)

        features[name] = feat

    return features


def select_encoding_params(table: pa.Table) -> Dict[str, Any]:
    """
    Select per-column Parquet encoding parameters based on column features.

    Returns a dict with:
      use_dictionary  – list of column names that should use dictionary encoding
                        (passed directly to ``pq.write_table(use_dictionary=...)``)
      column_encoding – dict[column_name, str] | None  (per-column encoding override)
      features        – dict[column_name, feature_dict]

    Rules (CodecDB-inspired):
      * cardinality_ratio < CARDINALITY_THRESHOLD → dictionary (PyArrow default)
      * high-cardinality sorted integers           → DELTA_BINARY_PACKED
      * high-cardinality strings                   → DELTA_BYTE_ARRAY
      * otherwise                                  → PLAIN (no dictionary)
    """
    features = compute_column_features(table)

    dict_columns: list[str] = []
    column_encoding: Dict[str, str] = {}

    for name, feat in features.items():
        cr = feat.get("cardinality_ratio", 1.0)
        is_sorted = feat.get("is_sorted", False)
        dtype_str = feat.get("dtype", "")
        is_int = "int" in dtype_str
        is_str = "string" in dtype_str or "utf8" in dtype_str

        if cr < CARDINALITY_THRESHOLD:
            dict_columns.append(name)
        else:
            if is_int and is_sorted:
                column_encoding[name] = "DELTA_BINARY_PACKED"
            elif is_str:
                column_encoding[name] = "DELTA_BYTE_ARRAY"
            else:
                column_encoding[name] = "PLAIN"

    return {
        "use_dictionary": dict_columns,
        "column_encoding": column_encoding if column_encoding else None,
        "features": features,
    }
