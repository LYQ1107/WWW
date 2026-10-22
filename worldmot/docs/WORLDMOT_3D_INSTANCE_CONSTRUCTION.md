# 3-D instance construction

For each current synchronized output, the builder groups pixels by PanSt3R's shared query id, filters non-finite points and low confidence, trims spatial outliers, and stores a robust center, covariance, support points, confidence, visible cameras, and native masks. A query key is scoped to one joint forward only. It is never copied into `global_id`.

The support points come from the predicted instance mask, not from the entire 2-D box. The same query observed by both cameras produces one `WorldInstance` with two visibility entries.
