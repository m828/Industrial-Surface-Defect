# Boundary metric implementation audit

Original implementation: `compute_mechanism_metrics.py` lines 17-46.

- `band(m,t) = cv2.dilate(m,K3,iterations=t) - cv2.erode(m,K3,iterations=t)`, clipped to [0,1].
- `K3` is a 3x3 all-ones uint8 structuring element.
- BF1 uses one-erosion contours, OpenCV distance transforms, and tolerance thresholds 1/2/3 px.
- Boundary IoU returns `None`/NA only when the union of both bands is zero.
- BF1 returns `None`/NA only when both contours are empty; one-sided empty cases return 0.
- No explicit OpenCV border argument is set; OpenCV defaults therefore apply.

Synthetic empty-rule outputs are stored in `boundary_empty_rule_tests.json`.
Named-sample outputs are stored in `boundary_named_sample_audit.json`.

## A/B/C centers
- Crazing BIoU@2 A: delta=-0.004527838299992, n=371/370, base=0.719010159957400, A2=0.714482321657408
- Crazing BIoU@2 B: delta=-0.006506278073246, n=368, base=0.724871655826618, A2=0.718365377753372
- Crazing BIoU@2 C: delta=-0.006577775634490, n=364, base=0.732837278418119, A2=0.726259502783629

Bootstrap intervals use 5000 image-level resamples, seed 20261018, shared indices, percentile intervals, and fixed cached masks; see `bootstrap_reproduction_summary.csv`.
