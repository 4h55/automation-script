import read_config
type="num1"
if type in ["num1", "num2", "num3", "num4", "mine"]:
    current_type = "numAndMine"
else:
    current_type = type
cfg = read_config.read_config()
score_thresh = cfg.get(f"threshParam.thresh_{current_type}")
print(score_thresh)