import json
from tensorboardX import SummaryWriter

logdir = "/home/xavie/logdir/dreamer/20251109T171831"
writer = SummaryWriter(logdir + "/tb")

with open(f"{logdir}/metrics.jsonl") as f:
    for line in f:
        data = json.loads(line)
        step = data.get("step", data.get("global_step", 0))
        for key, value in data.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(key, value, step)

writer.close()