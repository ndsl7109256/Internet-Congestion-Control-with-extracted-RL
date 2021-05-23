# Copyright 2019 Nathan Jay and Noga Rotman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import matplotlib.pyplot as plt
import numpy as np
import sys

if (not (len(sys.argv) == 2)) or (sys.argv[1] == "-h") or (sys.argv[1] == "--help"):
    print("usage: python3 draw_reward.py <reward.json>")
    exit(0)

filename = sys.argv[1]

data = {}
with open(filename) as f:
    rew_data = json.load(f)
#print (rew_data)
time_data=np.arange(0,len(rew_data))
"""
time_data = [float(event["Time"]) for event in data["Events"][1:]]
rew_data = [float(event["Reward"]) for event in data["Events"][1:]]
"""

fig, axes = plt.subplots(1, figsize=(10, 12))
rew_axis = axes

rew_axis.plot(time_data, rew_data)
rew_axis.set_ylabel("Reward")


rew_axis.set_xlabel("Monitor Interval")
x = filename.split(".", 1)
fig.suptitle("Summary Graph for %s" % sys.argv[1])
xtmp=x[0]+".pdf"
fig.savefig(xtmp)
