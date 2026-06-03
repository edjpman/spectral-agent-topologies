'''

[DEVELOPMENT ONLY]

Simple test to determine the general efficiency of decoding inference.

'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import time
from inference.agent_wiring import Agent

agent = Agent(mock=False, device='cuda')

start = time.time()

for i in range(3):
    print(f"\n--- Step {i+1} ---")
    agent.execute_task(
        task="Increment the Value by 1",
        context={'Value': 74, 'Parity': 'A', 'Level': 3}
    )

print("\nTotal time:", time.time() - start)


