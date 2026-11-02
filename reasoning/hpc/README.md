# Running Model 3 on the NUS HPC

The laptop GPU (4 GB) is too small for LoRA training, so the heavy steps run on the NUS HPC
(NVIDIA A40 GPUs, 48 GB). The code is the same as on the laptop (`reasoning/src/`); the settings live in
`hpc.conf`.

**How it works:** you send a "job" (a small script) to a queue. When a GPU is free, the HPC runs it.
Your laptop can be off meanwhile.

## Every time
Be on the NUS network: NUS Wi-Fi, or **Cisco Secure Client** VPN → "NUS Student - TLS - Auto Select Nearest
Location". If you see `Could not resolve hostname vanda.nus.edu.sg`, the VPN is off.

## First time only (already done)
| Step | Command (from the `BN4601A` folder on the laptop) | What it does |
|---|---|---|
| 1 | `powershell -ExecutionPolicy Bypass -File reasoning\hpc\install_key.ps1` | key login, so no password next time |
| 2 | `powershell -ExecutionPolicy Bypass -File reasoning\hpc\sync_up.ps1` | copy code + data to `/scratch/e0959158/bn4601a/BN4601A` |
| 3 | `ssh nus-vanda "cd /scratch/e0959158/bn4601a/BN4601A && bash reasoning/hpc/setup_once.sh"` | Python + libraries + MedGemma download (~2 h, slow disk) |
| 4 | `ssh nus-vanda "cd /scratch/e0959158/bn4601a/BN4601A && bash reasoning/hpc/submit.sh probe"` | 10-min test: do we get a GPU? |

## Running jobs
All commands below are typed on the laptop. Each one logs in, does one thing and comes back.

    ssh nus-vanda "cd /scratch/e0959158/bn4601a/BN4601A && bash reasoning/hpc/submit.sh lora mixed"
    ssh nus-vanda bash /scratch/e0959158/bn4601a/BN4601A/reasoning/hpc/status.sh

| Job | Command | Time on an A40 |
|---|---|---|
| one LoRA arm | `submit.sh lora mixed` / `emergency` / `two_stage` | ~50 min per 30k stage (mixed ~10 h in total) |
| vectors + heads + eval | `submit.sh embed` | ~1–2 h |
| final comparison | `qsub ... jobs/eval_arms.pbs` (see the top of that file) | ~10 min |

**`status.sh`** shows the queue and each arm's progress: stage, val loss, MedQA.
In the queue, `Q` = waiting, `R` = running, `H` = held (waiting for another job to finish).

**Good to know**
- Queue: jobs go to `auto_free` (free), which forwards them to the `gpu` queue. Sending to `gpu` directly is refused.
- At most 2 of your jobs run at once. Each job can run up to 48 h; training saves and stops at 46 h.
- Training saves every 30 min and after every stage. If a job stops, run the same `submit.sh lora <arm>` again.
  It continues from the last save.
- A job's printed log (`reasoning/outputs/logs/<job id>.OU`) appears only **after** the job ends.
  While it runs, use `status.sh`.
- Code changes: copy the changed file up (`scp`) before the job starts. Jobs read the Python files when they
  start.

## Bring results back to the laptop

    powershell -ExecutionPolicy Bypass -File reasoning\hpc\sync_down.ps1

It copies the adapters to `reasoning/models/lora/`, and the metrics and logs to `reasoning/outputs/`.

## Files
| File | Job |
|---|---|
| `hpc.conf` | settings: folders, queue, GPU size, learning rate, batch size |
| `setup_once.sh` | build the Python environment and download MedGemma (login node has internet; GPU nodes do not) |
| `submit.sh` | send jobs: `probe`, `lora <arm>`, `embed`, `all` |
| `status.sh` | progress report |
| `jobs/*.pbs` | the job scripts the HPC runs |
| `check_system.sh` | read-only: which GPUs, queues and folders exist |
| `install_key.ps1`, `sync_up.ps1`, `sync_down.ps1` | laptop side: key login, copy up, copy down |
