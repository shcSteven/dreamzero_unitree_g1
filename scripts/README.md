# Scripts — Unitree G1 Upper Body (DreamZero)

End-to-end workflow: convert a LeRobot v2 dataset to GEAR format, then launch DreamZero LoRA fine-tuning.

## Prerequisites

- Conda environment `dreamzero` activated
- Pretrained weights downloaded (the training script auto-downloads if missing):
  - `Wan2.1-I2V-14B-480P` → `/localhome/local-haochens/ckpts/Wan2.1-I2V-14B-480P`
  - `umt5-xxl` → `/localhome/local-haochens/ckpts/umt5-xxl`
  - `DreamZero-AgiBot` → `/localhome/local-haochens/ckpts/DreamZero-AgiBot`

## 1. Data Conversion

### Expected input

A LeRobot v2 dataset for Unitree G1 Dex3 upper body:

```
your_dataset/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       └── ...
├── videos/
│   └── chunk-000/
│       ├── observation.images.cam_room/
│       ├── observation.images.cam_left_wrist/
│       └── observation.images.cam_right_wrist/
└── meta/
    ├── info.json          # must have: features, total_episodes, fps
    └── tasks.jsonl        # task descriptions (task_index → text)
```

The parquet files should contain:
- `observation.state` — 28-dim float32 (left_arm[7], right_arm[7], left_hand[7], right_hand[7])
- `action` — 28-dim float32 (same layout)
- `task_index` — integer referencing tasks.jsonl

### Run conversion

```bash
cd /localhome/local-haochens/dreamzero_unitree_g1

./scripts/data/prepare_unitree_g1_upper.sh \
    /path/to/lerobot_dataset \
    /path/to/gear_output
```

Example:

```bash
./scripts/data/prepare_unitree_g1_upper.sh \
    /localhome/local-haochens/data/assemble_trocar_lerobot/assemble_trocar_lerobot \
    /localhome/local-haochens/data/assemble_trocar_gear
```

### What it does

1. Runs `convert_lerobot_to_gear.py` — copies the dataset to the output path and generates GEAR metadata (`modality.json`, `embodiment.json`, `stats.json`, `relative_stats_dreamzero.json`, `episodes.jsonl`)
2. Restores `tasks.jsonl` from the original dataset (the converter can't extract text from `task_index`-only columns)
3. Adds the `annotation.task` entry to `modality.json` (maps to `task_index`)
4. Runs a pre-training checklist to verify everything is correct

### State/action key mapping

| Sub-key | Indices | Dims | Joints |
|---|---|---|---|
| `left_arm` | [0, 7) | 7 | Shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw |
| `right_arm` | [7, 14) | 7 | Same as left |
| `left_hand` | [14, 21) | 7 | Thumb (3), middle (2), index (2) |
| `right_hand` | [21, 28) | 7 | Same as left |

All four sub-keys use relative actions during training.

## 2. Training

### WandB setup

The training script reports to WandB by default. Export your API key before launching:

```bash
export WANDB_API_KEY=<your_key>
```

To skip WandB logging entirely, append `report_to=none`:

```bash
./scripts/train/unitree_g1_upper_training.sh report_to=none
```

### Launch

```bash
cd /localhome/local-haochens/dreamzero_unitree_g1

./scripts/train/unitree_g1_upper_training.sh
```

### Override defaults with environment variables

```bash
DATA_ROOT=/path/to/gear_dataset \
OUTPUT_DIR=/path/to/checkpoints \
NUM_GPUS=4 \
    ./scripts/train/unitree_g1_upper_training.sh
```

### Key parameters

| Parameter | Default | Description |
|---|---|---|
| `DATA_ROOT` | `/localhome/local-haochens/data/assemble_trocar_gear` | GEAR-converted dataset |
| `OUTPUT_DIR` | `/localhome/local-haochens/ckpts/dreamzero_unitree_g1_upper_lora` | Checkpoint output |
| `NUM_GPUS` | auto-detected | Number of GPUs |
| `WAN_CKPT_DIR` | `/localhome/local-haochens/ckpts/Wan2.1-I2V-14B-480P` | Wan2.1 weights |
| `TOKENIZER_DIR` | `/localhome/local-haochens/ckpts/umt5-xxl` | Tokenizer weights |

### Training config

- **Architecture**: LoRA fine-tuning on DreamZero-AgiBot
- **Batch size**: 1 per GPU (× NUM_GPUS)
- **Learning rate**: 1e-5 with 5% warmup
- **Resolution**: 320×176
- **Views**: 3 cameras (room, left wrist, right wrist)
- **Action horizon**: 24 steps
- **DeepSpeed**: ZeRO Stage 2

## Full pipeline example

```bash
conda activate dreamzero
cd /localhome/local-haochens/dreamzero_unitree_g1

# Step 1: Convert dataset
./scripts/data/prepare_unitree_g1_upper.sh \
    /localhome/local-haochens/data/my_new_task/my_new_task \
    /localhome/local-haochens/data/my_new_task_gear

# Step 2: Train
DATA_ROOT=/localhome/local-haochens/data/my_new_task_gear \
    ./scripts/train/unitree_g1_upper_training.sh
```
