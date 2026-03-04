# Underwater Scene Understanding for AUV Navigation
### NVIDIA Cosmos Cookoff 2026 — Post-Training Submission

Fine-tuning **Cosmos-Reason2-2B** on underwater video data to create a specialized reasoning model for autonomous underwater vehicle (AUV) navigation. The model identifies marine species, assesses visibility conditions, detects hazards, and reasons about underwater scenes in structured, navigation-relevant terms.

---

![Pipeline](diagram.png)

## Quick Start

### Prerequisites

- Google Colab with H100 GPU (recommended) or A100 40GB
- HuggingFace account with access to `nvidia/Cosmos-Reason2-2B` and `nvidia/Cosmos-Reason2-8B`

### Setup

1. Add your HuggingFace token to Colab Secrets as `HF_TOKEN`
2. Create a HuggingFace model repo to host the fine-tuned adapter
3. Update `HF_REPO_ID` in notebooks 05, 06, and 07 with your repo ID

### Run the full pipeline

```
01 → 02 → 03 → 04 → 05 → 06 → 07
                              └──→ 08 (deployment — run alongside ROS2 + Isaac Sim)
```

Each notebook generates what the next one needs. Videos are downloaded automatically via fiftyone in notebook 02. The fine-tuned adapter is uploaded to HuggingFace Hub at the end of notebook 05 and loaded from there in notebooks 06, 07, and 08.

### Run the demo only (skip training)

Open `07_demo.ipynb` and run all cells. The adapter loads directly from HuggingFace Hub:

```python
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

base = Qwen3VLForConditionalGeneration.from_pretrained(
    "nvidia/Cosmos-Reason2-2B", dtype="auto", device_map="auto"
)
model = PeftModel.from_pretrained(base, "15juneee/cosmos-reason2-underwater-auv")
model.eval()
```

---

## Notebooks

| Notebook | Description |
|---|---|
| `01_data_exploration.ipynb` | Explore WebUOT-238-Test structure, annotations, and video contents |
| `02_vqa_generation.ipynb` | Generate LLaVA-format VQA pairs from tracking annotations |
| `03_zero_shot_baseline.ipynb` | Evaluate Cosmos-Reason2-2B zero-shot on underwater scenes |
| `04_distillation.ipynb` | Use Cosmos-Reason2-8B as teacher to regenerate open-ended answers |
| `05_sft_training.ipynb` | QLoRA fine-tuning on distilled dataset |
| `06_evaluation.ipynb` | MCQ accuracy, ROUGE-L, and qualitative before/after comparison |
| `07_demo.ipynb` | Gradio demo — side-by-side zero-shot vs fine-tuned inference |
| `08_inference_server.ipynb` | FastAPI + ngrok inference server for ROS2 bridge integration |

---

## Results

| Model | MCQ Accuracy | ROUGE-L | Notes |
|---|---|---|---|
| Zero-shot baseline | 46.7% (14/30) | 0.317 | 77% A-bias, prediction distribution: `{A:23, B:7}` |
| **QLoRA SFT + 8B Distillation** | **63.3% (19/30)** | **0.327** | Balanced predictions: `{A:14, B:16}` |

**+16.6pp MCQ improvement** on identical 30-sample evaluation set. **+3.2% ROUGE-L** on 100 open-ended samples.

The zero-shot model predicts option A for 77% of samples regardless of visual content (`{A: 23, B: 7}`). The fine-tuned model distributes predictions based on what it sees (`{A: 14, B: 16}`). Neither model predicted C, but the zero-shot never predicted B for environment or visibility questions. The fine-tuned model learned to reason about scene content rather than defaulting to a fixed response.

MCQ accuracy improved by +16.6pp because the model learned specific visual attributes: target presence/absence, environment type, and visibility level. ROUGE-L improved by a smaller margin (+3.2%) because the zero-shot model already produces fluent English that overlaps lexically with the ground truth. The qualitative examples below show the real difference: the fine-tuned model frames answers in AUV navigation terms, which ROUGE-L does not fully capture.

---

## Why Underwater?

Underwater environments present significant challenges for visual AI:

- Extreme visibility variation from turbidity, backscatter, and color attenuation
- No existing Cosmos recipes covering underwater scene understanding
- Direct applications in AUV navigation, marine infrastructure inspection, coral reef monitoring, and search and rescue

General-purpose vision models describe underwater footage generically. An AUV needs domain-specific reasoning: *"visibility is low, coral structures create collision risk, target is moving right at moderate speed."* That is what this fine-tuned model produces.

---

## Dataset

**WebUOT-1M** — *"WebUOT-1M: Advancing Deep Underwater Object Tracking with A Million-Scale Benchmark"* (NeurIPS 2024)

- 1,500 video clips, 1.1M frames, 408 target categories across 12 superclasses
- Per-video annotations: bounding boxes, language prompts, 23 tracking attributes (illumination variation, camouflage, low visibility, fast motion, etc.), absent labels, environment metadata
- Source: real-world YouTube/BiliBili footage across sea, river, lake, and fish tank environments

We used the 238-video test subset available on HuggingFace (`Voxel51/WebUOT-238-Test`), repurposing the tracking annotations to generate VQA training pairs.

### VQA Generation Pipeline

From each video and its annotations, we programmatically generated 5 question types:

| Type | Example Question | Ground Truth Source |
|---|---|---|
| **Object Identification** | What is the main subject? Describe it. | Language prompts + category labels |
| **Scene Assessment** | Assess visibility and AUV navigation suitability. | Environment + attribute metadata |
| **Hazard Detection** | What navigation hazards are present? | Tracking attributes (clutter, camouflage, etc.) |
| **Spatial Reasoning** | Where is the target and what direction is it moving? | Bounding box trajectories |
| **Categorical MCQ** | What environment type? What visibility level? | Verified dataset annotations |

**Final dataset:** 4,652 training pairs + 821 validation pairs from 238 videos.

### 8B Teacher Distillation

For open-ended question types, programmatic labels contain noise. We used **Cosmos-Reason2-8B as a teacher model** to regenerate high-quality answers, replacing template-generated ground truth with 8B model outputs. MCQ labels from verified dataset annotations were kept as-is.

The full pipeline uses only NVIDIA models: base student model, teacher model, and fine-tuned adapter are all from NVIDIA.

---

## Training

**Model:** `nvidia/Cosmos-Reason2-2B` (Qwen3-VL backbone)  
**Method:** QLoRA (4-bit quantization + LoRA adapters)  
**Hardware:** NVIDIA H100 80GB  

| Parameter | Value |
|---|---|
| Training pairs | 4,652 |
| Epochs | 3 |
| Learning rate | 2e-4 (cosine decay) |
| LoRA rank / alpha | 32 / 32 |
| Target modules | q, k, v, o, gate, up, down proj |
| Frames per video | 8 |
| Batch size | 1 (grad accum 4) |
| Training time | ~50 minutes |
| Peak VRAM | 6.82 GB |
| Final loss | 0.217 |

The 8-frame sampling strategy follows the finding from the Cosmos Cookbook Intelligent Transportation recipe: higher resolution per frame outperforms more frames at lower resolution.

---

## Qualitative Examples

### Scene Assessment
**Q:** Assess the underwater visibility and environmental conditions. How suitable are these conditions for AUV navigation?

| | Response |
|---|---|
| 🔴 Zero-Shot | *"The underwater visibility in the scene is clear, allowing for a detailed view of the coral reef and the fish. The water is clear, and the sunlight is..."* |
| 🟢 Fine-Tuned | *"This footage is from a sea environment. Underwater visibility is moderate, allowing for adequate AUV navigation with some limitations in detection range..."* |

### Hazard Detection
**Q:** What navigation hazards or visual challenges are present that an AUV would need to handle?

| | Response |
|---|---|
| 🔴 Zero-Shot | *"The primary navigation challenge is the presence of numerous fish swimming in the water column. These fish can pose a collision..."* |
| 🟢 Fine-Tuned | *"The underwater environment presents several navigation hazards for an AUV. The coral reef creates complex terrain with uneven surfaces that pose collision risks. The presence of multiple fish adds dynamic obstacles..."* |

### Object Identification
**Q:** What underwater subject is the focus of this video clip? Describe what you observe.

| | Response |
|---|---|
| 🔴 Zero-Shot | *"The video captures a vibrant underwater scene, showcasing a coral reef teeming with life. The reef is a complex structure, with various types of coral..."* |
| 🟢 Fine-Tuned | *"The video captures a vibrant coral reef teeming with life. The reef is composed of various types of coral, including branching and brain corals, which provide a diverse habitat for numerous fish species..."* |

### Spatial Reasoning
**Q:** Where is the primary subject located and what direction is it moving? Important for AUV target tracking.

| | Response |
|---|---|
| 🔴 Zero-Shot | *"The primary subject, a manta ray, is located in the center of the frame, swimming towards the right side of the image..."* |
| 🟢 Fine-Tuned | *"The primary subject, a stingray, is located in the center of the frame, slightly to the right, and is moving towards the left side of the frame."* |

---

## Fine-Tuned Adapter

🤗 [`15juneee/cosmos-reason2-underwater-auv`](https://huggingface.co/15juneee/cosmos-reason2-underwater-auv)

---

## End-to-End Deployment (ROS2 + Isaac Sim)

Beyond the Gradio demo, the fine-tuned model is integrated into a live robotics pipeline:

```
NVIDIA Isaac Sim (local)              Google Colab H100
────────────────────────              ─────────────────
/auv/camera/image_raw                 08_inference_server.ipynb
        │                             FastAPI  POST /infer
        ▼                                     │
 bridge_node.py  ──── HTTP / ngrok ─────────► model inference
        │                                     │
 /auv/cmd_vel   ◄──── JSON response ──────────┘
        │
  Isaac Sim ROV
```

**NB-08** starts a FastAPI server on Colab H100 exposed via ngrok. The ROS2 bridge node (`bridge_node.py`) subscribes to the simulated camera feed, POSTs frames to the inference server, parses the `recommended_action` field from the JSON response, and publishes velocity commands to the ROV.

**To run the deployment:**

1. Run NB-08 on Colab — copy the printed ngrok URL
2. On local machine with ROS2 Humble + Isaac Sim:
```bash
export INFERENCE_URL="https://xxxx.ngrok.io"
ros2 run blue_recon bridge_node
```
3. Launch Isaac Sim with the underwater scene — the ROV will navigate based on Cosmos Reason 2 decisions.


---

## References

- [Cosmos-Reason2-2B on HuggingFace](https://huggingface.co/nvidia/Cosmos-Reason2-2B)
- [Cosmos Cookbook](https://github.com/nvidia-cosmos/cosmos-cookbook)
- [WebUOT-1M Paper (NeurIPS 2024)](https://arxiv.org/abs/2405.19818)
- [WebUOT-238-Test on HuggingFace](https://huggingface.co/datasets/Voxel51/WebUOT-238-Test)
- [Cosmos Reason 2 Post-Training Guide](https://github.com/nvidia-cosmos/cosmos-reason2/blob/main/examples/cosmos_rl/README.md)
