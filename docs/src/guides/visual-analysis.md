# Visual Analysis

Visual analysis is enabled with `--visual` and is only valid for video inputs.

```sh
canscribe --visual ~/Videos/meeting.mp4
```

The pipeline uses frame routing before invoking heavier analysis:

| Route | Condition | Analysis |
| --- | --- | --- |
| `SPEAKING_FACE` | Face detected with mouth movement | Face analysis |
| `STATIC_FACE` | Face detected without mouth movement | Face analysis |
| `NO_FACE` | No face detected | Scene description |

Face routing and embeddings use InsightFace. Emotion detection uses `dima806/facial_emotions_image_detection`. Scene descriptions use `Qwen/Qwen3-VL-4B-Instruct`.

Visual mode adds model load time and GPU memory pressure. Use `canscribe check <video-file>` first when validating a new machine or media source.
