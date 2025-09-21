from transformers import VoxtralForConditionalGeneration, AutoProcessor
import torch

device = "cuda"
repo_id = "/media/f/T21/weights/Voxtral-Mini-3B-2507"

processor = AutoProcessor.from_pretrained(repo_id)
model = VoxtralForConditionalGeneration.from_pretrained(repo_id, torch_dtype=torch.bfloat16, device_map=device)


conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                # "path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/winning_call.mp3",
                "path": "/media/f/E/project/git/whisper/audio/multi_person.mp3",
            },
        ],
    }
]
inputs = processor.apply_chat_template(conversation)
inputs = inputs.to(device, dtype=torch.bfloat16)

outputs = model.generate(**inputs, max_new_tokens=500)
decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

print("\nGenerated response:")
print("=" * 80)
print(decoded_outputs[0])
print("=" * 80)

"""
(voxtral) f@f:/media/f/E/project/git/voxtral$ python transcription.py 
`torch_dtype` is deprecated! Use `dtype` instead!
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.72it/s]

Generated response:
================================================================================
The audio describes a dramatic moment in a baseball game, with a key play involving Edgar Martinez, a player known for his clutch hitting. Here's a breakdown of the events:

1. **Edgar Martinez at the Plate**: The audio begins with the pitcher, designated as "01," facing Edgar Martinez. Martinez is known for his ability to hit with runners in scoring position, and this play is likely a crucial moment in the game.

2. **Swing and a Line Drive**: Martinez swings at the pitch and hits a line drive. The ball is initially hit towards left field, but it is a home run.

3. **Joy and Junior**: The audio mentions "Joy" and "Junior" in relation to the play. This could refer to the umpires or possibly the commentators, but it's not clear from the audio.

4. **Late Throw to the Plate**: The play is described as "the throw to the plate will be late." This suggests that the runner at third base, likely Junior, is trying to score, but the throw to the plate is delayed, giving the runner a chance to advance.

5. **American League Championship**: The audio ends with a reference to the "American League Championship," indicating that the game is part of a playoff series.

Overall, the audio captures a tense and exciting moment in a baseball game, with a key play involving Edgar Martinez and a potential run-scoring opportunity.
================================================================================
(voxtral) f@f:/media/f/E/project/git/voxtral$ python transcription.py 
`torch_dtype` is deprecated! Use `dtype` instead!
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.90it/s]

Generated response:
================================================================================
The audio discusses the integration of different speech recognition models, specifically focusing on Whisper and FSR (Fast Speech Recognition) models. The conversation highlights the challenges and advantages of using these models, particularly in supporting multiple languages and handling large audio files. The speakers also discuss the need for distinguishing between short and long sentences, which is crucial for accurate speech recognition. The audio emphasizes the importance of choosing the right model based on the specific requirements and constraints, such as the support for different languages and the ability to handle large audio files. The speakers also touch on the potential for integrating these models into a unified framework, which could streamline the process and improve efficiency. The discussion concludes with a plan to test the proposed solutions and make necessary adjustments before finalizing the implementation.
================================================================================

"""