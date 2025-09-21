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
                "type": "text",
                "text": "Why should AI models be open-sourced?",
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
(voxtral) f@f:/media/f/E/project/git/voxtral$ python textonly.py 
`torch_dtype` is deprecated! Use `dtype` instead!
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:54<00:00, 27.38s/it]

Generated response:
================================================================================
Open-sourcing AI models can bring about several significant benefits, both for the scientific community and society at large. Here are some compelling reasons:

1. **Transparency and Reproducibility**:
   - **Transparency**: Open-sourcing allows others to inspect the model's code, data, and training processes. This transparency helps build trust and ensures that the model's claims are verifiable.
   - **Reproducibility**: By providing the code and data, others can reproduce the model's results, which is crucial for validating the model's performance and identifying any biases or errors.

2. **Collaboration and Innovation**:
   - **Collaboration**: Open-sourcing encourages collaboration among researchers, developers, and practitioners. This collaborative effort can lead to faster innovation and better solutions.
   - **Innovation**: By sharing models, researchers can build upon existing work, leading to more advanced and robust models.

3. **Accessibility and Inclusivity**:
   - **Accessibility**: Open-sourced models are freely available to anyone, regardless of their financial or institutional resources. This democratizes access to cutting-edge AI technologies.
   - **Inclusivity**: By making models open-source, researchers from diverse backgrounds can contribute to and benefit from the AI community, fostering a more inclusive and equitable research environment.

4. **Ethical Considerations**:
   - **Bias Mitigation**: Open-sourcing allows for the identification and mitigation of biases in AI models. By examining the code and data, researchers can identify and address potential biases that could lead to unfair or discriminatory outcomes.
   - **Accountability**: Open-sourcing can enhance accountability by making the model's development process transparent. This can help in holding developers and organizations accountable for the ethical implications of their AI models.

5. **Education and Training**:
   - **Education**: Open-sourced models can serve as educational tools for students and researchers. They can learn by studying the code, understanding the training processes, and experimenting with the models.
   - **Training**: Open-source projects often come with documentation and tutorials, which can be valuable resources for training and upskilling professionals in the field of AI.

6. **Regulatory and Policy Development**:
   - **Regulatory Compliance**: Open-sourcing can facilitate regulatory compliance by providing a clear and verifiable record of how the model was developed and trained.
   - **Policy Development**: By making models open-source, policymakers can better understand the capabilities
================================================================================

"""