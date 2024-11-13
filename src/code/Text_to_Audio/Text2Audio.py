import os
import scipy
from transformers import pipeline, AutoProcessor, MusicgenForConditionalGeneration

def text2audio(prompt, output_path):
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

    # Input from user is stored into userInput
    userInput = prompt

    # analysis stores the classified user input for with the different confidence level of the emotions detected
    analysis = classifier(userInput)
    filtered_analysis = sorted([item for item in analysis[0] if item['score'] > 0.4], key=lambda x: x['score'], reverse=True)

    labels_string = "mood: " + ", ".join([item['label'] for item in filtered_analysis])
    print(labels_string)

    edited_user_input = "\n".join([userInput, labels_string])
    print(edited_user_input)

    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

    inputs = processor(
        text=[edited_user_input],
        padding=True,
        return_tensors="pt",
    )

    audio_values = model.generate(**inputs, max_new_tokens=400)

    # Output the final audio into a .wav file
    sampling_rate = model.config.audio_encoder.sampling_rate
    scipy.io.wavfile.write(os.join(output_path, "musicgen_out.wav"), rate=sampling_rate, data=audio_values[0, 0].numpy())
    
    print(f"Audio generated successfully in: {output_path}")
    return True, edited_user_input