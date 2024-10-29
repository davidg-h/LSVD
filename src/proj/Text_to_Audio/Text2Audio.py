from transformers import pipeline
import scipy

classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# Input from user is stored into userInput
userInput = input("Enter your ideal advertisement description: ")

# analysis stores the classified user input for with the different confidence level of the emotions detected
analysis = classifier(userInput)
filtered_analysis = sorted([item for item in analysis[0] if item['score'] > 0.4], key=lambda x: x['score'], reverse=True)

labels_string = "mood: " + ", ".join([item['label'] for item in filtered_analysis])
print(labels_string)

edited_user_input = "\n".join([userInput, labels_string])
print(edited_user_input)

synthesiser = pipeline("text-to-audio", "facebook/musicgen-small")

# Use the edited_user_input which contains additional information from the sentimental analysis
music = synthesiser(edited_user_input, forward_params={"do_sample": True})

# Output the final audio into a .wav file
scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])