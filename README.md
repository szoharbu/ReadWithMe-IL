# ReadWithMe-IL
AI-Powered Interactive Reading Assistant for Hebrew-Speaking Children
ReadWithMe-IL is a desktop application designed to help children (specifically ages 7–9) improve their Hebrew reading fluency and pronunciation. Using Speech-to-Text (STT) technology, the app "listens" to the child reading aloud and provides real-time visual feedback by highlighting correctly spoken words.

🚀 Overview
Learning to read Hebrew with correct grammar and Niqqud (vowels) can be challenging. This tool provides a patient, non-judgmental environment for children to practice. By bridging the gap between complex AI speech models and a child-friendly interface, ReadWithMe-IL turns a standard reading exercise into an interactive experience.

✨ Key Features
Real-time Hebrew STT: Leverages Google Speech Recognition API specifically configured for Hebrew (he-IL).

Vowel (Niqqud) Support: Displays fully vocalized Hebrew text while performing "clean" text comparison in the background.

Dynamic Progress Tracking: Words turn green as the child reads them correctly. It includes a smart mapping logic to ensure the highlight matches the original punctuated text.

Child-Friendly UI: A modern, high-contrast Dark Mode interface built with CustomTkinter, featuring large buttons and clear status indicators.

Robust Architecture: Utilizes multi-threading to ensure the voice recognition process doesn't freeze the GUI, and includes an "Event Sync" mechanism for stable performance.

🛠 Tech Stack
Language: Python 3.x

GUI: CustomTkinter (A modern wrapper for Tkinter).

Speech Processing: SpeechRecognition & PyAudio.

Logic: Regular Expressions (Regex) for Niqqud stripping and custom string-mapping algorithms.

⚙️ Installation & Usage
Clone the repository:

Bash
git clone https://github.com/szoharbu/ReadWithMe-IL.git
cd ReadWithMe-IL
Install dependencies:

Bash
pip install customtkinter speechrecognition pyaudio
Run the app:

Bash
python main.py
🗺 Roadmap
[ ] Story Mode: Support for JSON-based story files containing multiple pages.

[ ] Visuals: Integration of illustrations and images for each reading slide.

[ ] Offline Support: Integration with the Vosk engine for offline, low-latency processing.

[ ] Gamification: Adding sound effects and achievement badges for finishing sentences.

Author: Zohar Buchris
Mail: szoharbu@gmail.com

License: MIT
