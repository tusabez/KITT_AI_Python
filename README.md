# KITT AI - Knight Rider Assistant

Welcome to the KITT AI project! This project brings the famous KITT AI from the "Knight Rider" TV series to life. It uses various Python libraries to provide voice interaction, image processing, and control of smart devices, emulating the functionality of the original KITT.

## Features

- **Voice Interaction**: Talk to KITT using voice commands powered by GPT-4o.
- **Wake Word Detection**: KITT can be activated using the wake word **"Hey KITT"**.
- **Weather Updates**: Get current weather information and forecasts.
- **News and Sports**: Receive the latest news and top sports updates.
- **Image Processing**: Capture and analyze images with KITT's witty commentary.
- **Smart Bulb Control**: Control a Tapo TP-Link smart bulb with voice commands.
- **Lakers Scores**: As a Lakers fan, you can ask KITT for the latest Lakers scores (but feel free to modify the code to suit your preferences).

## Repository Structure

> **Note:** All scripts and MP3 files must be in the same folder for the project to work correctly.

- `KITT.py`: The main script to run KITT AI.
- `modules/`: Various helper modules such as `weather.py` and `news.py`.
- `sounds/`: Various MP3 files used for waiting responses and sound effects.
- `chatbot1.txt`: Stores conversation history for the AI.

## Requirements

To run this project, you'll need the following Python libraries:

- Python 3.8 or higher
- `openai`: For interacting with OpenAI's API.
- `elevenlabs`: For text-to-speech functionality.
- `openwakeword`: For wake word detection.
- `pyaudio`: For audio input/output.
- `numpy`: For numerical operations.
- `sounddevice`: For real-time audio playback.
- `soundfile`: For reading and writing sound files.
- `opencv-python`: For image processing.
- `requests`: For making HTTP requests.
- `pyl530` (PyP100): For controlling Tapo TP-Link smart bulbs. (Note: The original module is no longer maintained. You can find the updated version [here](https://github.com/almottier/TapoP100).)

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/tusabez/KITT_AI_Python.git
   cd KITT_AI_Python
2. **Install Dependencies**

   ```bash
   pip install openai elevenlabs pyaudio numpy sounddevice soundfile opencv-python requests openwakeword pytz nba.api pandas openmeteo_requests request-cache retry-requests
   ```

   If you need the updated pyl530 library, you can install it directly from the GitHub repository:
   ```bash
   pip install git+https://github.com/almottier/TapoP100.git
4. **Set Up Keys**

   OpenAI: Store your OpenAI API key in your environment variables or add it directly to the script.
   ElevenLabs: Store your ElevenLabs API key in your environment variables or add it directly to the script.

5. **Create Your KITT Voice**
   
   You will need to create your own KITT voice since the original voice cannot be legally distributed. You can do this by:

   Obtaining voice samples from the "Knight Rider" TV show or the "Knight Rider 2000" movie.
   
   Using ElevenLabs' Instant Voice Clone feature to create a voice that mimics KITT.
   
   Note: Please do not publicly share the voice you create to avoid any potential copyright issues.

7. **Configure Smart Bulb (Optional)**

   If you want to control a Tapo TP-Link smart bulb, you'll need to configure the bulb's IP address, your email, and your password in the script:

   ```bash
   bulb = setup_bulb("Your bulb IP Address", "Your email", "Your password")
8. **Run the script**

   Start the main loop:

   ```bash
   python KITT.py

## Usage

Once the script is running, KITT will listen for the wake word "Hey KITT". After activation, you can interact with KITT using natural language commands such as:

"What's the weather right now?"
"Tell me a joke."
"Turn on the light."
"What time is it?"
"What’s the latest Lakers score?"
Since KITT is powered by GPT-4o, you can ask him about anything, and he'll respond in character as KITT from "Knight Rider."

## Compatibility

This project was tested on a Windows 11 computer, but it can be adapted to work on a Raspberry Pi 5 with some modifications to the code. I used a USB microphone and Raspberry Pi camera module 3. If you're planning to run this on a Raspberry Pi, you might need to adjust the setup for audio input/output, camera access, and performance optimizations. You'll also need to set up a virtual environment before you install the libraries. Update: (use KITT_Pi.py instead of KITT.py for the Raspberry Pi 5 as it has been optimized for it).

## Troubleshooting

Wake Word Detection Issues: Ensure your microphone is working correctly and that the wake word model is properly initialized.
Smart Bulb Control Issues: Verify the IP address, email, and password for the smart bulb are correct and the bulb is reachable on your network.
API Key Errors: Double-check that your API keys are correctly stored and valid.
Run models.py to download needed models to have the wakeword work. This is only done once.

## Contributing

Feel free to submit issues or pull requests if you find bugs or want to add new features. Contributions are welcome! While I’m a Lakers fan and have included functions to get Lakers scores, you can easily modify the code to follow your favorite team.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

 
