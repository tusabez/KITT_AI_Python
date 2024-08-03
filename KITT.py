import os
import time
import base64
import pyaudio
import numpy as np
import sounddevice as sd
import soundfile as sf
import cv2
import random
import threading
import concurrent.futures
from openai import OpenAI
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from openwakeword.model import Model
from typing import IO
from io import BytesIO
import subprocess
import weather
from news import get_latest_news, get_top_sports_news
import logging
import importlib
from datetime import datetime
import pytz
import requests
from PyP100 import PyL530

# Configure logging
logging.basicConfig(filename='chatbot_debug.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Lists of MP3 files
general_waiting_mp3s = [
    '1.mp3', '2.mp3', '3.mp3', '4.mp3', '5.mp3', 
    '6.mp3', '7.mp3', '8.mp3', 'Dont_you_have.mp3', 
    'Give_me_a_second.mp3', 'Let_me_think_about_that.mp3'
]
weather_waiting_mp3s = ['weather1.mp3', 'weather2.mp3', 'weather3.mp3']
win_mp3_files = ['lakers1.mp3', 'lakers2.mp3', 'lakers3.mp3', 'lakers4.mp3', 'lakers5.mp3']
lose_mp3_files = ['lakers6.mp3', 'lakers7.mp3', 'lakers8.mp3', 'lakers9.mp3', 'lakers10.mp3']

# Global variable for the bulb
bulb = None

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    first_line = lines[0].strip()
    remaining_lines = ''.join(lines[1:]).strip()
    return first_line, remaining_lines

def write_file(filepath, first_line, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(first_line + '\n' + content)

def record_audio_with_silence_detection():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    THRESHOLD = 500
    SILENCE_DURATION = 2  # seconds

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording... Speak now.")

    frames = []
    silent_frames = 0
    has_started = False

    try:
        while True:
            data = stream.read(CHUNK)
            frames.append(data)

            audio_data = np.frombuffer(data, dtype=np.int16)
            volume = np.abs(audio_data).mean()

            if volume > THRESHOLD:
                silent_frames = 0
                has_started = True
            elif has_started:
                silent_frames += 1

            if has_started and silent_frames > int(SILENCE_DURATION * RATE / CHUNK):
                print("Silence detected. Stopping recording.")
                break

    except KeyboardInterrupt:
        print("Recording stopped by user")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    print("Recording finished.")

    filename = 'myrecording.wav'
    wf = sf.SoundFile(filename, mode='w', samplerate=RATE, channels=CHANNELS, subtype='PCM_16')
    wf.write(np.frombuffer(b''.join(frames), dtype=np.int16))
    wf.close()

    return filename

def transcribe_audio(filename):
    with open(filename, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        return transcription.text

# Initialize OpenAI client
client = OpenAI() #store your OpenAI API key in your Environmental Variables or add here

# Set ElevenLabs API key
clnt = ElevenLabs(api_key="Your API Key") # Replace with your ElevenLabs API key or store in your environmental variables 

# Load initial conversation history
first_line, conversation_history = open_file('chatbot1.txt')

# Initialize openWakeWord with the hey_kitt.onnx model
try:
    detector = Model(wakeword_models=["hey_kitt.onnx"], inference_framework='onnx')
    print("Model initialized successfully.")
except ValueError as e:
    print("Error initializing openwakeword model:", e)
    exit(1)

def text_to_speech_stream(text: str) -> IO[bytes]:
    response = clnt.text_to_speech.convert(
        voice_id="Your cloned KITT voice ID from ElevenLabs",  # KITT's voice
        optimize_streaming_latency="0",
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(
            stability=0.7,
            similarity_boost=1.0,
            style=0.3,
            use_speaker_boost=True
        ),
    )
    audio_stream = BytesIO()
    for chunk in response:
        if chunk:
            audio_stream.write(chunk)
    audio_stream.seek(0)
    return audio_stream

def play_audio_file(file_path):
    audio_data, samplerate = sf.read(file_path)
    sd.play(audio_data, samplerate=samplerate)
    sd.wait()

def play_random_mp3(mp3_list):
    file_path = random.choice(mp3_list)
    play_audio_file(file_path)

def play_audio(audio_stream):
    audio_data, samplerate = sf.read(audio_stream)
    sd.play(audio_data, samplerate=samplerate)
    sd.wait()

def listen_for_wake_word(detector, chunk_size=1280):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = chunk_size
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("Listening for wake word...")
    while True:
        data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        prediction = detector.predict(data)
        if prediction["hey_kitt"] > 0.3:  # Adjust threshold as needed
            print("Wake word detected!")
            stream.stop_stream()
            stream.close()
            audio.terminate()
            return True

def capture_and_save_image():
    cap = cv2.VideoCapture(0)
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Camera opened: {cap.isOpened()}")
    print(f"Camera resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    # Allow more time for the camera to warm up
    time.sleep(3)

    max_attempts = 5
    for attempt in range(max_attempts):
        # Capture a single frame
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not capture image (attempt {attempt + 1}/{max_attempts}).")
            continue

        # Check if the image is too dark
        if np.mean(frame) < 10:  # Adjust this threshold as needed
            print(f"Image too dark, retrying... (attempt {attempt + 1}/{max_attempts})")
            time.sleep(1)  # Wait a bit before the next attempt
            continue

        # If we get here, we have a valid frame
        break
    else:
        print("Error: Failed to capture a sufficiently bright image after multiple attempts.")
        cap.release()
        return None

    # Release the camera
    cap.release()

    # Save the image to a file
    filename = 'captured_image.jpg'
    max_size = 19 * 1024 * 1024  # 19 MB (leaving some buffer)

    # Encode image as JPEG
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    _, encoded_image = cv2.imencode('.jpg', frame, encode_param)

    # If the image is too large, reduce quality until it's under the size limit
    while encoded_image.nbytes > max_size:
        encode_param[1] -= 5
        if encode_param[1] < 20:  # Minimum quality threshold
            print("Error: Unable to compress image to acceptable size.")
            return None
        _, encoded_image = cv2.imencode('.jpg', frame, encode_param)

    # Save the compressed image
    with open(filename, 'wb') as f:
        f.write(encoded_image)

    print(f"Image captured and saved as {filename}")
    print(f"Image size: {os.path.getsize(filename) / (1024 * 1024):.2f} MB")
    return filename

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        response = client.chat.completions.create(
            model="gpt-4o",  # Make sure to use 4o or 4o-mini
            messages=[
                {
                    "role": "system",
                    "content": "You are KITT from the TV show Knight Rider. Describe the image in a snarky and funny way, as if you're narrating what you see to a friend. Be concise but descriptive. If the image shows anything interesting, make a big deal about it!"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What do you see in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in analyze_image: {str(e)}")
        return "I'm sorry, but I'm having trouble processing the image. My visual circuits must be on the fritz again."

def handle_query(query, play_waiting_mp3=True):
    global first_line, conversation_history
    bot_response = ""  # Initialize bot_response at the start of the function

    if "what do you see" in query.lower():
        image_path = capture_and_save_image()
        if image_path:
            analysis = analyze_image(image_path)
            print("Analysis:", analysis)
            
            # Update conversation history
            conversation_history_lines = conversation_history.split('\n')
            conversation_history_lines = conversation_history_lines[-30:] # Keep only the last 30 lines
            conversation_history_lines.append(f"User: {query}\nKITT: {analysis}\n")
            conversation_history = '\n'.join(conversation_history_lines)
            write_file('chatbot1.txt', first_line, conversation_history)
            
            # Stream the audio response using Eleven Labs API
            audio_stream = text_to_speech_stream(analysis)
            play_audio(audio_stream)
        else:
            error_message = "I'm sorry, but I'm having trouble with my visual sensors. Could you check my camera connection?"
            print(error_message)
            
            # Update conversation history
            conversation_history_lines = conversation_history.split('\n')
            conversation_history_lines = conversation_history_lines[-20:]
            conversation_history_lines.append(f"User: {query}\nKITT: {error_message}\n")
            conversation_history = '\n'.join(conversation_history_lines)
            write_file('chatbot1.txt', first_line, conversation_history)

            audio_stream = text_to_speech_stream(error_message)
            play_audio(audio_stream)

    elif "what's the weather right now" in query.lower():
        # Get current weather info and generate speech concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            weather_future = executor.submit(weather.get_weather)
            
            # Start playing waiting response in a separate thread
            if play_waiting_mp3:
                waiting_thread = threading.Thread(target=play_random_mp3, args=(weather_waiting_mp3s,))
                waiting_thread.start()

            weather_info = weather_future.result()
            # Ensure consistency in formatting
            weather_info = weather_info.replace("F ", "Fahrenheit ").replace("%", " percent")
            if play_waiting_mp3:
                waiting_thread.join()  # Ensure waiting response finishes before proceeding
            print("Current Weather Info:", weather_info)

            # Update conversation history
            conversation_history_lines = conversation_history.split('\n')
            conversation_history_lines = conversation_history_lines[-20:]
            conversation_history_lines.append(f"User: {query}\nKITT: {weather_info}\n")
            conversation_history = '\n'.join(conversation_history_lines)
            write_file('chatbot1.txt', first_line, conversation_history)

            # Generate and play speech response concurrently
            audio_stream_future = executor.submit(text_to_speech_stream, weather_info)
            play_audio(audio_stream_future.result())

    elif any(phrase in query.lower() for phrase in [" weather forecast", " forecast", " weather prediction"]):
        print("DEBUG: About to call get_7_day_forecast()")
        
        # Get 7-day weather forecast and generate speech concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Force reload of weather module to ensure we get fresh data
            importlib.reload(weather)
            
            forecast_future = executor.submit(weather.get_7_day_forecast)
            
            # Start playing waiting response in a separate thread
            if play_waiting_mp3:
                waiting_thread = threading.Thread(target=play_random_mp3, args=(weather_waiting_mp3s,))
                waiting_thread.start()

            forecast_info = forecast_future.result()
            
            if play_waiting_mp3:
                waiting_thread.join()  # Ensure waiting response finishes before proceeding
            
            print("7-Day Forecast Info:")
            print(forecast_info)  # This will print the full forecast to the console

            # Update conversation history
            conversation_history_lines = conversation_history.split('\n')
            conversation_history_lines = conversation_history_lines[-20:]
            conversation_history_lines.append(f"User: {query}\nKITT: {forecast_info}\n")
            conversation_history = '\n'.join(conversation_history_lines)
            write_file('chatbot1.txt', first_line, conversation_history)

            # Generate and play speech response concurrently
            audio_stream_future = executor.submit(text_to_speech_stream, forecast_info)
            play_audio(audio_stream_future.result())

    elif "latest news" in query.lower():
        # Get latest news and generate speech concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            news_future = executor.submit(get_latest_news)
            
            # Start playing waiting response in a separate thread
            if play_waiting_mp3:
                waiting_thread = threading.Thread(target=play_random_mp3, args=(general_waiting_mp3s,))
                waiting_thread.start()

            latest_news = news_future.result()

            if play_waiting_mp3:
                waiting_thread.join()  # Ensure waiting response finishes before proceeding
            print("Latest News:", latest_news)

            # Update conversation history
            conversation_history_lines = conversation_history.split('\n')
            conversation_history_lines = conversation_history_lines[-20:]
            conversation_history_lines.append(f"User: {query}\nKITT: {latest_news}\n")
            conversation_history = '\n'.join(conversation_history_lines)
            write_file('chatbot1.txt', first_line, conversation_history)

            # Generate and play speech response concurrently
            audio_stream_future = executor.submit(text_to_speech_stream, latest_news)
            play_audio(audio_stream_future.result())

    elif "top sports news" in query.lower():
        # Get top sports news and generate speech concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            sports_news_future = executor.submit(get_top_sports_news)
            
            # Start playing waiting response in a separate thread
            if play_waiting_mp3:
                waiting_thread = threading.Thread(target=play_random_mp3, args=(general_waiting_mp3s,))
                waiting_thread.start()

            sports_news = sports_news_future.result()

            if play_waiting_mp3:
                waiting_thread.join()  # Ensure waiting response finishes before proceeding
            print("Top Sports News:", sports_news)

            # Update conversation history
            conversation_history_lines = conversation_history.split('\n')
            conversation_history_lines = conversation_history_lines[-20:]
            conversation_history_lines.append(f"User: {query}\nKITT: {sports_news}\n")
            conversation_history = '\n'.join(conversation_history_lines)
            write_file('chatbot1.txt', first_line, conversation_history)

            # Generate and play speech response concurrently
            audio_stream_future = executor.submit(text_to_speech_stream, sports_news)
            play_audio(audio_stream_future.result())

    elif "what time is it" in query.lower() or "current time" in query.lower():
        time_info = get_current_time()
        print("Current Time Info:", time_info)

        # Update conversation history
        conversation_history_lines = conversation_history.split('\n')
        conversation_history_lines = conversation_history_lines[-20:]
        conversation_history_lines.append(f"User: {query}\nKITT: {time_info}\n")
        conversation_history = '\n'.join(conversation_history_lines)
        write_file('chatbot1.txt', first_line, conversation_history)

        # Generate and play speech response
        audio_stream = text_to_speech_stream(time_info)
        play_audio(audio_stream)

    elif "tell me a joke" in query.lower() or "got any jokes" in query.lower():
        joke = get_joke()
        print("KITT:", joke)

        # Update conversation history
        conversation_history_lines = conversation_history.split('\n')
        conversation_history_lines = conversation_history_lines[-20:]
        conversation_history_lines.append(f"User: {query}\nKITT: {joke}\n")
        conversation_history = '\n'.join(conversation_history_lines)
        write_file('chatbot1.txt', first_line, conversation_history)

        # Generate and play speech response
        audio_stream = text_to_speech_stream(joke)
        play_audio(audio_stream)

        return joke
    
    elif "turn on the light" in query.lower():
        bot_response = turn_on_bulb()
    elif "turn off the light" in query.lower():
        bot_response = turn_off_bulb()
    elif "set brightness" in query.lower():
        try:
            brightness = int(query.split()[-1].rstrip('%'))
            bot_response = set_bulb_brightness(brightness)
        except ValueError:
            bot_response = "I'm sorry, I couldn't understand the brightness level. Please specify a percentage."
    elif "set color" in query.lower():
        try:
            parts = query.lower().split()
            hue = int(parts[parts.index("hue") + 1])
            saturation = int(parts[parts.index("saturation") + 1])
            bot_response = set_bulb_color(hue, saturation)
        except (ValueError, IndexError):
            bot_response = "I'm sorry, I couldn't understand the color settings. Please specify hue and saturation values."
    elif "set color temperature" in query.lower():
        try:
            temp = int(query.split()[-1])
            bot_response = set_bulb_color_temp(temp)
        except ValueError:
            bot_response = "I'm sorry, I couldn't understand the color temperature. Please specify a value in Kelvin."

    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            response_future = executor.submit(client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": first_line + '\n' + conversation_history},
                    {"role": "user", "content": query},
                ],
                stream=True
            )

            # Only play waiting response if play_waiting_mp3 is True
            if play_waiting_mp3:
                waiting_thread = threading.Thread(target=play_random_mp3, args=(general_waiting_mp3s,))
                waiting_thread.start()

            response = response_future.result()

            bot_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    bot_response += chunk.choices[0].delta.content

            if play_waiting_mp3:
                waiting_thread.join()
            
            print("KITT:", bot_response)

    # If bot_response is still empty, set a default message
    if not bot_response:
        bot_response = "I'm sorry, I didn't understand that command."

    # Update conversation history
    conversation_history_lines = conversation_history.split('\n')
    conversation_history_lines = conversation_history_lines[-20:]
    conversation_history_lines.append(f"User: {query}\nKITT: {bot_response}\n")
    conversation_history = '\n'.join(conversation_history_lines)
    write_file('chatbot1.txt', first_line, conversation_history)

    # Generate and play speech response if not already done
    if not any(phrase in query.lower() for phrase in ["what do you see", "what's the weather", "weather forecast", "latest news", "top sports news", "what time is it", "tell me a joke"]):
        audio_stream = text_to_speech_stream(bot_response)
        play_audio(audio_stream)

    return bot_response
def get_current_time():
    # Set the time zone to Pacific Time
    pacific_tz = pytz.timezone('US/Pacific')
    
    # Get the current time in Pacific Time
    current_time = datetime.now(pacific_tz)
    
    # Format the time string
    time_string = current_time.strftime("%I:%M %p")
    
    # Remove leading zero from hour if present
    if time_string.startswith("0"):
        time_string = time_string[1:]
    
    return f"The current time is {time_string}."

def get_joke():
    url = "https://official-joke-api.appspot.com/random_joke"
    response = requests.get(url)
    
    if response.status_code == 200:
        joke = response.json()
        setup = joke['setup']
        punchline = joke['punchline']
        return f"Alright, here's one for you: {setup} ... {punchline} Ha! I do hope my humor algorithms are functioning correctly."
    else:
        return "I'm sorry, my joke circuits seem to be malfunctioning. Perhaps I should stick to driving and crime-fighting."

def setup_bulb(ip_address, email, password):
    global bulb
    try:
        bulb = PyL530.L530(ip_address, email, password)
        bulb.handshake()
        bulb.login()
        print("Successfully connected to the bulb.")
        return bulb
    except Exception as e:
        print(f"Error connecting to the bulb: {str(e)}")
        return None

def turn_on_bulb():
    global bulb
    try:
        bulb.turnOn()
        return "The light has been turned on."
    except Exception as e:
        return f"Error turning on the light: {str(e)}"

def turn_off_bulb():
    global bulb
    try:
        bulb.turnOff()
        return "The light has been turned off."
    except Exception as e:
        return f"Error turning off the light: {str(e)}"

def set_bulb_brightness(brightness):
    global bulb
    try:
        bulb.setBrightness(brightness)
        return f"Brightness set to {brightness}%."
    except Exception as e:
        return f"Error setting brightness: {str(e)}"

def set_bulb_color(hue, saturation):
    global bulb
    try:
        bulb.setColor(hue, saturation)
        return f"Color set to hue {hue} and saturation {saturation}."
    except Exception as e:
        return f"Error setting color: {str(e)}"

def set_bulb_color_temp(temp):
    global bulb
    try:
        bulb.setColorTemp(temp)
        return f"Color temperature set to {temp} Kelvin."
    except Exception as e:
        return f"Error setting color temperature: {str(e)}"

def main_loop():
    while True:
        if listen_for_wake_word(detector):
            time.sleep(0.2)  # Slight delay to ensure state reset
            continuous_conversation = True
            first_query = True
            while continuous_conversation:
                user_message_file = record_audio_with_silence_detection()
                user_message = transcribe_audio(user_message_file)
                print(f"User said: {user_message}")  # Print user's message for debugging
                bot_response = handle_query(user_message, play_waiting_mp3=first_query)
                first_query = False
                
                # Check if the bot's response ends with a question mark
                if not bot_response.strip().endswith('?'):
                    continuous_conversation = False
                    print("Conversation ended. Listening for wake word...")
                else:
                    print("KITT asked a question. Waiting for user response...")
            detector.reset()

# Initialize the bulb. Using a Tapo TP-Link L535 smart bulb
bulb = setup_bulb("Your bulb IP Address", "Your email", "Your password") # Replace with your bulb's IP address, email, and password

if bulb is None:
    print("Failed to initialize the bulb. Some light-related functions may not work.")

if __name__ == "__main__":
    main_loop()
