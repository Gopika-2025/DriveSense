import pyttsx3

# Initialize TTS engine
engine = pyttsx3.init()

def alert(message):
    """
    Speak the alert message.

    Args:
        message (str): The message to announce.
    """
    engine.say(message)
    engine.runAndWait()


# Example usage
if __name__ == "__main__":
    alert("Warning! Pedestrian detected ahead.")
    alert("Traffic light is red. Stop the vehicle.")
