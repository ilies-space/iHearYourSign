print("protoype hand sign speaker")

from gtts import gTTS
import os


mytext = 'مرحبا'

textCharachter = []


def speak(text):
    language = 'ar'
    myobj = gTTS(text=text, lang=language, slow=False)
    myobj.save("soundFileHolder.mp3")
    # Playing the converted file
    os.system("mpg321 soundFileHolder.mp3")


def read():
    val = input("Reading: ")
    if val == "read":
        print(textCharachter)
        print(''.join(textCharachter))
        speak(''.join(textCharachter))
    else:
        textCharachter.append(val)
        os.system("mpg321 beep.mp3")
        read()


