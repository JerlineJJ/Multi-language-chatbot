import tkinter as tk
from tkinter import *
from tkinter import filedialog

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
from tensorflow.python.framework import ops
import random
import json
import pickle

import threading
import sounddevice
import wavio
from scipy.io.wavfile import write
import speech_recognition as sr

from gtts import gTTS # google text to speech
import random
import playsound # to play an audio file
import os # to remove created audio files

from google_trans_new import google_translator

#################################### This is a GUI part declaring TKinter Window######################################
root = Tk()
root.title("Multi-Language Chat Bot")
root.geometry("1000x1200")
bg = PhotoImage(file='img.png')
######################################################################################################################

filename =""
tkvar = StringVar(root)
lang_select=""

words =""
labels=""
data =""

global model

################################ This is Machine learning model section ################################################
def TrainingModel():
	global model
	global words
	global labels
	global data

	with open("intents1.json") as file:
		data = json.load(file)
	
	try:        
		with open("data.pickle", "rb") as f:
			words, labels, training, output = pickle.load(f)
	except:
		words = []
		labels = []
		docs_x = []
		docs_y = []

		for intent in data["intents"]:
			for pattern in intent["patterns"]:
				wrds = nltk.word_tokenize(pattern)
				words.extend(wrds)
				docs_x.append(wrds)
				docs_y.append(intent["tag"])

			if intent["tag"] not in labels:
				labels.append(intent["tag"])

		words = [stemmer.stem(w.lower()) for w in words if w != "?"]
		words = sorted(list(set(words)))

		labels = sorted(labels)

		training = []
		output = []

		out_empty = [0 for _ in range(len(labels))]

		for x, doc in enumerate(docs_x):
			bag = []

			wrds = [stemmer.stem(w.lower()) for w in doc]

			for w in words:
				if w in wrds:
					bag.append(1)
				else:
					bag.append(0)

			output_row = out_empty[:]
			output_row[labels.index(docs_y[x])] = 1

			training.append(bag)
			output.append(output_row)


		training = numpy.array(training)
		output = numpy.array(output)

		with open("data.pickle", "wb") as f:
			pickle.dump((words, labels, training, output), f)

	ops.reset_default_graph()

	net = tflearn.input_data(shape=[None, len(training[0])])
	net = tflearn.fully_connected(net, 8)
	net = tflearn.fully_connected(net, 8)
	net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
	net = tflearn.regression(net)

	model = tflearn.DNN(net)
	model.save("model.tflearn")

	try:
		model.load("model.tflearn")
		model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
	except:    
		print("Hello ecxeption")

	return model

def bag_of_words(s, words):
	bag = [0 for _ in range(len(words))]

	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i, w in enumerate(words):
			if w == se:
				bag[i] = 1
			
	return numpy.array(bag)

def chat(inp):	
	print("Start talking with the bot!") # (type quit to stop)!")
	global words
	global labels
	global data
	global model
	
	results = model.predict([bag_of_words(inp, words)])
	results_index = numpy.argmax(results)
	tag = labels[results_index]

	for tg in data["intents"]:
		if tg['tag'] == tag:
			responses = tg['responses']
	getResponse = random.choice(responses)
	print(getResponse)
	mytranslatorAfterProcess(getResponse)

#############################################End of Machine Learning #########################################################

#########################Starting from here is a code for GUI######################################################################

def openfile():
	global filename
	filename = filedialog.askopenfilename(title="Select A File", filetypes=(("wav files", "*.wav"), ("all files", "*.*")))
	tk.Label(root, text=filename).place(x=600, y=180)
	return filename


def RecordSound(filename):
	# record the audio
	second = 5
	fs = 4410
	fpath = os.path.dirname(__file__) + "\\" + filename
	print(r"recording...")
	tk.Label(root, text="Voice Command recording...").place(x=600, y=140)
	root.update()
	record_voice = sounddevice.rec(int(second * fs), samplerate=fs, channels=2)
	sounddevice.wait()
	wavio.write(fpath, record_voice, fs, sampwidth=2)
	print(r"stop recording...")
	tk.Label(root, text="Voice Command stop recording...").place(x=600, y=140)
	root.update()


def ConvertSoundToText(voice):
	global lang_select
	r = sr.Recognizer()    
	with sr.AudioFile(voice) as source:
		r.adjust_for_ambient_noise(source)
		
		print('Converting...')
		
		audio = r.listen(source)
		voice_data = ""
		lang = "vi-VN"
		try:
			if lang_select == "Spanish":
				lang = "es-ES"
			elif lang_select == "Hindi":
				lang = "hi-IN"
			else:
				lang = "vi-VN"
			voice_data = r.recognize_google(audio, language=lang)  # convert audio to text
		except sr.UnknownValueError: # error: recognizer does not understand
			speak('I did not get that', "en-US")
		except sr.RequestError:
			speak('Sorry, I did not get that', "en-US") # error: recognizer does not understand
		
		print(f">> {voice_data.lower()}") # print what user said
		#print('Success')
		sample_text = voice_data.lower()
		mytranslator(sample_text)

def mytranslator(sample_text):
	global lang_select	
	translator = google_translator()  
	#sample_text = 'hôm nay trời đẹp'
	translate_text = translator.translate(sample_text, lang_tgt="en-US")  
	print(translate_text)
	chat(translate_text)
	#return speak(translate_text, ln)

def mytranslatorAfterProcess(sample_text):
	global lang_select
	ln = ""
	if lang_select == "Spanish":
		ln = "es"
	elif lang_select == "Hindi":
		ln = "hi"
	else:
		ln = "vi"
	translator = google_translator()  
	#sample_text = 'hôm nay trời đẹp'
	
	translate_text = translator.translate(listToString(sample_text), lang_tgt=ln)  
	print(translate_text)
	print(ln)
	speak(translate_text, ln)


# get string and make a audio file to be played
def speak(audio_string, ln):
	tts = gTTS(text=audio_string, lang=ln) # text to speech(voice)
	#print(listToString(audio_string))
	r = random.randint(1,20000000)
	audio_file = 'audio' + str(r) + '.mp3'
	tts.save(audio_file) # save as mp3
	playsound.playsound(audio_file) # play the audio file
	print(f": {audio_string}") # print what app said
	os.remove(audio_file) # remove audio file

def listToString(s):
	# initialize an empty string 
    str1 = ""      
    # traverse in the string   
    for ele in s:  
        str1 += ele   
    
    # return string   
    return str1 

def change_dropdown(*args):
	global lang_select
	lang_select = tkvar.get()
	print(lang_select)

###################################################End of GUI code ##############################################################


###############################Main Program Thread########################################################################
class MultiLanguageChatBox():
	"""main thread to run the program
	"""
	TrainingModel()

	my_canvas = Canvas(root, width= 1200, height=1000)
	my_canvas.pack(fill='both', expand=True)
	my_canvas.create_image(0,0, image=bg, anchor='nw')
	my_canvas.create_text(500, 50, text="University Multi-Language Chat Bot", font='Ariel 40 bold', fill='dark blue')
	my_canvas.create_text(370, 113, text="Your Language", font='Ariel 12 bold', fill='black',  anchor='w')
	my_canvas.create_text(370, 153, text="Voice Command", font='Ariel 12 bold', fill='black',  anchor='w')
	my_canvas.create_text(370, 193, text="Choose a File", font='Ariel 12 bold', fill='black',  anchor='w')

	global tkvar
	global lang_select

	# Dictionary
	language = {'Spanish', 'Hindi', 'Vietnamese'}
	tkvar.set('Vietnamese') # set the default option

	popupMenu = OptionMenu(root, tkvar, *language)
	#Label(root, text="Please choose your language").grid(row = 1, column = 1)
	popupMenu.place(x=520, y=100)	

	# link function to change dropdown
	tkvar.trace('w', change_dropdown)

	open_btn = Button(root, text='Record', command=lambda: RecordSound("Test.wav"), width=7)
	open_btn.place(x=520, y=140)

	open_btn = Button(root, text='Browse', command=lambda: openfile(), width=7)
	open_btn.place(x=520, y=180)

	start_btn = Button(root, text='Ask Bot', command=lambda: ConvertSoundToText(filename), width=30)
	start_btn.place(x=400, y=220)		

	#Runs the application until we close
	root.mainloop()

if __name__ == "__main":
	MultiLanguageChatBox()