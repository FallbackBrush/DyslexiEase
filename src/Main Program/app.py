import streamlit
import nltk
import numpy
import pandas
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import time
#from transformers import pipeline
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
import cv2
import torchaudio
import matplotlib
import matplotlib.pyplot as plt
import io


@streamlit.cache_resource
def load_data():
    #summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
    model_id = "stabilityai/stable-diffusion-2"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    pipe = pipe.to('cuda')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    return pipe,bundle

def ner(text):
    # Load the English language model in spaCy
    nlp = spacy.load("en_core_web_sm")

    # Process the text with the language model
    doc = nlp(text)

    # Extract named entities
    named_entities = [(entity.text, entity.label_) for entity in doc.ents]

    # Extract noun chunks
    noun_chunks = [(chunk.text, chunk.root.text, chunk.root.dep_,
    chunk.root.head.text) for chunk in doc.noun_chunks]

    dict1={'name':[x[0] for x in named_entities],'type':[x[1] for x in named_entities]}
    # Print results
    return dict1

def summarise(text,summarizer):
    summarised  = summarizer(text, max_length=50, min_length=3, do_sample=False)
    return summarised[0]["summary_text"]

def keys(text):
    ans = nltk.pos_tag(nltk.word_tokenize(text))
    nouns=[]
    for i in ans:
        val=i[1]
        if(val=='NN' or val=='NNS'):# or val=='NNPS' or val=='NNP'):
            nouns.append(i[0])
    return nouns

def syns(text):
    synonyms = []
    antonyms = []
    wordlist=[]
    for x in nltk.word_tokenize(text):
        if(len(x)>7): wordlist.append(x)
    dict1={'word':[],'synlist':[],'anlist':[]}
    for x in wordlist:
        dict1['word'].append(x)
        syn=''
        an=''
        for synset in wordnet.synsets(x):
            for l in synset.lemmas():
                synonyms.append(l.name())
                syn=syn+" , "+str(l.name())
                if l.antonyms():
                    antonyms.append(l.antonyms()[0].name())
                    an=an+" , "+str(l.antonyms()[0].name())
        dict1['synlist'].append(syn)
        dict1['anlist'].append(an)
    
    
    return dict1

streamlit.title("DyslexiEase : A Text to Video Generator for Dyslexics")

summarizedPrompt = streamlit.text_input(label="Enter a promt. For example : A video of a man riding on a horse")
audioprompt = summarizedPrompt
streamlit.write("Please review your prompt before submission")
imagepipe,bundle = load_data()
processor = bundle.get_text_processor()
tacotron2 = bundle.get_tacotron2().to('cuda')
vocoder = bundle.get_vocoder().to('cuda')
#imagepipe.to('cuda')
if(streamlit.button(label="SUBMIT PROMPT",key="submitprompt")):
    #summarizedPrompt = summarise(prompt,summarizer)
    print(summarizedPrompt)
    if (len(summarizedPrompt.split(' '))>50):
        streamlit.warning('Prompt exceeds 50 words. Please limit prompt to 50 words', icon="⚠️")
    else:
        generated_images=[]
        generated_videos=[]
        with streamlit.spinner('Generating Resources...'):
            keywords = keys(summarizedPrompt)
            print(keywords)
            for i in keywords:
                imageprompt = "a minimal cartoon image of a "+i
                image = imagepipe(imageprompt).images[0]
                image = numpy.array(image)
                path = imageprompt+".png"
                generated_images.append(path)
                cv2.imwrite(path,image)
            
            summarizedPrompt = "a minimal cartoon image of a "+summarizedPrompt
            image = imagepipe(summarizedPrompt).images[0]
            image = numpy.array(image)
            path = summarizedPrompt+".png"
            generated_videos.append(path)
            cv2.imwrite(path,image)
        
        tabs=["Generated Images","Generated Videos"]
        imagetab,videotab = streamlit.tabs(tabs)
        with imagetab:
            for i in generated_images:
                streamlit.image(i,caption=str(i).split('.')[0])
        with videotab:
            for i in generated_videos:
                streamlit.image(i,caption=str(i).split('.')[0])

            with torch.inference_mode():
                processed, lengths = processor(audioprompt)
                processed = processed.to('cuda')
                lengths = lengths.to('cuda')
                spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
                waveforms, lengths = vocoder(spec, spec_lengths)
            print("done")
            fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(16, 9))
            ax1.imshow(spec[0].cpu().detach(), origin="lower", aspect="auto")
            ax2.plot(waveforms[0].cpu().detach())
            buffer = io.BytesIO()
            path = summarizedPrompt+'.wav'
            
            torchaudio.save(path,src=waveforms.detach().cpu(),sample_rate=22000)
            
            streamlit.audio(path)

