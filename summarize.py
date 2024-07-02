import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
from youtube_transcript_api import YouTubeTranscriptApi as yta
from urllib.parse import urlparse
from blingfire import text_to_sentences
from deepmultilingualpunctuation import PunctuationModel
import speech_recognition as sr
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'D:/Tesseract/tesseract.exe'
from PIL import Image
import imageio.v2 as iio


def bertSent_embeding(sentences):
    """
    Input a list of sentence tokens
    
    Output a list of latent vectors, each vector is a sentence representation
    
    
    """
    
    print("Token Embedding....")
    ## Add sentence head and tail as BERT requested
    marked_sent = ["[CLS] " +item + " [SEP]" for item in sentences]    
    
    print("Sentence embedding...")
    ## Bert tokenizization 
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    tokenized_sent = [tokenizer.tokenize(item) for item in marked_sent]
      
    print("Transformer positional embedding...")
    ## index to BERT vocabulary
    indexed_tokens = [tokenizer.convert_tokens_to_ids(item) for item in tokenized_sent]
    tokens_tensor = [torch.tensor([item]) for item in indexed_tokens]
    
    ## adding segment id as BERT requested
    segments_ids = [[1] * len(item) for ind,item in enumerate(tokenized_sent)]
    segments_tensors = [torch.tensor([item]) for item in segments_ids]
    
    print("Sending tokens into BERT pre-trained model...")
    ## load BERT base model and set to evaluation mode
    bert_model = BertModel.from_pretrained('bert-large-uncased')
    bert_model.eval()
    
    ## Output 24 layers of latent vector
    assert len(tokens_tensor) == len(segments_tensors)
    encoded_layers_list = []
    for i in range(len(tokens_tensor)):
        with torch.no_grad():
            encoded_layers, _ = bert_model(tokens_tensor[i], segments_tensors[i])
        encoded_layers_list.append(encoded_layers)
    
    print("Extracting the last layer vector from the BERT model...")
    ## Use only the last layer vetcor
    token_vecs_list = [layers[23][0] for layers in encoded_layers_list]
       
    ## Pooling word vector to sentence vector using mean
    sentence_embedding_list = [torch.mean(vec, dim=0).numpy() for vec in token_vecs_list]

    return sentence_embedding_list

def kmeans_sumIndex(sentence_embedding_list):
    """
    Input a list of embeded sentence vectors
    
    Output an list of indices of sentence in the paragraph, represent the clustering of key sentences
    
    Note: Kmeans is used here for clustering
    
    """
    n_clusters = np.ceil(len(sentence_embedding_list)**0.66)
    kmeans = KMeans(n_clusters=int(n_clusters))
    kmeans = kmeans.fit(sentence_embedding_list)
    
    sum_index,_ = pairwise_distances_argmin_min(kmeans.cluster_centers_, sentence_embedding_list,metric='euclidean')
    
    sum_index = sorted(sum_index)
    
    return sum_index


def bertSummarize(sentences):
    """
    Input a paragraph as string
    
    Output the summary including a few key sentences using BERT sentence embedding and clustering
    """
                
    print("=========================================================")  

    sentence_embedding_list = bertSent_embeding(sentences)

    print("Sentence embedded...passing it to Kmeans")

    sum_index = kmeans_sumIndex(sentence_embedding_list)
    
    summary = ' '.join([sentences[ind] for ind in sum_index])
    
    return summary


#=============================================================================================================================

print("Enter your option:")
print("1. PDF        2. Youtube        3.Speech       4.Image")
preference = int(input())

if preference == 1:
    print("Enter file name:")
    filename = input() 

    myFile = open(r"C:/Users/Reetik Raj\OneDrive\Desktop\pdf\{file}.pdf".format(file = filename),"rb")
    output_file = open("out.txt", "w", encoding="utf-8")
    pdfReader = PyPDF2.PdfReader(myFile)
    numOfPages = len(pdfReader.pages)
    print("Reading pdf file.....")
    for i in range(numOfPages):
        page = pdfReader.pages[i]
        text = page.extract_text()
        output_file.write(text)
    output_file.close()
    myFile.close()
    f = open("out.txt", "r",encoding="utf-8")
    data = f.read()

    #model = PunctuationModel()
    #punct_text = data
    #result = model.restore_punctuation(punct_text)
    
    print("Sending text for tokenization....")
     
    sentences = sent_tokenize(data)  
    sum = bertSummarize(sentences)

    print()
    print("Saving summarized text....")
    f = open("summarizedtext/{file}.txt".format(file = filename), "w",  encoding="utf-8")
    f.write(sum)

    f.close()
    f = open("summarizedtext/{file}.txt".format(file = filename), "r",  encoding="utf-8")
    text = f.read()
    newtext = text.replace("\n", " ")
    print("Fine tuning the text....")

    f.close()
    f = open("summarizedtext/{file}.txt".format(file = filename), "w",  encoding="utf-8")
    f.write(newtext)

    f.close()
    f = open(r"C:\Users\Reetik Raj\OneDrive\Desktop\pdf\{file}summarize.txt".format(file = filename), "w",  encoding="utf-8")
    f.write(newtext)
    print("Successfully saved.")

#-----------------------------------------------------------------------------------------------------------------------

elif preference == 2:
    print("Enter Youtube Video ID:")
    id = input()

    data = yta.get_transcript(id)
    print("Extracting the text from the video...")
    transcript = " "
    for value in data:
        for txt,values in value.items():
            if txt =="text":
                transcript = transcript + " " + values
    print("sending text to restore punctuation....")
    model = PunctuationModel()
    punct_text = transcript
    result = model.restore_punctuation(punct_text)
    sentences = text_to_sentences(result).split(".")
    print("Sending in for tokenization.....")
    sum = bertSummarize(sentences)
    
    
    print("Saving into local system...")
    f = open(r"C:\Users\Reetik Raj\OneDrive\Desktop\pdf\{file}summarize.txt".format(file = id), "w",  encoding="utf-8")
    f.write(sum)
    print("Successfully saved")

#-----------------------------------------------------------------------------------------------------------------------

elif preference == 3:
    r = sr.Recognizer()

# Start the microphone
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        r.pause_threshold = 1
        print("Speak now:")
        audio = r.listen(source)
    print("Recognizing Now .....")
    try:
        print("You said")
        text = r.recognize_google(audio)
    except Exception as e:
        print("Error"+str(e))
    print(text)
    
    print()
    print("Sending to restore punctuation...")
    model = PunctuationModel()
    punct_text = text
    result = model.restore_punctuation(punct_text)

   
    sentences = sent_tokenize(result)  
    print("Passing into BERT model for tokenization..")
    sum = bertSummarize(sentences)

    number = 1
    print("SAving file to local system....")
    f = open(r"C:\Users\Reetik Raj\OneDrive\Desktop\pdf\Lecture{number}summarize.txt".format(number = number), "w",  encoding="utf-8")
    number = number + 1
    f.write(sum)
    print("Successfully saved.") 

#------------------------------------------------------------------------------------------------------------------

elif preference == 4:
    print("Enter Image name:")
    imagename = input() 
             
    print("Reading image...") 
    
    img = iio.imread(r"C:/Users/Reetik Raj\OneDrive\Desktop\pdf\{file}.JPEG".format(file = imagename))    
    text = pytesseract.image_to_string(img)

    # Print the resulting text
    print("===========================================")

    print(text)

    print("Sent text into BERT model...")       
    sentences = sent_tokenize(text)  
    sum = bertSummarize(sentences)
            
    print("Saving file to local system....")
    f = open(r"C:\Users\Reetik Raj\OneDrive\Desktop\pdf\{imagename} summarize.txt".format(imagename = imagename), "w",  encoding="utf-8")
    f.write(sum)
    print("Successfully saved.")         