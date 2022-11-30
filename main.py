"""
Source Code of Furry Detector :O

Written by: @ZenithO_o on twitter (https://twitter.com/zenithO_o)

Updated and modified by: @madzombie11 
"""
# ML imports
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
import csv
import random
# Twitter imports
from tweepy import API
from tweepy import Paginator
from tweepy import OAuth1UserHandler
import tweepy

# GUI/Other imports
from tkinter import Tk, Canvas, Label, Entry, Button, Scale
from tkinter import N, NE, S, CENTER, HORIZONTAL, NORMAL, GROOVE, FLAT, DISABLED
from PIL import Image, ImageTk
import requests
from io import BytesIO
import json
import re
import time

"""
MAIN CLASSES
"""


class TwitterAuthenticator:
    """
    Does all the yucky
    """

    def authenticateTwitterApp(self):
        self.twitter_api_error_message = ""
        global twitterkey
        twitterkey = {}
        with open("FurryDetector-main/twitter_key.json") as json_file:
            
            twitterkey = json.load(json_file)
        if (
            (len(twitterkey["consumer_key"]) < 20)
            or (len(twitterkey["consumer_secret"]) < 30)
            or (len(twitterkey["access_token"]) < 30)
            or (len(twitterkey["access_token_secret"]) < 30)
        ):
            self.has_key = False
            self.twitter_api_error_message = '\nAn incorrect key has been detected. Please put the correct keys in the file "Twitterkey.json"'
            return None

        self.has_key = True
        auth = OAuth1UserHandler(twitterkey["consumer_key"], twitterkey["consumer_secret"])
        auth.set_access_token(
            twitterkey["access_token"], twitterkey["access_token_secret"]
        )
        return auth


class TwitterClient:
    """
    API handler
    """

    def __init__(self, twitter_user=None):
        self.authenticator = TwitterAuthenticator()
        self.authenticator.authenticateTwitterApp()
        self.twitter_user = twitter_user
        self.t_client = tweepy.Client(bearer_token = "", consumer_key =  twitterkey["consumer_key"] , access_token=twitterkey["access_token"],consumer_secret= twitterkey["consumer_secret"], access_token_secret= twitterkey["access_token_secret"],wait_on_rate_limit=False)

    def get_twitter_client_api(self):
        return self.t_client

    def get_user(self, user=None, id = None):
        print(user)
        return self.t_client.get_user(id = id,username = user, user_fields = ["public_metrics"] , expansions="pinned_tweet_id")
        #return self.twitter_client.get_user(user_id = user)


class TextParser:
    """
    Class for cleaning up text
    """

    def cleanRawText(self, rawText):

        # remove formatting (\n)
        rawText = rawText.rstrip()

        # remove punctuation
        punctuation = [".", "!", "?", ",", "-", "#", "|", "@"]
        for i in punctuation:
            rawText = rawText.replace(i, "")

        # remove excess whitespace
        rawText = re.sub(" +", " ", rawText)

        # fix punctuation
        return rawText

    def remove_u(self, word):
        word_u = (word.encode("unicode-escape")).decode("utf-8", "strict")
        if r"\u" in word_u:
            return None
        return word

    def deEmojify(self, inputString):
        return inputString.encode("ascii", "ignore").decode("ascii")

    def wordNumCount(self, words):
        wordDict = {}
        for word in words:
            if word != None:
                try:
                    wordDict[word] += 1
                except:
                    wordDict[word] = 1
        return wordDict

    def combineDict(self, dict1, dict2):
        for key in dict2:
            if key in dict1:
                dict1[key] += dict2[key]
            else:
                dict1[key] = dict2[key]
        return dict1

    def deleteGarbage(self, wordDict):
        with open("FurryDetector-main/data/stop_words.json") as json_file:
            garbage = json.load(json_file)

        for key in garbage:
            for word in garbage[key]:
                if word in wordDict:
                    del wordDict[word]
                if word.upper() in wordDict:
                    del wordDict[word.upper()]
                if word.capitalize() in wordDict:
                    del wordDict[word.capitalize()]

        for key in list(wordDict):
            try:
                if "https" in key:
                    del wordDict[key]
                if wordDict[key] == 1:
                    del wordDict[key]
                if r"\u" in key:
                    del wordDict[key]

                try:
                    int(key)
                    del wordDict[key]
                except ValueError:
                    pass

            except KeyError:
                pass

        return wordDict


class DatasetGenerator:
    """
    Class that does all the text processing for the ML model
    """
    def ezdata(self, tweets):
        rawText = ""
        #print(tweets)
        for text in [tweet["text"] for tweet in tweets]:
            rawText += " ~~~~~~ " + text
        #rawText = TextParser().cleanRawText(rawText)

        rawText = rawText.casefold()
        rawText = TextParser().deEmojify(rawText)
        words = rawText.split()
        #words = [TextParser().remove_u(eachWord) for eachWord in words]
        for key in words:
            try:
                if "https" in key:
                    words.remove(key)
            except KeyError:
                pass
        p = ' '.join(words)
        return str(p).split("~~~~~~")
    def generateDataset(self, tweets):

        rawText = ""
        #print(tweets)
        for text in [tweet["text"] for tweet in tweets]:
            rawText += " " + text
        rawText = TextParser().cleanRawText(rawText)

        userDict = self.processText(rawText)

        with open("FurryDetector-main/data/word_list.json", "r") as jsonfile:
            wordDict = json.load(jsonfile)

        MostUsedWordVal = 0
        for word in userDict:
            if userDict[word] > MostUsedWordVal:
                MostUsedWordVal = userDict[word]

        inputArr = []

        for word in wordDict["set"]:
            if word in userDict:
                if userDict[word] == MostUsedWordVal:
                    inputArr.append(1)
                else:
                    inputArr.append(userDict[word] / MostUsedWordVal)
            else:
                inputArr.append(0)
        return inputArr

    def processText(self, rawText):
        rawText = rawText.casefold()
        rawText = TextParser().deEmojify(rawText)
        words = rawText.split()
        words = [TextParser().remove_u(eachWord) for eachWord in words]

        wordDict = TextParser().wordNumCount(words)
        wordDict = TextParser().deleteGarbage(wordDict)
        return wordDict


class FurryDetector:
    """
    TensorFlow ML model class
    """

    def __init__(self):
        self.model = self.loadModel()

    def loadModel(self):
        # open json file
        json_file = open("FurryDetector-main/model/model.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()

        # convert to keras model
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights("FurryDetector-main/model/model.h5")

        return loaded_model

    def runModel(self, tweets):
        userDataset = [DatasetGenerator().generateDataset(tweets)]
        userDataset = np.array(userDataset)

        print(f"Created Dataset for {user.data['username']}!\nRunning now...")
        prediction = self.model.predict(userDataset)[0][0]
        print(f"Prediction for {user.data['username']}: {round((prediction*100),2)}%")

        
        if prediction >= 0.5:
            print(f"{user.data['username']} is most likely a furry!")
        else:
            print(f"{user.data['username']} is most likely NOT a furry!")
        
        return prediction


"""
Helper Functions
"""


def app_setup():
    detector = FurryDetector()
    client = TwitterClient()
    return detector, client

# Data variables
user = None
user_statuses = None
user_data = None
user_image = None

client = None
tweet_limit = 100
tick_interval = 400
tweet_amt = 100
detector = None

keras_model_error_message = ""
twitter_api_error_message = ""
has_key = False

detector, client = app_setup()
has_key = client.authenticator.has_key
twitter_api_error_message = client.authenticator.twitter_api_error_message


"""
Tkinter functions
"""


def testUser(user1= None,id = None):
    input_user = user1

    if input_user != "Input Twitter user here... ex: @zenithO_o":
        global user
        global user_image
        global user_statuses
        global tweet_limit
        
        try:
            # Get User Object
            user = client.get_user(user = user1,id = id)
            #print(user)
            #print()
            # Get number of statuses
            user_statuses = user.data.public_metrics["tweet_count"]
            if user_statuses > tweet_limit:
                tweet_limit = 100
            else:
                tweet_limit = user_statuses
            
        except tweepy.error.TweepError as e:
            print(e)

        except:
            print("guh")

    else:
        print("guh2")


def runDetector():
    tweets = []
    print(f"Colletcting {tweet_amt} tweets")

    tweets.extend(
        Paginator(
            client.t_client.get_users_tweets,
            id=user.data.id
        ).flatten(tweet_amt)
    )
    #print(tweets)

    print("Collected Data")

    pred = detector.runModel(tweets)
    return pred
    #updatePred(pred)

def updateScale(tweet_limit):
    global tick_interval

    tick_interval = tweet_limit 

    tweet_scale.configure(state=NORMAL, to=tweet_limit, tickinterval=tick_interval)
    pass


def updateAmt(sce):
    global tweet_amt
    tweet_amt = int(sce)

def check_user(id = None , usern = None):
    testUser(id=id, user1 = usern)
    updateAmt(sce = tweet_limit)
    return runDetector()




users_for_data_set = []

def getdatausers(id1):
    users_to_get_data = []
    users = client.t_client.get_users_followers(id=id1, user_fields=["public_metrics"], max_results = 1000)
    topf = []
    for i in users.data[0:10]:
        topf.append(i)
    for user2 in users.data:
        fc = int(user2.public_metrics["followers_count"])
        r = None
        for i in topf:
            if int(i.public_metrics["followers_count"]) < fc:
                if r != None and int(i.public_metrics["followers_count"]) < r.public_metrics["followers_count"]:
                    r = i
                elif r == None:
                    r = i
        if r != None:
            b = topf.index(r)
            topf = topf[:b]+[user2]+topf[b+1:]
    print(topf)
    for i in topf:
        if check_user(id = i.id) > .70:
            users_to_get_data.append(i)
            users_for_data_set.append(i.username)
    return users_to_get_data

def bruh(te):
    k = []
    for i in te:
        for b in getdatausers(i.id):
            k.append(b)
            users_for_data_set.append(str(b.username))
    return k

def the_big_list():
    for i in getdatausers("371847843"):
        getdatausers(i.id)

   

    with open('readme3.txt', 'w') as f:
        f.write(str(users_for_data_set))

#check_user(id =)
#the_big_list()
#getdatausers("")

dataer = []
def get_tweets_for_data(usern =None,id = None):
    user = client.get_user(user = usern ,id = id)
    try:
        user_statuses = user.data.public_metrics["tweet_count"]
    except:
        tl = 500
    if user_statuses > 1000:
        tl = 1000
    else:
        tl = user_statuses
    tweets = []
    tweets.extend(
        Paginator(
            client.t_client.get_users_tweets,
            id=user.data.id,
            exclude=["retweets", "replies"]
        ).flatten(tl)
    )
    time.sleep(40)
    userDataset = DatasetGenerator().ezdata(tweets)
    print(len(userDataset))
    #   print(userDataset)
    return userDataset

testlist = ['']
testlist = list(set(testlist))
comp = 0
print(len(testlist))
for i in testlist:
    try:
        dataer += get_tweets_for_data(usern= i)
        comp +=1
        print(f"{(comp / len(testlist))*100}%")
    except:
        print("error")
        with open('data.csv', 'w') as f:
            write = csv.writer(f)
            write.writerow(dataer)
        break
    

random.shuffle(dataer)
with open('data.csv', 'w') as f:
      
    write = csv.writer(f)
      
    write.writerow(dataer)
