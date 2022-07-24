class Sentiment_Analyzer:
    def post_find(self):
        import pymongo
        
        client = pymongo.MongoClient(
            "mongodb+srv://showyou:showyou@showyou-aznp8.mongodb.net/test?retryWrites=true&w=majority"
        )
        db = client.get_database('ShowYou')
        collection = db.get_collection('post')
        doc = collection.find()
        # for result in doc :
        #     print(result)
        client.close()
        return doc

    def sentiment_analysis_result_insert(self,list):
        import pymongo
        
        client = pymongo.MongoClient(
            "mongodb+srv://showyou:showyou@showyou-aznp8.mongodb.net/test?retryWrites=true&w=majority"
        )
        db = client.get_database('ShowYou')
        collection = db.get_collection('sentiment_analysis_result')
        collection.drop() 
        collection.insert(list)
        client.close()

    def get_model(self):
        import keras
        model=keras.models.load_model('./Sentiment Analyzer/model')
        return(model)

    def get_tokenizer(self):
        import pandas as pd

        train_data = pd.read_csv("./Sentiment Analyzer/training dataset.csv")

        stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

        import konlpy
        from konlpy.tag import Okt
        okt = Okt()
        X_train = []
        for sentence in train_data['title']:
          temp_X = []
          temp_X = okt.morphs(sentence, stem=True) # 토큰화
          temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
          X_train.append(temp_X)

        from keras.preprocessing.text import Tokenizer
        max_words = 35000
        tokenizer = Tokenizer(num_words = max_words)
        tokenizer.fit_on_texts(X_train)

        return(tokenizer)

    def get_keywordses(self,tokenizer):
        import pandas as pd

        test_data = self.post_find()

        stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

        import konlpy
        from konlpy.tag import Okt
        okt = Okt()
        X_test = []
        for sentence in test_data:
          temp_X = []
          temp_X = okt.morphs(sentence['post'], stem=True) # 토큰화
          temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
          X_test.append(temp_X)

        X_test = tokenizer.texts_to_sequences(X_test)

        from keras.preprocessing.sequence import pad_sequences
        max_len = 20 # 전체 데이터의 길이를 20로 맞춘다

        X_test = pad_sequences(X_test, maxlen=max_len)

        return(X_test)

    def set_sentiments(self,model,keywords):
        predict = model.predict(keywords)

        import numpy as np
        predict_labels = np.argmax(predict, axis=1)
        
        sentiments=[]
        for index in range(0,len(predict_labels)):
            row={}
            row['post_id']=index
            row['sentiment']=str(predict_labels[index]-1)
            sentiments.append(row)

        print(sentiments)
        self.sentiment_analysis_result_insert(sentiments)

    def __init__(self):
        self.model=self.get_model()
        self.tokenizer=self.get_tokenizer()

    def analyze(self):
        keywords=self.get_keywordses(self.tokenizer)
        self.set_sentiments(self.model,keywords)