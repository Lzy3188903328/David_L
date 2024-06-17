import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.utils import shuffle
import string
import nltk
from nltk.corpus import stopwords

# 读取数据
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# 添加标志跟踪真假
fake['target'] = 'fake'
true['target'] = 'true'

# 进行数据合并
data = pd.concat([fake, true]).reset_index(drop=True)

# 数据进行洗牌以防止偏差
data = shuffle(data)
data = data.reset_index(drop=True)

# 进行数据清理
# 删除日期(不会将其用于分析)
data.drop(["date"], axis=1, inplace=True)

# 删除标题(只使用文本)
data.drop(["title"], axis=1, inplace=True)

# 转换文本为小写
data['text'] = data['text'].apply(lambda x: x.lower())

# 删除标点符号
def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str

data['text'] = data['text'].apply(punctuation_removal)

# 删除停止词
nltk.download('stopwords')
stop = stopwords.words('english')
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# 数据探索
# 每个主题有多少篇文章
print(data.groupby(['subject'])['text'].count())
data.groupby(['subject'])['text'].count().plot(kind="bar")
plt.show()
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 筛选出假新闻的数据
fake_data = data[data["target"] == "fake"]

# 将所有文本合并成一个字符串
all_words = ' '.join([text for text in fake_data.text])

# 生成词云
wordcloud = WordCloud(width=800, height=500,
                      max_font_size=110,
                      collocations=False).generate(all_words)

# 显示词云
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

import nltk
from nltk.tokenize import WhitespaceTokenizer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 创建一个空格分词器实例
token_space = WhitespaceTokenizer()

def counter(data, column_text, quantity):
    all_words = ' '.join([text for text in data[column_text]])
    token_phrase = token_space.tokenize(all_words)
    frequency = nltk.FreqDist(token_phrase)
    
    df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                 "Frequency": list(frequency.values())})
    df_frequency = df_frequency.nlargest(columns="Frequency", n=quantity)
    
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=df_frequency, x="Word", y="Frequency", color='blue')
    ax.set(ylabel="Count")
    plt.xticks(rotation='vertical')
    plt.show()

# 示例调用



import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import itertools

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f' if normalize else 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

from sklearn.linear_model import LogisticRegressionpipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', LogisticRegression())])# Fitting the model
model = pipe.fit(X_train, y_train)# Accuracy
prediction = model.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

from sklearn.tree import DecisionTreeClassifier# Vectorizing and applying TF-IDF
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', DecisionTreeClassifier(criterion= 'entropy',
                                           max_depth = 20, 
                                           splitter='best', 
                                           random_state=42))])
# Fitting the model
model = pipe.fit(X_train, y_train)# Accuracy
prediction = model.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

from sklearn.ensemble import RandomForestClassifierpipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', RandomForestClassifier(n_estimators=50, criterion="entropy"))])model = pipe.fit(X_train, y_train)
prediction = model.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))