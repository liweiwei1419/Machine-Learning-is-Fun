# Bag of Words Meets Bags of Popcorn

![image-20190509205512819](https://ws4.sinaimg.cn/large/006tNc79ly1g2vczolw20j31je0c641i.jpg)

网址：https://www.kaggle.com/c/word2vec-nlp-tutorial

### Data Set

The labeled data set consists of 50,000 IMDB movie reviews, specially selected for sentiment analysis. The sentiment of reviews is binary, meaning the IMDB rating < 5 results in a sentiment score of 0, and rating >=7 have a sentiment score of 1. No individual movie has more than 30 reviews. The 25,000 review labeled training set does not include any of the same movies as the 25,000 review test set. In addition, there are another 50,000 IMDB reviews provided without any rating labels.

数据集
标记数据集包含50,000个IMDB电影评论，专门用于情绪分析。评论的情绪是二元的，意味着IMDB评级<5导致情绪评分为0，评级> = 7的情绪评分为1.没有个别电影的评论超过30。标记为训练集的25,000个评论不包括与25,000个评论测试集相同的任何电影。此外，还提供了另外50,000个IMDB评论，没有任何评级标签。



### File descriptions

- **labeledTrainData** - The labeled training set. The file is tab-delimited and has a header row followed by 25,000 rows containing an id, sentiment, and text for each review.  
- **testData** - The test set. The tab-delimited file has a header row followed by 25,000 rows containing an id and text for each review. Your task is to predict the sentiment for each one. 
- **unlabeledTrainData** - An extra training set with no labels. The tab-delimited file has a header row followed by 50,000 rows containing an id and text for each review. 
- **sampleSubmission** - A comma-delimited sample submission file in the correct format.

文件说明

 -  labeledTrainData  - 标记的训练集。该文件以制表符分隔，并且有一个标题行，后跟25,000行，每行包含一个id，情绪和文本。
 -  testData  - 测试集。制表符分隔文件有一个标题行，后跟25,000行，每行包含一个ID和文本。你的任务是预测每个人的情绪。
 -  unlabeledTrainData  - 没有标签的额外训练集。制表符分隔文件有一个标题行，后跟50,000行，每行包含一个ID和文本。
 -  sampleSubmission  - 以正确格式的逗号分隔的样本提交文件。

### Data fields

- **id** - Unique ID of each review
- **sentiment** - Sentiment of the review; 1 for positive reviews and 0 for negative reviews
- **review** - Text of the review

数据字段

 -  id  - 每个评论的唯一ID
 - 情绪 - 评论的情绪; 1表示正面评论，0表示负面评论
 - 审查 - 审查文本

### Code

Full tutorial code lives in this [github repo](https://github.com/wendykan/DeepLearningMovies)
