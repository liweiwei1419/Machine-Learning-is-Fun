# @Time    : 18/3/27 下午2:29
# @Author  : liweiwei1419
# @Site    : http://www.liwei.party/
# @Contact : liweiwei1419@gmail.com


from sklearn.datasets import fetch_20newsgroups

news = fetch_20newsgroups(subset='all')

X,y = news.data,news.target