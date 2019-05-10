# Crowdflower Search Results Relevance

Crowdflower 搜索结果相关性（Relevance）

网址：https://www.kaggle.com/c/crowdflower-search-relevance

So many of our favorite daily activities are mediated by proprietary search algorithms. Whether you're trying to find a stream of that reality TV show on cat herding or shopping an eCommerce site for a new set of Japanese sushi knives, the relevance of search results is often responsible for your (un)happiness. Currently, small online businesses have no good way of evaluating the performance of their search algorithms, making it difficult for them to provide an exceptional customer experience.

我们最喜欢的日常活动是由专有搜索算法调解的。无论您是想在猫放牧中找到真人秀节目的流，还是在电子商务网站上购买一套新的日本寿司刀，搜索结果的相关性通常都是您（非）幸福的原因。目前，小型在线企业没有很好的方法来评估其搜索算法的性能，这使他们难以提供卓越的客户体验。

The goal of this competition is to create an open-source model that can be used to measure the relevance of search results. In doing so, you'll be helping enable small business owners to match the experience provided by more resource rich competitors. It will also provide more established businesses a model to test against. Given the queries and resulting product descriptions from leading eCommerce sites, this competition asks you to evaluate the accuracy of their search algorithms.

本次竞赛的目标是创建一个开源模型，用于衡量搜索结果的相关性。通过这样做，您将帮助小型企业所有者匹配更多资源丰富的竞争对手提供的体验。它还将为更成熟的企业提供一个可以测试的模型。鉴于来自领先电子商务网站的查询和最终产品说明，本次竞赛要求您评估其搜索算法的准确性。

*Make a first submission with this Python benchmark on Kaggle scripts.* 

在 Kaggle 脚本上使用此 Python 基准（benchmark）进行首次提交。

---

[See this script for a quick exploration of the data](https://www.kaggle.com/users/993/ben-hamner/crowdflower-search-relevance/exploring-the-crowdflower-data)

To evaluate search relevancy, CrowdFlower has had their crowd evaluate searches from a handful of eCommerce websites. A total of 261 search terms were generated, and CrowdFlower put together a list of products and their corresponding search terms. Each rater in the crowd was asked to give a product search term a score of 1, 2, 3, 4, with 4 indicating the item completely satisfies the search query, and 1 indicating the item doesn't match the search term.

请参阅此脚本以快速浏览数据

为了评估搜索相关性（To evaluate search relevancy），CrowdFlower 让他们的人群评估来自少数电子商务网站的搜索。共生成了 261 个搜索字词，CrowdFlower 汇总了产品列表及其相应的搜索字词。人群中的每个评估者被要求给出产品搜索项得分1,2,3,4，其中 4 表示该项完全满足搜索查询，1表示该项与搜索项不匹配。

The challenge in this competition is to predict the relevance score given the product description and product title. To ensure that your algorithm is robust enough to handle any noisy HTML snippets in the wild real world, the data provided in the product description field is raw and contains information that is irrelevant to the product.

本次比赛的挑战是根据产品描述和产品名称预测相关性得分。为确保您的算法足够强大，可以处理野外现实世界中任何嘈杂的HTML片段，产品说明字段中提供的数据是原始数据，包含与产品无关的信息。

To discourage hand-labeling the data, CrowdFlower has also provided extra data that was not labeled by the crowd in the test set. This data is ignored when calculating your score.

为了阻止对数据进行手工标记，CrowdFlower还提供了额外的数据，这些数据未被测试集中的人群标记。计算得分时会忽略此数据。

Ready to explore the data? [Scripts](https://www.kaggle.com/c/crowdflower-search-relevance/scripts) is the most frictionless way to get familiar with the competition dataset! [See the data at a glance here.](https://www.kaggle.com/users/993/ben-hamner/crowdflower-search-relevance/data-samples) No download needed to start publishing and forking code in R and Python. It's already pre-loaded with our favorite packages and ready for you to start competing!

准备好探索数据？脚本是熟悉竞争数据集的最无摩擦的方式！在这里查看数据一目了然。无需下载即可开始在R和Python中发布和分叉代码。它已经预装了我们最喜欢的包装，随时准备开始竞争！

## File and data descriptions

文件和数据描述

- **train.csv** - the training data set includes:，train.csv  - 训练数据集包括：

- - id: Product id，id：产品ID
  - query: Search term used，查询：使用的搜索词
  - product_description: The full product description along with HTML formatting tags，product_description：完整的产品说明以及HTML格式标记
  - median_relevance: Median relevance score by 3 raters. This value is an integer between 1 and 4. ，median_relevance：3 个评分者的中位数相关性得分。该值是 1 到 4 之间的整数。
  - relevance_variance: Variance of the relevance scores given by raters. ，relevant_variance：评分者给出的相关性分数的变化。

- **test.csv** - the test set，test.csv  - 测试集

- - id: Product id，id：产品ID
  - query: Search term used，查询：使用的搜索词
  - product_description: The full product description along with HTML formatting tags，product_description：完整的产品说明以及HTML格式标记

- **sampleSubmission.csv** - a sample submission file in the correct format，sampleSubmission.csv  - 格式正确的示例提交文件

External data, such as dictionaries, thesaurus, language corpuses, are allowed. However, they must not be directly related to this specific dataset. The source of your external data must be posted to the forum to ensure fairness for all the participants in the community.

允许使用外部数据，例如词典，词库，语言语料库。但是，它们不得与此特定数据集直接相关。必须将您的外部数据来源发布到论坛，以确保社区中所有参与者的公平性。





Submissions are scored based on the quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, the metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Results have 4 possible ratings, 1,2,3,4.  Each search record is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of search records that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:



An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

From these three matrices, the quadratic weighted kappa is calculated as: 



## 

