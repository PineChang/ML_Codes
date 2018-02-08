import warnings
warnings.filterwarnings("ignore")

import jieba as jieba 
import numpy as np 
import codecs as codecs 
import pandas as pd 
import matplotlib.pyplot as plt 

import matplotlib 
matplotlib.rcParams['figure.figsize'] = (10.0,5.0)




from  wordcloud import WordCloud   

if __name__ == '__main__':
	df_fail = pd.read_csv("../../../../data/success_fail_reasons/query_result_failed_small.csv")
	df_fail.dropna()
	# 扔掉fail_reason和track_content 字段都为null的字段
	df_fail = df_fail.dropna(axis=0,how='any',subset=('fail_reason','track_content'))
	# 加载自定义词典 这样自定义词典中的词是一定能分开的
	jieba.load_userdict('../../../../data/success_fail_reasons/userdict.txt')

	# 将切的词 添加进数组values 将 series 转为2d Array tolist()将 2dArray 转变为 数组
	content = df_fail['fail_reason'].values.tolist()


	# 1 将每句话分词后，变为数组,每一行中每一个ci变为一项
	segment = []
	# jieba.cut 以及 jieba.cut_for_search 返回的结构都是一个可迭代的 generator，可以使用 for 循环来获得分词后得到的每一个词语(unicode)，或者用
	# jieba.lcut 以及 jieba.lcut_for_search 直接返回 list
	for line in content:
		try:
			segs = jieba.lcut(line)
			[segment.append(seg) for seg in segs if len(seg)>1 and seg!='\r\n']
		except:
			print(line)
			continue

	# 2 去掉停用词

	words_df = pd.DataFrame({'segment':segment})

	# (271, 1)

	# 读取停用词
	# 由txt生成dataframe 
	stopwords_df = pd.read_csv("../../../../data/success_fail_reasons/stopwords.txt",index_col=False,quoting=3,sep="\t",names=["stopword"],encoding='utf-8')
	# (2613, 1) isin 判断左边df中的每一项是否在右边df中 ~ 代表对series或者ndarray取非
	# 拿到去除停用词的df
	words_df=words_df[~words_df.segment.isin(stopwords_df.stopword)]
	# (191, 1)
	# print(words_df.segment.unique()) 
	# print(words_df.segment.value_counts()) 
	# groupby后 by的字段 最后变成了索引，但是在groupby() 后面可以访问，但是执行完后就不可以访问了
	words_stat = words_df.groupby(by=['segment'])['segment'].agg({"count_num":np.size})
# 	         count_num
# segment           
# 单交              11
# 多年              26
# 投保              64
# 渠道              64
# 进店              26

	# 重建索引,把segment解放出来
#   segment  count_num
# 0      单交         11
# 1      多年         26
# 2      投保         64
# 3      渠道         64
# 4      进店         26
	words_stat = words_stat.reset_index().sort_values(by=['count_num'],ascending=False)

# 第三步 做词云

	wordcloud = WordCloud(font_path="../../../../data/success_fail_reasons/simhei.ttf",background_color="white",max_font_size=80)
	word_frequence = {x[0]:x[1] for x in words_stat.head(1000).values}
	wordcloud=wordcloud.fit_words(word_frequence)
	plt.imshow(wordcloud)
