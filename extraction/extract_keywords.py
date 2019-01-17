import numpy as np 
import jieba 
import jieba.posseg as pseg
import jieba.analyse
  
class MyTextRank(object):
	def __init__(self, content, window_size, alpha, iteration_limit): 
	    self.content = content 
	    self.window_size = window_size 
	    self.alpha = alpha 
	    self.edge_dict = {} #记录节点的边连接字典 
	    self.iteration_limit = iteration_limit #迭代次数 
  
	#分词 
	def cut_content(self): 
		jieba.analyse.set_stop_words("./stopWord.txt") # 加载自定义停用词表（中科院计算所中文自然语言处理开放平台发布的中文停用词表，包含了1208个停用词）
		siever = ['n','nz','v','vd','vn','l','a','d'] 
		segment = pseg.cut(self.content) 
		self.word_list = [s.word for s in segment if s.flag in siever] 
		# print(self.word_list) 

  	#根据窗口，构建每个节点的相邻节点，返回边的集合 
	def get_graph(self):
		tmp_list = [] 
		length = len(self.word_list) # 单词列表长度
		for index, word in enumerate(self.word_list): 
			if word not in self.edge_dict.keys(): 
				tmp_list.append(word) 
				tmp_set = set()
				left = index - self.window_size + 1 #窗口左边界
				right = index + self.window_size #窗口右边界
				if left < 0:
					left = 0
				if right >= length:
					right = length
				for i in range(left, right):
					if i == index:
						continue
					tmp_set.add(self.word_list[i])
				self.edge_dict[word] = tmp_set

    
	def get_matrix(self):		#根据边的相连关系，构建矩阵 
		self.matrix = np.zeros([len(set(self.word_list)), len(set(self.word_list))])
		self.index = {} 		#记录词的index
		self.index_dict = {} 	#记录节点index对应的词

		for i, v in enumerate(set(self.word_list)):
			self.index[v] = i
			self.index_dict[i] = v
		for key in self.edge_dict.keys():
			for w in self.edge_dict[key]:
				self.matrix[self.index[key]][self.index[w]] = 1
				self.matrix[self.index[w]][self.index[key]] = 1
		#归一化 
		for j in range(self.matrix.shape[1]):
			sum = 0
			for i in range(self.matrix.shape[0]):
				sum += self.matrix[i][j]
			for i in range(self.matrix.shape[0]):
				self.matrix[i][j] /= sum

	
	def get_weights(self): 		#根据textrank公式计算权重
		self.PR = np.ones([len(set(self.word_list)), 1])
		for i in range(self.iteration_limit):
			self.PR = (1 - self.alpha) + self.alpha * np.dot(self.matrix, self.PR)
  
	
	def get_result(self): 		#输出词和相应的权重 
		PR_dict = {} 
		for i in range(len(self.PR)): 
			PR_dict[self.index_dict[i]] = self.PR[i][0]
		res = sorted(PR_dict.items(), key = lambda x : x[1], reverse=True)
		# print(res)
		return res
 
def extract_keywords(content, topK = 4):
	my_textrank = MyTextRank(content, 3, 0.85, 800)
	my_textrank.cut_content()
	my_textrank.get_graph()
	my_textrank.get_matrix()
	my_textrank.get_weights()
	# my_textrank.get_result()

	result =  my_textrank.get_result()
	keywords = []
	for i in range(topK):
		keywords.append(result[i][0])
	print(keywords)
	return keywords


if __name__ == '__main__':
	text_test = '''今天凌晨5点28分，我国在西昌卫星发射中心用长征四号丙运载火箭，成功地将嫦娥四号任务“鹊桥号”中继星发射升空。“鹊桥”号是世界上首颗运行于地月拉格朗日L2点的通信卫星，将为年底实施的嫦娥四号月球背面软着陆探测任务提供地月间的中继通信。
按照计划，我国在今年年底将执行嫦娥四号月球探测任务，实现国际首次月球背面软着陆和巡视勘察。由于月球的自转和公转周期相同，因此月球永远有一面一直背对地球。但地球与月球背面无法直接通信，“鹊桥”号就将承担地月之间的通信和数据传输任务。
发射成功之后，“鹊桥”将最终进入环绕地月拉格朗日L2点的使命轨道，这也是人类航天器首次实现在这一轨道长期环绕飞行。'''
	extract_keywords(text_test)