
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
		jieba.analyse.set_stop_words("titletrigger/extraction/stopWord.txt") # 加载自定义停用词表（中科院计算所中文自然语言处理开放平台发布的中文停用词表，包含了1208个停用词）
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
	# print(keywords)
	return keywords


if __name__ == '__main__':
	text_test = '''中共十三届全国人大常委会党组21日召开会议，专题学习习近平总书记关于人民代表大会制度的思想。全国人大常委会委员长、党组书记栗战书主持会议并讲话。会议指出，党的十八大以来，以习近平同志为核心的党中央高度重视、全面加强党对人大工作的领导，推动人大工作取得历史性成就。习近平总书记就坚持和完善人民代表大会制度、发展社会主义民主政治提出一系列新理念新思想新战略，拓展了人民代表大会制度的科学内涵、基本特征和本质要求，标志着党对人民代表大会制度的规律性认识达到新的高度，为做好新时代人大工作指明了方向、提供了遵循。会议强调，习近平总书记关于人民代表大会制度的思想，是习近平新时代中国特色社会主义思想的重要组成部分。通过学习，大家更加深刻认识到人民代表大会制度在党和国家事业发展的历史进程中，展现出巨大的政治优势和组织功效，必须坚定人民代表大会制度自信；人民代表大会制度是坚持党的领导、人民当家作主、依法治国有机统一的根本政治制度安排，具有强大的生机和活力，必须适应时代要求，不断推动人民代表大会制度与时俱进、完善发展。会议提出，要紧紧围绕习近平总书记关于人民代表大会制度的新思想，加强研究阐释，统一思想行动。要宣传好习近平总书记关于人民代表大会制度的思想，讲好中国人大故事，让国家根本政治制度深入人心。'''
	extract_keywords(text_test)