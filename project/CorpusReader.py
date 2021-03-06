def corpusReader():
	'''
	Returns [totalNoOfWordsInCorpus, hashInput, hashCode]
	hashInput is a list of tuples containing encoded values for each bigram
	(encoded value of bigram: [1,0,0,0] for the first word, [0,1,0,0] for the second word...)
	hashCode is a dictionary of words and their encoded vector
	'''
	
	from nltk.corpus import PlaintextCorpusReader
	corpus_root = './OwnCorpus'
	wordlists = PlaintextCorpusReader(corpus_root, '.*txt$')
	
	from nltk import word_tokenize 
	# tokenize the words and find out the number of words in the set. 
	from nltk.tokenize import RegexpTokenizer
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(wordlists.raw())
	totalNoOfWordsInCorpus = len(sorted(set(tokens))) # RETURN VALUE
	
	# change all words to lowercase
	lowerCaseWords=[]
	for word in tokens:
		lowerCaseWords.append(word.lower())
	#print lowerCaseWords 

	#Encode every word : Onehot encoding
	encodeVector = [0]*totalNoOfWordsInCorpus
	import collections
	hashCode = collections.OrderedDict()
	#hashCode = {}
	indexval = 0

	import copy
	for word in lowerCaseWords:
		if word not in hashCode:
			encodeVector[indexval] = 1 #generate one hot coding for word
			temp=copy.deepcopy(encodeVector)
			hashCode[word] = temp; #add the number and word to dictionary
			#print hashCode[word]
			encodeVector[indexval] = 0
			indexval+=1

	# make bigrams for each word in the list.
	from nltk import bigrams
	dataBigram = bigrams(lowerCaseWords)
	[words1, words2] = map(list, zip(*dataBigram))
	encodeWord1 = []
	encodeWord2 = []
	for word in words1:
		encodeWord1.append(hashCode[word])
	for word in words2:
		encodeWord2.append(hashCode[word])

	#hashInput = zip(encodeWord1,encodeWord2)

	from pybrain.datasets import SupervisedDataSet
        dataset = SupervisedDataSet(totalNoOfWordsInCorpus,totalNoOfWordsInCorpus )
	for i in range (0,len(encodeWord1)):
		dataset.addSample(encodeWord1[i], encodeWord2[i])
	print dataset;	
	#print totalNoOfWordsInCorpus
	#print hashInput
	return totalNoOfWordsInCorpus, dataset, hashCode

if __name__ == "__main__":
	corpusReader()
