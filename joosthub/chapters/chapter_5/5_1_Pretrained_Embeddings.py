# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/mycodehtml/NLP/joosthub/chapters/chapter_5 && \
# rm e.l && python 5_1_Pretrained_Embeddings.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
import torch
import torch.nn as nn
from tqdm import tqdm
from annoy import AnnoyIndex
import numpy as np

# ================================================================================
class PreTrainedEmbeddings(object):
    """ A wrapper around pre-trained word vectors and their use """
    def __init__(self, word_to_index, word_vectors):
        """
        Args:
            word_to_index (dict): mapping from word to integers
            word_vectors (list of numpy arrays)
        """
        self.word_to_index = word_to_index
        self.word_vectors = word_vectors
        self.index_to_word = {v: k for k, v in self.word_to_index.items()}

        self.index = AnnoyIndex(len(word_vectors[0]), metric='euclidean')
        print("Building Index!")
        for _, i in self.word_to_index.items():
            self.index.add_item(i, self.word_vectors[i])
        self.index.build(50)
        print("Finished!")
        
    @classmethod
    def from_embeddings_file(cls, embedding_file):
        """Instantiate from pre-trained vector file.
        
        Vector file should be of the format:
            word0 x0_0 x0_1 x0_2 x0_3 ... x0_N
            word1 x1_0 x1_1 x1_2 x1_3 ... x1_N
        
        Args:
            embedding_file (str): location of the file
        Returns: 
            instance of PretrainedEmbeddigns
        """
        word_to_index = {}
        word_vectors = []

        with open(embedding_file) as fp:
            for line in fp.readlines():
                line = line.split(" ")
                word = line[0]
                vec = np.array([float(x) for x in line[1:]])
                
                word_to_index[word] = len(word_to_index)
                word_vectors.append(vec)

        # print("np.array(word_vectors).shape",np.array(word_vectors).shape)
        # (400000, 100)

        # print("word_to_index",word_to_index)
        # {'the': 0, ',': 1, '.': 2, 'of': 3, 'to': 4, 'and': 5, 'in': 6, 'a': 7, '"': 8, "'s": 9, 'for': 10, '-': 11, 'that': 12, 'on': 13, 'is': 14, 'was': 15, 'said': 16, 'with': 17, 'he': 18, 'as': 

        return cls(word_to_index, word_vectors)
    
    def get_embedding(self, word):
        """
        Args:
            word (str)
        Returns
            an embedding (numpy.ndarray)
        """
        idx_of_word=self.word_to_index[word]
        # print("word",word)
        # man

        # print("idx_of_word",idx_of_word)
        # 300

        vec_of_word=self.word_vectors[idx_of_word]
        # print("vec_of_word",vec_of_word)
        # [ 3.7293e-01  3.8503e-01  7.1086e-01 -6.5911e-01 -1.0128e-03  9.2715e-01

        return vec_of_word

    def get_closest_to_vector(self, vector, n=1):
        """Given a vector, return its n nearest neighbors
        
        Args:
            vector (np.ndarray): should match the size of the vectors 
                in the Annoy index
            n (int): the number of neighbors to return
        Returns:
            [str, str, ...]: words that are nearest to the given vector. 
                The words are not ordered by distance 
        """
        nn_indices = self.index.get_nns_by_vector(vector, n)
        # print("nn_indices",nn_indices)
        # [67, 18, 787, 332]
        
        near_vecs=[]
        for neighbor in nn_indices:
            word_str=self.index_to_word[neighbor]
            near_vecs.append(word_str)
        # print("near_vecs",near_vecs)
        # ['she', 'he', 'woman', 'never']

        return near_vecs
    
    def compute_and_print_analogy(self, word1, word2, word3):
        """Prints the solutions to analogies using word embeddings

        Analogies are word1 is to word2 as word3 is to __
        This method will print: word1 : word2 :: word3 : word4
        
        Args:
            word1 (str)
            word2 (str)
            word3 (str)
        """

        vec1 = self.get_embedding(word1)
        vec2 = self.get_embedding(word2)
        vec3 = self.get_embedding(word3)
        
        # ================================================================================
        # Distance difference between vec2 and vec1
        spatial_relationship = vec2 - vec1
        # print("spatial_relationship",spatial_relationship)
        # [-0.25043   -0.443863  -0.47428    0.37034   -0.0271682 -0.61191
        #  -0.205921   0.220673   0.215317   0.00582    0.39623   -0.373654

        # Calculate vect4
        vec4 = vec3 + spatial_relationship
        # print("vec4",vec4)
        # [ 0.34325    0.004387   0.11892    0.444474   0.0842418  0.66739
        #  -0.039361   0.461373   0.605767   0.33348   -0.35411   -0.023584

        # ================================================================================
        closest_words = self.get_closest_to_vector(vec4, n=4)
        existing_words = set([word1, word2, word3])
        closest_words = [word for word in closest_words 
                             if word not in existing_words] 

        if len(closest_words) == 0:
            print("Could not find nearest neighbors for the computed vector!")
            return
        
        for word4 in closest_words:
            print("{} : {} :: {} : {}".format(word1, word2, word3, word4))

# ================================================================================
# embeddings = PreTrainedEmbeddings.from_embeddings_file('data/glove/glove.6B.100d.txt')
embeddings = PreTrainedEmbeddings.from_embeddings_file('/mnt/1T-5e7/mycodehtml/NLP/joosthub/glove.6B.100d.txt')

# ================================================================================
# Question 'man':'he'::'woman':x

embeddings.compute_and_print_analogy('man', 'he', 'woman')
# man : he :: woman : she
# man : he :: woman : never

embeddings.compute_and_print_analogy('fly', 'plane', 'sail')
# fly : plane :: sail : ship
# fly : plane :: sail : vessel

embeddings.compute_and_print_analogy('cat', 'kitten', 'dog')
# cat : kitten :: dog : puppy
# cat : kitten :: dog : rottweiler
# cat : kitten :: dog : puppies

embeddings.compute_and_print_analogy('blue', 'color', 'dog')
# blue : color :: dog : pig
# blue : color :: dog : viewer
# blue : color :: dog : bites

embeddings.compute_and_print_analogy('leg', 'legs', 'hand')
# leg : legs :: hand : fingers
# leg : legs :: hand : ears
# leg : legs :: hand : stick
# leg : legs :: hand : eyes

embeddings.compute_and_print_analogy('toe', 'foot', 'finger')
# toe : foot :: finger : hand
# toe : foot :: finger : kept
# toe : foot :: finger : ground

embeddings.compute_and_print_analogy('talk', 'communicate', 'read')
# talk : communicate :: read : correctly
# talk : communicate :: read : transmit
# talk : communicate :: read : distinguish

embeddings.compute_and_print_analogy('blue', 'democrat', 'red')
# blue : democrat :: red : republican
# blue : democrat :: red : congressman
# blue : democrat :: red : senator

embeddings.compute_and_print_analogy('man', 'king', 'woman')
# man : king :: woman : queen
# man : king :: woman : monarch
# man : king :: woman : throne

embeddings.compute_and_print_analogy('man', 'doctor', 'woman')
# man : doctor :: woman : nurse
# man : doctor :: woman : physician

embeddings.compute_and_print_analogy('fast', 'fastest', 'small')
# fast : fastest :: small : oldest
# fast : fastest :: small : quarters
