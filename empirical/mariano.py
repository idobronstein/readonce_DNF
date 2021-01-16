from consts import *
from data import *
from utilits import *


class mariano():

	def __init__(self):
		pass

	def SQ(self, h, X, Y):
		N = X.shape[0]
		res = 0 
		for i in range(N):
			if h(X[i]) == Y[i]:
				res += 1
		return res / N

	def literal(self, i, G, X, Y):
		I_1 = lambda x: POSITIVE if x[i] == POSITIVE else NEGATIVE
		I_0 = lambda x: POSITIVE if x[i] == NEGATIVE else NEGATIVE
		gamma_1 = self.SQ(I_1, X, Y)
		gamma_2 = self.SQ(I_0, X, Y)
		if abs(gamma_1 - gamma_2) < G:
			return 0
		else:
			return np.sign(gamma_1 - gamma_2)

	def SameTerm(self, i, j, G, X, Y):
		I_1 = lambda x: POSITIVE if x[i] == NEGATIVE and x[j] == POSITIVE else NEGATIVE
		I_0 = lambda x: POSITIVE if x[i] == NEGATIVE and x[j] == NEGATIVE else NEGATIVE
		gamma_1 = self.SQ(I_1, X, Y)
		gamma_2 = self.SQ(I_0, X, Y)
		return abs(gamma_1 - gamma_2) < G

	def createDNF(self, all_terms):
		dnf = []
		for t in all_terms:
			term = np.zeros([D])
			for a in t:
				term[a[0]] = a[1]
			dnf.append(term)
		return ReadOnceDNF(specifiec_DNF=dnf)

	def LeanMaxEnt(self, X, Y):
		m = X.shape[0]
		best_DNF = None
		best_DNF_score = 0
		start = 1 / (15 * np.sqrt(m))
		end = 15 /  np.sqrt(m)
		number_of_steps = 40
		for G in np.linspace(start, end, number_of_steps):
			S = np.zeros([D])
			all_terms = []
			for i in range(D):
				S[i] = self.literal(i, G, X, Y)
			for i in range(D):
				if S[i] != 0:
					all_terms.append([(i, S[i])])
					for j in range(i+1, D):
						if S[j] !=0:
							if self.SameTerm(i, j, G, X, Y) and self.SameTerm(j, i, G, X, Y):
								all_terms[-1].append((j, S[j]))
								S[j] = 0
			readonce = self.createDNF(all_terms)
			score = readonce.evaluate(X, Y)
			if score > best_DNF_score:
				best_DNF = readonce
				best_DNF_score = score
		print(best_DNF.DNF)
		return best_DNF



	def run(self, train_set, test_set):
		dnf = self.LeanMaxEnt(train_set[0], train_set[1])
		
		train_accuracy = dnf.evaluate(train_set[0], train_set[1])
		print('mariano Train accuracy: {0}'.format(train_accuracy)) 

		test_accuracy = dnf.evaluate(test_set[0], test_set[1])
		print('mariano Test accuracy: {0}'.format(test_accuracy)) 

		return 0, test_accuracy
