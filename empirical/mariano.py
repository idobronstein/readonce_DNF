from consts import *
from data import *
from utilits import *


class mariano():

	def __init__(self):
		pass


	def get_sammple_by_sign(self, X, Y, sign):
		X_sign = X[[i for i in range(X.shape[0]) if Y[i] == sign]]
		Y_sign = np.ones([X_sign.shape[0]], dtype=TYPE) * sign
		return X_sign , Y_sign

	def SQ(self, h, X, Y, m):
		res = 0 
		for i in range(X.shape[0]):
			if h(X[i]) == Y[i]:
				res += 1
		return res / m

	def literal(self, i, G, X, Y, m):
		I_1 = lambda x: POSITIVE if x[i] == POSITIVE else NEGATIVE
		I_0 = lambda x: POSITIVE if x[i] == NEGATIVE else NEGATIVE
		X_positve, Y_positve = self.get_sammple_by_sign(X, Y, POSITIVE)
		gamma_1 = self.SQ(I_1, X_positve, Y_positve, m)
		gamma_2 = self.SQ(I_0, X_positve, Y_positve, m)
		if abs(gamma_1 - gamma_2) < G:
			return 0
		else:
			return np.sign(gamma_1 - gamma_2)

	def SameTerm(self, i, j, G, X, Y, m):
		I_1 = lambda x: POSITIVE if x[i] == NEGATIVE and x[j] == POSITIVE else NEGATIVE
		I_0 = lambda x: POSITIVE if x[i] == NEGATIVE and x[j] == NEGATIVE else NEGATIVE
		X_positve, Y_positve = self.get_sammple_by_sign(X, Y, POSITIVE)
		gamma_1 = self.SQ(I_1, X_positve, Y_positve, m)
		gamma_2 = self.SQ(I_0, X_positve, Y_positve, m)
		return abs(gamma_1 - gamma_2) < G

	def remove_inaccurate_terms(self, all_terms, X, Y, epsilon, m):
		all_terms_final = []
		for term in all_terms:
			X_negative, Y_negative = self.get_sammple_by_sign(X, Y, NEGATIVE)
			Y_negative = -1 * Y_negative
			
			def term_check(x):
				for s in term:
					if x[s[0]] != s[1]:
						return NEGATIVE
				return POSITIVE
			term_value = self.SQ(term_check, X_negative, Y_negative, m)
			if term_value <= epsilon / (4 * D):
				all_terms_final.append(term)

		return all_terms_final

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
		const_approximation = self.SQ(lambda x: NEGATIVE, X, Y, m)
		#print(const_approximation)
		for epsilon in np.linspace(START_EPSILON, END_EPSILON, NUMBER_OF_EPSILONS):
			G = epsilon ** 2 / (2 * D)
			#if const_approximation < epsilon / 2 or const_approximation > 1 - epsilon / 2 or False:
			#	readonce = ReadOnceDNF(specifiec_DNF=[[0] * D])
			#else:
			S = np.zeros([D])
			all_terms = []
			for i in range(D):
				S[i] = self.literal(i, G / 2, X, Y, m)
			for i in range(D):
				if S[i] != 0:
					all_terms.append([(i, S[i])])
					for j in range(i+1, D):
						if S[j] !=0:
							if self.SameTerm(i, j, G / 2, X, Y, m) and self.SameTerm(j, i, G / 2, X, Y, m):
								all_terms[-1].append((j, S[j]))
								S[j] = 0
			#all_terms = self.remove_inaccurate_terms(all_terms, X, Y, epsilon, m)
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
