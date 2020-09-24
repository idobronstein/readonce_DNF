from consts import *

to_array =lambda x: np.array(x, dtype=TYPE) 

def add_image_to_final_set(final_x, final_y, current_x, current_y, positive_numbers, negative_numbers):
	current_x[current_x < BINARY_THRESHOLD ] = -1
	current_x[current_x >= BINARY_THRESHOLD ] = 1
	if current_y in positive_numbers:
		final_x.append(current_x.reshape([-1]))
		final_y.append(POSITIVE)
	if current_y in negative_numbers:
		final_x.append(current_x.reshape([-1]))
		final_y.append(NEGATIVE)	

def get_binary_mnist_db(positive_numbers, negative_numbers):
	x_train, y_train, x_test, y_test = mnist.mnist()
	final_x_train, final_y_train, final_x_test, final_y_test = [], [], [] ,[]
	for x, y in zip(x_train, y_train):
		add_image_to_final_set(final_x_train, final_y_train, x, y, positive_numbers, negative_numbers)
	for x, y in zip(x_test, y_test):
		add_image_to_final_set(final_x_test, final_y_test, x, y, positive_numbers, negative_numbers)
	train_set_size = int(TRAIN_SET_PRECENT * len(final_x_train))
	final_x_train_without_validation, final_y_train_without_validation = final_x_train[:train_set_size], final_y_train[:train_set_size]
	final_x_validation, final_y_validation = final_x_train[train_set_size:], final_y_train[train_set_size:]
	return (to_array(final_x_train_without_validation), to_array(final_y_train_without_validation)), (to_array(final_x_validation), to_array(final_y_validation)), (to_array(final_x_test), to_array(final_y_test))


def validate_figure_with_term(figure, term):
	figure_reshape = figure.reshape([-1])
	return np.dot(figure, term)	== np.sum(np.abs(term))

def validate_dataset_with_all_terms(dataset, all_terms):
	success_num = 0
	for x, y in zip(dataset[0], dataset[1]):
		flag = False
		for term in all_terms:
			if validate_figure_with_term(x, term):
				if y == POSITIVE:
					flag = True
					success_num += 1
					break
				else:
					flag = True
					break
		if not flag and y == NEGATIVE:
			success_num += 1
	return success_num / float(dataset[0].shape[0])