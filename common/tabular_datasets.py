from consts import *

def process_kr_kp():
    df = pd.read_csv(KR_VS_KP_PATH)
    columns = df.columns
    for col in columns:
        unique_vals = list(set(df[col].tolist()))
        col_vals = df[col].tolist()
        if len(unique_vals) == 2:
            new_vals = [*map(lambda x: NEGATIVE if x == unique_vals[0] else POSITIVE, col_vals)]
            df[col] = pd.Series(new_vals)
        else:
            for u in unique_vals:
                unique_val = u[1]
                new_col = col + "_" + unique_val
                new_vals = [*map(lambda x: NEGATIVE if x == u else POSITIVE, col_vals)]
                df[new_col] = pd.Series(new_vals)
            df = df.drop(col, 1)
    df.to_csv(KR_VS_KP_BIMARY_PATH)


def get_kr_kp_db():
    df = pd.read_csv(KR_VS_KP_BIMARY_PATH, index_col=0)
    all_Y = np.array(df['class'], dtype=TYPE)
    all_X = np.array(df.drop('class', 1), dtype=TYPE)
    number_of_samples = all_X.shape[0]
    all_samples_indexes = list(range(number_of_samples))
    np.random.shuffle(all_samples_indexes)
    all_Y = all_Y[all_samples_indexes]
    all_X = all_X[all_samples_indexes]
    train_set_size = int(number_of_samples * TRAIN_SET_SIZE)
    train_set = (all_X[:train_set_size], all_Y[:train_set_size])
    test_set = (all_X[train_set_size:], all_Y[train_set_size:])
    return train_set, test_set


def process_splice():
    df = pd.read_csv(SLICE_PATH)
    columns = df.columns
    for col in columns:
        unique_vals = list(set(df[col].tolist()))
        if col == 'Instance_name':
            continue
        if col == 'Class':
            print(unique_vals)
            col_vals = df[col].tolist()
            new_vals = [*map(lambda x: NEGATIVE if x == 'N' else POSITIVE, col_vals)]
            df[col] = pd.Series(new_vals)
        else:
            col_vals = df[col].tolist()
            for u in unique_vals:
                unique_val = u
                new_col = col + "_" + unique_val
                new_vals = [*map(lambda x: POSITIVE if x == u else NEGATIVE, col_vals)]
                df[new_col] = pd.Series(new_vals)
            df = df.drop(col, 1)
    df.to_csv(SLICE_BIMARY_PATH)

def get_splice_db():
    df = pd.read_csv(SLICE_BIMARY_PATH, index_col=0)
    all_Y = np.array(df['Class'], dtype=TYPE)
    all_X = np.array(df.drop(['Class', 'Instance_name'], 1), dtype=TYPE)
    number_of_samples = all_X.shape[0]
    all_samples_indexes = list(range(number_of_samples))
    np.random.shuffle(all_samples_indexes)
    all_Y = all_Y[all_samples_indexes]
    all_X = all_X[all_samples_indexes]
    train_set_size = int(number_of_samples * TRAIN_SET_SIZE)
    train_set = (all_X[:train_set_size], all_Y[:train_set_size])
    test_set = (all_X[train_set_size:], all_Y[train_set_size:])
    return train_set, test_set


def process_diabetes():
    df = pd.read_csv(DIABETES_PATH)
    columns = df.columns
    for col in columns:
        col_vals = df[col].tolist()
        if col == 'Age':
            ages = [16,30,50,70,91]
            for i in range(len(ages)-1):
                new_col = col + "_" + str(i)
                new_vals = [*map(lambda x: POSITIVE if (ages[i] <= x < ages[i + 1]) else NEGATIVE, col_vals)]
                df[new_col] = pd.Series(new_vals)
            df = df.drop(col, 1)
        else:
            unique_vals = list(set(df[col].tolist()))
            new_vals = [*map(lambda x: NEGATIVE if x == unique_vals[0] else POSITIVE, col_vals)]
            df[col] = pd.Series(new_vals)

    df.to_csv(DIABETES_BIMARY_PATH)

def get_diabetes():
    df = pd.read_csv(DIABETES_BIMARY_PATH, index_col=0)
    all_Y = np.array(df['class'], dtype=TYPE)
    all_X = np.array(df.drop('class', 1), dtype=TYPE)
    number_of_samples = all_X.shape[0]
    all_samples_indexes = list(range(number_of_samples))
    np.random.shuffle(all_samples_indexes)
    all_Y = all_Y[all_samples_indexes]
    all_X = all_X[all_samples_indexes]
    train_set_size = int(number_of_samples * TRAIN_SET_SIZE)
    train_set = (all_X[:train_set_size], all_Y[:train_set_size])
    test_set = (all_X[train_set_size:], all_Y[train_set_size:])
    return train_set, test_set

def process_pima_diabetes():
    df = pd.read_csv(PIMA_PATH)
    columns = df.columns
    for col in columns:
        col_vals = df[col].tolist()
        if col == 'class':
            unique_vals = list(set(df[col].tolist()))
            new_vals = [*map(lambda x: NEGATIVE if x == unique_vals[0] else POSITIVE, col_vals)]
            df[col] = pd.Series(new_vals)
        else:
            num_intervals = 5.0
            min_col = min(col_vals)
            max_col = max(col_vals)
            interval = (max_col + 1 - min_col)/num_intervals
            for i in range(int(num_intervals)):
                new_col = col + "_" + str(i)
                new_vals = [*map(lambda x: POSITIVE if (min_col + interval*i <= x < min_col + interval*(i+1)) else NEGATIVE, col_vals)]
                df[new_col] = pd.Series(new_vals)
            df = df.drop(col, 1)
    df.to_csv(PIMA_BIMARY_PATH)


def get_pima():
    df = pd.read_csv(PIMA_BIMARY_PATH, index_col=0)
    all_Y = np.array(df['class'], dtype=TYPE)
    all_X = np.array(df.drop('class', 1), dtype=TYPE)
    number_of_samples = all_X.shape[0]
    all_samples_indexes = list(range(number_of_samples))
    np.random.shuffle(all_samples_indexes)
    all_Y = all_Y[all_samples_indexes]
    all_X = all_X[all_samples_indexes]
    train_set_size = int(number_of_samples * TRAIN_SET_SIZE)
    train_set = (all_X[:train_set_size], all_Y[:train_set_size])
    test_set = (all_X[train_set_size:], all_Y[train_set_size:])
    return train_set, test_set


def process_balance():
    df = pd.read_csv(BALANCE_PATH)
    columns = df.columns
    for col in columns:
        unique_vals = list(set(df[col].tolist()))
        if col == 'Class':
            print(unique_vals)
            col_vals = df[col].tolist()
            new_vals = [*map(lambda x: NEGATIVE if x == 'R' else POSITIVE, col_vals)]
            df[col] = pd.Series(new_vals)
        else:
            col_vals = df[col].tolist()
            for u in unique_vals:
                unique_val = u
                new_col = col + "_" + str(unique_val)
                new_vals = [*map(lambda x: POSITIVE if x == u else NEGATIVE, col_vals)]
                df[new_col] = pd.Series(new_vals)
            df = df.drop(col, 1)
    df.to_csv(BALANCE_BIMARY_PATH)

def get_balance():
    df = pd.read_csv(BALANCE_BIMARY_PATH, index_col=0)
    all_Y = np.array(df['Class'], dtype=TYPE)
    all_X = np.array(df.drop(['Class'], 1), dtype=TYPE)
    number_of_samples = all_X.shape[0]
    all_samples_indexes = list(range(number_of_samples))
    np.random.shuffle(all_samples_indexes)
    all_Y = all_Y[all_samples_indexes]
    all_X = all_X[all_samples_indexes]
    train_set_size = int(number_of_samples * TRAIN_SET_SIZE)
    train_set = (all_X[:train_set_size], all_Y[:train_set_size])
    test_set = (all_X[train_set_size:], all_Y[train_set_size:])
    return train_set, test_set