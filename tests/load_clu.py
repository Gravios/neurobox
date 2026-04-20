import numpy as np

def load_clu(file_name):
    try:
        with open(file_name, 'r') as fp:
            # Read the first line to get the number of clusters
            n_clusters = int(fp.readline().strip())

            # Read the rest of the file to get the Clu array
            clu = np.array([int(x) for x in fp.read().split()])

            # Update n_clusters to be the maximum value in clu
            n_clusters = np.max(clu)

            return clu, n_clusters

    except FileNotFoundError:
        raise FileNotFoundError(f"Could not open file {file_name}")

