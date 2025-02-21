import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import re
import pywt

# init model once
model = SentenceTransformer("Qwen/Qwen2.5-0.5B-Instruct")
parquet_file_path = './train-00000-of-00001.parquet'

def compute_top_frequencies_and_amplitudes(data, top_n=5):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    if data.ndim != 2 or data.shape[1] != 896:
        raise ValueError("input must in L,896 as 896 is Embedding Dimension for Qwen 0.5B")
    
    L, _ = data.shape
    
    frequencies = np.zeros((top_n, 896))
    amplitudes = np.zeros((top_n, 896))
    
    for col in range(896):
        fft_result = np.fft.fft(data[:, col])
        
        amplitudes_col = np.abs(fft_result)
        
        sampling_rate = 2000
        frequencies_col = np.fft.fftfreq(L, d=1/sampling_rate)
        nonzero_indices = np.where(frequencies_col > 0)[0]
        top_indices = np.argsort(amplitudes_col[nonzero_indices])[-top_n:][::-1]
        frequencies[:, col] = frequencies_col[nonzero_indices][top_indices]
        amplitudes[:, col] = amplitudes_col[nonzero_indices][top_indices]
    
    return frequencies, amplitudes

def compute_top_frequencies_and_amplitudes_cwt(data, top_n=5, cwt_period=500):
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float32)

    if data.ndim != 2 or data.shape[1] != 896:
        raise ValueError("input must in L,896 as 896 is Embedding Dimension for Qwen 0.5B")

    L, _ = data.shape

    frequencies = np.zeros((top_n, 896), dtype=np.float32)
    amplitudes = np.zeros((top_n, 896), dtype=np.float32)

    wavelet = 'cmor'
    scales = np.arange(1, cwt_period + 1) 

    # Apply cwt on each Embedding Dimension
    for col in range(896):
        column_data = data[:, col]

        # padding zero
        if len(column_data) < cwt_period:
            column_data = np.pad(column_data, (0, cwt_period - len(column_data)), mode='constant')

        # CWT
        coefficients, frequencies_col = pywt.cwt(column_data, scales, wavelet)

        # get amplitudes
        amplitudes_col = np.abs(coefficients)

        # get top amplitudes
        top_indices = np.argsort(amplitudes_col.max(axis=1))[-top_n:][::-1]
        frequencies[:, col] = frequencies_col[top_indices]
        amplitudes[:, col] = amplitudes_col[top_indices].max(axis=1)

    return frequencies, amplitudes

def plot_3d_frequencies_and_amplitudes(frequencies, amplitudes, ax, alpha):
    # just color as rainbow
    colors = ['red', 'yellow', 'blue', 'green', 'purple']
    
    for i in range(5):
        x = np.arange(896)  # x 896 Embedding Dimension
        y = frequencies[i, :]  # y frequence
        z = amplitudes[i, :]  # z amplitudes
        df = pd.DataFrame(z)
        # use describe to check if is order by amplitudes
        description = df.describe()
        print(i)
        print(description)
        ax.scatter(x, y, z, c=colors[i], alpha=alpha)
    
def embed_sentence(sentence):
    if isinstance(sentence, (list, np.ndarray)):
        sentence = sentence[0]  # for thinking_trajectories

    # remove space or new line as for 0.5B there is no embedding for space or new line
    cleaned_string = re.sub(r'\s+', '', sentence)

    embeddings_matrix = np.zeros((len(cleaned_string), 896), dtype=np.float32)
    for i, char in enumerate(cleaned_string):
        char_embedding = model.encode(char)
        embeddings_matrix[i, :] = char_embedding
    return embeddings_matrix

def process_batch(sentences, n=10, alpha=0.01, ax=None):
    for sentence in sentences:
        embedding = embed_sentence(sentence)
        print(f"Sentence: {sentence}")
        print(f"Embedding shape: {embedding.shape}")
        # or use compute_top_frequencies_and_amplitudes
        frequencies, amplitudes = compute_top_frequencies_and_amplitudes_cwt(embedding, top_n=n)
        plot_3d_frequencies_and_amplitudes(frequencies, amplitudes, ax, alpha)
    
def main(sentences, alpha, batch_size=100, n=5, filename='combined_spectrum_plot.png'):
    # init 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
        
    # num_batches = (len(sentences) + batch_size - 1) // batch_size
    num_batches = 1
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(sentences))
        batch = sentences[start:end]
        print(f"Processing batch {i + 1}/{num_batches} (sentences {start + 1}-{end})")
        process_batch(batch, n=n, alpha=alpha, ax=ax)
    
    ax.set_xlabel('Dimension (896 Columns)')
    ax.set_ylabel('Frequency')
    ax.set_zlabel('Amplitude')
    ax.set_title('3D Plot of Top 1 Frequencies and Amplitudes')

    ax.view_init(elev=0, azim=0)  
    plt.savefig(f"{filename}_x_axis.png", bbox_inches='tight')
        
    ax.view_init(elev=0, azim=90)
    plt.savefig(f"{filename}_y_axis.png", bbox_inches='tight')
        
    ax.view_init(elev=90, azim=0)
    plt.savefig(f"{filename}_z_axis.png", bbox_inches='tight')
    
    plt.close()
    print(f"Combined spectrum plot saved as {filename}")

#thinking_trajectories
#question
field='thinking_trajectories'
df = pd.read_parquet(parquet_file_path, columns=[field])

column_values = df[field].values

sentences = column_values.tolist()

#main(sentences, batch_size=1, n=5, filed='question', filename='new_Qbatch1n1size1.png')
#main(sentences, batch_size=10, n=5, filed='question', filename='new_Qbatch10n1size1.png')
#main(sentences, field='question', batch_size=1, n=5, filename='Q_CWT_1_5')
main(sentences, alpha=0.1, batch_size=1, n=5, filename='T_CWT_1_5_001')
#main(sentences, field='thinking_trajectories', batch_size=1, n=5, filename='T_CWT_1_5')
#main(sentences, field='thinking_trajectories', batch_size=10, n=5, filename='T_CWT_10_5')

#main(sentences, batch_size=10, n=5, filename='new_Tbatch10n1size1.png')