import numpy as np

# Caminho para o arquivo
arquivo_npz = r"D:\Usuario\Downloads\normalization_stats.npz"

# Carregando o conteúdo do arquivo
dados = np.load(arquivo_npz)

# Verificando as chaves (nomes dos arrays salvos)
print("Chaves disponíveis:", dados.files)

# Acessando os arrays
for chave in dados.files:
    print(f"{chave}: {dados[chave]}")
