def count_pairs(tokens):
    """
    Contabiliza pares de tokens adjacentes.

    Args:
        tokens (list): Lista de tokens a serem contabilizados.
    Returns:
        counts (dict): Dicionário contendo pares adjacentes e quantidades.
    """
    counts = {}

    # Itera sobre os elementos consecutivos
    for pair in zip(tokens, tokens[1:]):
        # Realiza a contagem dos pares de elementos adjacentes
        counts[pair] = counts.get(pair, 0) + 1

    return counts


def merge_tokens(tokens, pair, new_token_value):
    """
    Função para fundir tokens adjacentes com maior quantia de ocorrências.

    Args:
        tokens (list): Lista de tokens a serem alterados.
        pair (tuple): Par de tokens adjacentes com maior ocorrência.
        new_token_value (int): Novo valor de token que representará a junção.
    Returns:
        new_tokens (list): Nova lista de tokens com junções realizadas.
    """

    new_tokens = []

    i = 0
    while i < len(tokens):
        # Verifica se não está no final da lista e se a adjacência é igual a fornecida
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            # Se é o par de maior adjacência, troca para o novo valor
            new_tokens.append(new_token_value)
            i += 2  # Avança para após a tupla analisada
        else:
            # Se não for igual a maior ocorrência, mantém o valor
            new_tokens.append(tokens[i])
            i += 1

    return new_tokens
