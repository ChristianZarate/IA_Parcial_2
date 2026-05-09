# Búsqueda en Profundidad (DFS - Depth-First Search)
# Explora lo más profundo posible antes de retroceder. Usa una pila (stack).

def dfs(grafo, inicio, objetivo):
    pila = [[inicio]]
    visitados = set()

    while pila:
        camino = pila.pop()
        nodo = camino[-1]

        if nodo in visitados:
            continue
        visitados.add(nodo)

        print(f"Visitando: {nodo}")

        if nodo == objetivo:
            return camino

        for vecino in reversed(grafo.get(nodo, [])):
            if vecino not in visitados:
                pila.append(camino + [vecino])

    return None


def dfs_recursivo(grafo, nodo, objetivo, visitados=None, camino=None):
    if visitados is None:
        visitados = set()
    if camino is None:
        camino = []

    visitados.add(nodo)
    camino = camino + [nodo]
    print(f"Visitando: {nodo}")

    if nodo == objetivo:
        return camino

    for vecino in grafo.get(nodo, []):
        if vecino not in visitados:
            resultado = dfs_recursivo(grafo, vecino, objetivo, visitados, camino)
            if resultado:
                return resultado

    return None


if __name__ == "__main__":
    grafo = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }

    inicio = 'A'
    objetivo = 'F'

    print(f"DFS iterativo desde '{inicio}' hasta '{objetivo}'")
    print("-" * 40)
    resultado = dfs(grafo, inicio, objetivo)
    if resultado:
        print(f"\nCamino encontrado: {' -> '.join(resultado)}")

    print(f"\nDFS recursivo desde '{inicio}' hasta '{objetivo}'")
    print("-" * 40)
    resultado = dfs_recursivo(grafo, inicio, objetivo)
    if resultado:
        print(f"\nCamino encontrado: {' -> '.join(resultado)}")