# Búsqueda en Anchura (BFS - Breadth-First Search)
# Explora nodo por nodo nivel a nivel, garantiza encontrar el camino más corto (en grafos sin pesos)

from collections import deque

def bfs(grafo, inicio, objetivo):
    cola = deque([[inicio]])
    visitados = set([inicio])

    while cola:
        camino = cola.popleft()
        nodo = camino[-1]

        print(f"Visitando: {nodo}")

        if nodo == objetivo:
            return camino

        for vecino in grafo.get(nodo, []):
            if vecino not in visitados:
                visitados.add(vecino)
                cola.append(camino + [vecino])

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

    print(f"BFS desde '{inicio}' hasta '{objetivo}'")
    print("-" * 35)
    resultado = bfs(grafo, inicio, objetivo)

    if resultado:
        print(f"\nCamino encontrado: {' -> '.join(resultado)}")
        print(f"Longitud del camino: {len(resultado) - 1} pasos")
    else:
        print("No se encontró camino.")