# Propagación de Restricciones - Consistencia de Arco (AC-3)
# AC-3 (Arc Consistency Algorithm 3): garantiza que para cada valor de una variable
# exista al menos un valor compatible en cada variable vecina.
# Reduce dominios antes o durante la búsqueda.

from collections import deque

class CSP:
    def __init__(self, variables, dominios, vecinos, restriccion_fn):
        self.variables = variables
        self.dominios = {v: list(d) for v, d in dominios.items()}
        self.vecinos = vecinos
        self.restriccion = restriccion_fn

    def es_consistente(self, xi, x, xj, y):
        return self.restriccion(xi, x, xj, y)


def ac3(csp):
    """
    Algoritmo AC-3.
    Elimina valores de dominios que no tienen soporte en los vecinos.
    Retorna False si algún dominio queda vacío (inconsistencia detectada).
    """
    cola = deque()
    for xi in csp.variables:
        for xj in csp.vecinos.get(xi, []):
            cola.append((xi, xj))

    revisiones = 0
    print(f"Arcos iniciales en cola: {len(cola)}\n")

    while cola:
        xi, xj = cola.popleft()
        revisiones += 1

        if revisar(csp, xi, xj):
            print(f"Dominio de {xi} reducido a: {csp.dominios[xi]}")

            if not csp.dominios[xi]:
                print(f"-> Dominio de {xi} quedó vacío. CSP inconsistente.")
                return False

            # Propagar: agregar arcos de vecinos de xi (excepto xj)
            for xk in csp.vecinos.get(xi, []):
                if xk != xj:
                    cola.append((xk, xi))

    print(f"\nAC-3 completado. Revisiones de arco: {revisiones}")
    return True


def revisar(csp, xi, xj):
    """Elimina de dom(xi) los valores sin soporte en dom(xj)."""
    revisado = False
    for x in csp.dominios[xi][:]:
        # x tiene soporte si existe algún y en dom(xj) compatible
        if not any(csp.es_consistente(xi, x, xj, y) for y in csp.dominios[xj]):
            csp.dominios[xi].remove(x)
            revisado = True
    return revisado


def backtracking_ac3(csp, asignacion=None):
    """Backtracking con AC-3 antes de cada asignación."""
    if asignacion is None:
        asignacion = {}
        print("Ejecutando AC-3 como preprocesamiento...")
        if not ac3(csp):
            return None
        print("\nDominios después de AC-3:")
        for v, d in csp.dominios.items():
            print(f"  {v}: {d}")
        print()

    if len(asignacion) == len(csp.variables):
        return asignacion

    var = min((v for v in csp.variables if v not in asignacion),
              key=lambda v: len(csp.dominios[v]))

    for valor in list(csp.dominios[var]):
        dominios_guardados = {v: list(d) for v, d in csp.dominios.items()}
        asignacion[var] = valor
        csp.dominios[var] = [valor]

        if ac3(csp):
            resultado = backtracking_ac3(csp, asignacion)
            if resultado is not None:
                return resultado

        csp.dominios = dominios_guardados
        del asignacion[var]

    return None


if __name__ == "__main__":
    variables = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']
    colores = ['R', 'G', 'B']
    dominios = {v: colores[:] for v in variables}
    vecinos = {
        'WA': ['NT', 'SA'],
        'NT': ['WA', 'SA', 'Q'],
        'SA': ['WA', 'NT', 'Q', 'NSW', 'V'],
        'Q':  ['NT', 'SA', 'NSW'],
        'NSW': ['Q', 'SA', 'V'],
        'V':  ['SA', 'NSW'],
        'T':  []
    }

    def diferente(v1, c1, v2, c2):
        return c1 != c2

    csp = CSP(variables, dominios, vecinos, diferente)

    print("=== Propagación de Restricciones - AC-3 ===")
    print("Coloreo del mapa de Australia\n")

    solucion = backtracking_ac3(csp)
    if solucion:
        print("Solución final:")
        for v, c in solucion.items():
            print(f"  {v}: {c}")
    else:
        print("Sin solución.")