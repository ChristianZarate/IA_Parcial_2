# Salto Atrás Dirigido por Conflictos (Conflict-Directed Backjumping - CBJ)
# Mejora el backtracking al identificar el CONJUNTO DE CONFLICTOS de una variable:
# las variables pasadas que causaron el fallo. Salta directamente al conflicto más
# reciente, evitando retrocesos inútiles.

class CSP:
    def __init__(self, variables, dominios, vecinos, restriccion_fn):
        self.variables = variables
        self.dominios = {v: list(d) for v, d in dominios.items()}
        self.vecinos = vecinos
        self.restriccion = restriccion_fn
        self.saltos = 0

    def es_consistente(self, var, val, asignacion):
        for vecino in self.vecinos.get(var, []):
            if vecino in asignacion:
                if not self.restriccion(var, val, vecino, asignacion[vecino]):
                    return False
        return True

    def conflictos_con(self, var, val, asignacion):
        """Devuelve el conjunto de variables pasadas que entran en conflicto con val."""
        conf = set()
        for vecino in self.vecinos.get(var, []):
            if vecino in asignacion:
                if not self.restriccion(var, val, vecino, asignacion[vecino]):
                    conf.add(vecino)
        return conf


FALLO = "FALLO"

def cbj(csp, nivel=0, asignacion=None, conjunto_conflictos=None):
    if asignacion is None:
        asignacion = {}
    if conjunto_conflictos is None:
        conjunto_conflictos = {v: set() for v in csp.variables}

    if len(asignacion) == len(csp.variables):
        return asignacion, None

    var = csp.variables[nivel]
    conflicto_acumulado = set()

    for valor in csp.dominios[var]:
        conf = csp.conflictos_con(var, valor, asignacion)

        if not conf:  # valor consistente
            asignacion[var] = valor
            print(f"  Asignando {var}={valor} | Conflictos acum.: {conjunto_conflictos[var]}")

            resultado, conf_regresado = cbj(csp, nivel + 1, asignacion, conjunto_conflictos)

            if resultado != FALLO:
                return resultado, None

            # Si el conflicto regresado no involucra a var, saltar más arriba
            if var not in conf_regresado:
                conjunto_conflictos[var] |= conf_regresado
                del asignacion[var]
                csp.saltos += 1
                print(f"  -> Salto desde {var}: conflictos {conf_regresado}")
                return FALLO, conf_regresado

            conjunto_conflictos[var] |= (conf_regresado - {var})
            del asignacion[var]
        else:
            conflicto_acumulado |= conf

    conjunto_conflictos[var] |= conflicto_acumulado
    return FALLO, conjunto_conflictos[var]


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

    print("=== Salto Atrás Dirigido por Conflictos (CBJ) ===")
    print("Coloreo del mapa de Australia\n")

    resultado, _ = cbj(csp)

    if resultado != FALLO:
        print("\nSolución:")
        for v, c in resultado.items():
            print(f"  {v}: {c}")
        print(f"\nSaltos realizados (backjumps): {csp.saltos}")
    else:
        print("Sin solución.")