"""
32 - Teoría de Juegos: Equilibrios y Mecanismos
================================================
La Teoría de Juegos estudia la toma de decisiones estratégica entre
agentes cuyas recompensas dependen de las acciones de los demás.

Conceptos implementados:
  1. Juego en forma normal (Matriz de pagos)
  2. Estrategias dominantes
  3. Equilibrio de Nash (Puro y Mixto)
  4. Equilibrio de Nash en juegos de suma cero (Minimax)
  5. Juegos iterados: dilema del prisionero con estrategias
  6. Mecanismo de subasta de precio sellado
"""

import random
import itertools


# ─────────────────── Juego en Forma Normal ───────────────────────────

class JuegoNormal:
    """
    Juego de 2 jugadores en forma normal.
    acciones_j1  : lista de acciones del jugador 1 (filas)
    acciones_j2  : lista de acciones del jugador 2 (columnas)
    pagos        : dict {(a1,a2): (u1,u2)}
    """
    def __init__(self, acciones_j1, acciones_j2, pagos):
        self.A1    = acciones_j1
        self.A2    = acciones_j2
        self.pagos = pagos

    def u(self, a1, a2):
        return self.pagos[(a1, a2)]

    def imprimir_matriz(self, titulo="Matriz de Pagos"):
        col_w = 18
        print(f"\n  {titulo}")
        header = " " * 14 + "".join(f"{str(a):>{col_w}}" for a in self.A2)
        print(f"  {header}")
        for a1 in self.A1:
            fila = f"  {str(a1):12s}"
            for a2 in self.A2:
                u1, u2 = self.u(a1, a2)
                fila += f"  ({u1:4.1f},{u2:4.1f})   "
            print(fila)

    # ── Estrategias Dominantes ──

    def estrategia_dominante_j1(self):
        """Retorna la estrategia estrictamente dominante del J1, si existe."""
        for a1 in self.A1:
            if all(self.u(a1, a2)[0] > self.u(b1, a2)[0]
                   for b1 in self.A1 if b1 != a1
                   for a2 in self.A2):
                return a1
        return None

    def estrategia_dominante_j2(self):
        for a2 in self.A2:
            if all(self.u(a1, a2)[1] > self.u(a1, b2)[1]
                   for b2 in self.A2 if b2 != a2
                   for a1 in self.A1):
                return a2
        return None

    # ── Equilibrio de Nash Puro ──

    def equilibrios_nash_puros(self):
        """Devuelve todos los perfiles de acción que son EN puros."""
        equilibrios = []
        for a1 in self.A1:
            for a2 in self.A2:
                u1, u2 = self.u(a1, a2)
                # J1 no quiere desviarse
                es_mejor_j1 = all(u1 >= self.u(b1, a2)[0] for b1 in self.A1)
                # J2 no quiere desviarse
                es_mejor_j2 = all(u2 >= self.u(a1, b2)[1] for b2 in self.A2)
                if es_mejor_j1 and es_mejor_j2:
                    equilibrios.append((a1, a2, u1, u2))
        return equilibrios

    # ── Equilibrio de Nash Mixto (2×2) ──

    def nash_mixto_2x2(self):
        """
        Para juegos 2×2 calcula la estrategia mixta de Nash.
        J1 mezcla con probabilidad p; J2 con probabilidad q.
        Se resuelve haciendo indiferentes a los jugadores.
        """
        if len(self.A1) != 2 or len(self.A2) != 2:
            return None
        a1, b1 = self.A1
        a2, b2 = self.A2
        # J1 indifferente entre a2,b2 cuando J2 mezcla con q:
        # U1(a1,a2)*q + U1(a1,b2)*(1-q) = U1(b1,a2)*q + U1(b1,b2)*(1-q)
        u1_a1a2 = self.u(a1,a2)[0]; u1_a1b2 = self.u(a1,b2)[0]
        u1_b1a2 = self.u(b1,a2)[0]; u1_b1b2 = self.u(b1,b2)[0]
        denom_q = (u1_a1a2 - u1_b1a2) - (u1_a1b2 - u1_b1b2)
        if abs(denom_q) < 1e-9:
            q = None
        else:
            q = (u1_b1b2 - u1_a1b2) / denom_q

        # J2 indiferente:
        u2_a1a2 = self.u(a1,a2)[1]; u2_a1b2 = self.u(a1,b2)[1]
        u2_b1a2 = self.u(b1,a2)[1]; u2_b1b2 = self.u(b1,b2)[1]
        denom_p = (u2_a1a2 - u2_a1b2) - (u2_b1a2 - u2_b1b2)
        if abs(denom_p) < 1e-9:
            p = None
        else:
            p = (u2_b1b2 - u2_a1b2) / denom_p

        return p, q   # p=P(J1 elige a1), q=P(J2 elige a2)


# ─────────────────── Equilibrio Minimax (suma cero) ──────────────────

def minimax_suma_cero(juego: JuegoNormal):
    """
    Para juegos de suma cero: J1 max, J2 min.
    Resuelve por enumeración para juegos pequeños.
    Retorna (valor_juego, acción_maxmin_J1, acción_minmax_J2).
    """
    # Estrategia maxmin de J1: max_a1 min_a2 u1(a1,a2)
    maxmin_val = float('-inf')
    maxmin_a1  = None
    for a1 in juego.A1:
        val = min(juego.u(a1, a2)[0] for a2 in juego.A2)
        if val > maxmin_val:
            maxmin_val, maxmin_a1 = val, a1

    # Estrategia minmax de J2: min_a2 max_a1 u1(a1,a2)
    minmax_val = float('inf')
    minmax_a2  = None
    for a2 in juego.A2:
        val = max(juego.u(a1, a2)[0] for a1 in juego.A1)
        if val < minmax_val:
            minmax_val, minmax_a2 = val, a2

    return maxmin_val, maxmin_a1, minmax_a2


# ─────────────────── Dilema del Prisionero iterado ───────────────────

def dilema_prisionero_iterado(n_rondas=10, estrategia1='tit_for_tat',
                               estrategia2='traidor'):
    """Simula el dilema del prisionero repetido."""
    hist1, hist2 = [], []
    puntos1, puntos2 = 0, 0

    # Pagos: (Cooperar, Cooperar)=(3,3); (Traicionar, Cooperar)=(5,0)
    #        (Cooperar, Traicionar)=(0,5); (T,T)=(1,1)
    pagos = {('C','C'):(3,3),('C','T'):(0,5),('T','C'):(5,0),('T','T'):(1,1)}

    def accion(estrategia, mi_hist, rival_hist):
        if estrategia == 'cooperador':
            return 'C'
        if estrategia == 'traidor':
            return 'T'
        if estrategia == 'tit_for_tat':
            return rival_hist[-1] if rival_hist else 'C'
        if estrategia == 'aleatorio':
            return random.choice(['C','T'])
        return 'C'

    for _ in range(n_rondas):
        a1 = accion(estrategia1, hist1, hist2)
        a2 = accion(estrategia2, hist2, hist1)
        p1, p2 = pagos[(a1, a2)]
        puntos1 += p1; puntos2 += p2
        hist1.append(a1); hist2.append(a2)

    return hist1, hist2, puntos1, puntos2


# ─────────────────── Subasta de Precio Sellado ───────────────────────

def subasta_primer_precio(valoraciones):
    """
    Subasta de primer precio: el que oferta más gana y paga su oferta.
    Estrategia Bayesiana de Nash: pujar n-1/n · valoración.
    """
    n = len(valoraciones)
    ofertas = {i: (n-1)/n * v for i, v in enumerate(valoraciones)}
    ganador = max(ofertas, key=ofertas.get)
    return ganador, ofertas[ganador], valoraciones[ganador]


def subasta_segundo_precio(valoraciones):
    """
    Subasta de Vickrey (segundo precio): ganador paga la segunda oferta más alta.
    Estrategia dominante: pujar la valoración verdadera.
    """
    ganador = max(range(len(valoraciones)), key=lambda i: valoraciones[i])
    segundo = sorted(valoraciones, reverse=True)[1]
    excedente = valoraciones[ganador] - segundo
    return ganador, segundo, excedente


# ─────────────────────────── demo ────────────────────────────────────

def demo():
    print("=" * 65)
    print("  Teoría de Juegos: Equilibrios y Mecanismos")
    print("=" * 65)

    # ── Dilema del Prisionero ──
    dp = JuegoNormal(
        ['Cooperar','Traicionar'],
        ['Cooperar','Traicionar'],
        {('Cooperar','Cooperar'):(3,3), ('Cooperar','Traicionar'):(0,5),
         ('Traicionar','Cooperar'):(5,0), ('Traicionar','Traicionar'):(1,1)}
    )
    dp.imprimir_matriz("Dilema del Prisionero")
    print(f"\n  Estrategia dominante J1: {dp.estrategia_dominante_j1()}")
    print(f"  Estrategia dominante J2: {dp.estrategia_dominante_j2()}")
    en = dp.equilibrios_nash_puros()
    print(f"  Equilibrios de Nash puros: {[(a1,a2) for a1,a2,_,_ in en]}")

    # ── Batalla de los Sexos ──
    bs = JuegoNormal(
        ['Fútbol','Ópera'], ['Fútbol','Ópera'],
        {('Fútbol','Fútbol'):(2,1), ('Fútbol','Ópera'):(0,0),
         ('Ópera','Fútbol'):(0,0), ('Ópera','Ópera'):(1,2)}
    )
    bs.imprimir_matriz("Batalla de los Sexos")
    en2 = bs.equilibrios_nash_puros()
    print(f"\n  Equilibrios de Nash puros: {[(a1,a2) for a1,a2,_,_ in en2]}")
    p, q = bs.nash_mixto_2x2()
    if p is not None and 0 <= p <= 1 and 0 <= q <= 1:
        print(f"  EN Mixto: J1 juega Fútbol con p={p:.3f}, "
              f"J2 juega Fútbol con q={q:.3f}")

    # ── Piedra-Papel-Tijeras (suma cero) ──
    ppt = JuegoNormal(
        ['P','Pa','T'], ['P','Pa','T'],
        {('P','P'):(0,0),('P','Pa'):(-1,1),('P','T'):(1,-1),
         ('Pa','P'):(1,-1),('Pa','Pa'):(0,0),('Pa','T'):(-1,1),
         ('T','P'):(-1,1),('T','Pa'):(1,-1),('T','T'):(0,0)}
    )
    ppt.imprimir_matriz("Piedra-Papel-Tijeras (suma cero)")
    v, a1_mm, a2_mm = minimax_suma_cero(ppt)
    print(f"\n  Valor minimax: {v}  |  maxmin J1: {a1_mm}  |  minmax J2: {a2_mm}")

    # ── Dilema iterado ──
    random.seed(1)
    print("\n  Dilema Iterado (10 rondas): Tit-for-Tat vs Traidor")
    h1, h2, p1, p2 = dilema_prisionero_iterado(10, 'tit_for_tat','traidor')
    print(f"  J1 (TfT)  : {' '.join(h1)}  Total={p1}")
    print(f"  J2 (Traid): {' '.join(h2)}  Total={p2}")

    # ── Subastas ──
    vals = [80, 65, 90, 72]
    print(f"\n  Subasta — Valoraciones: {vals}")
    g1, oferta, val = subasta_primer_precio(vals)
    print(f"  Primer precio  → Ganador={g1}  Paga={oferta:.1f}  "
          f"Valoración={val}  Excedente={val-oferta:.1f}")
    g2, pago, exc = subasta_segundo_precio(vals)
    print(f"  Segundo precio → Ganador={g2}  Paga={pago:.1f}  "
          f"Valoración={vals[g2]}  Excedente={exc:.1f}")
    print()


if __name__ == "__main__":
    demo()

