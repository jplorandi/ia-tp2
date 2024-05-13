from dataclasses import dataclass, field
from collections import deque
import heapq  # Para la cola de prioridad
import math
from ia.graphing import generar_arbol


@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: object = field(compare=False)


@dataclass(eq=True, frozen=True)
class EstadoDiscreto:
    x: int = 0
    y: int = 0

    def __repr__(self):
        return f"{self.x},{self.y}"


@dataclass
class Accion:
    tipo: str
    condicion: callable
    ejecutar: callable
    costo: callable

    def __repr__(self):
        return self.tipo


def generar_acciones(MIN_X, MAX_X, MIN_Y, MAX_Y, delta_j=1, delta_k=1):
    return [
        Accion("MoverIzquierda", lambda s: s.x > MIN_X, lambda s: EstadoDiscreto(s.x - delta_j, s.y),
               lambda s: 1.0),
        Accion("MoverDerecha", lambda s: s.x < MAX_X, lambda s: EstadoDiscreto(s.x + delta_j, s.y),
               lambda s: 1.0),
        Accion("MoverArriba", lambda s: s.y > MIN_Y, lambda s: EstadoDiscreto(s.x, s.y - delta_k),
               lambda s: 1.0),
        Accion("MoverAbajo", lambda s: s.y < MAX_Y, lambda s: EstadoDiscreto(s.x, s.y + delta_k),
               lambda s: 1.0),
        Accion("MoverArribaIzquierda", lambda s: MAX_X > s.x > MIN_X and MAX_Y > s.y > MIN_Y,
               lambda s: EstadoDiscreto(s.x - delta_j, s.y + delta_k), lambda s: 0.7),
        Accion("MoverArribaDerecha", lambda s: MAX_X > s.x > MIN_X and MAX_Y > s.y > MIN_Y,
               lambda s: EstadoDiscreto(s.x + delta_j, s.y + delta_k), lambda s: 0.7),
        Accion("MoverAbajoIzquierda", lambda s: MAX_X > s.x > MIN_X and MAX_Y > s.y > MIN_Y,
               lambda s: EstadoDiscreto(s.x - delta_j, s.y - delta_k), lambda s: 0.7),
        Accion("MoverAbajoDerecha", lambda s: MAX_X > s.x > MIN_X and MAX_Y > s.y > MIN_Y,
               lambda s: EstadoDiscreto(s.x + delta_j, s.y - delta_k), lambda s: 0.7),
        Accion("Mantener", lambda s: MAX_X > s.x > MIN_X and MAX_Y > s.y > MIN_Y,
               lambda s: s, lambda s: 0.0),
    ]


def dfs_solucion(estado_inicial, objetivo, acciones):
    frontera = [estado_inicial]
    visitados = set([estado_inicial])  # Marcar el estado inicial como visitado inmediatamente
    camino = {estado_inicial: None}
    soluciones = []

    while frontera:
        estado_actual = frontera.pop()

        if estado_actual == objetivo:
            return reconstruir_camino(camino, estado_actual), visitados

        for accion in acciones:
            if accion.condicion(estado_actual):
                nuevo_estado = accion.ejecutar(estado_actual)
                if nuevo_estado not in visitados:
                    frontera.append(nuevo_estado)
                    camino[nuevo_estado] = (estado_actual, accion.tipo)
                    visitados.add(nuevo_estado)  # Marcar como visitado cuando se agrega a la frontera

    return soluciones


def bfs_solucion(estado_inicial, objetivo, acciones):
    # Implementación de búsqueda en amplitud
    # Frontera es una cola con los estados a explorar hacia adelante
    frontera = deque([estado_inicial])
    # Visitados es un conjunto con los estados que ya se han explorado, para no repetirlos
    visitados = set([estado_inicial])
    # Camino es un diccionario que guarda el estado anterior y la acción que llevó a un estado
    # para poder reconstruir el camino después. Camino almacena el estado actual como clave y
    # una tupla con el estado anterior y la acción como valor. El estado inicial no tiene estado anterior.
    # Esta solucion ignora el costo de las acciones y tambien la posibilidad de que multiples acciones y estados anteriores
    # lleven al mismo estado. Finalmente, no se exploraran todos los estados en el espacio, sino aquellos que sean
    # alcanzables desde el estado inicial.
    camino = {estado_inicial: None}
    count = 0
    # Mientras haya estados por explorar
    while frontera:
        estado_actual = frontera.popleft()

        if estado_actual == objetivo:  # Valvula de escape -- Objetivo encontrado
            generar_arbol(camino, "BFS")
            return reconstruir_camino(camino, estado_actual), visitados

        # visitados.add(estado_actual)

        for accion in acciones:
            if accion.condicion(estado_actual):
                nuevo_estado = accion.ejecutar(estado_actual)
                if nuevo_estado not in visitados:
                    frontera.append(nuevo_estado)
                    camino[nuevo_estado] = (estado_actual, accion.tipo)
                    visitados.add(nuevo_estado)

    return None


def heuristica(estado, objetivo, tipo: str = 'manhattan'):
    if tipo == 'manhattan':
        # Distancia Manhattan: suma de las diferencias absolutas de las coordenadas
        return abs(estado.x - objetivo.x) + abs(estado.y - objetivo.y)
    elif tipo == 'euclidiana':
        # Distancia Euclidiana: raíz cuadrada de la suma de los cuadrados de las diferencias de las coordenadas
        return math.sqrt(
            (estado.x - objetivo.x) ** 2 + (estado.y - objetivo.y) ** 2)
    else:
        raise ValueError("Tipo de heurística no soportado: debe ser 'manhattan' o 'euclidiana'")


def a_estrella_solucion(estado_inicial, objetivo, acciones, distancia='euclidiana'):
    frontera = []
    heapq.heappush(frontera, PrioritizedItem(0, estado_inicial))
    visitados = set()
    camino = {estado_inicial: None}
    costos = {estado_inicial: 0}

    while frontera:
        actual = heapq.heappop(frontera).item

        if actual == objetivo:
            generar_arbol(camino, f"A* {distancia}")
            return reconstruir_camino(camino, actual), visitados

        visitados.add(actual)

        for accion in acciones:
            if accion.condicion(actual):
                nuevo_estado = accion.ejecutar(actual)
                nuevo_costo = costos[actual] + accion.costo(actual)

                if nuevo_estado not in visitados or nuevo_costo < costos.get(nuevo_estado, float('inf')):
                    costos[nuevo_estado] = nuevo_costo
                    prioridad = nuevo_costo + heuristica(nuevo_estado, objetivo, distancia)
                    heapq.heappush(frontera, PrioritizedItem(prioridad, nuevo_estado))
                    camino[nuevo_estado] = (actual, accion.tipo)
                    visitados.add(nuevo_estado)

    return None


def best_first_solucion(estado_inicial, objetivo, acciones, distancia='manhattan'):
    frontera = []
    heapq.heappush(frontera, PrioritizedItem(0, estado_inicial))
    visitados = set()
    camino = {estado_inicial: None}

    while frontera:
        actual = heapq.heappop(frontera).item

        if actual == objetivo:
            generar_arbol(camino, f"Best first {distancia}")
            return reconstruir_camino(camino, actual), visitados

        visitados.add(actual)

        for accion in acciones:
            if accion.condicion(actual):
                nuevo_estado = accion.ejecutar(actual)

                if nuevo_estado not in visitados:
                    heapq.heappush(frontera,
                                   PrioritizedItem(heuristica(nuevo_estado, objetivo, distancia), nuevo_estado))
                    camino[nuevo_estado] = (actual, accion.tipo)
                    visitados.add(nuevo_estado)

    return None


def reconstruir_camino(camino, estado_final):
    acciones = []
    paso = camino[estado_final]
    estado_previo = estado_final
    while paso:
        estado, accion = paso
        acciones.append(accion + "-" + estado_previo.__str__())
        paso = camino[estado]
        estado_previo = estado
    acciones.reverse()
    return acciones


def informar(tipo, solucion, visitados):
    if solucion is not None:
        dispersion = len(solucion) / len(visitados)
        # print(
        #     f"Solución encontrada [{tipo}] de longitud {len(solucion)} nodos generados: "
        #     f"{len(visitados)} P: {dispersion:.4f} : {solucion}")
        print(
            f"Solución encontrada [{tipo}] de longitud {len(solucion)} nodos generados: "
            f"{len(visitados)} P: {dispersion:.4f}")
    else:
        print(f"No se encontró solución [{tipo}]")


if __name__ == '__main__':
    # Ejemplo de uso
    # estado_inicial = EstadoDiscreto(-13, 4)
    estado_inicial = EstadoDiscreto(-7, 4)
    objetivo = EstadoDiscreto(0, 0)
    acciones = generar_acciones(-20, 20, -20, 20, 1, 1)
    print(f"Acciones generadas: {acciones}")
    solucion, visitados = bfs_solucion(estado_inicial, objetivo, acciones)
    informar("BFS", solucion, visitados)
    solucion, visitados = dfs_solucion(estado_inicial, objetivo, acciones)
    informar("DFS", solucion, visitados)
    solucion, visitados = a_estrella_solucion(estado_inicial, objetivo, acciones, 'euclidiana')
    informar("A* Euclideana", solucion, visitados)
    solucion, visitados = a_estrella_solucion(estado_inicial, objetivo, acciones, 'manhattan')
    informar("A* Manhattan", solucion, visitados)
    solucion, visitados = best_first_solucion(estado_inicial, objetivo, acciones, 'euclidiana')
    informar("Best First Euclideana", solucion, visitados)
    solucion, visitados = best_first_solucion(estado_inicial, objetivo, acciones, 'manhattan')
    informar("Best First Manhattan", solucion, visitados)
