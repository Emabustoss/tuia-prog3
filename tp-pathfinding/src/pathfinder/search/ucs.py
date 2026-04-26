from ..models.grid import Grid
from ..models.frontier import PriorityQueueFrontier
from ..models.solution import NoSolution, Solution
from ..models.node import Node

class UniformCostSearch:
    @staticmethod
    def search(grid: Grid) -> Solution:
        """
        Encuentra el camino optimo utilizando UCS.
        
        Es un algoritmo no informado que utiliza una Cola de Prioridad. Expande 
        siempre el nodo con el menor costo acumulado g(n) desde el inicio. 
        Garantiza encontrar el camino mas corto (optimo) incluso en grafos donde 
        los movimientos tienen costos diferentes (casillas con peso).
        """
        root = Node("", state=grid.initial, cost=0, parent=None, action=None)
        # En UCS, el diccionario guarda el estado y su costo minimo conocido
        reached = {root.state: 0} 
        # Frontera ordenada por prioridad (menor costo acumulado g(n))
        frontera = PriorityQueueFrontier()
        # La prioridad inicial es el costo de la raíz (0)
        frontera.add(root, root.cost)

        while not frontera.is_empty():
            n = frontera.pop()
            # Test objetivo al extraer (necesario para garantizar el camino más barato)
            if grid.objective_test(n.state):
                return Solution(n, reached)
            # Expansión de vecinos
            for a in grid.actions(n.state):
                s = grid.result(n.state, a)
                # Calculamos cuanto costaria llegar a este vecino
                nuevo_costo = n.cost + grid.individual_cost(n.state, a)
                # Solo creamos y exploramos este nodo si:
                # Nunca habíamos llegado a este estado o si ya habíamos llegado, pero descubrimos una ruta mas barata
                if s not in reached or nuevo_costo < reached[s]:
                    son = Node(
                        "",
                        state=s,
                        cost=nuevo_costo,
                        parent=n,
                        action=a,
                    )
                    # Actualizamos el diccionario con el nuevo costo minimo
                    reached[s] = nuevo_costo 
                    # Lo agregamos a la frontera con su prioridad (el costo acumulado)
                    frontera.add(son, son.cost)

        return NoSolution(reached)