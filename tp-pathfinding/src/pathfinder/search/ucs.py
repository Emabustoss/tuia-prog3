from ..models.grid import Grid
from ..models.frontier import PriorityQueueFrontier
from ..models.solution import NoSolution, Solution
from ..models.node import Node

class UniformCostSearch:
    @staticmethod
    def search(grid: Grid) -> Solution:
        root = Node("", state=grid.initial, cost=0, parent=None, action=None)
        reached = {root.state: 0} 
        frontera = PriorityQueueFrontier()

        frontera.add(root, root.cost)

        while not frontera.is_empty():
            n = frontera.pop()

            if grid.objective_test(n.state):
                return Solution(n, reached)

            for a in grid.actions(n.state):
                s = grid.result(n.state, a)
                nuevo_costo = n.cost + grid.individual_cost(n.state, a)

                if s not in reached or nuevo_costo < reached[s]:
                    son = Node(
                        "",
                        state=s,
                        cost=nuevo_costo,
                        parent=n,
                        action=a,
                    )
                    reached[s] = nuevo_costo 
                    frontera.add(son, son.cost)

        return NoSolution(reached)