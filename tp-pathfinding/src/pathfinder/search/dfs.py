from ..models.grid import Grid
from ..models.frontier import StackFrontier
from ..models.solution import NoSolution, Solution
from ..models.node import Node


class DepthFirstSearch:
    @staticmethod
    def search(grid: Grid) -> Solution:
        root = Node("", state=grid.initial, cost=0, parent=None, action=None)

        frontera = StackFrontier()
        frontera.add(root)

        expanded = dict()

        if grid.objective_test(root.state):
            return Solution(root, expanded)

        while True:
            if frontera.is_empty():   # o sin () si en tu clase es atributo
                return NoSolution(expanded)

            n = frontera.remove()

            if n.state in expanded:
                continue

            expanded[n.state] = True

            for a in grid.actions(n.state):
                s = grid.result(n.state, a)

                if s not in expanded:
                    son = Node(
                        "",
                        state=s,
                        cost=n.cost + grid.individual_cost(n.state, a),
                        parent=n,
                        action=a,
                    )

                    if grid.objective_test(s):
                        return Solution(son, expanded)

                    frontera.add(son)


        return NoSolution(expanded)