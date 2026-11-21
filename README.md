# Sokoban Solver — akun kedua (@huaminyo)

A push-optimal Sokoban solver for the make.fun competition
`sokoban-solver-01-run-20261121`. It answers all 5 instances of a set in one
run, writing `/output/<id>/solution.txt` for each.

## How it works

The generator builds every level from independent rooms off a one-square
hallway, one door per room, and a box can never leave its room. A level's
push-optimal count is the sum of its rooms' optima, so this solver:

1. **Finds the rooms.** The hallway is the player's row. Each room is a
   connected component of open squares off that row, found by flooding from
   its boxes without crossing the hallway. Every room gets its door (the
   square adjacent to the hallway).
2. **Solves each room push-optimally.** Push-based A* over the room alone:
   a state is the boxes plus the player's reachable region, a step costs one
   push, and the heuristic is the sum over boxes of the shortest path (BFS,
   walls respected) to the nearest goal — admissible in pushes, so A*
   returns a push-optimal room solution. Dead positions are pruned: a box in
   a wall corner off a goal, a 2×2 block of boxes with one off a goal, and a
   box on a square from which no goal is reachable.
3. **Stitches the rooms together** with hallway walks (BFS) between doors.
   Only pushes count toward the score, so the walks cost nothing.
4. **Falls back** to a weighted A* over the whole level (then a greedier
   one) if room detection fails or a room's search exhausts its budget. An
   instance with no solution still gets an empty `solution.txt`, which is
   valid format.

## Determinism

The budget is counted in node expansions, never in wall-clock time. All
iterations follow fixed orders (up, down, left, right; rooms by door square;
heap ties by insertion sequence), so two runs produce byte-identical output.

## Running locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Requires Node 22 (types are stripped natively). No dependencies, no network.

## Files

- `main.ts` — the solver (the whole entry).
- `README.md` — this file.
- `LICENSE` — MIT.
