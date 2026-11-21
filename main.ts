// Sokoban solver for sokoban-solver-01-run-20261121 (akun kedua @huaminyo).
//
// Idea: the generator builds each level from independent rooms off a hallway,
// one door per room, and a box can never leave its room. A level's push-optimal
// count is the sum of its rooms' optima, so this solver finds the rooms, solves
// each one push-optimally on its own, and stitches the room solutions together
// with hallway walks. Solving four 4-box rooms is far cheaper than searching
// one 16-box level, and it is exactly optimal when every room is.
//
// Per room the search is a push-based A*: a state is the boxes plus the player's
// reachable region, a step costs one push, and the heuristic is the sum over
// boxes of the shortest path (BFS, walls respected, other boxes ignored) to the
// nearest goal -- admissible in pushes, so A* returns a push-optimal room
// solution. Dead positions are pruned: a box in a wall corner off a goal, a
// 2x2 block of boxes with one off a goal, and a box on a square from which no
// goal is reachable at all. The budget is counted in node expansions, never in
// wall-clock time, so two runs are byte-identical.
//
// If room detection fails, or a room's search exhausts its budget, the solver
// falls back to a weighted A* over the whole level, then to a greedier one.
// An instance that still has no solution gets an empty solution.txt (valid
// format, zero points) so the entry still passes.
//
// Local run: mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Work budgets, counted in node expansions (deterministic: never the clock).
const ROOM_EXPANSIONS = 120_000;
const WHOLE_EXPANSIONS = 250_000;
const GREEDY_EXPANSIONS = 150_000;

// Directions in fixed order: up, down, left, right. Every iteration in this
// program follows this order, so every tie breaks the same way every run.
const DIRS: { dx: number; dy: number; ch: string }[] = [
  { dx: 0, dy: -1, ch: 'u' },
  { dx: 0, dy: 1, ch: 'd' },
  { dx: -1, dy: 0, ch: 'l' },
  { dx: 1, dy: 0, ch: 'r' },
];

type Grid = {
  w: number;
  h: number;
  wall: Uint8Array; // 1 where a wall (or outside the level) stands
  goal: Uint8Array; // 1 where a goal square is
  boxes: number[]; // sorted box squares
  player: number; // player square
};

function parseLevel(text: string): Grid {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const w = Math.max(...rows.map((r) => r.length));
  const h = rows.length;
  const wall = new Uint8Array(w * h).fill(1);
  const goal = new Uint8Array(w * h);
  const boxes: number[] = [];
  let player = 0;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goal[i] = 1;
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goal, boxes, player };
}

const idx = (g: Grid, x: number, y: number): number => y * g.w + x;
const inBounds = (g: Grid, x: number, y: number): boolean => x >= 0 && x < g.w && y >= 0 && y < g.h;

// Breadth-first distances from one start square over the open squares of a
// region (a square is open when region[i] is 1), ignoring boxes. Unreachable
// squares keep Infinity.
function bfsDistances(g: Grid, start: number, region: Uint8Array): number[] {
  const dist: number[] = new Array(g.w * g.h).fill(Infinity);
  const queue: number[] = [start];
  dist[start] = 0;
  for (let head = 0; head < queue.length; head++) {
    const cur = queue[head]!;
    const cx = cur % g.w;
    const cy = Math.floor(cur / g.w);
    for (const d of DIRS) {
      const nx = cx + d.dx;
      const ny = cy + d.dy;
      if (!inBounds(g, nx, ny)) continue;
      const ni = idx(g, nx, ny);
      if (!region[ni] || dist[ni] !== Infinity) continue;
      dist[ni] = dist[cur]! + 1;
      queue.push(ni);
    }
  }
  return dist;
}

// The player's walk from `from` to `to` over open squares, avoiding the boxes
// in `boxSet`, as a string of u/d/l/r. Deterministic: BFS in fixed direction
// order, first path found. Returns null when `to` is unreachable.
function walkPath(g: Grid, from: number, to: number, boxSet: Set<number>): string | null {
  if (from === to) return '';
  const prev: number[] = new Array(g.w * g.h).fill(-1);
  const prevCh: string[] = new Array(g.w * g.h).fill('');
  const queue: number[] = [from];
  prev[from] = from;
  for (let head = 0; head < queue.length; head++) {
    const cur = queue[head]!;
    if (cur === to) break;
    const cx = cur % g.w;
    const cy = Math.floor(cur / g.w);
    for (const d of DIRS) {
      const nx = cx + d.dx;
      const ny = cy + d.dy;
      if (!inBounds(g, nx, ny)) continue;
      const ni = idx(g, nx, ny);
      if (g.wall[ni] || boxSet.has(ni) || prev[ni] !== -1) continue;
      prev[ni] = cur;
      prevCh[ni] = d.ch;
      queue.push(ni);
    }
  }
  if (prev[to] === -1) return null;
  let path = '';
  for (let cur = to; cur !== from; cur = prev[cur]!) path = prevCh[cur] + path;
  return path;
}

// The set of squares the player can reach from `from` over open squares,
// avoiding the boxes in `boxSet`. Returned as a Uint8Array mask.
function reachableMask(g: Grid, from: number, boxSet: Set<number>): Uint8Array {
  const seen = new Uint8Array(g.w * g.h);
  const queue: number[] = [from];
  seen[from] = 1;
  for (let head = 0; head < queue.length; head++) {
    const cur = queue[head]!;
    const cx = cur % g.w;
    const cy = Math.floor(cur / g.w);
    for (const d of DIRS) {
      const nx = cx + d.dx;
      const ny = cy + d.dy;
      if (!inBounds(g, nx, ny)) continue;
      const ni = idx(g, nx, ny);
      if (g.wall[ni] || boxSet.has(ni) || seen[ni]) continue;
      seen[ni] = 1;
      queue.push(ni);
    }
  }
  return seen;
}

// A binary heap of [priority, tiebreak, sequence, state], smallest first.
// The sequence number makes every pop deterministic.
class Heap {
  private a: [number, number, number, number][] = [];
  get size(): number {
    return this.a.length;
  }
  push(e: [number, number, number, number]): void {
    const a = this.a;
    let i = a.length;
    a.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!Heap.less(a[i]!, a[p]!)) break;
      [a[i], a[p]] = [a[p]!, a[i]!];
      i = p;
    }
  }
  pop(): [number, number, number, number] {
    const a = this.a;
    const top = a[0]!;
    const last = a.pop()!;
    if (a.length > 0) {
      a[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= a.length) break;
        if (c + 1 < a.length && Heap.less(a[c + 1]!, a[c]!)) c++;
        if (!Heap.less(a[c]!, a[i]!)) break;
        [a[i], a[c]] = [a[c]!, a[i]!];
        i = c;
      }
    }
    return top;
  }
  private static less(x: [number, number, number, number], y: [number, number, number, number]): boolean {
    return x[0] < y[0] || (x[0] === y[0] && (x[1] < y[1] || (x[1] === y[1] && x[2] < y[2])));
  }
}

type SearchSpace = {
  grid: Grid;
  region: Uint8Array; // squares the boxes and the player may occupy
  goals: number[]; // goal squares inside the region
  distToGoal: number[][]; // distToGoal[goal][square]: BFS pushes lower bound
  minDist: number[]; // minDist[square]: nearest-goal distance (Infinity = dead)
};

// Precompute, for one search space, the BFS distance from every goal to every
// square of the region (walls respected, boxes ignored). A box's distance to
// its nearest goal is a lower bound on the pushes it still needs.
function makeSpace(g: Grid, region: Uint8Array, goals: number[]): SearchSpace {
  const distToGoal: number[][] = goals.map((goal) => bfsDistances(g, goal, region));
  const minDist: number[] = new Array(g.w * g.h).fill(Infinity);
  for (let i = 0; i < g.w * g.h; i++) {
    if (!region[i]) continue;
    let m = Infinity;
    for (const d of distToGoal) if (d[i]! < m) m = d[i]!;
    minDist[i] = m;
  }
  return { grid: g, region, goals, distToGoal, minDist };
}

const DEBUG = process.env.MAKEFUN_DEBUG === '1';
function debug(...args: unknown[]): void {
  if (DEBUG) console.error('[sokoban]', ...args);
}

// True when a box layout can never be solved: a box in a wall corner off its
// goal, a 2x2 block of boxes with one off a goal, or a box on a square from
// which no goal is reachable at all.
function isDead(space: SearchSpace, boxes: number[]): boolean {
  const g = space.grid;
  const goal = g.goal;
  for (const b of boxes) {
    if (goal[b]) continue;
    if (space.minDist[b] === Infinity) return true;
    const x = b % g.w;
    const y = Math.floor(b / g.w);
    // Wall corner: two perpendicular neighbours are walls (outside the region).
    const up = !inBounds(g, x, y - 1) || !space.region[idx(g, x, y - 1)];
    const down = !inBounds(g, x, y + 1) || !space.region[idx(g, x, y + 1)];
    const left = !inBounds(g, x - 1, y) || !space.region[idx(g, x - 1, y)];
    const right = !inBounds(g, x + 1, y) || !space.region[idx(g, x + 1, y)];
    if ((up && left) || (up && right) || (down && left) || (down && right)) return true;
  }
  // 2x2 block: four boxes in a square, at least one off a goal, can never move.
  for (const b of boxes) {
    const x = b % g.w;
    const y = Math.floor(b / g.w);
    if (!inBounds(g, x + 1, y + 1)) continue;
    const b2 = idx(g, x + 1, y);
    const b3 = idx(g, x, y + 1);
    const b4 = idx(g, x + 1, y + 1);
    if (boxes.includes(b2) && boxes.includes(b3) && boxes.includes(b4)) {
      if (!goal[b] || !goal[b2] || !goal[b3] || !goal[b4]) return true;
    }
  }
  return false;
}

// Push-based A* over one search space. `weight` scales the heuristic: 1 is
// optimal A*, higher is a faster, greedier search. Returns the move string
// (walks + pushes, lowercase) with the expansions spent, or null when the
// budget runs out.
function pushSearch(
  space: SearchSpace,
  startBoxes: number[],
  startPlayer: number,
  maxExpansions: number,
  weight: number,
): { moves: string; expansions: number } | null {
  const g = space.grid;
  const goalSet = new Set(space.goals);
  const solved = (bs: number[]): boolean => bs.every((b) => goalSet.has(b));
  const heuristic = (bs: number[]): number => {
    let h = 0;
    for (const b of bs) {
      const d = space.minDist[b]!;
      if (d === Infinity) return Infinity;
      h += d;
    }
    return h;
  };

  // State storage, parallel arrays indexed by state id.
  const stateBoxes: number[][] = [startBoxes.slice().sort((a, b) => a - b)];
  const statePlayer: number[] = [startPlayer];
  const stateG: number[] = [0];
  const parent: number[] = [-1];
  const pushBox: number[] = [-1]; // box pushed to reach this state (from parent)
  const pushDir: number[] = [-1]; // direction index of that push
  // Transposition: best push count per (boxes, player region). The region is
  // identified by its lowest square: two player positions in one connected
  // region (with the same boxes) are the same state for pushing purposes.
  const best = new Map<string, number>();
  const keyOf = (bs: number[], regionMin: number): string => bs.join(',') + '|' + regionMin;

  const startMask = reachableMask(g, startPlayer, new Set(startBoxes));
  let startMin = Infinity;
  for (let i = 0; i < startMask.length; i++) if (startMask[i]) { startMin = i; break; }
  best.set(keyOf(stateBoxes[0]!, startMin), 0);
  const heap = new Heap();
  let seq = 0;
  const h0 = heuristic(stateBoxes[0]!);
  if (h0 === Infinity || isDead(space, stateBoxes[0]!)) return null;
  heap.push([h0 * weight, 0, seq++, 0]);

  let expanded = 0;
  while (heap.size > 0 && expanded < maxExpansions) {
    const [, , , s] = heap.pop();
    const bs = stateBoxes[s]!;
    const player = statePlayer[s]!;
    // Stale heap entry: a better path to this state was already recorded.
    const mask = reachableMask(g, player, new Set(bs));
    let regionMin = Infinity;
    for (let i = 0; i < mask.length; i++) if (mask[i]) { regionMin = i; break; }
    const k = keyOf(bs, regionMin);
    if (best.get(k) !== stateG[s]) continue;
    if (solved(bs)) {
      // Rebuild the move string: for each push, the player's walk to the push
      // square (recomputed by BFS, deterministic) plus the push step itself.
      const pushes: { box: number; dir: number }[] = [];
      for (let t = s; t > 0; t = parent[t]!) pushes.push({ box: pushBox[t]!, dir: pushDir[t]! });
      pushes.reverse();
      let moves = '';
      let curPlayer = startPlayer;
      let curBoxes = startBoxes.slice().sort((a, b) => a - b);
      for (const p of pushes) {
        const d = DIRS[p.dir]!;
        const pushPos = p.box - (d.dy * g.w + d.dx);
        const walk = walkPath(g, curPlayer, pushPos, new Set(curBoxes));
        if (walk === null) return null; // cannot happen: the search checked reachability
        moves += walk + d.ch;
        const at = curBoxes.indexOf(p.box);
        curBoxes[at] = p.box + (d.dy * g.w + d.dx);
        curBoxes.sort((a, b) => a - b);
        curPlayer = p.box;
      }
      return { moves, expansions: expanded };
    }
    expanded++;
    const boxSet = new Set(bs);
    for (let bi = 0; bi < bs.length; bi++) {
      const b = bs[bi]!;
      const bx = b % g.w;
      const by = Math.floor(b / g.w);
      for (let di = 0; di < DIRS.length; di++) {
        const d = DIRS[di]!;
        const px = bx - d.dx;
        const py = by - d.dy;
        const dx2 = bx + d.dx;
        const dy2 = by + d.dy;
        if (!inBounds(g, px, py) || !inBounds(g, dx2, dy2)) continue;
        const pushPos = idx(g, px, py);
        const dest = idx(g, dx2, dy2);
        if (!mask[pushPos] || !space.region[dest] || boxSet.has(dest)) continue;
        const nb = bs.slice();
        nb[bi] = dest;
        nb.sort((a, b) => a - b);
        if (isDead(space, nb)) continue;
        const h = heuristic(nb);
        if (h === Infinity) continue;
        const ng = stateG[s]! + 1;
        // The player ends on the pushed box's old square; its region there is
        // what the next state is keyed on.
        const nmask = reachableMask(g, b, new Set(nb));
        let nMin = Infinity;
        for (let i = 0; i < nmask.length; i++) if (nmask[i]) { nMin = i; break; }
        const nk = keyOf(nb, nMin);
        const seen = best.get(nk);
        if (seen !== undefined && seen <= ng) continue;
        const t = stateBoxes.length;
        stateBoxes.push(nb);
        statePlayer.push(b);
        stateG.push(ng);
        parent.push(s);
        pushBox.push(b);
        pushDir.push(di);
        best.set(nk, ng);
        heap.push([ng + weight * h, -ng, seq++, t]);
      }
    }
  }
  return null;
}

type Room = {
  squares: number[]; // open squares of the room (region)
  boxes: number[];
  goals: number[];
  door: number; // room square adjacent to the hallway (player entry)
  hallSquare: number; // hallway square just outside the door
};

// Split a level into rooms. The hallway is the player's row; each room is a
// connected component of open squares off that row, found by flooding from
// its boxes without crossing the hallway. Returns null when the split does
// not cover every box exactly once, or a room holds no goal.
function findRooms(g: Grid): Room[] | null {
  const hallY = Math.floor(g.player / g.w);
  const open = new Uint8Array(g.w * g.h);
  for (let i = 0; i < g.w * g.h; i++) open[i] = g.wall[i] ? 0 : 1;
  const boxRoom = new Map<number, number>();
  const rooms: Room[] = [];
  for (const b of g.boxes) {
    if (boxRoom.has(b)) continue;
    const by = Math.floor(b / g.w);
    if (by === hallY) return null; // a box in the hallway: not a room level
    // Flood this box's side of the hallway (never crossing the hallway row).
    const region: number[] = [];
    const seen = new Set<number>([b]);
    const queue: number[] = [b];
    for (let head = 0; head < queue.length; head++) {
      const cur = queue[head]!;
      region.push(cur);
      const cx = cur % g.w;
      const cy = Math.floor(cur / g.w);
      for (const d of DIRS) {
        const nx = cx + d.dx;
        const ny = cy + d.dy;
        if (!inBounds(g, nx, ny) || ny === hallY) continue;
        const ni = idx(g, nx, ny);
        if (!open[ni] || seen.has(ni)) continue;
        seen.add(ni);
        queue.push(ni);
      }
    }
    const roomBoxes = g.boxes.filter((x) => seen.has(x));
    const roomGoals = region.filter((x) => g.goal[x]);
    if (roomBoxes.length === 0 || roomGoals.length === 0) return null;
    // The door: the room square adjacent to a hallway square.
    let door = -1;
    let hallSquare = -1;
    for (const s of region) {
      const sx = s % g.w;
      const sy = Math.floor(s / g.w);
      if (Math.abs(sy - hallY) !== 1) continue;
      const hs = idx(g, sx, hallY);
      if (!open[hs]) continue;
      door = s;
      hallSquare = hs;
      break;
    }
    if (door === -1) return null;
    const roomIndex = rooms.length;
    for (const x of roomBoxes) boxRoom.set(x, roomIndex);
    rooms.push({ squares: region, boxes: roomBoxes, goals: roomGoals, door, hallSquare });
  }
  if (boxRoom.size !== g.boxes.length) return null;
  // Deterministic room order: by door square.
  rooms.sort((a, b) => a.door - b.door);
  return rooms;
}

// Solve one level. Rooms first (each push-optimal), stitched with hallway
// walks; whole-level weighted search as the fallback.
function solveLevel(text: string, id: string): string {
  const g = parseLevel(text);
  const rooms = findRooms(g);
  debug(id, 'rooms:', rooms === null ? 'detection FAILED' : rooms.length);
  if (rooms !== null) {
    let moves = '';
    let player = g.player;
    const allBoxes = new Set(g.boxes);
    let ok = true;
    for (const room of rooms) {
      // Walk the hallway to this room's door. The room search starts with
      // the player on the hallway square outside the door -- never on the
      // door square itself, because a box may start there (the player then
      // pushes it into the room as the first move).
      const toDoor = walkPath(g, player, room.hallSquare, allBoxes);
      if (toDoor === null) { ok = false; break; }
      moves += toDoor;
      // Solve the room push-optimally. The search space is the room plus
      // the one hallway square outside the door, so the player can stand
      // there to push a box off the door. A box pushed out onto that
      // square is caught as dead: three of its neighbours are outside the
      // space, which the corner check reads as walls.
      const region = new Uint8Array(g.w * g.h);
      for (const s of room.squares) region[s] = 1;
      region[room.hallSquare] = 1;
      const space = makeSpace(g, region, room.goals);
      const result = pushSearch(space, room.boxes, room.hallSquare, ROOM_EXPANSIONS, 1);
      debug(id, 'room door', room.door, 'boxes', room.boxes.length, 'goals', room.goals.length,
        result === null ? 'FAILED (budget)' : `solved expansions=${result.expansions} moves=${result.moves.length}`);
      if (result === null) { ok = false; break; }
      const roomMoves = result.moves;
      moves += roomMoves;
      // Replay the room solution to find where the player ends up, then walk
      // back out to the hallway square outside the door.
      let curPlayer = room.hallSquare;
      const curBoxes = new Set(room.boxes);
      for (const ch of roomMoves) {
        const d = DIRS.find((dd) => dd.ch === ch)!;
        const next = curPlayer + d.dy * g.w + d.dx;
        if (curBoxes.has(next)) {
          curBoxes.delete(next);
          curBoxes.add(next + d.dy * g.w + d.dx);
        }
        curPlayer = next;
      }
      const outWalk = walkPath(g, curPlayer, room.hallSquare, curBoxes);
      if (outWalk === null) { ok = false; break; }
      moves += outWalk;
      player = room.hallSquare;
      // The room's boxes are solved now; they stay where they are.
      for (const b of room.boxes) allBoxes.delete(b);
      for (const b of curBoxes) allBoxes.add(b);
    }
    if (ok) return moves;
  }
  // Fallback: weighted A* over the whole level, then a greedier pass.
  debug(id, 'fallback: whole-level weighted A*');
  const region = new Uint8Array(g.w * g.h);
  for (let i = 0; i < g.w * g.h; i++) region[i] = g.wall[i] ? 0 : 1;
  const goals: number[] = [];
  for (let i = 0; i < g.w * g.h; i++) if (g.goal[i]) goals.push(i);
  const space = makeSpace(g, region, goals);
  const whole = pushSearch(space, g.boxes, g.player, WHOLE_EXPANSIONS, 2);
  debug(id, 'whole w=2:', whole === null ? 'FAILED' : `solved expansions=${whole.expansions}`);
  if (whole !== null) return whole.moves;
  const greedy = pushSearch(space, g.boxes, g.player, GREEDY_EXPANSIONS, 6);
  debug(id, 'whole w=6:', greedy === null ? 'FAILED' : `solved expansions=${greedy.expansions}`);
  return greedy === null ? '' : greedy.moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'), id);
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
