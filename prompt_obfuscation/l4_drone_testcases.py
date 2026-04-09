"""
L4 AirSim Drone Test Cases
===========================
25 queries spanning 3 difficulty tiers.
Each entry:
  {
    "id":       str,
    "tier":     "easy" | "medium" | "hard",
    "query":    str,                         # user natural-language request
    "expected_states": [[Δx,Δy,Δz,Δyaw],…] # ground-truth per numerical_evaluator.txt
                                             # None = trajectory scoring skipped
  }

State-change convention (NED world frame):
  [Δx, Δy, Δz, Δyaw]
  Δx > 0 → North,  Δy > 0 → East,  Δz < 0 → Up
  Δyaw > 0 → clockwise (right),  < 0 → counter-clockwise (left)
"""

TEST_CASES = [

    # ── EASY ─────────────────────────────────────────────────────────────────
    # 1  Simple takeoff and vertical climb
    {
        "id": "E01",
        "tier": "easy",
        "query": "Take off and fly 10 meters up.",
        "expected_states": [
            [0, 0, -10, 0],
        ],
    },
    # 2  Fly north in world frame
    {
        "id": "E02",
        "tier": "easy",
        "query": "Take off, then fly 5 meters forward (north).",
        "expected_states": [
            [5, 0, 0, 0],
        ],
    },
    # 3  Fly east (world frame)
    {
        "id": "E03",
        "tier": "easy",
        "query": "Take off, then fly 8 meters to the east.",
        "expected_states": [
            [0, 8, 0, 0],
        ],
    },
    # 4  Land after hovering
    {
        "id": "E04",
        "tier": "easy",
        "query": "Take off, climb 3 meters, then land.",
        "expected_states": [
            [0, 0, -3, 0],
        ],
    },
    # 5  Pure yaw rotation
    {
        "id": "E05",
        "tier": "easy",
        "query": "Take off, then turn 90 degrees clockwise (right).",
        "expected_states": [
            [0, 0, 0, 90],
        ],
    },
    # 6  Fly north + east (two world-frame legs)
    {
        "id": "E06",
        "tier": "easy",
        "query": "Take off, fly 6 meters north, then fly 4 meters east.",
        "expected_states": [
            [6, 0, 0, 0],
            [0, 4, 0, 0],
        ],
    },
    # 7  Fly south (negative X)
    {
        "id": "E07",
        "tier": "easy",
        "query": "Take off, fly 5 meters south.",
        "expected_states": [
            [-5, 0, 0, 0],
        ],
    },
    # 8  Turn left (CCW)
    {
        "id": "E08",
        "tier": "easy",
        "query": "Take off, turn 45 degrees to the left.",
        "expected_states": [
            [0, 0, 0, -45],
        ],
    },

    # ── MEDIUM ────────────────────────────────────────────────────────────────
    # 9  Turn right then fly forward in body frame
    {
        "id": "M01",
        "tier": "medium",
        "query": "Take off, turn 90 degrees right, then fly 5 meters forward in the drone's body frame.",
        # After 90° CW: heading East. Body-forward (+x') → world +y
        # world_x = 5*cos(-90°) + 0 = 0, world_y = -5*sin(-90°) + 0 = 5
        "expected_states": [
            [0, 0, 0, 90],
            [0, 5, 0, 0],
        ],
    },
    # 10 Turn left then fly body-forward
    {
        "id": "M02",
        "tier": "medium",
        "query": "Take off, turn 90 degrees left, then fly 4 meters forward in the drone's body frame.",
        # After -90° yaw: heading West. Body-forward → world -y
        # world_x = 4*cos(90°) = 0, world_y = -4*sin(90°) = -4
        "expected_states": [
            [0, 0, 0, -90],
            [0, -4, 0, 0],
        ],
    },
    # 11 Fly body-right (no prior rotation, heading North)
    {
        "id": "M03",
        "tier": "medium",
        "query": "Take off, fly 3 meters to the right in the drone's body frame (drone is facing north).",
        # body_y = +3, yaw = 0 → world_x=0, world_y=3
        "expected_states": [
            [0, 3, 0, 0],
        ],
    },
    # 12 Fly body-left (heading North)
    {
        "id": "M04",
        "tier": "medium",
        "query": "Take off, fly 6 meters to the left in the drone's body frame.",
        # body_y = -6, yaw=0 → world_y = -6
        "expected_states": [
            [0, -6, 0, 0],
        ],
    },
    # 13 Turn 180° then fly forward
    {
        "id": "M05",
        "tier": "medium",
        "query": "Take off, turn 180 degrees, then fly 5 meters forward.",
        # heading South after 180°; body-forward → world -x
        "expected_states": [
            [0, 0, 0, 180],
            [-5, 0, 0, 0],
        ],
    },
    # 14 Fly to absolute position
    {
        "id": "M06",
        "tier": "medium",
        "query": "Take off, then fly to position [10, 5, -3] in world coordinates.",
        "expected_states": [
            [10, 5, -3, 0],
        ],
    },
    # 15 Sequential turns
    {
        "id": "M07",
        "tier": "medium",
        "query": "Take off, turn 45 degrees right, then turn another 45 degrees right.",
        "expected_states": [
            [0, 0, 0, 45],
            [0, 0, 0, 45],
        ],
    },
    # 16 Climb then fly north
    {
        "id": "M08",
        "tier": "medium",
        "query": "Take off, climb 5 meters, then fly 10 meters north.",
        "expected_states": [
            [0, 0, -5, 0],
            [10, 0, 0, 0],
        ],
    },
    # 17 Turn 90° right, fly body-left
    {
        "id": "M09",
        "tier": "medium",
        "query": "Take off, turn 90 degrees clockwise, then fly 4 meters to the left in the drone's body frame.",
        # heading East (yaw=90). body_y=-4 → radian_yaw = rad(-90) = -π/2
        # world_x = 0*cos(-π/2) + (-4)*sin(-π/2) = 4
        # world_y = 0*(-sin(-π/2)) + (-4)*cos(-π/2) = 0
        "expected_states": [
            [0, 0, 0, 90],
            [4, 0, 0, 0],
        ],
    },

    # ── HARD ──────────────────────────────────────────────────────────────────
    # 18 Simple 4×4 square (world frame, facing north throughout)
    {
        "id": "H01",
        "tier": "hard",
        "query": (
            "Take off, then fly a 4×4 meter square route starting north: "
            "go 4m north, 4m east, 4m south, 4m west, while always facing north."
        ),
        "expected_states": [
            [4, 0, 0, 0],
            [0, 4, 0, 0],
            [-4, 0, 0, 0],
            [0, -4, 0, 0],
        ],
    },
    # 19 Square perimeter inspection (perpendicular inward, CW)
    {
        "id": "H02",
        "tier": "hard",
        "query": (
            "Take off, then fly a 4×4 meter square inspection route. "
            "Before each side, turn 90 degrees clockwise so the drone faces inward. "
            "Route: forward, right, backward, left."
        ),
        # From numerical_evaluator.txt reference:
        # [0,0,0,90],[4,0,0,0],[0,0,0,90],[0,4,0,0],
        # [0,0,0,90],[-4,0,0,0],[0,0,0,90],[0,-4,0,0]
        "expected_states": [
            [0, 0, 0, 90],
            [4, 0, 0, 0],
            [0, 0, 0, 90],
            [0, 4, 0, 0],
            [0, 0, 0, 90],
            [-4, 0, 0, 0],
            [0, 0, 0, 90],
            [0, -4, 0, 0],
        ],
    },
    # 20 Square inspection + reverse path
    {
        "id": "H03",
        "tier": "hard",
        "query": (
            "Take off, fly a 4×4 meter perpendicular inspection square (CW turns), "
            "then fly the reverse path."
        ),
        # From numerical_evaluator.txt:
        # Forward square: [0,0,0,90],[4,0,0,0],[0,0,0,90],[0,4,0,0],
        #                  [0,0,0,90],[-4,0,0,0],[0,0,0,90],[0,-4,0,0]
        # Reverse: [0,4,0,0],[0,0,0,-90],[4,0,0,0],[0,0,0,-90],
        #           [0,-4,0,0],[0,0,0,-90],[-4,0,0,0]... see evaluator
        "expected_states": [
            [0, 0, 0, 90], [4, 0, 0, 0],
            [0, 0, 0, 90], [0, 4, 0, 0],
            [0, 0, 0, 90], [-4, 0, 0, 0],
            [0, 0, 0, 90], [0, -4, 0, 0],
            [0, 4, 0, 0],
            [0, 0, 0, -90], [4, 0, 0, 0],
            [0, 0, 0, -90], [0, -4, 0, 0],
            [0, 0, 0, -90], [-4, 0, 0, 0],
        ],
    },
    # 21 3-leg L-shaped path with turns
    {
        "id": "H04",
        "tier": "hard",
        "query": (
            "Take off, fly 5 meters north, turn 90 degrees right, "
            "fly 3 meters forward in the drone's body frame, "
            "turn 90 degrees right, fly 5 meters forward in the drone's body frame."
        ),
        # leg1: [5,0,0,0]; turn CW: [0,0,0,90]; heading East
        # leg2 body-forward with yaw=90: world_x=0, world_y=3 → [0,3,0,0]
        # turn CW: [0,0,0,90]; heading South
        # leg3 body-forward with yaw=180: world_x=-5, world_y=0 → [-5,0,0,0]
        "expected_states": [
            [5, 0, 0, 0],
            [0, 0, 0, 90],
            [0, 3, 0, 0],
            [0, 0, 0, 90],
            [-5, 0, 0, 0],
        ],
    },
    # 22 Mixed climb + square inspection
    {
        "id": "H05",
        "tier": "hard",
        "query": (
            "Take off, climb 5 meters, then fly a 4×4 meter square (always facing north): "
            "4m north, 4m east, 4m south, 4m west."
        ),
        "expected_states": [
            [0, 0, -5, 0],
            [4, 0, 0, 0],
            [0, 4, 0, 0],
            [-4, 0, 0, 0],
            [0, -4, 0, 0],
        ],
    },
    # 23 360-degree orbit simulation (4 × 90° CW turns, fly north between each)
    {
        "id": "H06",
        "tier": "hard",
        "query": (
            "Take off, then execute a diamond pattern: fly 5m north, "
            "turn 90° right, fly 5m forward, turn 90° right, "
            "fly 5m forward, turn 90° right, fly 5m forward."
        ),
        # leg1 N: [5,0,0,0]; turn: [0,0,0,90]; heading E
        # leg2 body-fwd E: world [0,5,0,0]; turn: [0,0,0,90]; heading S
        # leg3 body-fwd S: world [-5,0,0,0]; turn: [0,0,0,90]; heading W
        # leg4 body-fwd W: world [0,-5,0,0]
        "expected_states": [
            [5, 0, 0, 0],
            [0, 0, 0, 90],
            [0, 5, 0, 0],
            [0, 0, 0, 90],
            [-5, 0, 0, 0],
            [0, 0, 0, 90],
            [0, -5, 0, 0],
        ],
    },
    # 24 Counter-clockwise square inspection
    {
        "id": "H07",
        "tier": "hard",
        "query": (
            "Take off, fly a 4×4 meter square inspection with CCW turns (left), "
            "facing inward on each side."
        ),
        # From evaluator.txt CCW reference:
        # turn -90, fly N [4,0,0,0], turn -90, fly W [0,-4,0,0],
        # turn -90, fly S [-4,0,0,0], turn -90, fly E [0,4,0,0]
        "expected_states": [
            [0, 0, 0, -90], [4, 0, 0, 0],
            [0, 0, 0, -90], [0, -4, 0, 0],
            [0, 0, 0, -90], [-4, 0, 0, 0],
            [0, 0, 0, -90], [0, 4, 0, 0],
        ],
    },
]


def get_cases_by_tier(tier: str) -> list[dict]:
    return [tc for tc in TEST_CASES if tc["tier"] == tier]


def print_summary():
    tiers = ["easy", "medium", "hard"]
    print(f"Total test cases: {len(TEST_CASES)}")
    for t in tiers:
        n = len(get_cases_by_tier(t))
        print(f"  {t.capitalize()}: {n}")
    print()
    for tc in TEST_CASES:
        gt = tc["expected_states"]
        print(f"  [{tc['id']}] ({tc['tier']:6s}) {tc['query'][:70]}")
        if gt:
            print(f"           Expected steps: {len(gt)}")


if __name__ == "__main__":
    print_summary()
