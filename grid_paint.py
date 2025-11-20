import pygame

# ---------- Core cooperative DFS planning ----------

N, M = 6, 10
mid = M // 2
dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def dfs(r, c, col_min, col_max, visited, path):
    if r < 0 or r >= N or c < col_min or c > col_max:
        return
    if visited[r][c]:
        return
    visited[r][c] = True
    path.append((r, c))
    for dr, dc in dirs:
        dfs(r + dr, c + dc, col_min, col_max, visited, path)


def compute_paths():
    visitedA = [[False] * M for _ in range(N)]
    visitedB = [[False] * M for _ in range(N)]
    pathA, pathB = [], []

    dfs(0, 0, 0, mid - 1, visitedA, pathA)        # Robot A
    dfs(0, M - 1, mid, M - 1, visitedB, pathB)    # Robot B

    return pathA, pathB


# ---------- Pygame visualization ----------

CELL_SIZE = 60
WIDTH = M * CELL_SIZE
HEIGHT = N * CELL_SIZE

BG_COLOR = (25, 25, 35)
GRID_COLOR = (70, 70, 90)
A_COLOR = (80, 180, 255)
B_COLOR = (255, 200, 80)
TEXT_COLOR = (230, 230, 230)


def draw_grid(screen, grid):
    for r in range(N):
        for c in range(M):
            x = c * CELL_SIZE
            y = r * CELL_SIZE
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

            # base cell
            pygame.draw.rect(screen, GRID_COLOR, rect, width=1)

            if grid[r][c] == 'A':
                pygame.draw.rect(screen, A_COLOR, rect.inflate(-4, -4))
            elif grid[r][c] == 'B':
                pygame.draw.rect(screen, B_COLOR, rect.inflate(-4, -4))


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT + 50))
    pygame.display.set_caption("Cooperative Grid Painting Agents")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 22)

    pathA, pathB = compute_paths()
    grid = [['.' for _ in range(M)] for _ in range(N)]
    max_steps = max(len(pathA), len(pathB))

    step = 0
    running = True

    while running:
        clock.tick(6)  # FPS / animation speed

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update step
        if step < max_steps:
            if step < len(pathA):
                r, c = pathA[step]
                grid[r][c] = 'A'
            if step < len(pathB):
                r, c = pathB[step]
                grid[r][c] = 'B'
            step += 1

        # Draw
        screen.fill(BG_COLOR)
        draw_grid(screen, grid)

        # Legend bar
        legend_rect = pygame.Rect(0, HEIGHT, WIDTH, 50)
        pygame.draw.rect(screen, (15, 15, 25), legend_rect)

        txt = font.render("Robot A (Blue) | Robot B (Yellow)  -  Cooperative Grid Painting", True, TEXT_COLOR)
        screen.blit(txt, (10, HEIGHT + 10))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
