import pygame

# Configuration de la fenêtre
WIDTH, HEIGHT = 500, 500  # Taille de la fenêtre
MARGIN = 50  # Marge pour les axes
AXIS_MIN, AXIS_MAX = 0, 10  # Échelle des axes


# Conversion pixel -> coordonnées (0,10)
def pixel_to_coord(x, y):
    coord_x = (x - MARGIN) / (WIDTH - 2 * MARGIN) * (AXIS_MAX - AXIS_MIN)
    coord_y = (HEIGHT - MARGIN - y) / (HEIGHT - 2 * MARGIN) * (AXIS_MAX - AXIS_MIN)
    return round(coord_x, 2), round(coord_y, 2)


# Callback lors d'un clic
def on_click(x, y):
    print(f"Click détecté aux coordonnées: {x}, {y}")


# Fonction pour dessiner les axes
def draw_axes(screen, font):
    pygame.draw.line(screen, (0, 0, 0), (MARGIN, HEIGHT - MARGIN), (WIDTH - MARGIN, HEIGHT - MARGIN), 2)
    pygame.draw.line(screen, (0, 0, 0), (MARGIN, HEIGHT - MARGIN), (MARGIN, MARGIN), 2)

    for i in range(AXIS_MIN, AXIS_MAX + 1):
        x_pos = MARGIN + i * (WIDTH - 2 * MARGIN) / (AXIS_MAX - AXIS_MIN)
        y_pos = HEIGHT - MARGIN - i * (HEIGHT - 2 * MARGIN) / (AXIS_MAX - AXIS_MIN)

        pygame.draw.line(screen, (0, 0, 0), (x_pos, HEIGHT - MARGIN - 5), (x_pos, HEIGHT - MARGIN + 5), 2)
        pygame.draw.line(screen, (0, 0, 0), (MARGIN - 5, y_pos), (MARGIN + 5, y_pos), 2)

        screen.blit(font.render(str(i), True, (0, 0, 0)), (x_pos - 5, HEIGHT - MARGIN + 10))
        screen.blit(font.render(str(i), True, (0, 0, 0)), (MARGIN - 25, y_pos - 5))

    screen.blit(font.render("X", True, (0, 0, 0)), (WIDTH - MARGIN + 10, HEIGHT - MARGIN - 10))
    screen.blit(font.render("Y", True, (0, 0, 0)), (MARGIN - 10, MARGIN - 20))


# Fonction pour dessiner une droite passant par (0,0) et (x,y)
def draw_line_from_origin(screen, x, y):
    if x == MARGIN:
        x2, y2 = MARGIN, MARGIN  # Verticale
    else:
        slope = (y - (HEIGHT - MARGIN)) / (x - MARGIN)

        if abs(slope) <= 1:
            x2 = WIDTH - MARGIN if x > MARGIN else MARGIN
            y2 = (slope * (x2 - MARGIN)) + (HEIGHT - MARGIN)
        else:
            y2 = MARGIN if y < HEIGHT - MARGIN else HEIGHT - MARGIN
            x2 = ((y2 - (HEIGHT - MARGIN)) / slope) + MARGIN

        x2 = max(MARGIN, min(x2, WIDTH - MARGIN))
        y2 = max(MARGIN, min(y2, HEIGHT - MARGIN))

    pygame.draw.line(screen, (255, 0, 0), (MARGIN, HEIGHT - MARGIN), (x2, y2), 2)


# Initialisation de Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Espace 2D Interactif")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 24)  # Police pour les labels

running = True
mouse_x, mouse_y = MARGIN, HEIGHT - MARGIN

while running:
    screen.fill((255, 255, 255))  # Fond blanc

    draw_axes(screen, font)  # Dessiner les axes
    draw_line_from_origin(screen, mouse_x, mouse_y)  # Dessiner la ligne dynamique

    # Gérer les événements
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            coord_x, coord_y = pixel_to_coord(x, y)
            on_click(coord_x, coord_y)
        elif event.type == pygame.MOUSEMOTION:
            mouse_x, mouse_y = event.pos

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
