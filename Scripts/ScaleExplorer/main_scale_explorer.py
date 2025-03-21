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


# Initialisation de Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Espace 2D Interactif")
clock = pygame.time.Clock()

running = True
while running:
    screen.fill((255, 255, 255))  # Fond blanc

    # Dessiner les axes
    pygame.draw.line(screen, (0, 0, 0), (MARGIN, HEIGHT - MARGIN), (WIDTH - MARGIN, HEIGHT - MARGIN), 2)
    pygame.draw.line(screen, (0, 0, 0), (MARGIN, HEIGHT - MARGIN), (MARGIN, MARGIN), 2)

    # Gérer les événements
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            coord_x, coord_y = pixel_to_coord(x, y)
            on_click(coord_x, coord_y)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
