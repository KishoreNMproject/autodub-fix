from __future__ import annotations


def chase_step(enemy, player):
    dx = 0
    dy = 0
    if player.x > enemy.x:
        dx = enemy.speed
    elif player.x < enemy.x:
        dx = -enemy.speed
    if player.y > enemy.y:
        dy = enemy.speed
    elif player.y < enemy.y:
        dy = -enemy.speed
    return dx * 0.5, dy * 0.5
