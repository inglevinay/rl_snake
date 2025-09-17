# play.py

import torch
from agent import Agent
from snake_game import SnakeGame, GRID_SIZE, CELL_SIZE, GAME_SPEED
import tkinter as tk

EMPTY_CELL = 0
BODY_CELL = 1
HEAD_CELL = 2
FOOD_CELL = 3

GRID_SIZE = 20
CELL_SIZE = 20  # Size of each cell in pixels for rendering
SNAKE_COLOR = "green"
FOOD_COLOR = "red"
GAME_SPEED = 50  # Milliseconds

def play():
    # 1. Initialize Agent and Game
    agent = Agent()
    game = SnakeGame()

    # 2. Load the trained model
    agent.model.load()

    # --- Set up Tkinter for visualization ---
    root = tk.Tk()
    root.title("Snake AI")
    canvas = tk.Canvas(root, width=GRID_SIZE * CELL_SIZE, height=GRID_SIZE * CELL_SIZE, bg="black")
    canvas.pack()

    # 3. Define rendering and game loop functions
    def draw_board():
        """Renders the game state from the game object's board onto the canvas."""
        canvas.delete("all")
        board = game.get_board()
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                cell_type = board[row, col]
                if cell_type == EMPTY_CELL: # Skip drawing empty cells for performance
                    continue
                
                x1, y1 = col * CELL_SIZE, row * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE

                if cell_type == HEAD_CELL:
                    canvas.create_rectangle(x1, y1, x2, y2, fill=SNAKE_COLOR, outline="white")
                elif cell_type == BODY_CELL:
                    canvas.create_rectangle(x1, y1, x2, y2, fill=SNAKE_COLOR)
                elif cell_type == FOOD_CELL:
                    canvas.create_oval(x1, y1, x2, y2, fill=FOOD_COLOR)

    def game_loop():
        if not game.game_over:
            # 1. Get current state
            state_old = agent.get_state(game)

            # 2. Get move (NO RANDOMNESS)
            # Temporarily set epsilon to 0 for pure inference
            original_epsilon = agent.epsilon
            agent.epsilon = 0
            final_move = agent.get_action(state_old)
            agent.epsilon = original_epsilon # Restore it if needed elsewhere

            # 3. Perform move
            _, game.game_over, score = game.play_step(final_move)
            
            # 4. Redraw and repeat
            draw_board()
            root.after(GAME_SPEED, game_loop)
        else:
            print("Game Over! Final Score:", game.score)
            root.destroy()
    
    # Start the game loop
    game_loop()
    root.mainloop()

if __name__ == '__main__':
    play()