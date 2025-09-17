import numpy as np
import tkinter as tk
import math

# --- Game Constants ---
EMPTY_CELL = 0
BODY_CELL = 1
HEAD_CELL = 2
FOOD_CELL = 3

GRID_SIZE = 20
CELL_SIZE = 20  # Size of each cell in pixels for rendering
SNAKE_COLOR = "green"
FOOD_COLOR = "red"
GAME_SPEED = 150  # Milliseconds

class SnakeGame:
    """
    Handles the core logic of the Snake game. The internal 'board' is
    updated incrementally with every move for efficiency.
    """
    def __init__(self):
        self.board = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.GRID_SIZE = GRID_SIZE
        self.frame_iteration = 0
        # Call reset() to perform the initial setup, avoiding code duplication.
        self.reset()

    def spawn_food(self):
        while True:
            position = (np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE))
            if position not in self.snake:
                return position

    def change_direction(self, new_direction):
        # Prevent the snake from reversing on itself
        if self.direction != (-new_direction[0], -new_direction[1]):
            self.direction = new_direction

    def move_snake(self):
        """
        Moves the snake one step, checks for collisions, and incrementally
        updates the internal board state.
        """
        if self.game_over:
            return

        old_head = self.snake[0]
        new_head = (old_head[0] + self.direction[0], old_head[1] + self.direction[1])

        # Check for wall collision
        if not (0 <= new_head[0] < GRID_SIZE and 0 <= new_head[1] < GRID_SIZE):
            self.game_over = True
            return

        # Check for self collision
        if new_head in self.snake:
            self.game_over = True
            return
        
        # --- Incremental Board Updates ---
        # 1. Update the old head to be a body part
        self.board[old_head] = BODY_CELL

        # Move the snake's body
        self.snake.insert(0, new_head)
        
        # 2. Update the new head position on the board
        self.board[new_head] = HEAD_CELL

        # Check for food consumption
        if new_head == self.food:
            self.score += 1
            self.food = self.spawn_food()
            self.board[self.food] = FOOD_CELL
        else:
            # 3. If no food is eaten, remove the tail and clear it from the board
            tail = self.snake.pop()
            self.board[tail] = EMPTY_CELL

    def play_step(self, action):
        """
        Takes an action, updates the game, and returns a SHAPED reward.
        """
        # --- 1. Calculate distance to food BEFORE the move ---
        old_head = self.snake[0]
        dist_before = math.sqrt((old_head[0] - self.food[0])**2 + (old_head[1] - self.food[1])**2)

        # 2. Translate agent's action into a direction
        directions_cw = [(-1, 0), (0, 1), (1, 0), (0, -1)] # U, R, D, L
        current_dir_index = directions_cw.index(self.direction)
        if np.array_equal(action, [1, 0, 0]): # Straight
            new_direction = self.direction
        elif np.array_equal(action, [0, 1, 0]): # Right turn
            next_dir_index = (current_dir_index + 1) % 4
            new_direction = directions_cw[next_dir_index]
        else: # Left turn [0, 0, 1]
            next_dir_index = (current_dir_index - 1) % 4
            new_direction = directions_cw[next_dir_index]
        self.direction = new_direction
        
        # 3. Perform the move
        self.move_snake()
        
        # --- 4. Define the rewards based on the outcome ---
        reward = 0

        if self.frame_iteration > 100 * len(self.snake):
            self.game_over = True
            reward = -10 # Punish it for being too slow
            return reward, self.game_over, self.score
        if self.game_over:
            reward = -10
            return reward, self.game_over, self.score
        
        if self.snake[0] == self.food:
            # Huge reward for eating the food
            reward = 10
        else:
            # --- 5. SHAPE the reward: encourage getting closer ---
            new_head = self.snake[0]
            dist_after = math.sqrt((new_head[0] - self.food[0])**2 + (new_head[1] - self.food[1])**2)
            
            if dist_after < dist_before:
                # It got closer! Give it a small nudge of encouragement.
                reward = 0.25
            else:
                # It moved away or stayed the same distance. Punish it slightly.
                reward = -0.2

        return reward, self.game_over, self.score

    def get_board(self):
        """Returns the current state of the game board."""
        return self.board

    def reset(self):
        """Resets the game to its initial state and sets up the board."""
        self.snake = [(GRID_SIZE // 2, GRID_SIZE // 2)]
        self.direction = (0, 1)
        self.game_over = False
        self.score = 0
        self.frame_iteration = 0
        
        # Perform the initial board setup
        self.board.fill(EMPTY_CELL)
        self.food = self.spawn_food()
        self.board[self.snake[0]] = HEAD_CELL
        self.board[self.food] = FOOD_CELL


# --- Main Application Block ---
if __name__ == "__main__":
    # 1. Initialize the game logic
    game = SnakeGame()

    # 2. Set up the Tkinter window and canvas
    root = tk.Tk()
    root.title("Snake Game")
    canvas = tk.Canvas(root, width=GRID_SIZE * CELL_SIZE, height=GRID_SIZE * CELL_SIZE, bg="black")
    canvas.pack()
    root.resizable(False, False)

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
        """The main loop that updates game state and redraws the board."""
        if not game.game_over:
            game.move_snake()
            draw_board()
            root.after(GAME_SPEED, game_loop)
        else:
            # Display Game Over message
            canvas.create_text(
                GRID_SIZE * CELL_SIZE / 2,
                GRID_SIZE * CELL_SIZE / 2,
                text=f"Game Over\nScore: {game.score}",
                fill="white",
                font=("Helvetica", 24),
                justify=tk.CENTER
            )

    # 4. Bind keyboard inputs to change the snake's direction
    root.bind("<Up>", lambda event: game.change_direction((-1, 0)))
    root.bind("<Down>", lambda event: game.change_direction((1, 0)))
    root.bind("<Left>", lambda event: game.change_direction((0, -1)))
    root.bind("<Right>", lambda event: game.change_direction((0, 1)))

    # 5. Start the game loop and the Tkinter main loop
    game_loop()
    root.mainloop()