# Hackman: A Hybrid HMM-RL Agent

This project implements a "smart" agent to play the game of Hangman. It uses a hybrid strategy that combines a powerful probabilistic model (similar to an HMM) with a Q-learning (Reinforcement Learning) agent to learn the optimal policy for guessing letters.

## Core Components

The agent is a combination of two components that work together:

1.  **Position-Aware Probabilistic Model (`HangmanHMM`)**
    * This is not a traditional HMM but a position-aware frequency model trained on a 50,000-word corpus.
    * For any given game state (e.g., `_ _ A _ E`), it filters the entire corpus for words that match this pattern.
    * It then calculates the frequency of unguessed letters in the blank spots of those matching words, providing a highly accurate probability for the next best guess.

2.  **Q-Learning Agent (Reinforcement Learning)**
    * This agent learns a "policy" (a Q-table) to maximize its long-term rewards (win the game).
    * **State:** The game state is defined as a unique string: `f"{masked_word}|{lives_left}|{sorted(guessed_letters)}"`.
    * **Rewards:** The agent receives shaped rewards to learn effective play:
        * **Correct Guess:** `+5 *` (number of letters revealed)
        * **Wrong Guess:** `-10`
        * **Win Game:** `+50`
        * **Lose Game:** `-50`

## 🧠 How it Works

The agent's strategy manages the exploration vs. exploitation trade-off:

* **Exploitation (Q-Value + HMM):** When exploiting, the agent doesn't just pick the move with the highest Q-value. It combines its learned experience with the HMM's logic:
    `combined_value = q_value + (hmm_probability * 10)`
    This ensures the agent's "best" move is a smart, probabilistic guess that is also informed by its long-term learned strategy.

* **Exploration (Guided):** When exploring (based on a decaying epsilon), the agent doesn't pick a purely random letter. It makes a **weighted random guess based on the HMM's probabilities**. This "guided exploration" ensures that even its exploratory moves are high-quality, speeding up learning.

## Performance

The notebook trains the agent for 20,000 episodes and evaluates it on 2,000 unseen test words.

* **Training Win Rate:** 97.1% (on final 1k episodes)
* **Test Win Rate (Unseen Words):** 19.75%
* **Overfitting:** The large drop in performance indicates that the Q-table (with 79,000+ states) memorized the training set rather than learning a generalizable policy.

## How to Use

1.  **Run the Notebook:** Execute the cells in `ml_hackathon.ipynb` from top to bottom.
2.  **Process:** The notebook will:
    * Load and pre-process the `words.txt` corpus.
    * Initialize the `HangmanGame`, `HangmanHMM`, and `QLearningAgent`.
    * Train the agent for 20,000 episodes and save the Q-table.
    * Evaluate the agent on a 2,000-word test set.
    * Run a comparative analysis against other strategies.
    * Generate all final reports, visualizations, and model files.

## Dependencies

The project requires the following Python libraries:
* `pandas`
* `numpy`
* `scikit-learn` (sklearn)
* `matplotlib`
* `tqdm`
* `hmmlearn`

## Output Files

Running the notebook will produce the following output files:

* `hangman_model.pkl`: The trained Q-table and HMM model.
* `evaluation_results.csv`: Detailed per-game results from the evaluation phase.
* `report_data.pkl`: Serialized Python object containing data for analysis.
* `FINAL_SUMMARY.txt`: A text file with the final performance metrics.
* `comprehensive_results.png`: A saved plot showing all key visualizations.
