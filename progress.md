# Progress

1. We verified that 8 is the optimal concurrency.
2. We verified that we need to use the maximum token limit instead of 4k.
3. We stopped the first Kimi-K2.6 L1.1 RL run after all observed rewards were zero, then changed reward shaping to penalize format and compile failures and use partial correctness.
