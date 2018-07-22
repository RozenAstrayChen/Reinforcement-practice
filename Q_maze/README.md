#Note

## Process

### Init Maze

### Do train

- choose_action
- feeback by env 
- input feeback data to train brain
- record next observation

train algorithm:
```python
q_target = r + self.gamma * self.q_table.loc[s_, :].max() 
```