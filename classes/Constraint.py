from dataclasses import dataclass
from classes.Event import Event

@dataclass
class Constraint:
  predecessor: Event
  successor: Event
  relation: str
  
  def __str__(self):
    return str(self.predecessor) + " " + self.relation + " " + str(self.successor)
