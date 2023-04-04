from dataclasses import dataclass, field
from graphviz import Digraph
from collections import defaultdict
from classes.State import State
from classes.TransitionSystem import TransitionSystem
import random
import json

COLORS = ["aquamarine3", "cornflowerblue", "cadetblue", "dodgerblue2", "cyan4"] #extend color pallette for more constraints; best would be to have dynamic one

#not sure whether it would be nicer to inherit from TransitionSystem
@dataclass
class AugmentedTransitionSystem:
  transition_system : TransitionSystem
  constraints: list = field(default_factory=list)
  additional_states: list = field(default_factory=list)
  additional_events: list = field(default_factory=list)    # list of abstracted events
  additional_transitions: dict = field(default_factory=dict)
  transitions_probabilities: dict = field(default_factory=dict)
  states_annotations: dict = field(default_factory=dict)
  
  #TODO: adapt this; base probability on both i) number of outgoing arcs and ii) how often a state was observed
  def calculate_probabilities_relative(self):
    merged_transitions = self.transition_system.transitions | self.additional_transitions
    counter_per_pre_state = defaultdict(int)
    for transition, count in merged_transitions.items():
      counter_per_pre_state[transition.state_1] = counter_per_pre_state[transition.state_1] + count
    transitions_relative_occurence = defaultdict(float)
    for transition,val in merged_transitions.items():
      counter = counter_per_pre_state[transition.state_1]
      transitions_relative_occurence[transition] = round(val/counter,2)
    self.transitions_probabilities = transitions_relative_occurence
    # calculate annotations {outgoing_state1: probability, outgoing_state2: probability} for each state
    merged_states = self.transition_system.states.union(self.additional_states)
    for state in merged_states:
      outgoing_states = {}
      for transition, count in merged_transitions.items():
        if transition.state_1 == state:
          outgoing_states.update({transition.state_2: self.transitions_probabilities[transition]})
      self.states_annotations[state] = outgoing_states
    
  def draw(self, outputpath, filename):
    d_graph = Digraph(name=str(filename), graph_attr={'rankdir':'LR', 'center':'true', 'margin':'1'})
    st = Digraph(name='states', node_attr={'shape':'circle', 'fixedsize':'true', 'height':'.3','width':'.3'})
    for tr,val in self.transition_system.transitions.items():
      st.node(str(tr.state_1))
      st.node(str(tr.state_2))
      d_graph.edge(str(tr.state_1), str(tr.state_2), label=str(tr.event)+"count"+str(val)+"probab"+str(self.transitions_probabilities[tr]))    
    d_graph.subgraph(st)
    legend = Digraph(name="legend")
    add_st = Digraph(name='additional_states', node_attr={'shape':'circle', 'fixedsize':'true', 'height':'.3','width':'.3'})
    for tr,val in self.additional_transitions.items():
      col = COLORS[self.constraints.index(tr.constraint)]
      add_st.node(str(tr.state_1))
      add_st.node(str(tr.state_2), color=col, fontcolor=col)
      d_graph.edge(str(tr.state_1), str(tr.state_2), label=str(tr.event)+"count"+str(val)+"probab"+str(self.transitions_probabilities[tr]), color=col, fontcolor=col)
      legend.node(str(tr.constraint), color=col)
    d_graph.subgraph(add_st)
    
    
    d_graph.subgraph(legend)
    d_graph.render(outputpath+"/"+filename, view=False, cleanup=True)
    return outputpath+"/"+filename+".pdf"
  
  '''  
  def predict(self,trace):
    trace.calculate_prefixes((next(iter(self.transition_system.states))).state_abstraction)   #be careful here, could go wrong if different states have different abstraction functions! we can think of including a check before creating the transition system as such; not nice but at least one option to check for consistency
    current_state = trace.prefixes[-1].state

    merged_states = self.transition_system.states.union(self.additional_states)
    if(current_state not in merged_states):
      return "I have never seen this state before. Check your model!"
    outgoing_states = self.states_annotations[current_state]
    max_prob = max(outgoing_states.values())
    possible_states = list()
    for possible_state, probability in outgoing_states.items():
      if probability == max_prob:
        possible_states.append(possible_state)
    # only one state possible
    if len(possible_states) == 1:
      res = ""
      if possible_states[0].origin_log == False:
        res = "THE FOLLOWING PREDICTION IS BASED ON CONSTRAINT: " + str(possible_states[0].constraint) + "\n"
      return res + "current state is "+str(current_state)+"upcoming state is "+str(possible_states[0])+"with probability "+str(max_prob)
    # multiple states possible
    res = "-----MULTIPLE STATES POSSIBLE-----"
    for state in possible_states:
      if state.origin_log == False:
        res = res + "\nTHE FOLLOWING PREDICTION IS BASED ON CONSTRAINT: " + str(state.constraint)
      res = res + "\ncurrent state is "+str(current_state)+"upcoming state is "+str(state)+"with probability "+str(max_prob)
    return res
  '''
  # predict for all possible prefixes within a trace
  def predict(self,trace):
    prefixes_prediction = []
    prefixes_true = []
    trace.calculate_prefixes((next(iter(self.transition_system.states))).state_abstraction)
    for prefix in trace.prefixes:
      current_state = prefix.state
      prefixes_true.append(current_state.abstracted_events[-1].get_conceptname())
      # case 1: unseen state
      if(current_state not in self.states_annotations.keys()):
        #print("Unable to predict as I have never seen this state before.")
        prefixes_prediction.append("unknown")
        continue
      # case 2: no prediction as this is the end state of the augmented transition system
      if self.states_annotations[current_state] == {}:
        #print("No prediction as this is the end state")
        prefixes_prediction.append("eoc")
        continue
      # case 3: observed state
      outgoing_states = self.states_annotations[current_state]
      max_prob = max(outgoing_states.values())
      possible_states = list()
      for possible_state, probability in outgoing_states.items():
        if probability == max_prob:
          possible_states.append(possible_state)
        # only one state possible
      if len(possible_states) == 1:
        #print(f"Next event lable for {current_state} is {possible_states[0].abstracted_events[-1]}")
        prefixes_prediction.append(possible_states[0].abstracted_events[-1].get_conceptname())
      else:
        # multiple states possible
        #TODO how to choose one prediction from multiple states?
        next_state = random.choice(possible_states)
        #print(f"MULTIPLE STATES POSSIBLE: next event lable for {current_state} is {next_state.abstracted_events[-1]}")
        prefixes_prediction.append(next_state.abstracted_events[-1].get_conceptname())
    prefixes_true.append("eoc")
    del prefixes_true[0]
    return prefixes_true, prefixes_prediction

  def online_predict(self,trace):
    trace.calculate_prefixes((next(iter(self.transition_system.states))).state_abstraction)
    current_state = trace.prefixes[-1].state
    # case 1: unseen state
    if(current_state not in self.states_annotations.keys()):
      return "Unable to predict as I have never seen this state before."
    # case 2: no prediction as this is the end state of the augmented transition system
    if self.states_annotations[current_state] == {}:
      return "No prediction as this is the end state"
    # case 3: observed state
    outgoing_states = self.states_annotations[current_state]
    max_prob = max(outgoing_states.values())
    possible_states = list()
    for possible_state, probability in outgoing_states.items():
      if probability == max_prob:
        possible_states.append(possible_state)
      # only one state possible
    if len(possible_states) == 1:
      res = ""
      if possible_states[0].origin_log == False:
        res = "THE FOLLOWING PREDICTION IS BASED ON CONSTRAINT: " + str(possible_states[0].constraint) + "\n"
      return res +"Next event lable for " + str(current_state) + "is " + str(possible_states[0].abstracted_events[-1].get_conceptname() + " with probability " + str(max_prob))
      # multiple states possible
    res = "MULTIPLE STATES POSSIBLE:"
    for state in possible_states:
      res = res + "\nNext event lable for "+str(current_state) + "is " + str(state.abstracted_events[-1].get_conceptname()) + " with probability " + str(max_prob)
    return res