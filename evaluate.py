from classes.Attribute import Attribute
from classes.Event import Event
from classes.Trace import Trace
from classes.Log import Log
from generate_ts import augment_ts, read_data
from collections import defaultdict
from sklearn import metrics 
import pickle
import argparse
import time


parser = argparse.ArgumentParser(
    description="ATS prediction")

parser.add_argument("--stream_path", 
    type=str, 
    default="data/stream/",   
    help="path to dataset")

parser.add_argument("--stream_name", 
    type=str, 
    default="synthetic.csv",   
    help="dataset name")

parser.add_argument("--constraint_path", 
    type=str, 
    default="data/constraint_patterns/",
    help="path to constraints")

parser.add_argument("--constraint_name", 
    type=str, 
    default="synthetic.csv",   
    help="constraint name")

parser.add_argument("--fformat", 
    type=str, 
    default="csv",   
    help="file format for stream dataset")

parser.add_argument("--lc_filter", 
    type=str, 
    default="[False,""]",
    help="lifecycle filter")

parser.add_argument("--state_abstraction", 
    type=str, 
    default="sequence",   
    help="abstraction for state representation")

parser.add_argument("--ts_model", 
    type=str, 
    default="output/synthetic.pickle",   
    help="path to ts model")

parser.add_argument("--ats_model", 
    type=str, 
    default="output/synthetic_augmented.pickle",   
    help="path to augmented_ts model")

parser.add_argument("--update", 
    type=lambda x: (str(x).lower() == 'true'), 
    default=False, 
    help="update the prediction model")

args = parser.parse_args()


# ATS: basic transition system without constraints
def ts_prediction(event_stream, model_path, update=False):
    picklefile = open(model_path, 'rb')
    ts = pickle.load(picklefile)
    picklefile.close()
    y_true = []
    y_pred = []
    for trace in event_stream.traces:
        prefixes_true, prefixes_pred = ts.predict(trace)
        y_true += prefixes_true
        y_pred += prefixes_pred 
        # update transitions as event stream evolves
        if update:
            if prefixes_pred[-1] == "unknown":
                ts = update_ts_unknown(ts, trace)
            else:
                ts = update_ts_known(ts, trace)
        # do not update if unknown behavior occur
        if not update:
            if prefixes_pred[-1] == "unknown":
                ts = ts
            else:
                ts = update_ts_known(ts, trace)

    token_true, token_pred = mapper(y_true, y_pred)
    accuracy = metrics.accuracy_score(token_true, token_pred)
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(token_true, token_pred, average="weighted")
    print("##########Prediction results for ATS##########")
    print('Accuracy across all prefixes:', accuracy)
    print('F-score across all prefixes:', fscore)
    print('Precision across all prefixes:', precision)
    print('Recall across all prefixes:', recall)
    
# AATS: augmented transition system including constraints
def ats_prediction(event_stream, model_path, update=False):
    picklefile = open(model_path, 'rb')
    augmented_ts = pickle.load(picklefile)
    picklefile.close()
    y_true = []
    y_pred = []
    for trace in event_stream.traces:
        prefixes_true, prefixes_pred = augmented_ts.predict(trace) 
        y_true += prefixes_true
        y_pred += prefixes_pred   
        # update transitions as event stream evolves
        if update:
            if prefixes_pred[-1] == "unknown":
                augmented_ts = update_ats_unknown(augmented_ts, trace, augmented_ts.constraints) # original constraints
            else:
                augmented_ts = update_ats_known(augmented_ts, trace)
        # do not update if unknown behavior occur
        if not update:
            if prefixes_pred[-1] == "unknown":
                augmented_ts = augmented_ts
            else:
                augmented_ts = update_ats_known(augmented_ts, trace)
        
    token_true, token_pred = mapper(y_true, y_pred)
    accuracy = metrics.accuracy_score(token_true, token_pred)
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(token_true, token_pred, average="weighted")
    print("##########Prediction results for AATS##########")
    print('Accuracy across all prefixes:', accuracy)
    print('F-score across all prefixes:', fscore)
    print('Precision across all prefixes:', precision)
    print('Recall across all prefixes:', recall)

def mapper(y_true, y_pred):
    token_true = []
    token_pred = []
    keys = set(y_true) | set(y_pred)
    val = range(len(keys))
    labels_dict = dict(zip(keys, val))
    for _y in y_true:
        token_true.append(labels_dict[_y])
    for _y in y_pred:
        token_pred.append(labels_dict[_y])
    return token_true, token_pred

# update transitions for ts if trace has been observed before 
def update_ts_known(ts, trace):
    # update according to involving stream regardless of predicted value is true or false
    tr_label_space, tr_state_space, tr_transitions = trace.calculate_transitions(args.state_abstraction)
    for transition in tr_transitions:
        ts.transitions[transition] += 1
    ts.calculate_probabilities_relative()
    return ts

# update transitions for ts if trace has not been observed before 
def update_ts_unknown(ts, trace):
    tr_label_space, tr_state_space, tr_transitions = trace.calculate_transitions(args.state_abstraction)
    ts.events.update(tr_label_space)
    ts.states.update(tr_state_space)
    for transition in tr_transitions:
        ts.transitions[transition] += 1
    ts.calculate_probabilities_relative()
    return ts

# update transitions for augmented_ts if trace has been observed before 
def update_ats_known(augmented_ts, trace):
    # update according to involving stream regardless of predicted value is true or false
    tr_label_space, tr_state_space, tr_transitions = trace.calculate_transitions(args.state_abstraction)
    for transition in tr_transitions:
        if transition in augmented_ts.transition_system.transitions:
            augmented_ts.transition_system.transitions[transition] += 1
        if transition in augmented_ts.additional_transitions:
            tr_constraint = None
            for constraint in augmented_ts.constraints:
                predecessor = constraint.predecessor.apply_event_abstraction()
                successor = constraint.successor.apply_event_abstraction()
                pre_state = transition.state_1.abstracted_events
                suc_state = transition.state_2.abstracted_events  
                if predecessor in pre_state and successor not in pre_state and successor in suc_state:
                    tr_constraint = constraint
            transition.origin_log = False
            transition.constraint = tr_constraint
            augmented_ts.additional_transitions[transition] += 1
    augmented_ts.calculate_probabilities_relative()
    return augmented_ts

# update transitions for augmented_ts if trace has not been observed before 
def update_ats_unknown(augmented_ts, trace, constraints):
    tr_label_space, tr_state_space, tr_transitions = trace.calculate_transitions(args.state_abstraction)
    # update the basic transition system
    updated_ts = augmented_ts.transition_system
    updated_ts.events.update(tr_label_space)
    updated_ts.states.update(tr_state_space)
    for transition in tr_transitions:
        updated_ts.transitions[transition] += 1
    # augment the updated transition system with original constraints
    new_ats = update_constraints(updated_ts, constraints)
    new_ats.calculate_probabilities_relative()
    return new_ats

def update_constraints(ts, constraints):
    augmented_ts = augment_ts(ts, constraints)
    return augmented_ts

if __name__ == "__main__":
    # ATS without constraints: no updates
    stream_ts, constraints = read_data(args.stream_path, args.stream_name, args.constraint_path, 
                                    args.constraint_name, args.fformat, args.lc_filter)  
    ts_prediction(stream_ts, args.ts_model, args.update)

    # AATS including constraints: no updates
    stream_ats, constraints = read_data(args.stream_path, args.stream_name, args.constraint_path, 
                                    args.constraint_name, args.fformat, args.lc_filter)
    start = time.time()
    ats_prediction(stream_ats, args.ats_model, args.update)
    
    end = time.time()
    print(f"##########Retrain and prediction time for AATS: {end-start}##########") 