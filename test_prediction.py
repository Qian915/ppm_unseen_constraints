from classes.Attribute import Attribute
from classes.Event import Event
from classes.Trace import Trace
import pickle


picklefile = open('output/synthetic_augmented.pickle', 'rb')
augmented_ats = pickle.load(picklefile)
picklefile.close()

attributes = list()
attributes.append(Attribute('case:concept:name', "test"))
attributes.append(Attribute('concept:name', "A"))
attributes.append(Attribute('time:timestamp', "2022-09-14T16:10:45.227+02:00"))
attributes.append(Attribute('lifecycle:transition', "calling"))
events = list()
events.append(Event(attributes))
trace = Trace("test",events)
print(augmented_ats.online_predict(trace))


attributes = list()
attributes.append(Attribute('case:concept:name', "test"))
attributes.append(Attribute('concept:name', "B"))
attributes.append(Attribute('time:timestamp', "2022-09-14T16:10:45.227+02:00"))
attributes.append(Attribute('lifecycle:transition', "calling"))
events.append(Event(attributes))
trace = Trace("test",events)
print(augmented_ats.online_predict(trace))


attributes = list()
attributes.append(Attribute('case:concept:name', "test"))
attributes.append(Attribute('concept:name', "C"))
attributes.append(Attribute('time:timestamp', "2022-09-14T16:10:45.227+02:00"))
attributes.append(Attribute('lifecycle:transition', "calling"))
events.append(Event(attributes))
trace = Trace("test",events)
print(augmented_ats.online_predict(trace))